"""Upload 20 local HD frame thumbs to the private cover store (pipeline pilot).

Uses R2 when ``R2_*`` is set, otherwise Supabase Storage (existing SUPABASE_*).

  PYTHONPATH=app python scripts/upload_cover_pilot.py
  PYTHONPATH=app python scripts/upload_cover_pilot.py --n 20 --verify
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app"))

import catalog  # noqa: E402
from cover_store import (  # noqa: E402
    CoverStoreError,
    backend,
    configured,
    get_cover,
    put_cover,
)
from frame_thumbs import CACHE_DIR, _ensure_thumb, frame_path, frames_root  # noqa: E402


def _sample_ids(n: int) -> list[str]:
    root = frames_root()
    ids: list[str] = []
    for card in catalog.featured_videos():
        vid = str(card["video_id"])
        if frame_path(vid, root=root) and vid not in ids:
            ids.append(vid)
    for vid in catalog.load_library()["video_id"].astype(str):
        if vid in ids:
            continue
        if frame_path(vid, root=root):
            ids.append(vid)
        if len(ids) >= n:
            break
    return ids[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--verify", action="store_true", help="Re-download each object after upload")
    args = parser.parse_args()

    if not configured():
        print("ERROR: no cover store. Set R2_* or SUPABASE_URL + SUPABASE_SECRET_KEY in .env")
        sys.exit(1)

    print("backend:", backend())
    ids = _sample_ids(args.n)
    print(f"uploading {len(ids)} covers from {CACHE_DIR} / frames")

    ok = 0
    t0 = time.perf_counter()
    for vid in ids:
        path = _ensure_thumb(vid, force=False)
        if path is None or not path.is_file():
            print("SKIP no thumb", vid)
            continue
        data = path.read_bytes()
        try:
            key = put_cover(vid, data, ext="jpg", content_type="image/jpeg")
        except CoverStoreError as exc:
            print("FAIL upload", vid, exc)
            continue
        if args.verify:
            # Drop local copy so the next read hits the cloud.
            cached = REPO / ".cache" / "cover_store" / backend() / f"{vid}.jpg"
            if cached.exists():
                cached.unlink()
            try:
                got = get_cover(vid, ext="jpg", force_remote=True)
            except CoverStoreError as exc:
                print("FAIL verify", vid, exc)
                continue
            if not got:
                print("FAIL verify empty", vid)
                continue
            print(f"OK  {vid}  key={key}  bytes={len(data)}  remote={len(got)}")
        else:
            print(f"OK  {vid}  key={key}  bytes={len(data)}")
        ok += 1

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"done {ok}/{len(ids)} in {elapsed:.0f} ms  backend={backend()}")
    if ok < len(ids):
        sys.exit(2)


if __name__ == "__main__":
    main()
