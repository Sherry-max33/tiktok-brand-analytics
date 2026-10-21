"""Upload local Apify HD covers from ``.cache/apify_covers/`` to R2.

Skips objects that already exist (same size). Safe to re-run.

  PYTHONPATH=app .venv/bin/python scripts/upload_apify_covers_r2.py
  PYTHONPATH=app .venv/bin/python scripts/upload_apify_covers_r2.py --limit 5 --verify
  PYTHONPATH=app .venv/bin/python scripts/upload_apify_covers_r2.py --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app"))

from cover_store import (  # noqa: E402
    CoverStoreError,
    backend,
    configured,
    get_cover,
    object_key,
    put_cover,
    r2_configured,
)

COVER_DIR = REPO / ".cache" / "apify_covers"


def _list_jpgs() -> list[Path]:
    if not COVER_DIR.is_dir():
        return []
    return sorted(p for p in COVER_DIR.glob("*.jpg") if p.is_file() and p.stat().st_size > 0)


def _already_remote(client, bucket: str, key: str, local_size: int) -> bool:
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return False
    return int(head.get("ContentLength") or 0) == local_size


def _upload_one(
    path: Path,
    *,
    skip_existing: bool,
    verify: bool,
    client,
    bucket: str,
) -> tuple[str, str, int]:
    """Return (status, video_id, bytes) where status is ok|skip|fail."""
    vid = path.stem
    data = path.read_bytes()
    key = object_key(vid, ext="jpg")
    if skip_existing and _already_remote(client, bucket, key, len(data)):
        return "skip", vid, len(data)
    try:
        put_cover(vid, data, ext="jpg", content_type="image/jpeg")
    except CoverStoreError as exc:
        return f"fail:{exc}", vid, len(data)
    if verify:
        cached = REPO / ".cache" / "cover_store" / "r2" / f"{vid}.jpg"
        if cached.exists():
            cached.unlink()
        try:
            got = get_cover(vid, ext="jpg", force_remote=True)
        except CoverStoreError as exc:
            return f"fail:verify {exc}", vid, len(data)
        if len(got) != len(data):
            return f"fail:size {len(got)}!={len(data)}", vid, len(data)
    return "ok", vid, len(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Upload at most N files (0=all)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify", action="store_true", help="Re-fetch after each upload (slow)")
    parser.add_argument("--no-skip", action="store_true", help="Re-upload even if remote exists")
    args = parser.parse_args()

    if not configured() or not r2_configured():
        print("ERROR: R2_* not configured in .env")
        sys.exit(1)

    print("backend:", backend())
    files = _list_jpgs()
    if args.limit > 0:
        files = files[: args.limit]
    print(f"candidates: {len(files)} from {COVER_DIR}")
    if not files:
        sys.exit(1)

    # Shared client for head_object skip checks
    from cover_store import _r2_client
    import llm

    client = _r2_client()
    bucket = llm.setting("R2_BUCKET")

    ok = skip = fail = 0
    t0 = time.perf_counter()
    workers = max(1, args.workers)

    def job(path: Path):
        return _upload_one(
            path,
            skip_existing=not args.no_skip,
            verify=args.verify,
            client=client,
            bucket=bucket,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, p): p for p in files}
        done = 0
        for fut in as_completed(futures):
            status, vid, nbytes = fut.result()
            done += 1
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"FAIL {vid} {status}")
            if done % 100 == 0 or done == len(files):
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed else 0
                print(
                    f"progress {done}/{len(files)}  ok={ok} skip={skip} fail={fail}  "
                    f"{rate:.1f}/s"
                )

    elapsed = time.perf_counter() - t0
    print(f"done ok={ok} skip={skip} fail={fail} in {elapsed:.1f}s  backend={backend()}")
    if fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
