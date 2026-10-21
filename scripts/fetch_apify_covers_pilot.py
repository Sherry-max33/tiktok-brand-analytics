"""Pilot: Apify official HD covers for N videos → private Supabase/R2 cover store.

Only downloads covers (no MP4). Overwrites pilot keys ``covers/{video_id}.jpg``.

  PYTHONPATH=src:app python scripts/fetch_apify_covers_pilot.py --n 20
  PYTHONPATH=src:app python scripts/fetch_apify_covers_pilot.py --n 20 --upload
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "app"))

import catalog  # noqa: E402
from cover_store import backend, configured, put_cover  # noqa: E402
from frame_thumbs import frame_path, frames_root  # noqa: E402
from tiktok_brand.crawl.apify_video_client import (  # noqa: E402
    download_url_to_file,
    fetch_videos_by_post_urls,
    get_apify_token,
    video_id_from_apify_item,
)


def _sample_ids(n: int) -> list[str]:
    """Same pilot set as the frame/R2 upload: featured-with-frames, then library."""
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


def _post_url(video_id: str) -> str:
    card = catalog.get_video(video_id)
    if card and card.get("url"):
        return str(card["url"])
    # Fallback shape TikTok accepts
    return f"https://www.tiktok.com/@tiktok/video/{video_id}"


def _cover_urls(item: dict) -> list[str]:
    """Prefer Apify-hosted downloaded covers; then origin/original cover CDN URLs."""
    urls: list[str] = []
    for key in ("mediaUrls", "downloadedCovers", "covers"):
        val = item.get(key)
        if isinstance(val, list):
            urls.extend(str(u) for u in val if u)
        elif isinstance(val, str) and val.strip():
            urls.append(val.strip())
        elif isinstance(val, dict):
            urls.extend(str(u) for u in val.values() if u)

    video_meta = item.get("videoMeta") if isinstance(item.get("videoMeta"), dict) else {}
    for key in (
        "coverUrl",
        "originCover",
        "originalCover",
        "dynamicCover",
        "zoomCover",
    ):
        val = item.get(key) or video_meta.get(key)
        if isinstance(val, str) and val.startswith("http"):
            urls.append(val)
        elif isinstance(val, dict):
            # zoomCover often maps size → url
            for u in val.values():
                if isinstance(u, str) and u.startswith("http"):
                    urls.append(u)

    # Dedupe, prefer apify.com hosts (stable KV download)
    seen = set()
    ordered: list[str] = []
    for prefer_apify in (True, False):
        for u in urls:
            if u in seen:
                continue
            is_apify = "api.apify.com" in u
            if prefer_apify and not is_apify:
                continue
            if not prefer_apify and is_apify:
                continue
            seen.add(u)
            ordered.append(u)
    return ordered


def _to_jpeg(path: Path) -> bytes:
    """Normalize any downloaded cover to high-quality JPEG bytes."""
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        # Keep native resolution — this is the official cover, do not downscale.
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95, optimize=True, progressive=True)
        return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Push covers into the configured private store (Supabase/R2)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / ".cache" / "apify_covers_pilot",
        help="Local folder for downloaded covers (gitignored via .cache/)",
    )
    args = parser.parse_args()

    if not get_apify_token():
        print("ERROR: APIFY_API_TOKEN missing in .env")
        sys.exit(1)
    if args.upload and not configured():
        print("ERROR: cover store not configured (SUPABASE_* or R2_*)")
        sys.exit(1)

    ids = _sample_ids(args.n)
    urls = [_post_url(vid) for vid in ids]
    print(f"Apify cover-only fetch for {len(urls)} posts…")
    print("backend:", backend() if configured() else "(no upload)")

    items, meta = fetch_videos_by_post_urls(
        urls,
        download_videos=False,
        download_covers=True,
    )
    print(f"Apify returned {len(items)} items  run={meta.get('id') or meta}")

    args.out.mkdir(parents=True, exist_ok=True)
    by_id = {}
    for item in items:
        vid = video_id_from_apify_item(item)
        if vid:
            by_id[vid] = item

    ok = 0
    fail = []
    for vid in ids:
        item = by_id.get(vid)
        if not item:
            fail.append((vid, "not in Apify dataset"))
            continue
        cover_urls = _cover_urls(item)
        if not cover_urls:
            # dump keys once for debugging
            fail.append((vid, f"no cover urls; keys={sorted(item.keys())[:30]}"))
            continue
        dest = args.out / f"{vid}.bin"
        got = None
        last_err = None
        for u in cover_urls:
            try:
                download_url_to_file(u, dest)
                # Skip obvious non-images if tiny
                if dest.stat().st_size < 500:
                    last_err = RuntimeError(f"too small {dest.stat().st_size}")
                    continue
                got = dest
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if got is None:
            fail.append((vid, f"download failed: {last_err}"))
            continue
        try:
            jpeg = _to_jpeg(got)
        except Exception as exc:  # noqa: BLE001
            fail.append((vid, f"decode failed: {exc}"))
            continue
        jpg_path = args.out / f"{vid}.jpg"
        jpg_path.write_bytes(jpeg)
        print(f"OK  {vid}  {jpg_path.stat().st_size/1024:.0f}KB  from {cover_urls[0][:60]}…")
        if args.upload:
            put_cover(vid, jpeg, ext="jpg", content_type="image/jpeg")
            # Invalidate local cover-store cache so Streamlit re-fetches
            cache = REPO / ".cache" / "cover_store" / backend() / f"{vid}.jpg"
            if cache.exists():
                cache.unlink()
        ok += 1

    print(f"done {ok}/{len(ids)}")
    for vid, reason in fail:
        print(f"FAIL {vid}: {reason}")
    if fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
