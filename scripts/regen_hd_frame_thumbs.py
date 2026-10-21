"""Regenerate high-quality list thumbs from Drive frame_000.jpg (no Apify).

Writes:
  .cache/frame_thumbs/hd1/{video_id}.jpg
  .cache/frame_thumbs_review/index.html   ← open in a browser to judge sharpness

  PYTHONPATH=app python scripts/regen_hd_frame_thumbs.py
  PYTHONPATH=app python scripts/regen_hd_frame_thumbs.py --n 20
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app"))

import catalog  # noqa: E402
from frame_thumbs import (  # noqa: E402
    CACHE_DIR,
    CACHE_TAG,
    THUMB_JPEG_QUALITY,
    THUMB_MAX_EDGE,
    _ensure_thumb,
    frame_path,
    frames_root,
)

REVIEW_DIR = REPO / ".cache" / "frame_thumbs_review"


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


def _write_review(rows: list[dict]) -> Path:
    """Copy thumbs next to index.html so file:// and http:// both work in Chrome."""
    import shutil

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = REVIEW_DIR / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        vid = row["video_id"]
        src = CACHE_DIR / f"{vid}.jpg"
        if src.is_file():
            shutil.copy2(src, img_dir / f"{vid}.jpg")
        cards.append(
            f"""
            <figure>
              <img src="img/{html.escape(vid)}.jpg" alt="{html.escape(vid)}" loading="lazy">
              <figcaption>
                <code>{html.escape(vid)}</code><br>
                thumb {row['thumb_kb']:.0f} KB · src {row['src_kb']:.0f} KB ·
                {row['width']}×{row['height']}
              </figcaption>
            </figure>
            """
        )
    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>HD frame thumbs review ({CACHE_TAG})</title>
<style>
  body {{ margin: 24px; font: 14px/1.4 -apple-system, Helvetica, sans-serif; background: #f4f1ec; color: #1f1e1c; }}
  h1 {{ font-weight: 500; font-size: 22px; }}
  .meta {{ color: #6f6a64; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 20px; }}
  figure {{ margin: 0; background: #fff; border: 1px solid #e7e2da; }}
  img {{ display: block; width: 100%; aspect-ratio: 9/16; object-fit: cover; background: #111; }}
  figcaption {{ padding: 10px 12px; font-size: 12px; color: #55524d; word-break: break-all; }}
</style>
</head><body>
<h1>HD frame thumbs · {html.escape(CACHE_TAG)}</h1>
<p class="meta">max_edge={THUMB_MAX_EDGE} · jpeg_quality={THUMB_JPEG_QUALITY} ·
n={len(rows)} · source=frame_000.jpg (not TikTok official cover)</p>
<p class="meta">Preferred: <code>http://127.0.0.1:8765/</code>
(from <code>.cache/frame_thumbs_review</code> via <code>python -m http.server 8765</code>)</p>
<div class="grid">{"".join(cards)}</div>
</body></html>
"""
    out = REVIEW_DIR / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    root = frames_root()
    print("FRAMES_ROOT:", root)
    ids = _sample_ids(args.n)
    print(f"regenerating {len(ids)} thumbs → {CACHE_DIR}")

    rows = []
    from PIL import Image

    for vid in ids:
        src = frame_path(vid, root=root)
        assert src is not None
        path = _ensure_thumb(vid, root=root, force=True)
        if path is None:
            print("FAIL", vid)
            continue
        with Image.open(path) as im:
            w, h = im.size
        rows.append(
            {
                "video_id": vid,
                "source": str(src),
                "thumb_kb": path.stat().st_size / 1024,
                "src_kb": src.stat().st_size / 1024,
                "width": w,
                "height": h,
            }
        )
        print(
            f"OK {vid}  {w}x{h}  thumb={path.stat().st_size/1024:.0f}KB  "
            f"src={src.stat().st_size/1024:.0f}KB"
        )

    review = _write_review(rows)
    print("review:", review)
    print("open:", f"file://{review}")


if __name__ == "__main__":
    main()
