"""
Crawl TikTok comments for selected videos via Apify.

Usage (project root):
  # Single video (mock if no APIFY_API_TOKEN)
  python scripts/crawl_comments.py --video-id 7391028884474547489 --comments-per-video 10

  # Multiple videos
  python scripts/crawl_comments.py --video-id 7391028884474547489 --video-id 7517962734051331350

  # Top N videos from feature table (after build_dataset.py)
  python scripts/crawl_comments.py --from-feature-table --limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from tiktok_brand.common.io import write_jsonl
from tiktok_brand.common.logging import get_logger
from tiktok_brand.common.time import now_ts
from tiktok_brand.crawl.comment_crawler import crawl_comments, crawl_video_comments

log = get_logger("scripts.crawl_comments")


def main() -> None:
    ap = argparse.ArgumentParser(description="Crawl TikTok comments for video(s)")
    ap.add_argument("--video-id", action="append", default=[], help="Video ID or URL (repeatable)")
    ap.add_argument(
        "--from-feature-table",
        action="store_true",
        help="Select top videos from processed feature table",
    )
    ap.add_argument("--limit", type=int, help="Max videos when using --from-feature-table")
    ap.add_argument("--comments-per-video", type=int, help="Comments to fetch per video")
    ap.add_argument("--batch-size", type=int, default=25, help="Videos per Apify run batch")
    args = ap.parse_args()

    project_cfg = yaml.safe_load((ROOT / "configs/project.yaml").read_text(encoding="utf-8"))
    tz = project_cfg["time"]["timezone"]
    raw_dir = Path(project_cfg["output"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    comments_per_video = args.comments_per_video or int(
        project_cfg.get("sampling", {}).get("comments_per_video", 100)
    )

    if args.from_feature_table:
        from tiktok_brand.crawl.comment_crawler import crawl_comments_from_config

        crawl_comments_from_config(limit=args.limit)
        return

    if not args.video_id:
        raise SystemExit("Provide --video-id (one or more) or --from-feature-table")

    ts = now_ts()
    if len(args.video_id) == 1:
        rows = crawl_video_comments(args.video_id[0], comments_per_video, tz)
        out = raw_dir / f"tiktok_comments_{args.video_id[0]}_{ts}.jsonl"
    else:
        rows = crawl_comments(
            args.video_id,
            comments_per_video,
            tz,
            batch_size=args.batch_size,
        )
        out = raw_dir / f"tiktok_comments_batch_{len(args.video_id)}_{ts}.jsonl"

    if not rows:
        raise SystemExit("No comment records returned. Check APIFY_API_TOKEN and video IDs.")

    write_jsonl(out, rows)
    log.info("Wrote %s comment records to %s", len(rows), out)

    sample = rows[0]
    print(
        "Sample:",
        {
            "video_id": sample.get("video_id"),
            "comment_id": sample.get("comment_id"),
            "comment_text": (sample.get("comment_text") or "")[:80],
            "comment_author_username": sample.get("comment_author_username"),
        },
    )


if __name__ == "__main__":
    main()
