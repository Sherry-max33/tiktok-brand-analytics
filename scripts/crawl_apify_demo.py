"""
Small Apify crawl demo.

Usage (project root):
  export APIFY_API_TOKEN=apify_api_...
  python scripts/crawl_apify_demo.py --hashtag nike --count 5
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
from tiktok_brand.common.time import now_ts
from tiktok_brand.crawl.video_hashtag_crawler import crawl_hashtag
from tiktok_brand.crawl.video_user_crawler import crawl_user


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hashtag", help="Seed hashtag to crawl via Apify")
    ap.add_argument("--user", help="Profile username to crawl via Apify")
    ap.add_argument("--brand", default="", help="Brand label: nike or adidas")
    ap.add_argument("--count", type=int, default=5, help="Max videos to fetch")
    args = ap.parse_args()

    if bool(args.hashtag) == bool(args.user):
        raise SystemExit("Provide exactly one of --hashtag or --user")

    project_cfg = yaml.safe_load((ROOT / "configs/project.yaml").read_text(encoding="utf-8"))
    tz = project_cfg["time"]["timezone"]
    raw_dir = Path(project_cfg["output"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    ts = now_ts()
    if args.hashtag:
        rows = crawl_hashtag(args.hashtag, args.count, tz, brand=args.brand)
        out = raw_dir / f"tiktok_hashtag_{args.brand or 'unknown'}_{args.hashtag}_{ts}.jsonl"
    else:
        rows = crawl_user(args.user, args.count, tz, brand=args.brand or None)
        out = raw_dir / f"tiktok_user_{args.brand or 'unknown'}_{args.user}_{ts}.jsonl"

    if not rows:
        raise SystemExit("No records returned. Check APIFY_API_TOKEN and Apify quota.")

    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} records to {out}")

    sample = rows[0]
    print(
        "Sample:",
        {
            "video_id": sample.get("video_id"),
            "author_username": sample.get("author_username"),
            "video_duration_sec": sample.get("video_duration_sec"),
            "view_count": sample.get("view_count"),
            "hashtags": sample.get("hashtags")[:5],
        },
    )


if __name__ == "__main__":
    main()
