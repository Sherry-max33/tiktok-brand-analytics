from __future__ import annotations
from pathlib import Path
import yaml

from tiktok_brand.common.io import write_jsonl
from tiktok_brand.common.logging import get_logger
from tiktok_brand.crawl.hashtag_crawler import crawl_hashtags_from_config

log = get_logger("scripts.crawl_hashtags")

def main() -> None:
    # For quick testing, use smaller sample size
    per_hashtag = 50  # Override config for faster testing
    log.info(f"Starting hashtag crawl with per_hashtag={per_hashtag}")
    crawl_hashtags_from_config(per_hashtag=per_hashtag)

if __name__ == "__main__":
    main()
