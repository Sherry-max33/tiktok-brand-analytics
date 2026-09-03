from __future__ import annotations
from pathlib import Path
import yaml

from tiktok_brand.common.logging import get_logger
from tiktok_brand.crawl.video_hashtag_crawler import crawl_hashtags_from_config

log = get_logger("scripts.crawl_hashtags")


def main() -> None:
    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    per_hashtag = int(project_cfg["sampling"].get("videos_per_hashtag", 100))
    log.info("Starting hashtag crawl with per_hashtag=%s", per_hashtag)
    crawl_hashtags_from_config(per_hashtag=per_hashtag)


if __name__ == "__main__":
    main()
