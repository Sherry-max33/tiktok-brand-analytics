from __future__ import annotations
from pathlib import Path
import yaml

from tiktok_brand.common.io import write_jsonl
from tiktok_brand.common.logging import get_logger
from tiktok_brand.crawl.hashtag_crawler import crawl_hashtag

log = get_logger("scripts.crawl_hashtags")

def main() -> None:
    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    hashtags_cfg = yaml.safe_load(Path("configs/hashtags.yaml").read_text(encoding="utf-8"))
    tz = project_cfg["time"]["timezone"]
    per_hashtag = project_cfg["sampling"]["per_hashtag"]
    raw_dir = Path(project_cfg["output"]["raw_dir"])

    for brand in ["nike","adidas"]:
        for tag in hashtags_cfg.get(brand, []):
            rows = crawl_hashtag(seed_hashtag=tag, count=per_hashtag, tz_name=tz)
            out = raw_dir / f"tiktok_hashtag_{brand}_{tag}.jsonl"
            write_jsonl(out, rows)
            log.info("Wrote %s rows to %s", len(rows), out)

if __name__ == "__main__":
    main()
