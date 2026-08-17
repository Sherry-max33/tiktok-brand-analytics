from __future__ import annotations
from pathlib import Path
import yaml

from tiktok_brand.common.io import write_jsonl
from tiktok_brand.common.logging import get_logger
from tiktok_brand.crawl.user_crawler import crawl_user
from tiktok_brand.common.time import now_ts

log = get_logger("scripts.crawl_users")

def main() -> None:
    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    accounts_cfg = yaml.safe_load(Path("configs/accounts.yaml").read_text(encoding="utf-8"))
    tz = project_cfg["time"]["timezone"]
    per_user = project_cfg["sampling"].get(
        "videos_per_keyword_target",
        project_cfg["sampling"].get("per_user", 200),
    )
    raw_dir = Path(project_cfg["output"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    for brand, users in (accounts_cfg.get("official_accounts") or {}).items():
        for u in users:
            rows = crawl_user(username=u, count=per_user, tz_name=tz, brand=brand)
            if rows:
                out = raw_dir / f"tiktok_user_{brand}_{u}_{now_ts()}.jsonl"
                write_jsonl(out, rows)
                log.info("Wrote %s rows to %s", len(rows), out)

if __name__ == "__main__":
    main()
