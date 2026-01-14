from __future__ import annotations
from pathlib import Path
import yaml

from tiktok_brand.common.io import write_jsonl
from tiktok_brand.common.logging import get_logger
from tiktok_brand.crawl.user_crawler import crawl_user

log = get_logger("scripts.crawl_users")

def main() -> None:
    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    accounts_cfg = yaml.safe_load(Path("configs/accounts.yaml").read_text(encoding="utf-8"))
    tz = project_cfg["time"]["timezone"]
    per_user = project_cfg["sampling"]["per_user"]
    raw_dir = Path(project_cfg["output"]["raw_dir"])

    usernames = []
    for _, users in (accounts_cfg.get("official_accounts") or {}).items():
        usernames.extend(users)

    for u in usernames:
        rows = crawl_user(username=u, count=per_user, tz_name=tz)
        out = raw_dir / f"tiktok_user_{u}.jsonl"
        write_jsonl(out, rows)
        log.info("Wrote %s rows to %s", len(rows), out)

if __name__ == "__main__":
    main()
