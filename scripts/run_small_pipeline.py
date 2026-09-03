from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiktok_brand.crawl.video_hashtag_crawler import crawl_hashtags_from_config
from tiktok_brand.etl.build_clean_table import build_clean_table
from tiktok_brand.etl.feature_table import build_feature_table


def run_hashtag_crawl(per_hashtag: int = 40) -> None:
    """Small test crawl; writes JSONL under data/raw."""
    crawl_hashtags_from_config(per_hashtag=per_hashtag)


def run_etl() -> None:
    """raw JSONL → clean + feature parquet under data/processed/."""
    project_cfg = yaml.safe_load((ROOT / "configs/project.yaml").read_text(encoding="utf-8"))
    output = project_cfg["output"]
    raw_dir = ROOT / output["raw_dir"]
    clean_dir = ROOT / output.get("clean_dir", "data/processed/clean")
    feature_test_path = ROOT / output["processed_dir"] / "feature" / "feature_test.parquet"

    raw_paths = sorted(raw_dir.glob("tiktok_hashtag_*.jsonl"))
    if not raw_paths:
        print(f"No raw files found under {raw_dir}, please run crawl first.")
        return

    clean_df = build_clean_table(
        raw_paths=raw_paths,
        accounts_cfg_path=str(ROOT / "configs/accounts.yaml"),
        project_cfg_path=str(ROOT / "configs/project.yaml"),
        hashtags_cfg_path=str(ROOT / "configs/hashtags.yaml"),
    )
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_out = clean_dir / "clean_test.parquet"
    clean_df.to_parquet(clean_out, index=False)
    print(f"Wrote clean table to {clean_out} (rows={len(clean_df)})")

    feature_df = build_feature_table(clean_df)
    feature_test_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(feature_test_path, index=False)
    print(f"Wrote feature table to {feature_test_path} (rows={len(feature_df)})")

    cols = ["brand", "product_categories", "content_type", "weighted_engagement_rate"]
    existing = [c for c in cols if c in feature_df.columns]
    print(feature_df[existing].head())


if __name__ == "__main__":
    run_hashtag_crawl(per_hashtag=40)
    run_etl()
