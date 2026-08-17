from pathlib import Path
import sys

import pandas as pd
import yaml

# 确保 src/ 在 Python 路径中，这样可以 import tiktok_brand 包
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiktok_brand.crawl.hashtag_crawler import crawl_hashtags_from_config
from tiktok_brand.etl.build_clean_table import build_clean_table
from tiktok_brand.etl.feature_table import build_feature_table


def run_hashtag_crawl(per_hashtag: int = 40) -> None:
    """
    小规模测试用：根据当前 configs/hashtags.yaml 爬少量 hashtag 视频。

    无 APIFY_API_TOKEN 时用 Apify 格式 mock；有 token 时走真实 crawl。
    输出 JSONL 与 data/raw 现有格式一致。
    """
    crawl_hashtags_from_config(per_hashtag=per_hashtag)


def run_etl() -> None:
    """
    从 data/raw 读取刚爬到的 JSONL，跑一遍 clean_table → feature_table，
    并各输出一个测试用 Parquet 文件，最后打印几列核心字段做 sanity check。
    """
    root = ROOT
    project_cfg = yaml.safe_load((root / "configs/project.yaml").read_text(encoding="utf-8"))
    raw_dir = root / project_cfg["output"]["raw_dir"]

    raw_paths = sorted(raw_dir.glob("tiktok_hashtag_*.jsonl"))
    if not raw_paths:
        print(f"No raw files found under {raw_dir}, please run crawl first.")
        return

    clean_df = build_clean_table(
        raw_paths=raw_paths,
        hashtags_cfg_path=str(root / "configs/hashtags.yaml"),
        accounts_cfg_path=str(root / "configs/accounts.yaml"),
        project_cfg_path=str(root / "configs/project.yaml"),
    )
    clean_out = root / "data/clean/clean_test.parquet"
    clean_out.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(clean_out, index=False)
    print(f"Wrote clean table to {clean_out} (rows={len(clean_df)})")

    feature_df = build_feature_table(clean_df)
    feature_out = root / "data/feature/feature_test.parquet"
    feature_out.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(feature_out, index=False)
    print(f"Wrote feature table to {feature_out} (rows={len(feature_df)})")

    cols = ["brand", "product_category", "content_type", "engagement_rate"]
    existing = [c for c in cols if c in feature_df.columns]
    print(feature_df[existing].head())


if __name__ == "__main__":
    # 1) 小规模爬 seed（mock 或 Apify），每个最多 40 条
    run_hashtag_crawl(per_hashtag=40)
    # 2) 跑一遍 clean → feature 并打印结果
    run_etl()

