"""
Smoke test: 小规模爬虫 + 全链路 ETL + 关键字段校验。

用法（项目根目录）:
  python scripts/smoke_test.py

- 先用 configs/hashtags.yaml 里当前 seed 爬一小批（per_hashtag=50）。
  目前仍是 mock；接上真实 TikTok API 后，同一条命令即对真数据做 smoke test。
- 再跑 raw → clean → feature，并检查：
  - 必要列存在、行数>0
  - brand / product_category / engagement_rate 等非全空、取值合理
"""
from pathlib import Path
import os
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiktok_brand.crawl.hashtag_crawler import crawl_hashtags_from_config
from tiktok_brand.etl.build_clean_table import build_clean_table
from tiktok_brand.etl.feature_table import build_feature_table


REQUIRED_FEATURE_COLS = [
    "video_id",
    "brand",
    "caption_text",
    "hashtags",
    "normalized_hashtags",
    "product_line",
    "product_category",
    "brand_style",
    "content_type",
    "engagement_rate",
    "view_count",
    "like_count",
    "crawled_at",
]
VALID_BRANDS = {"nike", "adidas", "both"}
VALID_PRODUCT_CATEGORIES = {"shoes", "apparel", "uncategorized"}


def _rate_non_null(s: pd.Series) -> float:
    if s is None:
        return 0.0
    return float(s.notna().mean()) if len(s) else 0.0


def _rate_non_empty_str(s: pd.Series) -> float:
    if s is None:
        return 0.0
    s2 = s.dropna().astype(str).str.strip()
    return float((s2 != "").mean()) if len(s) else 0.0


def _get_mode() -> str:
    """
    SMOKE_MODE:
      - mock (default): 宽松，保证管道可跑通即可
      - real: 严格，加入命中率/分布校验，适用于接上真实 API 后
    """
    return (os.environ.get("SMOKE_MODE") or "mock").strip().lower()


def run_crawl(per_hashtag: int = 50) -> None:
    crawl_hashtags_from_config(per_hashtag=per_hashtag)


def run_etl_and_return_feature_df():
    project_cfg = yaml.safe_load((ROOT / "configs/project.yaml").read_text(encoding="utf-8"))
    raw_dir = ROOT / project_cfg["output"]["raw_dir"]
    raw_paths = sorted(raw_dir.glob("tiktok_hashtag_*.jsonl"))
    if not raw_paths:
        raise FileNotFoundError(f"No tiktok_hashtag_*.jsonl under {raw_dir}. Run crawl first.")
    clean_df = build_clean_table(
        raw_paths=raw_paths,
        hashtags_cfg_path=str(ROOT / "configs/hashtags.yaml"),
        accounts_cfg_path=str(ROOT / "configs/accounts.yaml"),
        project_cfg_path=str(ROOT / "configs/project.yaml"),
    )
    feature_df = build_feature_table(clean_df)
    return feature_df, clean_df, len(raw_paths)


def validate_feature_df(df: pd.DataFrame) -> list[str]:
    errors = []
    mode = _get_mode()
    if df.empty:
        errors.append("Feature table is empty.")
        return errors
    missing = [c for c in REQUIRED_FEATURE_COLS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
    if "brand" in df.columns:
        valid = df["brand"].dropna().astype(str).str.lower().isin(VALID_BRANDS)
        invalid_brand = ~valid
        if invalid_brand.any():
            bad = df.loc[invalid_brand, "brand"].dropna().unique().tolist()
            errors.append(f"Unexpected brand values: {bad}")
    if "product_category" in df.columns:
        cats = df["product_category"].dropna().astype(str).str.lower().unique()
        bad = [c for c in cats if c not in VALID_PRODUCT_CATEGORIES]
        if bad:
            errors.append(f"Unexpected product_category values: {bad}")
    if "engagement_rate" in df.columns:
        rate = pd.to_numeric(df["engagement_rate"], errors="coerce")
        out_of_range = (rate.dropna() < 0) | (rate.dropna() > 1)
        if out_of_range.any():
            errors.append("engagement_rate has values outside [0, 1].")
    if "view_count" in df.columns:
        views = pd.to_numeric(df["view_count"], errors="coerce")
        if (views.dropna() < 0).any():
            errors.append("view_count has negative values.")

    # --- Business-ish sanity checks ---
    # mock 模式不强制命中率（mock caption/hashtags 很单一，避免误判失败）
    if mode == "real":
        # brand 分布：至少出现 nike 和 adidas（你当前 smoke seed 是 2+2）
        if "brand" in df.columns:
            brands = set(df["brand"].dropna().astype(str).str.lower().unique().tolist())
            if not {"nike", "adidas"}.issubset(brands):
                errors.append(f"Expected both nike and adidas in brand, got: {sorted(brands)}")

        # product_line 命中率（非空）
        if "product_line" in df.columns:
            hit = _rate_non_empty_str(df["product_line"])
            if hit < 0.10:
                errors.append(f"product_line hit-rate too low: {hit:.1%} (<10%)")

        # product_category 非 uncategorized 比例
        if "product_category" in df.columns:
            pc = df["product_category"].astype(str).str.lower()
            known = pc.isin({"shoes", "apparel"})
            known_rate = float(known.mean())
            if known_rate < 0.20:
                errors.append(f"product_category known-rate too low: {known_rate:.1%} (<20%)")

        # content_type 命中率
        if "content_type" in df.columns:
            hit = _rate_non_empty_str(df["content_type"])
            if hit < 0.10:
                errors.append(f"content_type hit-rate too low: {hit:.1%} (<10%)")

        # engagement_rate 非空比例（只在 view_count>0 的行上评估）
        if "engagement_rate" in df.columns and "view_count" in df.columns:
            views = pd.to_numeric(df["view_count"], errors="coerce").fillna(0)
            mask = views > 0
            if mask.any():
                non_null = _rate_non_null(df.loc[mask, "engagement_rate"])
                if non_null < 0.80:
                    errors.append(f"engagement_rate non-null too low: {non_null:.1%} (<80% for view_count>0)")
    return errors


def main() -> None:
    mode = _get_mode()
    print(f"Smoke test: crawl (small) + ETL + validation (mode={mode})")
    print("---")
    run_crawl(per_hashtag=50)
    print("---")
    feature_df, clean_df, num_raw_files = run_etl_and_return_feature_df()
    print(f"Raw files used: {num_raw_files}, clean rows: {len(clean_df)}, feature rows: {len(feature_df)}")
    print("---")
    errors = validate_feature_df(feature_df)
    if errors:
        print("SMOKE TEST FAILED")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("SMOKE TEST PASSED")
    print("  Required columns present, brand/product_category/engagement_rate look valid.")


if __name__ == "__main__":
    main()
