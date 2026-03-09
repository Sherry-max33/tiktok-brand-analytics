from __future__ import annotations
from pathlib import Path
import glob
import yaml

from tiktok_brand.etl.build_clean_table import build_clean_table
from tiktok_brand.etl.feature_table import build_feature_table, write_partitioned_parquet

def main() -> None:
    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    raw_dir = Path(project_cfg["output"]["raw_dir"])
    processed_dir = Path(project_cfg["output"]["processed_dir"])
    tz = project_cfg["time"]["timezone"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = sorted(glob.glob(str(raw_dir / "*.jsonl")))
    clean_df = build_clean_table(
        raw_paths=raw_paths,
        hashtags_cfg_path="configs/hashtags.yaml",
        accounts_cfg_path="configs/accounts.yaml",
        project_cfg_path="configs/project.yaml",
    )
    # Single-file output (backward compat)
    out_single = processed_dir / "tiktok_videos.parquet"
    clean_df.to_parquet(out_single, index=False)
    print(f"Wrote {len(clean_df):,} rows to {out_single}")

    # Feature table + partitioned Parquet
    feature_df = build_feature_table(clean_df, tz=tz)
    out_partitioned = processed_dir / "feature_table"
    write_partitioned_parquet(feature_df, out_partitioned, partition_cols=["brand", "post_date"])
    print(f"Wrote feature table ({len(feature_df):,} rows) to {out_partitioned}/")

if __name__ == "__main__":
    main()
