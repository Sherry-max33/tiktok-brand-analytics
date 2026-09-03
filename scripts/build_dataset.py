from __future__ import annotations
from pathlib import Path
import glob
import yaml

from tiktok_brand.etl.build_clean_table import build_clean_table
from tiktok_brand.etl.feature_table import build_feature_table, write_partitioned_parquet


def main() -> None:
    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    output = project_cfg["output"]
    raw_dir = Path(output["raw_dir"])
    clean_dir = Path(output.get("clean_dir", Path(output["processed_dir"]) / "clean"))
    feature_dir = Path(output.get("feature_dir", Path(output["processed_dir"]) / "feature"))
    tz = project_cfg["time"]["timezone"]

    clean_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = sorted(glob.glob(str(raw_dir / "*.jsonl")))
    clean_df = build_clean_table(
        raw_paths=raw_paths,
        accounts_cfg_path=output.get("accounts_cfg", "configs/accounts.yaml"),
        project_cfg_path="configs/project.yaml",
        hashtags_cfg_path=output.get("hashtags_cfg", "configs/hashtags.yaml"),
    )
    clean_out = clean_dir / "tiktok_videos.parquet"
    clean_df.to_parquet(clean_out, index=False)
    print(f"Wrote {len(clean_df):,} rows to {clean_out}")

    feature_df = build_feature_table(
        clean_df,
        tz=tz,
        taxonomy_cfg_path=output.get("taxonomy_cfg", "configs/taxonomy.yaml"),
    )
    out_partitioned = feature_dir / "feature_table"
    write_partitioned_parquet(feature_df, out_partitioned, partition_cols=["brand", "post_date"])
    print(f"Wrote feature table ({len(feature_df):,} rows) to {out_partitioned}/")


if __name__ == "__main__":
    main()
