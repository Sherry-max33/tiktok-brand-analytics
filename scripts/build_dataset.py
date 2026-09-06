from __future__ import annotations

import glob
import os
import tempfile
from pathlib import Path

import yaml

from tiktok_brand.etl.build_clean_comments import build_clean_comments
from tiktok_brand.etl.build_clean_table import build_clean_table
from tiktok_brand.etl.comment_feature_table import (
    build_comment_feature_table,
    write_comment_feature_parquet,
)
from tiktok_brand.etl.feature_table import build_feature_table, write_partitioned_parquet
from tiktok_brand.etl.rule_config import load_feature_rules


def _bulk_rules_path(base_rules: str, enable_mt: bool) -> tuple[str, str | None]:
    """Return rules path; if MT disabled, write a temp override. Second value is temp path to delete."""
    if enable_mt:
        return base_rules, None
    cfg = load_feature_rules(base_rules)
    sent = dict(cfg.get("sentiment") or {})
    sent["mt_enabled"] = False
    cfg = dict(cfg)
    cfg["sentiment"] = sent
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name, tmp.name


def main() -> None:
    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    output = project_cfg["output"]
    raw_dir = Path(output["raw_dir"])
    clean_dir = Path(output.get("clean_dir", Path(output["processed_dir"]) / "clean"))
    feature_dir = Path(output.get("feature_dir", Path(output["processed_dir"]) / "feature"))
    tz = project_cfg["time"]["timezone"]
    base_rules = output.get("feature_rules_cfg", "configs/feature_rules.yaml")

    clean_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = sorted(glob.glob(str(raw_dir / "*.jsonl")))
    build_mt = os.environ.get("BUILD_MT", "0").strip() == "1"
    rules_path, tmp_rules = _bulk_rules_path(base_rules, enable_mt=build_mt)
    if not build_mt:
        print("BUILD_MT=0: sentiment MT disabled for bulk run (EN VADER only). Set BUILD_MT=1 to enable.")

    try:
        clean_df = build_clean_table(
            raw_paths=raw_paths,
            accounts_cfg_path=output.get("accounts_cfg", "configs/accounts.yaml"),
            project_cfg_path="configs/project.yaml",
            hashtags_cfg_path=output.get("hashtags_cfg", "configs/hashtags.yaml"),
        )
        clean_out = clean_dir / "tiktok_videos.parquet"
        clean_df.to_parquet(clean_out, index=False)
        print(f"Wrote {len(clean_df):,} video rows to {clean_out}")

        feature_df = build_feature_table(
            clean_df,
            tz=tz,
            taxonomy_cfg_path=output.get("taxonomy_cfg", "configs/taxonomy.yaml"),
            feature_rules_path=rules_path,
        )
        out_partitioned = feature_dir / "feature_table"
        # Partition by brand only — post_date creates >1024 hive partitions.
        write_partitioned_parquet(feature_df, out_partitioned, partition_cols=["brand"])
        flat_out = feature_dir / "feature_table.parquet"
        feature_df.to_parquet(flat_out, index=False)
        print(f"Wrote video feature table ({len(feature_df):,} rows) to {out_partitioned}/ and {flat_out}")

        comment_clean = build_clean_comments(raw_paths, video_clean_df=clean_df)
        comment_clean_out = clean_dir / "tiktok_comments.parquet"
        comment_clean.to_parquet(comment_clean_out, index=False)
        print(f"Wrote {len(comment_clean):,} comment rows to {comment_clean_out}")

        if len(comment_clean) > 0:
            comment_feature = build_comment_feature_table(
                comment_clean,
                sentiment_rules_path=rules_path,
                disable_mt=not build_mt,
            )
            comment_feature_out = feature_dir / "comment_feature_table.parquet"
            write_comment_feature_parquet(comment_feature, comment_feature_out)
            print(
                f"Wrote comment feature table ({len(comment_feature):,} rows) to {comment_feature_out}"
            )
        else:
            print("No comment raw files found; skipped comment feature table.")
    finally:
        if tmp_rules:
            Path(tmp_rules).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
