"""Recompute content_type + social_mechanic on the feature table (no full rebuild).

Usage:
  PYTHONPATH=src python -m scripts.refresh_content_labels
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from tiktok_brand.etl.content_type_rules import infer_content_types
from tiktok_brand.etl.feature_table import write_partitioned_parquet
from tiktok_brand.etl.rule_config import (
    get_content_type_rules,
    get_social_mechanic_rules,
    load_feature_rules,
)
from tiktok_brand.etl.social_mechanic_rules import infer_social_mechanics


def main() -> None:
    # Reload YAML after edits (lru_cache)
    load_feature_rules.cache_clear()
    get_content_type_rules.cache_clear()
    get_social_mechanic_rules.cache_clear()

    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    feature_dir = Path(project_cfg["output"].get("feature_dir", "data/processed/feature"))
    video_path = feature_dir / "feature_table.parquet"
    df = pd.read_parquet(video_path)

    caption = df["caption_raw"] if "caption_raw" in df.columns else df.get("caption_text")
    if caption is None:
        raise SystemExit("feature table missing caption_raw/caption_text")

    caps = caption.fillna("").astype(str)
    df["content_type"] = [infer_content_types(c) for c in caps.tolist()]
    df["social_mechanic"] = [infer_social_mechanics(c) for c in caps.tolist()]

    n = len(df)
    ct_hit = sum(1 for x in df["content_type"] if isinstance(x, list) and len(x) > 0)
    sm_hit = sum(1 for x in df["social_mechanic"] if isinstance(x, list) and len(x) > 0)
    print(f"content_type non-empty: {ct_hit}/{n} ({100*ct_hit/n:.1f}%)")
    print(f"social_mechanic non-empty: {sm_hit}/{n} ({100*sm_hit/n:.1f}%)")

    from collections import Counter

    ct_counts: Counter[str] = Counter()
    sm_counts: Counter[str] = Counter()
    for labels in df["content_type"]:
        if isinstance(labels, list):
            ct_counts.update(labels)
    for labels in df["social_mechanic"]:
        if isinstance(labels, list):
            sm_counts.update(labels)
    print("content_type label counts:", dict(ct_counts.most_common()))
    print("social_mechanic label counts:", dict(sm_counts.most_common()))

    df.to_parquet(video_path, index=False)
    write_partitioned_parquet(df, feature_dir / "feature_table", partition_cols=["brand"])
    print(f"Wrote → {video_path}")


if __name__ == "__main__":
    main()
