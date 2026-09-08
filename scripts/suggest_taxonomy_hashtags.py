"""Suggest taxonomy.yaml expansions from high-frequency unmapped hashtags.

Does NOT write configs/taxonomy.yaml. Review the CSV, then edit YAML yourself.

Usage (repo root):
  PYTHONPATH=src python -m scripts.suggest_taxonomy_hashtags
  PYTHONPATH=src python -m scripts.suggest_taxonomy_hashtags --top 80
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
import yaml

from tiktok_brand.etl.taxonomy_rules import load_taxonomy
from tiktok_brand.etl.text_prep import CLUSTER_STOPWORDS

# Brand / traffic tags that should not become style/line map keys.
_SKIP_TAGS = CLUSTER_STOPWORDS | {
    "nike",
    "adidas",
    "nikes",
    "adidass",
    "justdoit",
    "impossibleisnothing",
}


def _as_list(val: Any) -> List[Any]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return val
    try:
        import numpy as np

        if isinstance(val, np.ndarray):
            return val.tolist()
    except ImportError:
        pass
    return []


def _norm_tag(tag: Any) -> str:
    return str(tag).strip().lstrip("#").lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frequency scan of hashtags not in taxonomy maps (suggest-only)"
    )
    parser.add_argument("--top", type=int, default=100, help="Top unmapped tags to keep per view")
    parser.add_argument("--min-count", type=int, default=3, help="Min video frequency to report")
    parser.add_argument("--feature-table", default=None)
    parser.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    feature_dir = Path(project_cfg["output"].get("feature_dir", "data/processed/feature"))
    video_path = Path(args.feature_table) if args.feature_table else feature_dir / "feature_table.parquet"
    out_dir = Path(args.out_dir) if args.out_dir else feature_dir / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)

    load_taxonomy.cache_clear()
    tax = load_taxonomy(str(args.taxonomy))
    style_keys: Set[str] = {
        str(k).lower()
        for k in (
            tax.get("seed_style_map")
            or tax.get("account_style_map")
            or tax.get("brand_style_map")
            or {}
        )
    }
    line_keys: Set[str] = {str(k).lower() for k in (tax.get("product_line_map") or {})}
    cat_keys: Set[str] = {str(k).lower() for k in (tax.get("product_category_map") or {})}
    mapped_any = style_keys | line_keys | cat_keys

    df = pd.read_parquet(video_path)
    tag_col = "normalized_hashtags" if "normalized_hashtags" in df.columns else "hashtags"
    if tag_col not in df.columns:
        raise SystemExit("feature table missing hashtag columns")

    freq: Counter[str] = Counter()
    brand_freq: Dict[str, Counter[str]] = {}
    for _, row in df.iterrows():
        brand = str(row.get("brand") or "unknown")
        if brand not in brand_freq:
            brand_freq[brand] = Counter()
        seen_row: Set[str] = set()
        for t in _as_list(row.get(tag_col)):
            key = _norm_tag(t)
            if not key or key in seen_row:
                continue
            seen_row.add(key)
            freq[key] += 1
            brand_freq[brand][key] += 1

    rows: List[Dict[str, Any]] = []
    skipped_noise = 0
    for tag, count in freq.most_common():
        if count < int(args.min_count):
            continue
        if tag in _SKIP_TAGS:
            skipped_noise += 1
            continue
        in_style = tag in style_keys
        in_line = tag in line_keys
        in_cat = tag in cat_keys
        if in_style and in_line and in_cat:
            continue
        # Prefer reporting tags missing at least one of style/line (category alone is weaker signal)
        if tag in mapped_any and in_style and in_line:
            continue
        rows.append(
            {
                "hashtag": tag,
                "video_count": count,
                "in_seed_style_map": in_style,
                "in_product_line_map": in_line,
                "in_product_category_map": in_cat,
                "suggest_for": (
                    ",".join(
                        [
                            name
                            for name, hit in (
                                ("seed_style_map", not in_style),
                                ("product_line_map", not in_line),
                                ("product_category_map", not in_cat),
                            )
                            if hit
                        ]
                    )
                ),
                **{f"count_{b}": brand_freq.get(b, Counter()).get(tag, 0) for b in sorted(brand_freq)},
            }
        )

    # Keep unmapped-first ordering, then by frequency
    rows.sort(
        key=lambda r: (
            int(r["in_seed_style_map"]) + int(r["in_product_line_map"]) + int(r["in_product_category_map"]),
            -int(r["video_count"]),
        )
    )
    out_df = pd.DataFrame(rows[: int(args.top)])
    out_path = out_dir / "taxonomy_unmapped_hashtags.csv"
    out_df.to_csv(out_path, index=False)

    fully_unmapped = sum(
        1
        for r in rows
        if not r["in_seed_style_map"]
        and not r["in_product_line_map"]
        and not r["in_product_category_map"]
    )
    print(f"Unique hashtags: {len(freq)}")
    print(f"Skipped brand/traffic noise tags: {skipped_noise}")
    print(f"Candidates (min_count>={args.min_count}, partial/full unmapped): {len(rows)}")
    print(f"Fully unmapped among candidates: {fully_unmapped}")
    print(f"Wrote top {len(out_df)} → {out_path}")
    print("Next: review CSV → hand-edit taxonomy.yaml (no auto-write).")


if __name__ == "__main__":
    main()
