"""Build cluster_embedding_text + cluster_text_embedding (leave modeling embeddings intact).

Usage (repo root):
  PYTHONPATH=src python -m scripts.enrich_cluster_embeddings
  PYTHONPATH=src python -m scripts.enrich_cluster_embeddings --force   # re-encode all
  PYTHONPATH=src python -m scripts.enrich_cluster_embeddings --text-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from tiktok_brand.embeddings.caption_embed import compute_text_embeddings
from tiktok_brand.etl.feature_table import write_partitioned_parquet
from tiktok_brand.etl.text_prep import build_cluster_embedding_text


def _as_list(val) -> list:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write cluster_embedding_text / cluster_text_embedding without touching text_embedding"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only rebuild cluster_embedding_text (skip SBERT encode)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even when cluster_text_embedding already exists",
    )
    parser.add_argument(
        "--rules",
        default="configs/feature_rules.yaml",
        help="Feature rules (reuses text_embedding model settings)",
    )
    args = parser.parse_args()

    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    feature_dir = Path(project_cfg["output"].get("feature_dir", "data/processed/feature"))
    video_path = feature_dir / "feature_table.parquet"
    df = pd.read_parquet(video_path)

    # Prefer existing caption_clean; fall back to raw caption path inside builder
    captions = df["caption_raw"] if "caption_raw" in df.columns else df.get("caption_text")
    caption_clean = df["caption_clean"] if "caption_clean" in df.columns else None
    hashtags = df["hashtags"] if "hashtags" in df.columns else df.get("hashtag_list")

    texts: list[str] = []
    for i in range(len(df)):
        cap = None if captions is None else captions.iloc[i]
        clean = None if caption_clean is None else caption_clean.iloc[i]
        tags = _as_list(None if hashtags is None else hashtags.iloc[i])
        texts.append(
            build_cluster_embedding_text(
                cap,
                tags,
                caption_clean=None if clean is None or (isinstance(clean, float) and pd.isna(clean)) else str(clean),
            )
        )

    df["cluster_embedding_text"] = texts
    nonempty = sum(1 for t in texts if t and t.strip())
    print(
        f"cluster_embedding_text: {nonempty}/{len(df)} non-empty "
        f"(embedding_text / text_embedding untouched)",
        flush=True,
    )

    if args.text_only:
        df.to_parquet(video_path, index=False)
        write_partitioned_parquet(df, feature_dir / "feature_table", partition_cols=["brand"])
        print(f"Wrote cluster_embedding_text → {video_path}", flush=True)
        return

    # Encode: skip rows that already have vectors unless --force
    need_idx: list[int] = []
    need_texts: list[str] = []
    existing = df["cluster_text_embedding"] if "cluster_text_embedding" in df.columns else None
    for i, text in enumerate(texts):
        if not args.force and existing is not None:
            prev = existing.iloc[i]
            if prev is not None and not (isinstance(prev, float) and pd.isna(prev)):
                try:
                    if len(prev) > 0:
                        continue
                except TypeError:
                    pass
        need_idx.append(i)
        need_texts.append(text)

    print(f"Encoding cluster_text_embedding for {len(need_idx)}/{len(df)} rows…", flush=True)
    if need_idx:
        rows = compute_text_embeddings(need_texts, rules_path=args.rules)
        if "cluster_text_embedding" not in df.columns:
            df["cluster_text_embedding"] = None
        if "cluster_embedding_method" not in df.columns:
            df["cluster_embedding_method"] = None
        if "cluster_embedding_model" not in df.columns:
            df["cluster_embedding_model"] = None
        for i, row in zip(need_idx, rows):
            df.at[df.index[i], "cluster_text_embedding"] = row["text_embedding"]
            df.at[df.index[i], "cluster_embedding_method"] = row["embedding_method"]
            df.at[df.index[i], "cluster_embedding_model"] = row["embedding_model"]

    n_ok = sum(
        1
        for v in df["cluster_text_embedding"]
        if v is not None and not (isinstance(v, float) and pd.isna(v))
    )
    df.to_parquet(video_path, index=False)
    write_partitioned_parquet(df, feature_dir / "feature_table", partition_cols=["brand"])
    print(f"cluster_text_embedding ready: {n_ok}/{len(df)} → {video_path}", flush=True)


if __name__ == "__main__":
    main()
