"""Assign content_cluster_id from text embeddings (bottom-up clusters).

Unlike rule-based ``content_type`` (top-down taxonomy), clusters discover
cohesive caption-themes in embedding space.

Default: KMeans on multilingual BERT ``text_embedding`` (PCA optional).
Later you can swap/fuse with CLIP visual embeddings the same way.

Usage (repo root):
  PYTHONPATH=src python -m scripts.cluster_content --n-clusters 20
  PYTHONPATH=src python -m scripts.cluster_content --n-clusters 20 --pca 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tiktok_brand.etl.feature_table import write_partitioned_parquet


def _as_vector(x) -> np.ndarray | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, str):
        return None
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster videos by text embedding")
    parser.add_argument("--n-clusters", type=int, default=20, help="KMeans k")
    parser.add_argument(
        "--pca",
        type=int,
        default=50,
        help="PCA dims before KMeans (0 = no PCA)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=1,
        help="Reserved for future HDBSCAN; unused for KMeans",
    )
    args = parser.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    feature_dir = Path(project_cfg["output"].get("feature_dir", "data/processed/feature"))
    video_path = feature_dir / "feature_table.parquet"
    df = pd.read_parquet(video_path)
    if "content_cluster_id" not in df.columns:
        df["content_cluster_id"] = pd.NA

    vectors = []
    row_idxs = []
    for i, emb in enumerate(df["text_embedding"] if "text_embedding" in df.columns else []):
        v = _as_vector(emb)
        if v is None:
            continue
        vectors.append(v)
        row_idxs.append(i)

    if len(vectors) < max(2, int(args.n_clusters)):
        raise SystemExit(
            f"Not enough text embeddings to cluster "
            f"({len(vectors)} usable; need ≥ {args.n_clusters})."
        )

    X = np.vstack(vectors)
    # L2 normalize (SBERT usually already is)
    X = normalize(X)

    pca_dims = int(args.pca)
    if pca_dims > 0 and pca_dims < X.shape[1]:
        pca_dims = min(pca_dims, X.shape[0] - 1, X.shape[1])
        print(f"PCA → {pca_dims} dims on {X.shape[0]} rows", flush=True)
        X = PCA(n_components=pca_dims, random_state=int(args.seed)).fit_transform(X)
        X = normalize(X)

    k = int(args.n_clusters)
    print(f"KMeans k={k} on {X.shape[0]} × {X.shape[1]}", flush=True)
    km = KMeans(n_clusters=k, random_state=int(args.seed), n_init=10)
    labels = km.fit_predict(X)

    df["content_cluster_id"] = pd.NA
    for idx, lab in zip(row_idxs, labels):
        df.at[df.index[idx], "content_cluster_id"] = int(lab)

    df.to_parquet(video_path, index=False)
    write_partitioned_parquet(df, feature_dir / "feature_table", partition_cols=["brand"])

    counts = pd.Series(labels).value_counts().sort_index()
    print(f"Wrote content_cluster_id → {video_path}", flush=True)
    print(f"Cluster sizes: min={counts.min()} max={counts.max()} mean={counts.mean():.1f}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
