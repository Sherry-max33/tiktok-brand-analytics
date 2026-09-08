"""Assign content_cluster_id from BERT text embeddings (K-Means).

Pipeline (recommended):
  L2-normalize → PCA (no whiten) → K-Means → pick k → human-label clusters

Why K-Means (not HDBSCAN) as the main scheme:
  - every video gets a cluster id
  - fixed k → easy Nike vs Adidas comparison
  - new rows can ``predict`` into existing centroids
  - fast, stable enough for a modeling feature
  - HDBSCAN often leaves many points as noise (-1)

Usage (repo root):
  # Original modeling embeddings:
  PYTHONPATH=src python -m scripts.cluster_content --sweep

  # Cleaned clustering embeddings (after enrich_cluster_embeddings):
  PYTHONPATH=src python -m scripts.cluster_content --sweep \\
    --embedding-col cluster_text_embedding --id-col content_cluster_id_clean \\
    --text-col cluster_embedding_text \\
    --profiles-out data/processed/feature/content_cluster_profiles_clean.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tiktok_brand.etl.feature_table import write_partitioned_parquet
from tiktok_brand.etl.text_prep import CLUSTER_STOPWORDS, is_cluster_stopword

DEFAULT_K_GRID = (8, 10, 12, 15, 20, 25)
_TOKEN_RE = re.compile(r"[a-z0-9_#]+", re.I)
# Profile-only stopwords (do not change embeddings / cluster assignments).
_PROFILE_TOKEN_STOP = CLUSTER_STOPWORDS | {"hashtags"}


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


def _load_matrix(df: pd.DataFrame, *, embedding_col: str) -> tuple[np.ndarray, list[int]]:
    if embedding_col not in df.columns:
        raise SystemExit(f"feature table missing {embedding_col}")
    vectors: list[np.ndarray] = []
    row_idxs: list[int] = []
    for i, emb in enumerate(df[embedding_col]):
        v = _as_vector(emb)
        if v is None:
            continue
        vectors.append(v)
        row_idxs.append(i)
    if len(vectors) < 8:
        raise SystemExit(f"Not enough embeddings in {embedding_col} ({len(vectors)})")
    return np.vstack(vectors), row_idxs


def _fit_pca(
    X_norm: np.ndarray, *, pca_dims: int, seed: int
) -> tuple[np.ndarray, object | None, float]:
    from sklearn.decomposition import PCA

    if pca_dims <= 0 or pca_dims >= X_norm.shape[1]:
        return X_norm, None, 1.0
    n_comp = min(pca_dims, X_norm.shape[0] - 1, X_norm.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed, whiten=False)
    X_pca = pca.fit_transform(X_norm)
    explained = float(pca.explained_variance_ratio_.sum())
    return X_pca, pca, explained


def _fit_kmeans(X_pca: np.ndarray, *, k: int, seed: int, n_init: int):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    labels = km.fit_predict(X_pca)
    return km, labels


def _silhouette(X_norm: np.ndarray, labels: np.ndarray, *, seed: int) -> float:
    from sklearn.metrics import silhouette_score

    # Score on L2 space with cosine (matches semantic intent); sample for speed.
    return float(
        silhouette_score(
            X_norm,
            labels,
            metric="cosine",
            sample_size=min(3000, len(X_norm)),
            random_state=seed,
        )
    )


def _sweep(
    X_norm: np.ndarray,
    X_pca: np.ndarray,
    *,
    k_values: list[int],
    seed: int,
    n_init: int,
) -> pd.DataFrame:
    rows = []
    for k in k_values:
        if k >= len(X_pca):
            continue
        _, labels = _fit_kmeans(X_pca, k=k, seed=seed, n_init=n_init)
        sizes = np.bincount(labels, minlength=k)
        score = _silhouette(X_norm, labels, seed=seed)
        # cheap stability: second seed size-rank correlation proxy via silhouette delta
        _, labels2 = _fit_kmeans(X_pca, k=k, seed=seed + 1, n_init=n_init)
        score2 = _silhouette(X_norm, labels2, seed=seed)
        rows.append(
            {
                "k": k,
                "silhouette": round(score, 4),
                "silhouette_seed2": round(score2, 4),
                "sil_delta": round(abs(score - score2), 4),
                "smallest": int(sizes.min()),
                "largest": int(sizes.max()),
                "size_ratio": round(float(sizes.max()) / max(1, float(sizes.min())), 2),
                "mean_size": round(float(sizes.mean()), 1),
            }
        )
    return pd.DataFrame(rows)


def _tokens(text: str) -> list[str]:
    out = []
    for t in _TOKEN_RE.findall(text or ""):
        tok = t.lower()
        if len(tok) <= 2:
            continue
        if tok in _PROFILE_TOKEN_STOP or is_cluster_stopword(tok):
            continue
        out.append(tok)
    return out


def _hashtag_list(val) -> list[str]:
    if val is None:
        return []
    try:
        if isinstance(val, float) and pd.isna(val):
            return []
    except (TypeError, ValueError):
        pass
    # numpy / list-like from parquet
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, str):
        return [val.lstrip("#").lower()] if val.strip() else []
    if isinstance(val, (list, tuple)):
        out = []
        for x in val:
            if x is None:
                continue
            try:
                if isinstance(x, float) and pd.isna(x):
                    continue
            except (TypeError, ValueError):
                pass
            text = str(x).strip()
            if text:
                out.append(text.lstrip("#").lower())
        return out
    return []


def _row_hashtags(row: pd.Series) -> list[str]:
    for key in ("normalized_hashtags", "hashtags", "hashtag_list"):
        if key not in row.index:
            continue
        tags = _hashtag_list(row.get(key))
        if tags:
            return [t for t in tags if not is_cluster_stopword(t)]
    return []


def _example_text(row: pd.Series, text_col: str) -> str:
    for key in (text_col, "cluster_embedding_text", "caption_clean", "embedding_text"):
        if key not in row.index:
            continue
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        text = str(val).strip()
        if text:
            return text
    return ""


def _write_profiles(
    df: pd.DataFrame,
    *,
    row_idxs: list[int],
    labels: np.ndarray,
    X_pca: np.ndarray,
    centers: np.ndarray,
    out_path: Path,
    text_col: str,
    top_n: int = 15,
) -> None:
    """Nearest-to-centroid captions + brand mix + keywords for human naming."""
    profiles = []
    for cid in range(centers.shape[0]):
        member_pos = np.where(labels == cid)[0]
        if len(member_pos) == 0:
            continue
        # distances in PCA space to this centroid
        diffs = X_pca[member_pos] - centers[cid]
        dists = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dists)[:top_n]
        near_pos = member_pos[order]
        near_rows = [row_idxs[p] for p in near_pos]

        brand_counts: Counter[str] = Counter()
        token_counts: Counter[str] = Counter()
        tag_counts: Counter[str] = Counter()
        eng = []
        examples = []
        for ridx in near_rows:
            row = df.iloc[ridx]
            brand = str(row.get("brand") or "unknown")
            brand_counts[brand] += 1
            cap = _example_text(row, text_col)
            for tok in _tokens(cap):
                token_counts[tok] += 1
            for tag in _row_hashtags(row):
                tag_counts[tag] += 1
            w = row.get("weighted_engagement_rate")
            if w is not None and not (isinstance(w, float) and pd.isna(w)):
                eng.append(float(w))
            examples.append(
                {
                    "video_id": str(row.get("video_id")),
                    "brand": brand,
                    "caption": cap[:240],
                    "weighted_engagement_rate": None if not eng else float(w) if w == w else None,
                }
            )

        # brand mix on full cluster, not only nearest
        full_brands: Counter[str] = Counter()
        full_eng = []
        for p in member_pos:
            row = df.iloc[row_idxs[p]]
            full_brands[str(row.get("brand") or "unknown")] += 1
            w = row.get("weighted_engagement_rate")
            if w is not None and not (isinstance(w, float) and pd.isna(w)):
                full_eng.append(float(w))

        profiles.append(
            {
                "cluster_id": int(cid),
                "size": int(len(member_pos)),
                "brand_mix": dict(full_brands),
                "mean_weighted_engagement_rate": (
                    float(np.mean(full_eng)) if full_eng else None
                ),
                "top_tokens": [t for t, _ in token_counts.most_common(15)],
                "top_hashtags": [t for t, _ in tag_counts.most_common(15)],
                "nearest_examples": examples,
                "suggested_label": "",
                "human_label": "",
                "label_notes": "",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote cluster profiles → {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-Means content clusters on BERT text embeddings"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Final k to fit (required with --write unless --sweep only)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Evaluate k grid (silhouette / size balance); does not write",
    )
    parser.add_argument(
        "--k-grid",
        default=",".join(str(k) for k in DEFAULT_K_GRID),
        help="Comma-separated k values for --sweep",
    )
    parser.add_argument(
        "--pca",
        type=int,
        default=50,
        help="PCA dims before KMeans (0 = skip PCA). Try 32/50/64.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument(
        "--embedding-col",
        default="text_embedding",
        help="Embedding column: text_embedding (original) or cluster_text_embedding (cleaned)",
    )
    parser.add_argument(
        "--id-col",
        default="content_cluster_id",
        help="Cluster id column to write (use content_cluster_id_clean for cleaned run)",
    )
    parser.add_argument(
        "--text-col",
        default="embedding_text",
        help="Caption column for profile examples / top_tokens",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write cluster id column into feature_table.parquet",
    )
    parser.add_argument(
        "--profiles",
        action="store_true",
        help="Dump nearest-caption profiles JSON for human naming",
    )
    parser.add_argument(
        "--profiles-out",
        type=Path,
        default=None,
        help="Profiles JSON path (default depends on embedding-col)",
    )
    args = parser.parse_args()

    from sklearn.preprocessing import normalize

    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    feature_dir = Path(project_cfg["output"].get("feature_dir", "data/processed/feature"))
    video_path = feature_dir / "feature_table.parquet"
    df = pd.read_parquet(video_path)

    embedding_col = str(args.embedding_col)
    id_col = str(args.id_col)
    text_col = str(args.text_col)
    profiles_out = args.profiles_out
    if profiles_out is None:
        if embedding_col == "cluster_text_embedding":
            profiles_out = Path("data/processed/feature/content_cluster_profiles_clean.json")
        else:
            profiles_out = Path("data/processed/feature/content_cluster_profiles.json")

    X, row_idxs = _load_matrix(df, embedding_col=embedding_col)
    X_norm = normalize(X, norm="l2")
    X_pca, _pca, explained = _fit_pca(
        X_norm, pca_dims=int(args.pca), seed=int(args.seed)
    )
    print(
        f"Embedding={embedding_col} id_col={id_col}\n"
        f"Matrix: n={X_norm.shape[0]} dim={X_norm.shape[1]} → "
        f"PCA={X_pca.shape[1]} (explained_var={explained:.3f}, whiten=False)",
        flush=True,
    )

    if args.sweep or args.n_clusters is None:
        grid = [int(x) for x in str(args.k_grid).split(",") if x.strip()]
        print("\n=== k sweep (silhouette on L2+cosine; KMeans on PCA) ===", flush=True)
        sweep_df = _sweep(
            X_norm,
            X_pca,
            k_values=grid,
            seed=int(args.seed),
            n_init=int(args.n_init),
        )
        print(sweep_df.to_string(index=False), flush=True)
        print(
            "\nNote: embedding silhouettes are often modest — "
            "prefer interpretable, balanced clusters over max score alone.",
            flush=True,
        )
        if not args.write:
            if args.n_clusters is None:
                print("\nPick a k, then re-run with --n-clusters K --write --profiles")
            return

    k = int(args.n_clusters)
    if k >= len(X_pca):
        raise SystemExit(f"k={k} too large for n={len(X_pca)}")

    print(f"\nFitting final KMeans k={k} n_init={args.n_init}", flush=True)
    km, labels = _fit_kmeans(
        X_pca, k=k, seed=int(args.seed), n_init=int(args.n_init)
    )
    sil = _silhouette(X_norm, labels, seed=int(args.seed))
    sizes = np.bincount(labels, minlength=k)
    print(
        f"silhouette={sil:.4f} sizes min={sizes.min()} max={sizes.max()} "
        f"ratio={sizes.max()/max(1,sizes.min()):.2f}",
        flush=True,
    )

    if args.profiles:
        _write_profiles(
            df,
            row_idxs=row_idxs,
            labels=labels,
            X_pca=X_pca,
            centers=km.cluster_centers_,
            out_path=profiles_out,
            text_col=text_col,
        )

    if not args.write:
        print("Dry-run only (no --write). Feature table unchanged.")
        return

    df[id_col] = pd.NA
    for idx, lab in zip(row_idxs, labels):
        df.at[df.index[idx], id_col] = int(lab)

    df.to_parquet(video_path, index=False)
    write_partitioned_parquet(df, feature_dir / "feature_table", partition_cols=["brand"])
    print(f"Wrote {id_col} → {video_path}", flush=True)
    print(pd.Series(labels).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
