"""Discover comment topics for Theme 4 (NMF baseline + SBERT clusters).

Discovery finds semantic conversation patterns. Clusters are NOT final business
topics. Final labels should come from comment-level taxonomy classification:

  PYTHONPATH=src python -m scripts.discover_comment_topics --classify-lexicon

Optional cluster→topic mapping remains as a discovery assist / audit only.

Usage (repo root):
  PYTHONPATH=src python -m scripts.discover_comment_topics
  PYTHONPATH=src python -m scripts.discover_comment_topics --method both
  PYTHONPATH=src python -m scripts.discover_comment_topics --classify-lexicon
  PYTHONPATH=src python -m scripts.discover_comment_topics --apply-mapping PATH  # legacy assist

Outputs under data/processed/feature/audits/:
  comment_topic_nmf_summary.json
  comment_topic_cluster_cards.jsonl
  comment_topic_mapping_stub.yaml
  comment_feature_table.parquet  # topic_cluster_id; comment_topic if classify/map
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from tiktok_brand.analysis.sentiment_topics import (
    apply_cluster_mapping,
    apply_lexicon_topics,
    load_cluster_mapping,
    load_topic_taxonomy,
    topic_id_to_label,
)


def _comment_text(row: pd.Series) -> str:
    for col in ("caption_en", "comment_text_clean", "comment_text"):
        t = str(row.get(col) or "").strip()
        if t:
            return t
    return ""


def _as_embedding(val: Any) -> Optional[np.ndarray]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, np.ndarray):
        return val.astype(np.float32)
    if isinstance(val, (list, tuple)):
        try:
            return np.asarray(val, dtype=np.float32)
        except Exception:
            return None
    return None


def _top_terms(vectorizer: TfidfVectorizer, model: NMF, n_terms: int = 12) -> Dict[int, List[str]]:
    feature_names = np.array(vectorizer.get_feature_names_out())
    out: Dict[int, List[str]] = {}
    for i, comp in enumerate(model.components_):
        idx = np.argsort(comp)[::-1][:n_terms]
        out[i] = [str(feature_names[j]) for j in idx]
    return out


def run_nmf(
    texts: Sequence[str],
    *,
    n_topics: int = 12,
    max_features: int = 5000,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[int, List[str]], Dict[str, Any]]:
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.6,
        stop_words="english",
    )
    X = vec.fit_transform(texts)
    k = min(n_topics, max(2, X.shape[0] // 50))
    model = NMF(
        n_components=k,
        init="nndsvda",
        random_state=random_state,
        max_iter=400,
    )
    W = model.fit_transform(X)
    labels = W.argmax(axis=1).astype(int)
    terms = _top_terms(vec, model)
    meta = {"n_topics": k, "n_docs": int(X.shape[0]), "n_features": int(X.shape[1])}
    return labels, terms, meta


def run_embedding_kmeans(
    embeddings: np.ndarray,
    texts: Sequence[str],
    *,
    n_clusters: int = 12,
    random_state: int = 42,
    top_n_words: int = 12,
) -> Tuple[np.ndarray, Dict[int, List[str]], Dict[str, Any]]:
    """SBERT vectors → MiniBatchKMeans; top words via per-cluster TF-IDF."""
    k = min(n_clusters, max(2, len(texts) // 80))
    km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=1024, n_init=10)
    labels = km.fit_predict(embeddings)

    # c-TF-IDF style: TF-IDF on docs, average weight by cluster membership
    vec = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.7,
        stop_words="english",
    )
    X = vec.fit_transform(texts)
    feat = np.array(vec.get_feature_names_out())
    terms: Dict[int, List[str]] = {}
    for c in range(k):
        mask = labels == c
        if not mask.any():
            terms[c] = []
            continue
        centroid = np.asarray(X[mask].mean(axis=0)).ravel()
        idx = np.argsort(centroid)[::-1][:top_n_words]
        terms[c] = [str(feat[i]) for i in idx if centroid[i] > 0]
    meta = {"n_clusters": k, "n_docs": int(len(texts)), "algorithm": "minibatch_kmeans_sbert"}
    return labels.astype(int), terms, meta


def _seed_guess(top_words: Sequence[str], taxonomy: Dict[str, Any]) -> str:
    seeds = taxonomy.get("seed_keywords") or {}
    scores: Counter = Counter()
    bag = " ".join(top_words).lower()
    for topic_id, kws in seeds.items():
        for kw in kws:
            if kw.lower() in bag:
                scores[topic_id] += 1
    if not scores:
        return "other_unclear"
    return scores.most_common(1)[0][0]


def build_cards(
    labels: np.ndarray,
    terms: Dict[int, List[str]],
    texts: Sequence[str],
    brands: Sequence[str],
    *,
    method: str,
    taxonomy: Dict[str, Any],
    n_examples: int = 5,
) -> List[Dict[str, Any]]:
    cards = []
    for cid in sorted(set(int(x) for x in labels)):
        idx = np.flatnonzero(labels == cid)
        examples = [texts[i][:220] for i in idx[:n_examples]]
        brand_counts = Counter(str(brands[i]).lower() for i in idx)
        top = terms.get(cid, [])
        cards.append(
            {
                "method": method,
                "cluster_id": int(cid),
                "n": int(len(idx)),
                "top_words": top,
                "suggested_topic_id": _seed_guess(top, taxonomy),
                "brand_counts": dict(brand_counts),
                "examples": examples,
            }
        )
    cards.sort(key=lambda r: r["n"], reverse=True)
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover comment topic clusters")
    parser.add_argument(
        "--method",
        choices=["nmf", "sbert", "both"],
        default="both",
        help="nmf=TF-IDF+NMF baseline; sbert=KMeans on existing embeddings; both=run both, primary=sbert",
    )
    parser.add_argument("--n-topics", type=int, default=12)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Skip discovery; only apply lexicon topics onto existing comment table",
    )
    parser.add_argument(
        "--classify-lexicon",
        action="store_true",
        help="Assign comment_topic per comment via taxonomy lexicon (preferred final labels)",
    )
    parser.add_argument(
        "--apply-seed-map",
        action="store_true",
        help="DRAFT: write comment_topic from seed guess on whole clusters (prefer --classify-lexicon)",
    )
    parser.add_argument(
        "--apply-mapping",
        default=None,
        help="Legacy assist: cluster→topic YAML (prefer --classify-lexicon for reporting)",
    )
    parser.add_argument("--comments-path", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    feature_dir = Path(project_cfg["output"].get("feature_dir", "data/processed/feature"))
    comments_path = Path(args.comments_path) if args.comments_path else feature_dir / "comment_feature_table.parquet"
    out_dir = Path(args.out_dir) if args.out_dir else feature_dir / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_topic_taxonomy()
    df = pd.read_parquet(comments_path)
    print(f"Loaded {len(df):,} comments from {comments_path}")

    if args.classify_only:
        out = apply_lexicon_topics(df, cfg=taxonomy)
        print("Applied comment-level lexicon → comment_topic / comment_topics")
        print(out["comment_topic"].value_counts(dropna=False).head(16).to_string())
        if "topic_layer" in out.columns:
            print("Layers:")
            print(out["topic_layer"].value_counts(dropna=False).to_string())
        out.to_parquet(comments_path, index=False)
        print(f"Updated → {comments_path}")
        return

    df = df.copy()
    df["_text"] = df.apply(_comment_text, axis=1)
    usable = df["_text"].str.len() >= int(args.min_chars)
    work = df.loc[usable].copy()
    print(f"Usable texts (≥{args.min_chars} chars): {len(work):,}")

    texts = work["_text"].tolist()
    brands = work["brand"].astype(str).tolist() if "brand" in work.columns else ["?"] * len(work)

    cards_all: List[Dict[str, Any]] = []
    primary_labels: Optional[np.ndarray] = None
    primary_method = "sbert"

    if args.method in ("nmf", "both"):
        nmf_labels, nmf_terms, nmf_meta = run_nmf(texts, n_topics=int(args.n_topics))
        (out_dir / "comment_topic_nmf_summary.json").write_text(
            json.dumps({"meta": nmf_meta, "top_terms": {str(k): v for k, v in nmf_terms.items()}}, indent=2),
            encoding="utf-8",
        )
        cards_all.extend(
            build_cards(nmf_labels, nmf_terms, texts, brands, method="nmf", taxonomy=taxonomy)
        )
        print(f"NMF: k={nmf_meta['n_topics']} → {out_dir / 'comment_topic_nmf_summary.json'}")
        if args.method == "nmf":
            primary_labels = nmf_labels
            primary_method = "nmf"

    if args.method in ("sbert", "both"):
        emb_list = []
        keep_idx = []
        for i, val in enumerate(work.get("text_embedding", pd.Series([None] * len(work)))):
            vec = _as_embedding(val)
            if vec is None:
                continue
            emb_list.append(vec)
            keep_idx.append(i)
        if len(emb_list) < 50:
            print("SBERT path skipped: too few embeddings")
            if primary_labels is None and args.method == "both":
                primary_labels = nmf_labels
                primary_method = "nmf"
        else:
            E = np.vstack(emb_list)
            # Align texts/brands to rows with embeddings
            texts_e = [texts[i] for i in keep_idx]
            brands_e = [brands[i] for i in keep_idx]
            lab, terms, meta = run_embedding_kmeans(E, texts_e, n_clusters=int(args.n_topics))
            cards_all.extend(
                build_cards(lab, terms, texts_e, brands_e, method="sbert_kmeans", taxonomy=taxonomy)
            )
            print(f"SBERT KMeans: k={meta['n_clusters']} on {meta['n_docs']} docs")
            # Map back to full `work` index: default -1, fill embedding rows
            full = np.full(len(work), -1, dtype=int)
            for j, i in enumerate(keep_idx):
                full[i] = int(lab[j])
            # For rows without embedding, fall back to NMF label if available
            if args.method == "both":
                missing = full < 0
                full[missing] = nmf_labels[missing]
            primary_labels = full
            primary_method = "sbert_kmeans"

    cards_path = out_dir / "comment_topic_cluster_cards.jsonl"
    with cards_path.open("w", encoding="utf-8") as f:
        for card in cards_all:
            if card["method"] != primary_method and args.method == "both":
                # keep NMF cards too for baseline comparison
                pass
            f.write(json.dumps(card, ensure_ascii=False) + "\n")
    print(f"Wrote cluster cards → {cards_path}")

    # Stub mapping for primary method only
    primary_cards = [c for c in cards_all if c["method"] == primary_method]
    stub = {
        "_readme": (
            "Map each cluster_id → topic_id from configs/comment_topics.yaml. "
            "suggested_topic_id is lexicon-only draft — review with examples in cluster cards."
        ),
        "method": primary_method,
        "mappings": [
            {
                "cluster_id": c["cluster_id"],
                "suggested_topic_id": c["suggested_topic_id"],
                "topic_id": c["suggested_topic_id"],  # edit this
                "n": c["n"],
                "top_words": c["top_words"][:8],
            }
            for c in primary_cards
        ],
    }
    stub_path = out_dir / "comment_topic_mapping_stub.yaml"
    stub_path.write_text(yaml.safe_dump(stub, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote mapping stub → {stub_path}")

    # Attach primary cluster ids onto full comment table
    out = df.copy()
    out["topic_cluster_id"] = pd.NA
    out["topic_cluster_method"] = pd.NA
    if primary_labels is not None:
        out.loc[work.index, "topic_cluster_id"] = primary_labels
        out.loc[work.index, "topic_cluster_method"] = primary_method

    if args.classify_lexicon:
        out = apply_lexicon_topics(out, cfg=taxonomy)
        print("Applied comment-level lexicon → comment_topic / comment_topics")
        print(out["comment_topic"].value_counts(dropna=False).head(16).to_string())
        if "topic_layer" in out.columns:
            print("Layers:")
            print(out["topic_layer"].value_counts(dropna=False).to_string())
    elif args.apply_mapping:
        raw = yaml.safe_load(Path(args.apply_mapping).read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "mappings" in raw:
            mapping = {
                str(m["cluster_id"]): str(m.get("topic_id") or m.get("suggested_topic_id"))
                for m in raw["mappings"]
            }
        else:
            mapping = load_cluster_mapping(args.apply_mapping)
        out = apply_cluster_mapping(out, mapping)
        print("Applied cluster mapping → comment_topic (assist only; prefer --classify-lexicon)")
        print(out["comment_topic"].value_counts(dropna=False).head(12).to_string())
    elif args.apply_seed_map and primary_labels is not None:
        seed_map = {str(c["cluster_id"]): c["suggested_topic_id"] for c in primary_cards}
        out = apply_cluster_mapping(out, seed_map)
        print("Applied seed-guess map → comment_topic (DRAFT — prefer --classify-lexicon)")
        print(out["comment_topic"].value_counts(dropna=False).head(12).to_string())

    out = out.drop(columns=["_text"], errors="ignore")
    out.to_parquet(comments_path, index=False)
    print(f"Updated → {comments_path}")
    print(f"Taxonomy labels: {list(topic_id_to_label(taxonomy).values())}")


if __name__ == "__main__":
    main()
