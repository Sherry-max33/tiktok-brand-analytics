"""Theme 4: Sentiment & Topics (comment-centric brand perception).

Layers
------
1. Overall comment sentiment by brand (pos/neu/neg on comments)
2. Content type × sentiment × brand
3. Topic mix (comment-level taxonomy labels)
4. Topic × sentiment × brand

Topic pipeline
--------------
- Discovery (SBERT clusters / NMF) finds semantic conversation patterns.
- Business taxonomy is a separate interpretation layer.
- Final ``comment_topic`` is assigned **per comment** (lexicon/rules),
  not by bulk-mapping every member of a discovery cluster.

Volume tables use all labeled comments; sentiment tables use scored rows only
(typically ~84% of comments have a VADER score).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from .data import CLUSTER_COL, explode_list_col

SENTIMENT_ORDER = ("negative", "neutral", "positive")
SENTIMENT_CAT = pd.CategoricalDtype(categories=list(SENTIMENT_ORDER), ordered=True)

DEFAULT_TOPICS_CFG = Path("configs/comment_topics.yaml")

# Backward-compatible aliases from the first cluster-mapping taxonomy.
_TOPIC_ALIASES = {
    "design_style": "design_aesthetics",
    "comfort_fit": "fit_comfort",
    "availability_purchase": "availability",
    "quality": "quality_performance",
    "nostalgia_heritage": "other_unclear",
    "collaboration": "other_unclear",
}


def load_topic_taxonomy(path: Union[str, Path] = DEFAULT_TOPICS_CFG) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def topic_id_to_label(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    cfg = cfg or load_topic_taxonomy()
    return {t["id"]: t["label"] for t in (cfg.get("topics") or []) if "id" in t}


def topic_id_to_layer(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    cfg = cfg or load_topic_taxonomy()
    return {t["id"]: str(t.get("layer") or "residual") for t in (cfg.get("topics") or []) if "id" in t}


def substantive_topic_ids(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    layers = topic_id_to_layer(cfg)
    return [tid for tid, layer in layers.items() if layer == "substantive"]


def product_reaction_topic_ids(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    layers = topic_id_to_layer(cfg)
    return [tid for tid, layer in layers.items() if layer == "product_reaction"]


def social_topic_ids(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    layers = topic_id_to_layer(cfg)
    return [tid for tid, layer in layers.items() if layer == "social"]


def _example_signals_from_seeds(seeds: Sequence[str], *, max_examples: int = 8) -> str:
    """Pick readable lexicon examples: multi-word phrases first, then short tokens."""
    skip = {"$", "£", "€", "id?", "name?", "link?"}
    phrases: List[str] = []
    singles: List[str] = []
    seen = set()
    for raw in seeds:
        s = str(raw).strip()
        if not s or s in skip or s.lower() in seen:
            continue
        seen.add(s.lower())
        if " " in s:
            phrases.append(s)
        elif len(s) >= 3:
            singles.append(s)
    picked = (phrases + singles)[:max_examples]
    return ", ".join(picked)


def topic_lexicon_examples(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    layers: Sequence[str] = ("substantive",),
    max_examples: int = 8,
) -> pd.DataFrame:
    """Report table: topic label → example classification signals from ``seed_keywords``.

    These are lexicon rules (cluster inspection + business logic), not cluster IDs.
    """
    cfg = cfg or load_topic_taxonomy()
    labels = topic_id_to_label(cfg)
    layer_map = topic_id_to_layer(cfg)
    seeds = cfg.get("seed_keywords") or {}
    allowed = {str(x) for x in layers}
    # Preserve taxonomy order from configs/comment_topics.yaml topics list
    rows = []
    for t in cfg.get("topics") or []:
        tid = t.get("id")
        if not tid or layer_map.get(tid) not in allowed:
            continue
        kw = seeds.get(tid) or []
        rows.append(
            {
                "topic": labels.get(tid, tid),
                "layer": layer_map.get(tid),
                "example_signals": _example_signals_from_seeds(kw, max_examples=max_examples),
                "n_keywords": len(kw),
            }
        )
    return pd.DataFrame(rows)


def normalize_topic_id(topic_id: Any) -> str:
    tid = str(topic_id or "other_unclear").strip()
    return _TOPIC_ALIASES.get(tid, tid)


def add_polarity_bin(
    df: pd.DataFrame,
    *,
    score_col: str = "sentiment_score",
    out_col: str = "sentiment_polarity",
    neg_max: float = -0.05,
    pos_min: float = 0.05,
) -> pd.DataFrame:
    """Bin VADER compound into negative / neutral / positive."""
    out = df.copy()
    s = pd.to_numeric(out.get(score_col), errors="coerce")
    labels = np.where(
        s.isna(),
        None,
        np.where(s <= neg_max, "negative", np.where(s >= pos_min, "positive", "neutral")),
    )
    out[out_col] = pd.Series(labels, index=out.index, dtype=object)
    out[out_col] = out[out_col].astype(SENTIMENT_CAT)
    return out


def scored_comments(df: pd.DataFrame, *, score_col: str = "sentiment_score") -> pd.DataFrame:
    """Rows with a usable sentiment score."""
    work = df
    if "sentiment_polarity" not in work.columns:
        work = add_polarity_bin(work, score_col=score_col)
    mask = pd.to_numeric(work[score_col], errors="coerce").notna()
    return work.loc[mask].copy()


def brand_perception_summary(
    comments: pd.DataFrame,
    *,
    brand_col: str = "brand",
    brands: Sequence[str] = ("nike", "adidas"),
    score_col: str = "sentiment_score",
) -> pd.DataFrame:
    """§1: overall comment sentiment by brand (scored comments only)."""
    work = scored_comments(comments, score_col=score_col)
    rows = []
    for b in brands:
        sub = work.loc[work[brand_col].astype(str).str.lower() == b]
        n = len(sub)
        if n == 0:
            continue
        mix = sub["sentiment_polarity"].astype(str).value_counts(normalize=True)
        rows.append(
            {
                "brand": b,
                "n_comments": n,
                "n_videos": int(sub["video_id"].nunique()) if "video_id" in sub.columns else np.nan,
                "median_sentiment": float(pd.to_numeric(sub[score_col], errors="coerce").median()),
                "mean_sentiment": float(pd.to_numeric(sub[score_col], errors="coerce").mean()),
                "share_positive": float(mix.get("positive", 0.0)),
                "share_neutral": float(mix.get("neutral", 0.0)),
                "share_negative": float(mix.get("negative", 0.0)),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["n_all_comments"] = len(comments)
    out.attrs["n_scored"] = len(work)
    return out


def video_label_sentiment(
    comments: pd.DataFrame,
    videos: pd.DataFrame,
    label_col: str,
    *,
    brand_col: str = "brand",
    min_count: int = 30,
    score_col: str = "sentiment_score",
) -> pd.DataFrame:
    """Comment sentiment by a video-level list label (e.g. content_type, brand_styles)."""
    work = scored_comments(comments, score_col=score_col)
    if "video_id" not in work.columns or "video_id" not in videos.columns:
        raise KeyError("video_id required on comments and videos")
    if label_col not in videos.columns:
        raise KeyError(f"Missing {label_col} on videos")

    vcols = ["video_id", label_col]
    if brand_col in videos.columns and brand_col not in work.columns:
        vcols.append(brand_col)
    v = videos[vcols].drop_duplicates("video_id")
    merged = work.merge(v, on="video_id", how="inner", suffixes=("", "_vid"))
    sample = v[label_col].dropna().head(50)
    uses_lists = any(isinstance(x, (list, tuple, np.ndarray)) for x in sample)
    if uses_lists:
        merged = explode_list_col(merged, label_col)
    else:
        merged = merged.loc[merged[label_col].notna()].copy()
    g = merged.groupby([label_col, brand_col], dropna=False)
    out = g.size().rename("n").to_frame()
    out["median_sentiment"] = g[score_col].median()
    out["share_positive"] = g["sentiment_polarity"].apply(
        lambda s: float((s.astype(str) == "positive").mean()) if len(s) else np.nan
    )
    out["share_negative"] = g["sentiment_polarity"].apply(
        lambda s: float((s.astype(str) == "negative").mean()) if len(s) else np.nan
    )
    out = out.reset_index().rename(columns={label_col: "level", brand_col: "brand"})
    out = out[out["n"] >= min_count].sort_values(["level", "brand"])
    out.attrs["label_col"] = label_col
    out.attrs["source_comments"] = len(work)
    out.attrs["joined_comments"] = len(merged)
    return out.reset_index(drop=True)


def content_type_sentiment(
    comments: pd.DataFrame,
    videos: pd.DataFrame,
    *,
    brand_col: str = "brand",
    min_count: int = 30,
    score_col: str = "sentiment_score",
) -> pd.DataFrame:
    """§2: content_type × brand × comment sentiment."""
    return video_label_sentiment(
        comments,
        videos,
        "content_type",
        brand_col=brand_col,
        min_count=min_count,
        score_col=score_col,
    )


def brand_style_sentiment(
    comments: pd.DataFrame,
    videos: pd.DataFrame,
    *,
    brand_col: str = "brand",
    min_count: int = 30,
    score_col: str = "sentiment_score",
) -> pd.DataFrame:
    """§2: brand_styles × brand × comment sentiment."""
    return video_label_sentiment(
        comments,
        videos,
        "brand_styles",
        brand_col=brand_col,
        min_count=min_count,
        score_col=score_col,
    )


def content_cluster_sentiment(
    comments: pd.DataFrame,
    videos: pd.DataFrame,
    *,
    brand_col: str = "brand",
    min_count: int = 30,
    score_col: str = "sentiment_score",
    cluster_col: str = CLUSTER_COL,
) -> pd.DataFrame:
    """§2: text content_cluster × brand × comment sentiment."""
    return video_label_sentiment(
        comments,
        videos,
        cluster_col,
        brand_col=brand_col,
        min_count=min_count,
        score_col=score_col,
    )


def content_types_both_brands(
    tab: pd.DataFrame,
    *,
    brands: Sequence[str] = ("nike", "adidas"),
    level_col: str = "level",
    brand_col: str = "brand",
) -> List[str]:
    """Levels with at least one row for every brand (apples-to-apples)."""
    want = {str(b).lower() for b in brands}
    out = []
    for level, g in tab.groupby(level_col, dropna=False):
        have = {str(x).lower() for x in g[brand_col]}
        if want.issubset(have):
            out.append(level)
    return out


levels_both_brands = content_types_both_brands


# ---------------------------------------------------------------------------
# Comment-level topic classification
# ---------------------------------------------------------------------------


def _compile_keyword_patterns(seeds: Dict[str, Sequence[str]]) -> Dict[str, List[Tuple[str, re.Pattern]]]:
    """Longer phrases first; word-boundary for single tokens."""
    compiled: Dict[str, List[Tuple[str, re.Pattern]]] = {}
    for topic_id, kws in (seeds or {}).items():
        items: List[Tuple[str, re.Pattern]] = []
        for kw in sorted({str(k).lower().strip() for k in kws if str(k).strip()}, key=len, reverse=True):
            if " " in kw or "[" in kw or any(ord(ch) > 127 for ch in kw):
                pat = re.compile(re.escape(kw), re.IGNORECASE)
            else:
                pat = re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)", re.IGNORECASE)
            items.append((kw, pat))
        compiled[str(topic_id)] = items
    return compiled


def _comment_text_row(row: pd.Series) -> str:
    """Prefer English translation of the comment when present, else original text.

    On the comment feature table, ``caption_en`` is the MT English of the comment
    (not the video caption).
    """
    for col in ("caption_en", "comment_text_clean", "comment_text"):
        t = str(row.get(col) or "").strip()
        if t:
            return t
    return ""


def classify_comment_text(
    text: str,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    patterns: Optional[Dict[str, List[Tuple[str, re.Pattern]]]] = None,
) -> Tuple[str, List[str]]:
    """Return (primary_topic_id, all_matched_topic_ids).

    Layer preference: substantive > product_reaction > social > residual.
    Within a layer, ``primary_priority`` picks the primary label.
    """
    cfg = cfg or load_topic_taxonomy()
    seeds = cfg.get("seed_keywords") or {}
    pats = patterns or _compile_keyword_patterns(seeds)
    priority = list(cfg.get("primary_priority") or list(seeds.keys()) + ["other_unclear"])
    layers = topic_id_to_layer(cfg)

    raw = str(text or "").strip()
    if not raw:
        return "other_unclear", ["other_unclear"]

    norm = raw.lower().strip()
    scores: Dict[str, int] = {}
    for topic_id, items in pats.items():
        hit = 0
        for _kw, pat in items:
            if pat.search(norm):
                hit += 1
        if hit:
            scores[topic_id] = hit

    if not scores:
        stripped = re.sub(r"\[(?:photo|sticker|stickers)\]", " ", norm, flags=re.I)
        stripped = re.sub(r"[\W_]+", " ", stripped, flags=re.UNICODE).strip()
        if not stripped or len(stripped) <= 2:
            return "meme_chatter", ["meme_chatter"]
        if re.fullmatch(
            r"(fr|ngl|tbh|imo|idk|omg|smh|ikr|ong|lmao|lol|yes|ok|last|same)+",
            stripped,
        ):
            return "meme_chatter", ["meme_chatter"]
        if re.match(r"^@", norm) or re.search(r"(^|\s)@[a-z0-9._]+", norm):
            return "meme_chatter", ["meme_chatter"]
        return "other_unclear", ["other_unclear"]

    matched = list(scores.keys())
    layer_rank = {"substantive": 0, "product_reaction": 1, "social": 2, "residual": 3}
    best_layer = min(
        (layer_rank.get(layers.get(t, "residual"), 9) for t in matched),
        default=9,
    )
    pool = [
        t
        for t in matched
        if layer_rank.get(layers.get(t, "residual"), 9) == best_layer
    ]

    def _rank(tid: str) -> Tuple[int, int]:
        try:
            pri = priority.index(tid)
        except ValueError:
            pri = 999
        return (pri, -scores.get(tid, 0))

    primary = sorted(pool, key=_rank)[0]
    all_ids = sorted(matched, key=_rank)
    return primary, all_ids


def apply_lexicon_topics(
    comments: pd.DataFrame,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    text_col: Optional[str] = None,
    out_col: str = "comment_topic",
    multi_col: str = "comment_topics",
) -> pd.DataFrame:
    """Assign primary + multi-label topics at comment level."""
    cfg = cfg or load_topic_taxonomy()
    pats = _compile_keyword_patterns(cfg.get("seed_keywords") or {})
    out = comments.copy()

    if text_col and text_col in out.columns:
        texts = out[text_col].astype(str).fillna("")
    else:
        texts = out.apply(_comment_text_row, axis=1)

    primaries: List[str] = []
    multis: List[List[str]] = []
    for t in texts:
        primary, matched = classify_comment_text(t, cfg=cfg, patterns=pats)
        primaries.append(primary)
        multis.append(matched)

    out[out_col] = primaries
    out[multi_col] = multis
    out["topic_layer"] = out[out_col].map(topic_id_to_layer(cfg))
    return out


def apply_cluster_mapping(
    comments: pd.DataFrame,
    mapping: Dict[Any, str],
    *,
    cluster_col: str = "topic_cluster_id",
    out_col: str = "comment_topic",
) -> pd.DataFrame:
    """Map discovery cluster ids → taxonomy topic ids (discovery assist only).

    Prefer ``apply_lexicon_topics`` for final reporting labels.
    """
    out = comments.copy()
    if cluster_col not in out.columns:
        raise KeyError(f"Missing {cluster_col}")
    norm = {str(k): normalize_topic_id(v) for k, v in mapping.items()}

    def _map_one(x: Any) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "other_unclear"
        key = str(int(x)) if isinstance(x, (int, np.integer)) or (
            isinstance(x, float) and float(x).is_integer()
        ) else str(x)
        return norm.get(key, norm.get(str(x), "other_unclear"))

    out[out_col] = out[cluster_col].map(_map_one)
    return out


def load_cluster_mapping(path: Union[str, Path]) -> Dict[str, str]:
    """YAML: {cluster_id: topic_id, ...} or list of {cluster_id, topic_id}."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if isinstance(raw, dict) and "mappings" in raw:
        items = raw["mappings"]
        return {str(i["cluster_id"]): normalize_topic_id(i["topic_id"]) for i in items}
    if isinstance(raw, dict):
        return {
            str(k): normalize_topic_id(v)
            for k, v in raw.items()
            if not str(k).startswith("_")
        }
    raise ValueError(f"Unrecognized mapping format: {path}")


def _prepare_topic_frame(
    comments: pd.DataFrame,
    *,
    topic_col: str,
    taxonomy: Optional[Dict[str, str]],
    layers: Optional[Iterable[str]] = None,
    layer_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    work = comments.copy()
    if topic_col not in work.columns:
        raise KeyError(
            f"Missing {topic_col}. Run apply_lexicon_topics (preferred) or discovery mapping."
        )
    if work[topic_col].map(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
        work = explode_list_col(work, topic_col)
    else:
        work[topic_col] = work[topic_col].map(normalize_topic_id).astype(str)

    work = work.loc[work[topic_col].notna() & (work[topic_col].astype(str) != "")]
    work = work.loc[~work[topic_col].astype(str).isin(["nan", "None", "none"])]

    if layers is not None:
        lmap = layer_map or topic_id_to_layer()
        allowed = {str(x) for x in layers}
        work = work.loc[work[topic_col].map(lambda t: lmap.get(str(t), "residual") in allowed)]

    labels = taxonomy or topic_id_to_label()
    work["_topic_label"] = work[topic_col].map(lambda x: labels.get(str(x), str(x)))
    return work


def topic_volume_by_brand(
    comments: pd.DataFrame,
    *,
    topic_col: str = "comment_topic",
    brand_col: str = "brand",
    brands: Sequence[str] = ("nike", "adidas"),
    min_count: int = 20,
    taxonomy: Optional[Dict[str, str]] = None,
    layers: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """§3: topic volume + brand share (all labeled comments; not scored-only)."""
    work = _prepare_topic_frame(
        comments, topic_col=topic_col, taxonomy=taxonomy, layers=layers
    )
    total = len(work)
    rows = []
    for topic, g in work.groupby("_topic_label", dropna=False):
        n = len(g)
        if n < min_count:
            continue
        row = {
            "topic": topic,
            "n": n,
            "share": n / total if total else np.nan,
        }
        for b in brands:
            row[f"{b}_n"] = int((g[brand_col].astype(str).str.lower() == b).sum())
            row[f"{b}_share"] = row[f"{b}_n"] / n if n else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    out.attrs["total_topic_rows"] = total
    out.attrs["n_input_comments"] = len(comments)
    return out.reset_index(drop=True)


def topic_sentiment_by_brand(
    comments: pd.DataFrame,
    *,
    topic_col: str = "comment_topic",
    brand_col: str = "brand",
    brands: Sequence[str] = ("nike", "adidas"),
    min_count: int = 20,
    score_col: str = "sentiment_score",
    taxonomy: Optional[Dict[str, str]] = None,
    layers: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """§4: topic × brand sentiment (scored comments only)."""
    scored = scored_comments(comments, score_col=score_col)
    work = _prepare_topic_frame(
        scored, topic_col=topic_col, taxonomy=taxonomy, layers=layers
    )

    rows = []
    for topic, g in work.groupby("_topic_label", dropna=False):
        row: Dict[str, Any] = {"topic": topic, "n": int(len(g))}
        keep = False
        for b in brands:
            sub = g.loc[g[brand_col].astype(str).str.lower() == b]
            nb = len(sub)
            row[f"{b}_n"] = nb
            row[f"{b}_median_sentiment"] = (
                float(pd.to_numeric(sub[score_col], errors="coerce").median()) if nb else np.nan
            )
            row[f"{b}_share_positive"] = (
                float((sub["sentiment_polarity"].astype(str) == "positive").mean()) if nb else np.nan
            )
            row[f"{b}_share_negative"] = (
                float((sub["sentiment_polarity"].astype(str) == "negative").mean()) if nb else np.nan
            )
            if nb >= min_count:
                keep = True
        if keep or row["n"] >= min_count:
            rows.append(row)
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    out.attrs["n_scored"] = len(scored)
    out.attrs["n_topic_scored_rows"] = len(work)
    return out.reset_index(drop=True)


def topic_sentiment_gap(
    topic_sent: pd.DataFrame,
    *,
    brands: Sequence[str] = ("nike", "adidas"),
    metric: str = "share_positive",
) -> pd.DataFrame:
    """Compact Nike vs Adidas gap table for dumbbell charts / takeaways.

    ``metric`` may be ``share_positive``, ``share_negative``, or ``net``
    (positive − negative). Always attaches per-brand net columns when pos/neg exist.
    """
    if len(brands) != 2:
        raise ValueError("topic_sentiment_gap expects exactly two brands")
    a, b = brands
    out = topic_sent.copy()
    for brand in brands:
        pos_c, neg_c = f"{brand}_share_positive", f"{brand}_share_negative"
        if pos_c in out.columns and neg_c in out.columns:
            out[f"{brand}_net"] = out[pos_c] - out[neg_c]
    if f"{a}_net" in out.columns and f"{b}_net" in out.columns:
        out["net_gap"] = out[f"{b}_net"] - out[f"{a}_net"]

    if metric == "net":
        ca, cb = f"{a}_net", f"{b}_net"
    else:
        ca, cb = f"{a}_{metric}", f"{b}_{metric}"
    out["gap"] = out[cb] - out[ca]
    # Brand with higher value on this metric (for positive/net = warmer; for negative = more negative)
    out["lead_brand"] = np.where(
        out["gap"].isna(),
        None,
        np.where(out["gap"] > 0, b, np.where(out["gap"] < 0, a, "tie")),
    )
    out["warmer_brand"] = out["lead_brand"]  # alias; meaningful when metric is positive/net
    cols = list(
        dict.fromkeys(
            [
                "topic",
                "n",
                f"{a}_n",
                f"{b}_n",
                ca,
                cb,
                "gap",
                "lead_brand",
                "warmer_brand",
                f"{a}_share_positive",
                f"{b}_share_positive",
                f"{a}_share_negative",
                f"{b}_share_negative",
                f"{a}_net",
                f"{b}_net",
                "net_gap",
            ]
        )
    )
    return out[[c for c in cols if c in out.columns]].sort_values("gap", ascending=False).reset_index(
        drop=True
    )
