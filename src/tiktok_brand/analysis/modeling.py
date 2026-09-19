"""Engagement prediction: pre-publish features → WER → BRI / Top-Q screening.

No post-publish engagement/view features. Compare CatBoost vs XGBoost
under author-grouped CV.

Model selection protocol (classification / screening layer)
---------------------------------------------------------
Primary:     Lift@10
Secondary:   Q4 / Top-Q AUC, Precision@10
Diagnostic:  Spearman (expected score or P(top))
Supporting:  Macro-F1, Accuracy  (not for model picking)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder

from .data import WER_COL, as_list

# Selection hierarchy for screening / ranking models (not regression MAE).
MODEL_SELECTION_PROTOCOL = (
    ("primary", "lift_at_10"),
    ("secondary", "auc_top"),  # Q4 / Top-Q AUC
    ("secondary", "precision_at_10"),
    ("diagnostic", "spearman"),
    ("supporting", "macro_f1"),
    ("supporting", "accuracy"),
)

# ---------------------------------------------------------------------------
# Feature / leakage specs
# ---------------------------------------------------------------------------

LEAKAGE_COLS = frozenset(
    {
        "view_count",
        "like_count",
        "comment_count",
        "share_count",
        "collect_count",
        "engagement_count",
        "engagement_rate",
        "weighted_engagement_count",
        "weighted_engagement_rate",
        "brand_relative_engagement_index",
        "like_to_view_rate",
        "comment_to_view_rate",
        "share_to_view_rate",
        "engagement_to_view_rate",
        "comment_to_like_ratio",
    }
)

NUMERIC_COLS = [
    "author_follower_count",  # log1p applied in matrix
    "video_duration_sec",
    "caption_length_words",
    "hashtag_count",
    "mention_count",
    "sentiment_score",
    "emoji_count",
    "exclamation_count",
    "question_count",
]

# Progressive ablation groups (Model 0 → 4)
METADATA_NUM_COLS = ["author_follower_count", "video_duration_sec"]
ENGINEERED_TEXT_NUM_COLS = [
    "caption_length_words",
    "hashtag_count",
    "mention_count",
    "sentiment_score",
    "emoji_count",
    "exclamation_count",
    "question_count",
]
ALIGNMENT_NUM_COLS = ["clip_alignment_mean", "clip_alignment_max"]
VISUAL_AXIS_NUM_COLS = [
    "visual_format_score",
    "visual_setting_score",
    "vf_campaign_visual_score",
]

# Raw hour/weekday kept optional; cyclical preferred via use_cyclical_time
TIME_RAW_COLS = ["post_hour", "post_weekday"]

BOOL_COLS = [
    "author_verified",
    "is_official_brand",
    "has_cta",
    "has_purchase_cta",
    "has_discovery_traffic_cta",
    "has_promo_language",
    "has_music",
    "is_sample_trending_audio",
    "is_weekend",
    "has_hook_phrase",
    "has_interactive_question",
    "has_giveaway_promo_phrase",
]

METADATA_BOOL_COLS = [
    "author_verified",
    "is_official_brand",
    "has_music",
    "is_sample_trending_audio",
    "is_weekend",
]
ENGINEERED_TEXT_BOOL_COLS = [
    "has_cta",
    "has_purchase_cta",
    "has_discovery_traffic_cta",
    "has_promo_language",
    "has_hook_phrase",
    "has_interactive_question",
    "has_giveaway_promo_phrase",
]

# Optional continuous visual confidence scores (precomputed, not raw pixels)
NUMERIC_OPTIONAL = [
    "visual_format_score",
    "visual_setting_score",
    "vf_campaign_visual_score",
    # Caption–frame CLIP similarity (pre-publish; ~70%+ coverage when visual embeds exist)
    "clip_alignment_mean",
    "clip_alignment_max",
]

CAT_COLS = [
    "brand",
    "creator_tier",
    "creator_type",
    "visual_format",
    "visual_setting",
    "caption_lang",
    "content_cluster_id_clean_k12",
]

METADATA_CAT_COLS = ["brand", "creator_tier", "creator_type"]
ENGINEERED_TEXT_CAT_COLS = ["caption_lang"]
VISUAL_AXIS_CAT_COLS = ["visual_format", "visual_setting"]
TEXT_CLUSTER_CAT_COLS = ["content_cluster_id_clean_k12"]

# Derived categoricals for viral-signal interactions (built in prepare_model_frame)
INTERACTION_CAT_COLS = [
    "ix_tier_x_ctype",
    "ix_brand_x_ctype",
    "ix_brand_x_mechanic",
    "ix_ctype_x_hook",
    "ix_dur_x_ctype",
    "ix_hour_x_ctype",
]

LIST_COLS = [
    "content_type",
    "brand_styles",
    "social_mechanic",
]

EMB_COL = "cluster_text_embedding"
VISUAL_EMB_COL = "visual_embedding"
GROUP_COL = "author_id"
BRAND_COL = "brand"
DEFAULT_PCA_DIMS = 32
DEFAULT_VISUAL_PCA_DIMS = 24
MISSING = "__MISSING__"
LOG_EPS = 1e-6


@dataclass
class ModelData:
    """Prepared modeling frame (no fitted transformers yet)."""

    frame: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    brands: np.ndarray
    feature_names_hint: list[str] = field(default_factory=list)
    n_dropped_missing_y: int = 0
    n_dropped_missing_emb: int = 0


def assert_no_leakage(columns: Sequence[str]) -> None:
    bad = sorted(set(columns) & LEAKAGE_COLS)
    if bad:
        raise ValueError(f"Leakage columns present in feature set: {bad}")


def _caption_structure_features(text: pd.Series) -> pd.DataFrame:
    s = text.fillna("").astype(str)
    low = s.str.lower()
    # Rough emoji proxy: non-ascii non-letter symbols (good enough for tree models)
    emoji = s.map(lambda t: sum(1 for ch in t if ord(ch) > 127 and not ch.isalnum()))
    hook = low.str.contains(
        r"\b("
        r"stop scrolling|wait for it|you won'?t believe|pov\b|watch till|watch until|"
        r"top \d+|number \d+|day \d+|part \d+|here'?s why|don'?t make this mistake|"
        r"secret|hack|before you|must try|must see"
        r")\b",
        regex=True,
        na=False,
    )
    interactive_q = low.str.contains(
        r"("
        r"\?|"
        r"\b(what do you think|which one|drop a|comment below|tell me|would you|"
        r"have you ever|can you guess|vote|reply)\b"
        r")",
        regex=True,
        na=False,
    )
    giveaway = low.str.contains(
        r"\b("
        r"giveaway|gift card|free gift|win a|i'?ll pick|tag a friend|"
        r"link in bio|use code|discount|sale|promo code|limited time"
        r")\b",
        regex=True,
        na=False,
    )
    return pd.DataFrame(
        {
            "emoji_count": emoji.astype(np.int32),
            "exclamation_count": s.str.count(r"!").astype(np.int32),
            "question_count": s.str.count(r"\?").astype(np.int32),
            "has_hook_phrase": hook.astype(bool),
            "has_interactive_question": interactive_q.astype(bool),
            "has_giveaway_promo_phrase": giveaway.astype(bool),
        },
        index=text.index,
    )


def _primary_list_label(val) -> str:
    items = as_list(val)
    if not items:
        return MISSING
    x = items[0]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return MISSING
    return str(x)


def _hook_type_label(row: pd.Series) -> str:
    parts = []
    if bool(row.get("has_hook_phrase")):
        parts.append("hook")
    if bool(row.get("has_interactive_question")):
        parts.append("ask")
    if bool(row.get("has_giveaway_promo_phrase")):
        parts.append("giveaway")
    return "+".join(parts) if parts else "none"


def _add_interaction_columns(work: pd.DataFrame) -> pd.DataFrame:
    """Build primary labels + crossover cats for viral-signal interactions."""
    work = work.copy()
    work["content_type_primary"] = (
        work["content_type"].map(_primary_list_label) if "content_type" in work.columns else MISSING
    )
    work["social_mechanic_primary"] = (
        work["social_mechanic"].map(_primary_list_label)
        if "social_mechanic" in work.columns
        else MISSING
    )
    work["hook_type"] = work.apply(_hook_type_label, axis=1)

    dur = pd.to_numeric(work.get("video_duration_sec"), errors="coerce")
    work["duration_bucket"] = pd.cut(
        dur,
        bins=[-np.inf, 15, 30, 60, np.inf],
        labels=["le15", "15to30", "30to60", "gt60"],
    ).astype(str).replace("nan", MISSING)

    hour = pd.to_numeric(work.get("post_hour"), errors="coerce")
    work["post_hour_bucket"] = pd.cut(
        hour.fillna(-1),
        bins=[-np.inf, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"],
    ).astype(str).replace("nan", MISSING)

    n = len(work)
    tier = (
        work["creator_tier"].astype(str).fillna(MISSING)
        if "creator_tier" in work.columns
        else pd.Series([MISSING] * n, index=work.index)
    )
    brand = (
        work["brand"].astype(str).fillna(MISSING)
        if "brand" in work.columns
        else pd.Series([MISSING] * n, index=work.index)
    )
    ctype = work["content_type_primary"].astype(str)
    mech = work["social_mechanic_primary"].astype(str)
    hook = work["hook_type"].astype(str)
    dur_b = work["duration_bucket"].astype(str)
    hour_b = work["post_hour_bucket"].astype(str)

    work["ix_tier_x_ctype"] = tier + "|" + ctype
    work["ix_brand_x_ctype"] = brand + "|" + ctype
    work["ix_brand_x_mechanic"] = brand + "|" + mech
    work["ix_ctype_x_hook"] = ctype + "|" + hook
    work["ix_dur_x_ctype"] = dur_b + "|" + ctype
    work["ix_hour_x_ctype"] = hour_b + "|" + ctype
    return work


def prepare_model_frame(
    df: pd.DataFrame,
    *,
    require_embedding: bool = True,
) -> ModelData:
    """Filter rows and assemble raw columns used for modeling."""
    work = df.copy()
    y = pd.to_numeric(work[WER_COL], errors="coerce")
    miss_y = int(y.isna().sum())
    work = work.loc[y.notna()].copy()
    y = y.loc[work.index]

    miss_emb = 0
    if require_embedding:

        def _ok(v) -> bool:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return False
            try:
                arr = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                return False
            return arr.ndim == 1 and arr.size > 0 and np.isfinite(arr).all()

        ok = work[EMB_COL].map(_ok) if EMB_COL in work.columns else pd.Series(False, index=work.index)
        miss_emb = int((~ok).sum())
        work = work.loc[ok].copy()
        y = y.loc[work.index]

    if GROUP_COL not in work.columns:
        raise KeyError(f"Missing group column {GROUP_COL}")

    # Text structure from caption (prefer English translation if present)
    cap = work["caption_en"] if "caption_en" in work.columns else work.get("caption_clean")
    if cap is not None:
        struct = _caption_structure_features(cap)
        for c in struct.columns:
            work[c] = struct[c].to_numpy()

    if "post_weekday" in work.columns:
        wd = pd.to_numeric(work["post_weekday"], errors="coerce")
        work["is_weekend"] = wd.isin([5, 6]).astype(bool)

    if "create_time" in work.columns:
        ct = pd.to_datetime(work["create_time"], errors="coerce", utc=True)
        ct_naive = ct.dt.tz_localize(None)
        work["publish_week"] = ct_naive.dt.to_period("W").astype(str)
        work["publish_month"] = ct_naive.dt.to_period("M").astype(str)

    work = _add_interaction_columns(work)

    groups = work[GROUP_COL].astype(str).fillna("UNKNOWN_AUTHOR").to_numpy()
    brands = work[BRAND_COL].astype(str).to_numpy()
    return ModelData(
        frame=work.reset_index(drop=True),
        y=y.to_numpy(dtype=float),
        groups=groups,
        brands=brands,
        n_dropped_missing_y=miss_y,
        n_dropped_missing_emb=miss_emb,
    )


def _embedding_matrix(series: pd.Series) -> np.ndarray:
    rows = [np.asarray(v, dtype=float).ravel() for v in series]
    return np.vstack(rows)


def _embedding_matrix_with_fill(series: pd.Series, fill: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (n, d) matrix and boolean mask of originally-valid rows."""
    parsed: list[Optional[np.ndarray]] = []
    for v in series:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                parsed.append(None)
                continue
            arr = np.asarray(v, dtype=float).ravel()
            if arr.size > 0 and np.isfinite(arr).all():
                parsed.append(arr)
            else:
                parsed.append(None)
        except (TypeError, ValueError):
            parsed.append(None)
    valid = [p for p in parsed if p is not None]
    if not valid and fill is None:
        raise ValueError("No valid embeddings to infer dimension")
    dim = int(fill.size) if fill is not None else int(valid[0].size)
    if fill is None:
        fill_vec = np.mean(np.vstack(valid), axis=0)
    else:
        fill_vec = np.asarray(fill, dtype=float).ravel()
        if fill_vec.size != dim:
            tmp = np.zeros(dim, dtype=float)
            n = min(dim, fill_vec.size)
            tmp[:n] = fill_vec[:n]
            fill_vec = tmp
    mask = np.array([p is not None for p in parsed], dtype=bool)
    rows = []
    for p in parsed:
        if p is None:
            rows.append(fill_vec)
        elif p.size == dim:
            rows.append(p)
        else:
            fixed = np.zeros(dim, dtype=float)
            n = min(dim, p.size)
            fixed[:n] = p[:n]
            rows.append(fixed)
    return np.vstack(rows).astype(np.float64), mask


def _multi_hot(series: pd.Series, vocab: list[str]) -> np.ndarray:
    idx = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(series), len(vocab)), dtype=np.float32)
    for r, val in enumerate(series):
        for item in as_list(val):
            key = MISSING if item is None or (isinstance(item, float) and np.isnan(item)) else str(item)
            if key in idx:
                out[r, idx[key]] = 1.0
    return out


def _list_vocab(series: pd.Series, min_count: int = 5) -> list[str]:
    from collections import Counter

    cnt: Counter[str] = Counter()
    for val in series:
        items = as_list(val)
        if not items:
            cnt[MISSING] += 1
            continue
        for item in items:
            if item is None or (isinstance(item, float) and np.isnan(item)):
                cnt[MISSING] += 1
            else:
                cnt[str(item)] += 1
    return [k for k, n in cnt.most_common() if n >= min_count]


# ---------------------------------------------------------------------------
# Target transforms (always evaluate after inverse → raw WER)
# ---------------------------------------------------------------------------

def transform_y(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if mode == "raw":
        return y
    if mode == "log1p":
        # Near-identity for WER≈0.006 — kept for ablation honesty
        return np.log1p(np.clip(y, 0, None))
    if mode == "log":
        return np.log(np.clip(y, 0, None) + LOG_EPS)
    raise ValueError(f"Unknown target mode: {mode}")


def inverse_y(y_t: np.ndarray, mode: str) -> np.ndarray:
    y_t = np.asarray(y_t, dtype=float)
    if mode == "raw":
        return y_t
    if mode == "log1p":
        return np.clip(np.expm1(y_t), 0, None)
    if mode == "log":
        return np.clip(np.exp(y_t) - LOG_EPS, 0, None)
    raise ValueError(f"Unknown target mode: {mode}")


@dataclass
class FoldMatrices:
    X_train: np.ndarray
    X_test: np.ndarray
    feature_names: list[str]
    cat_indices: list[int]
    brand_baseline_train: dict[str, float]
    cat_levels: dict[str, list[str]] = field(default_factory=dict)


def brand_baselines(y: np.ndarray, brands: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for b in np.unique(brands):
        out[str(b)] = float(np.median(y[brands == b]))
    return out


def wer_to_bri(wer: np.ndarray, brands: np.ndarray, baselines: dict[str, float]) -> np.ndarray:
    base = np.array([baselines.get(str(b), np.nan) for b in brands], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bri = wer / base
    return bri


def _fold_target_priors(
    y_train: np.ndarray,
    groups_train: np.ndarray,
    brands_train: np.ndarray,
    groups_test: np.ndarray,
    brands_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Author/brand WER priors from train only (LOO on train, full mean on test)."""
    brand_mean = pd.Series(y_train).groupby(brands_train).mean()
    brand_med = pd.Series(y_train).groupby(brands_train).median()
    global_med = float(np.median(y_train))

    author_sum = pd.Series(y_train).groupby(groups_train).sum()
    author_cnt = pd.Series(y_train).groupby(groups_train).size()

    n_tr = len(y_train)
    prior_author_tr = np.empty(n_tr, dtype=np.float32)
    prior_n_tr = np.empty(n_tr, dtype=np.float32)
    for i in range(n_tr):
        a = groups_train[i]
        cnt = int(author_cnt.get(a, 0))
        if cnt <= 1:
            prior_author_tr[i] = float(brand_med.get(brands_train[i], global_med))
            prior_n_tr[i] = 0.0
        else:
            prior_author_tr[i] = float((author_sum[a] - y_train[i]) / (cnt - 1))
            prior_n_tr[i] = float(cnt - 1)

    prior_brand_tr = np.array(
        [float(brand_mean.get(b, global_med)) for b in brands_train], dtype=np.float32
    )

    n_te = len(groups_test)
    prior_author_te = np.empty(n_te, dtype=np.float32)
    prior_n_te = np.empty(n_te, dtype=np.float32)
    for i, a in enumerate(groups_test):
        if a in author_cnt.index:
            prior_author_te[i] = float(author_sum[a] / author_cnt[a])
            prior_n_te[i] = float(author_cnt[a])
        else:
            prior_author_te[i] = float(brand_med.get(brands_test[i], global_med))
            prior_n_te[i] = 0.0
    prior_brand_te = np.array(
        [float(brand_mean.get(b, global_med)) for b in brands_test], dtype=np.float32
    )

    names = ["prior_author_wer", "prior_author_n", "prior_brand_wer"]
    X_tr = np.column_stack([prior_author_tr, prior_n_tr, prior_brand_tr])
    X_te = np.column_stack([prior_author_te, prior_n_te, prior_brand_te])
    return X_tr, X_te, names


def _shrinkage_baseline(
    y: np.ndarray,
    author: np.ndarray,
    parent: np.ndarray,
    *,
    m: float = 10.0,
    loo: bool = False,
) -> np.ndarray:
    """Hierarchical shrink: (n*y_bar_a + m*y_bar_parent) / (n+m)."""
    parent_mean = pd.Series(y).groupby(parent).mean()
    global_mean = float(np.mean(y))
    a_sum = pd.Series(y).groupby(author).sum()
    a_cnt = pd.Series(y).groupby(author).size()
    out = np.empty(len(y), dtype=np.float32)
    for i in range(len(y)):
        a = author[i]
        p = parent[i]
        p_bar = float(parent_mean.get(p, global_mean))
        n = int(a_cnt.get(a, 0))
        if n <= 0:
            out[i] = p_bar
            continue
        if loo and n > 1:
            a_bar = float((a_sum[a] - y[i]) / (n - 1))
            n_eff = n - 1
        elif loo and n == 1:
            out[i] = p_bar
            continue
        else:
            a_bar = float(a_sum[a] / n)
            n_eff = n
        out[i] = (n_eff * a_bar + m * p_bar) / (n_eff + m)
    return out


def _fold_shrinkage_priors(
    data: ModelData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    m: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Author→tier / author→cluster shrinkage priors (train-only stats)."""
    frame = data.frame
    y_tr = data.y[train_idx]
    author_tr = data.groups[train_idx]
    author_te = data.groups[test_idx]

    tier_tr = (
        frame.iloc[train_idx]["creator_tier"].astype(str).fillna(MISSING).to_numpy()
        if "creator_tier" in frame.columns
        else np.full(len(train_idx), MISSING)
    )
    tier_te = (
        frame.iloc[test_idx]["creator_tier"].astype(str).fillna(MISSING).to_numpy()
        if "creator_tier" in frame.columns
        else np.full(len(test_idx), MISSING)
    )
    cl_tr = (
        frame.iloc[train_idx]["content_cluster_id_clean_k12"].map(
            lambda x: MISSING if pd.isna(x) else str(x)
        ).to_numpy()
        if "content_cluster_id_clean_k12" in frame.columns
        else np.full(len(train_idx), MISSING)
    )
    cl_te = (
        frame.iloc[test_idx]["content_cluster_id_clean_k12"].map(
            lambda x: MISSING if pd.isna(x) else str(x)
        ).to_numpy()
        if "content_cluster_id_clean_k12" in frame.columns
        else np.full(len(test_idx), MISSING)
    )

    # Train LOO shrinkage
    shrink_tier_tr = _shrinkage_baseline(y_tr, author_tr, tier_tr, m=m, loo=True)
    shrink_cl_tr = _shrinkage_baseline(y_tr, author_tr, cl_tr, m=m, loo=True)

    # Parent means + author stats for test (unseen author → pure parent)
    tier_mean = pd.Series(y_tr).groupby(tier_tr).mean()
    cl_mean = pd.Series(y_tr).groupby(cl_tr).mean()
    brand_mean = pd.Series(y_tr).groupby(data.brands[train_idx]).mean()
    global_mean = float(np.mean(y_tr))
    a_sum = pd.Series(y_tr).groupby(author_tr).sum()
    a_cnt = pd.Series(y_tr).groupby(author_tr).size()

    def _apply(author, parent, parent_mean_map):
        out = np.empty(len(author), dtype=np.float32)
        for i, a in enumerate(author):
            p_bar = float(parent_mean_map.get(parent[i], global_mean))
            if a in a_cnt.index:
                n = int(a_cnt[a])
                a_bar = float(a_sum[a] / n)
                out[i] = (n * a_bar + m * p_bar) / (n + m)
            else:
                out[i] = p_bar
        return out

    shrink_tier_te = _apply(author_te, tier_te, tier_mean)
    shrink_cl_te = _apply(author_te, cl_te, cl_mean)
    prior_brand_tr = np.array(
        [float(brand_mean.get(b, global_mean)) for b in data.brands[train_idx]], dtype=np.float32
    )
    prior_brand_te = np.array(
        [float(brand_mean.get(b, global_mean)) for b in data.brands[test_idx]], dtype=np.float32
    )
    prior_n_tr = np.array([float(max(a_cnt.get(a, 0) - 1, 0)) for a in author_tr], dtype=np.float32)
    prior_n_te = np.array([float(a_cnt.get(a, 0)) for a in author_te], dtype=np.float32)

    names = [
        "shrink_author_tier_wer",
        "shrink_author_cluster_wer",
        "prior_brand_wer",
        "prior_author_n",
    ]
    X_tr = np.column_stack([shrink_tier_tr, shrink_cl_tr, prior_brand_tr, prior_n_tr])
    X_te = np.column_stack([shrink_tier_te, shrink_cl_te, prior_brand_te, prior_n_te])
    return X_tr, X_te, names


class FeatureBuilder:
    """Fit/transform tabular + PCA(embeddings) inside each CV fold.

    Modality switches support progressive ablation (Model 0→4):
    metadata → +engineered text → +text emb → +visual emb/axes → +alignment.
    """

    def __init__(
        self,
        pca_dims: int = DEFAULT_PCA_DIMS,
        list_min_count: int = 5,
        *,
        use_cyclical_time: bool = True,
        keep_raw_time: bool = False,
        use_interactions: bool = False,
        use_engineered_text: bool = True,
        use_text_embedding: bool = True,
        use_visual_embedding: bool = False,
        use_visual_axes: bool = False,
        use_clip_alignment: bool = False,
        use_content_cluster: bool = True,
        visual_pca_dims: int = DEFAULT_VISUAL_PCA_DIMS,
    ):
        self.pca_dims = pca_dims
        self.list_min_count = list_min_count
        self.use_cyclical_time = use_cyclical_time
        self.keep_raw_time = keep_raw_time
        self.use_interactions = use_interactions
        self.use_engineered_text = use_engineered_text
        self.use_text_embedding = use_text_embedding
        self.use_visual_embedding = use_visual_embedding
        self.use_visual_axes = use_visual_axes
        self.use_clip_alignment = use_clip_alignment
        # Cluster id is a discrete summary of text embeddings — only with text emb
        self.use_content_cluster = use_content_cluster and use_text_embedding
        self.visual_pca_dims = visual_pca_dims
        self.list_vocabs: dict[str, list[str]] = {}
        self.ord_encoder: Optional[OrdinalEncoder] = None
        self.pca: Optional[PCA] = None
        self.visual_pca: Optional[PCA] = None
        self.visual_fill: Optional[np.ndarray] = None
        self.num_cols: list[str] = []
        self.bool_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.feature_names: list[str] = []
        self.cat_indices: list[int] = []

    def fit(self, frame: pd.DataFrame) -> "FeatureBuilder":
        num: list[str] = [c for c in METADATA_NUM_COLS if c in frame.columns]
        if self.use_engineered_text:
            num += [c for c in ENGINEERED_TEXT_NUM_COLS if c in frame.columns and c not in num]
        if self.use_visual_axes:
            num += [c for c in VISUAL_AXIS_NUM_COLS if c in frame.columns and c not in num]
        if self.use_clip_alignment:
            num += [c for c in ALIGNMENT_NUM_COLS if c in frame.columns and c not in num]
        if self.keep_raw_time or not self.use_cyclical_time:
            num = num + [c for c in TIME_RAW_COLS if c in frame.columns]
        self.num_cols = num

        bools = [c for c in METADATA_BOOL_COLS if c in frame.columns]
        if self.use_engineered_text:
            bools += [c for c in ENGINEERED_TEXT_BOOL_COLS if c in frame.columns and c not in bools]
        self.bool_cols = bools

        cats = [c for c in METADATA_CAT_COLS if c in frame.columns]
        if self.use_engineered_text:
            cats += [c for c in ENGINEERED_TEXT_CAT_COLS if c in frame.columns and c not in cats]
        if self.use_visual_axes:
            cats += [c for c in VISUAL_AXIS_CAT_COLS if c in frame.columns and c not in cats]
        if self.use_content_cluster:
            cats += [c for c in TEXT_CLUSTER_CAT_COLS if c in frame.columns and c not in cats]
        if self.use_interactions:
            cats = cats + [c for c in INTERACTION_CAT_COLS if c in frame.columns]
        self.cat_cols = cats

        self.list_vocabs = {}
        if self.use_engineered_text:
            for c in LIST_COLS:
                if c in frame.columns:
                    self.list_vocabs[c] = _list_vocab(frame[c], min_count=self.list_min_count)

        cat_df = frame[self.cat_cols].copy() if self.cat_cols else pd.DataFrame(index=frame.index)
        for c in self.cat_cols:
            cat_df[c] = cat_df[c].map(lambda x: MISSING if pd.isna(x) else str(x))
        self.ord_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        if self.cat_cols:
            self.ord_encoder.fit(cat_df)

        self.pca = None
        if self.use_text_embedding and EMB_COL in frame.columns:
            emb = _embedding_matrix(frame[EMB_COL])
            n_comp = min(self.pca_dims, emb.shape[0], emb.shape[1])
            self.pca = PCA(n_components=n_comp, random_state=42)
            self.pca.fit(emb)

        if self.use_visual_embedding and VISUAL_EMB_COL in frame.columns:
            v_mat, v_mask = _embedding_matrix_with_fill(frame[VISUAL_EMB_COL])
            self.visual_fill = np.mean(v_mat[v_mask], axis=0) if v_mask.any() else np.mean(v_mat, axis=0)
            v_mat, v_mask = _embedding_matrix_with_fill(frame[VISUAL_EMB_COL], fill=self.visual_fill)
            fit_mat = v_mat[v_mask] if v_mask.sum() >= 10 else v_mat
            n_v = min(self.visual_pca_dims, fit_mat.shape[0], fit_mat.shape[1])
            self.visual_pca = PCA(n_components=n_v, random_state=42)
            self.visual_pca.fit(fit_mat)
        else:
            self.visual_pca = None
            self.visual_fill = None

        names: list[str] = []
        for c in self.num_cols:
            names.append(f"{c}_log1p" if c == "author_follower_count" else c)
        if self.use_cyclical_time:
            names.extend(["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"])
        names.extend(self.bool_cols)
        for c, vocab in self.list_vocabs.items():
            names.extend([f"{c}={v}" for v in vocab])
        cat_start = len(names)
        names.extend(self.cat_cols)
        self.cat_indices = list(range(cat_start, cat_start + len(self.cat_cols)))
        if self.pca is not None:
            names.extend([f"emb_pca_{i}" for i in range(self.pca.n_components_)])
        if self.visual_pca is not None:
            names.append("has_visual_embedding")
            names.extend([f"vis_pca_{i}" for i in range(self.visual_pca.n_components_)])
        self.feature_names = names
        assert_no_leakage(self.feature_names)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        n = len(frame)
        if self.num_cols:
            num = frame[self.num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            for i, c in enumerate(self.num_cols):
                if c == "author_follower_count":
                    num[:, i] = np.log1p(np.clip(num[:, i], 0, None))
            num = np.nan_to_num(num, nan=-1.0)
            parts.append(num.astype(np.float32))
        if self.use_cyclical_time:
            hour = pd.to_numeric(frame.get("post_hour"), errors="coerce").fillna(12).to_numpy(dtype=float)
            wd = pd.to_numeric(frame.get("post_weekday"), errors="coerce").fillna(0).to_numpy(dtype=float)
            cyc = np.column_stack(
                [
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * wd / 7),
                    np.cos(2 * np.pi * wd / 7),
                ]
            ).astype(np.float32)
            parts.append(cyc)
        if self.bool_cols:
            bools = frame[self.bool_cols].fillna(False).astype(bool).to_numpy(dtype=np.float32)
            parts.append(bools)
        for c, vocab in self.list_vocabs.items():
            parts.append(_multi_hot(frame[c], vocab))
        if self.cat_cols and self.ord_encoder is not None:
            cat_df = frame[self.cat_cols].copy()
            for c in self.cat_cols:
                cat_df[c] = cat_df[c].map(lambda x: MISSING if pd.isna(x) else str(x))
            cats = self.ord_encoder.transform(cat_df).astype(np.float32)
            parts.append(cats)
        if self.pca is not None:
            emb = _embedding_matrix(frame[EMB_COL])
            parts.append(self.pca.transform(emb).astype(np.float32))
        if self.visual_pca is not None:
            v_mat, v_mask = _embedding_matrix_with_fill(frame[VISUAL_EMB_COL], fill=self.visual_fill)
            parts.append(v_mask.astype(np.float32).reshape(-1, 1))
            parts.append(self.visual_pca.transform(v_mat).astype(np.float32))
        X = np.hstack(parts) if parts else np.zeros((n, 0), dtype=np.float32)
        if X.shape[1] != len(self.feature_names):
            raise RuntimeError(
                f"Feature width mismatch: X={X.shape[1]} names={len(self.feature_names)}"
            )
        return X


def make_fold_matrices(
    data: ModelData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    pca_dims: int = DEFAULT_PCA_DIMS,
    use_cyclical_time: bool = True,
    use_target_priors: bool = False,
    use_shrinkage_priors: bool = False,
    shrinkage_m: float = 10.0,
    use_interactions: bool = False,
    use_engineered_text: bool = True,
    use_text_embedding: bool = True,
    use_visual_embedding: bool = False,
    use_visual_axes: Optional[bool] = None,
    use_clip_alignment: bool = False,
    use_content_cluster: bool = True,
    visual_pca_dims: int = DEFAULT_VISUAL_PCA_DIMS,
) -> FoldMatrices:
    if use_visual_axes is None:
        use_visual_axes = use_visual_embedding
    builder = FeatureBuilder(
        pca_dims=pca_dims,
        use_cyclical_time=use_cyclical_time,
        use_interactions=use_interactions,
        use_engineered_text=use_engineered_text,
        use_text_embedding=use_text_embedding,
        use_visual_embedding=use_visual_embedding,
        use_visual_axes=use_visual_axes,
        use_clip_alignment=use_clip_alignment,
        use_content_cluster=use_content_cluster,
        visual_pca_dims=visual_pca_dims,
    )
    builder.fit(data.frame.iloc[train_idx])
    X_train = builder.transform(data.frame.iloc[train_idx])
    X_test = builder.transform(data.frame.iloc[test_idx])
    names = list(builder.feature_names)
    cat_indices = list(builder.cat_indices)

    if use_shrinkage_priors:
        p_tr, p_te, p_names = _fold_shrinkage_priors(
            data, train_idx, test_idx, m=shrinkage_m
        )
        X_train = np.hstack([X_train, p_tr])
        X_test = np.hstack([X_test, p_te])
        names = names + p_names
    elif use_target_priors:
        p_tr, p_te, p_names = _fold_target_priors(
            data.y[train_idx],
            data.groups[train_idx],
            data.brands[train_idx],
            data.groups[test_idx],
            data.brands[test_idx],
        )
        X_train = np.hstack([X_train, p_tr])
        X_test = np.hstack([X_test, p_te])
        names = names + p_names

    baselines = brand_baselines(data.y[train_idx], data.brands[train_idx])
    cat_levels: dict[str, list[str]] = {}
    if builder.ord_encoder is not None and builder.cat_cols:
        for col, cats in zip(builder.cat_cols, builder.ord_encoder.categories_):
            cat_levels[col] = [str(c) for c in cats]
    return FoldMatrices(
        X_train=X_train,
        X_test=X_test,
        feature_names=names,
        cat_indices=cat_indices,
        brand_baseline_train=baselines,
        cat_levels=cat_levels,
    )


@dataclass
class CVResult:
    scores: pd.DataFrame
    oof_pred: np.ndarray
    oof_true: np.ndarray
    oof_brand: np.ndarray
    importances: Optional[pd.DataFrame] = None


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(a) < 3:
        return float("nan")
    r, _ = spearmanr(a, b)
    return float(r)


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    brands: np.ndarray,
    baselines: dict[str, float],
) -> dict[str, float]:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    # RMSLE / MAE on log(y+eps) — relative scale for small rates
    lt = np.log(y_true + LOG_EPS)
    lp = np.log(np.clip(y_pred, 0, None) + LOG_EPS)
    rmsle = float(np.sqrt(np.mean((lp - lt) ** 2)))
    mae_log = float(np.mean(np.abs(lp - lt)))
    # SMAPE on WER (stable when y near 0)
    denom = np.abs(y_true) + np.abs(y_pred) + LOG_EPS
    smape = float(np.mean(2.0 * np.abs(err) / denom))
    bri_true = wer_to_bri(y_true, brands, baselines)
    bri_pred = wer_to_bri(y_pred, brands, baselines)
    mae_bri = float(np.mean(np.abs(bri_pred - bri_true)))
    return {
        "mae": mae,
        "rmse": rmse,
        "rmsle": rmsle,
        "mae_log": mae_log,
        "smape": smape,
        "spearman": _spearman(y_true, y_pred),
        "mae_bri": mae_bri,
    }


def fit_catboost_simple(
    X_train,
    y_train,
    cat_indices: list[int],
    *,
    seed: int = 42,
    iterations: int = 300,
    loss_function: str = "MAE",
):
    from catboost import CatBoostRegressor, Pool

    n_features = X_train.shape[1]
    cols = [f"f{i}" for i in range(n_features)]
    train_df = pd.DataFrame(X_train, columns=cols)
    for i in cat_indices:
        if i < n_features:
            train_df[cols[i]] = train_df[cols[i]].round().astype(int)
    cat_names = [cols[i] for i in cat_indices if i < n_features]
    model = CatBoostRegressor(
        loss_function=loss_function,
        iterations=iterations,
        depth=6,
        learning_rate=0.06,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )
    pool = Pool(train_df, y_train, cat_features=cat_names if cat_names else None)
    model.fit(pool)
    model._tiktok_feature_cols = cols  # type: ignore[attr-defined]
    model._tiktok_cat_indices = [i for i in cat_indices if i < n_features]  # type: ignore[attr-defined]
    return model


def _catboost_predict(model, X) -> np.ndarray:
    cols = getattr(model, "_tiktok_feature_cols", [f"f{i}" for i in range(X.shape[1])])
    cat_indices = getattr(model, "_tiktok_cat_indices", [])
    df = pd.DataFrame(X, columns=cols)
    for i in cat_indices:
        df[cols[i]] = df[cols[i]].round().astype(int)
    return model.predict(df)


def fit_xgboost(
    X_train,
    y_train,
    *,
    seed: int = 42,
    n_estimators: int = 300,
    objective: str = "reg:absoluteerror",
):
    from xgboost import XGBRegressor

    model = XGBRegressor(
        objective=objective,
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


def _fit_predict_regressor(
    model_name: str,
    X_train: np.ndarray,
    y_train_t: np.ndarray,
    X_test: np.ndarray,
    cat_indices: list[int],
    *,
    seed: int,
    iterations: int,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns raw-scale predictions, feature importances, feature names unused."""
    if model_name == "catboost":
        loss = "RMSE" if target_mode in {"log", "log1p"} else "MAE"
        model = fit_catboost_simple(
            X_train, y_train_t, cat_indices, seed=seed, iterations=iterations, loss_function=loss
        )
        pred_t = _catboost_predict(model, X_test)
        fi = np.asarray(model.get_feature_importance(), dtype=float)
    elif model_name == "xgboost":
        obj = "reg:squarederror" if target_mode in {"log", "log1p"} else "reg:absoluteerror"
        model = fit_xgboost(X_train, y_train_t, seed=seed, n_estimators=iterations, objective=obj)
        pred_t = model.predict(X_test)
        fi = np.asarray(model.feature_importances_, dtype=float)
    elif model_name == "xgboost_huber":
        # Approx Huber via squarederror on winsorized raw target (caller should pass raw)
        model = fit_xgboost(
            X_train, y_train_t, seed=seed, n_estimators=iterations, objective="reg:squarederror"
        )
        pred_t = model.predict(X_test)
        fi = np.asarray(model.feature_importances_, dtype=float)
    else:
        raise ValueError(model_name)
    pred = inverse_y(pred_t, target_mode)
    return pred, fi, []


def run_grouped_cv(
    data: ModelData,
    *,
    n_splits: int = 5,
    pca_dims: int = DEFAULT_PCA_DIMS,
    seed: int = 42,
    models: Sequence[str] = ("catboost", "xgboost"),
    iterations: int = 300,
    target_mode: str = "raw",
    use_cyclical_time: bool = True,
    use_target_priors: bool = False,
) -> dict[str, CVResult]:
    """Author-grouped K-fold CV. Metrics always on raw WER after inverse transform."""
    gkf = GroupKFold(n_splits=n_splits)
    results: dict[str, CVResult] = {}

    for model_name in models:
        oof = np.full(len(data.y), np.nan, dtype=float)
        rows = []
        imp_acc: dict[str, list[float]] = {}
        for fold, (tr, te) in enumerate(gkf.split(np.arange(len(data.y)), data.y, data.groups)):
            mats = make_fold_matrices(
                data,
                tr,
                te,
                pca_dims=pca_dims,
                use_cyclical_time=use_cyclical_time,
                use_target_priors=use_target_priors,
            )
            y_tr_t = transform_y(data.y[tr], target_mode)
            pred, fi, _ = _fit_predict_regressor(
                model_name,
                mats.X_train,
                y_tr_t,
                mats.X_test,
                mats.cat_indices,
                seed=seed + fold,
                iterations=iterations,
                target_mode=target_mode,
            )
            for name, val in zip(mats.feature_names, fi):
                imp_acc.setdefault(name, []).append(float(val))

            oof[te] = pred
            m = _metrics(data.y[te], pred, data.brands[te], mats.brand_baseline_train)
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "rmsle": m["rmsle"],
                    "mae_log": m["mae_log"],
                    "smape": m["smape"],
                    "spearman": m["spearman"],
                    "mae_bri": m["mae_bri"],
                    "n_test": int(len(te)),
                }
            )

        imp_df = None
        if imp_acc:
            imp_df = (
                pd.DataFrame(
                    {
                        "feature": list(imp_acc.keys()),
                        "importance_mean": [float(np.mean(v)) for v in imp_acc.values()],
                    }
                )
                .sort_values("importance_mean", ascending=False)
                .reset_index(drop=True)
            )

        results[model_name] = CVResult(
            scores=pd.DataFrame(rows),
            oof_pred=oof,
            oof_true=data.y.copy(),
            oof_brand=data.brands.copy(),
            importances=imp_df,
        )
    return results


def summarize_cv(results: dict[str, CVResult]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        s = res.scores
        row = {
            "model": name,
            "mae_mean": s["mae"].mean(),
            "mae_std": s["mae"].std(),
            "rmse_mean": s["rmse"].mean(),
            "spearman_mean": s["spearman"].mean(),
            "mae_bri_mean": s["mae_bri"].mean(),
        }
        for col in ("rmsle", "mae_log", "smape"):
            if col in s.columns:
                row[f"{col}_mean"] = s[col].mean()
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)
    return out


def holdout_eval(
    data: ModelData,
    model_name: str,
    *,
    test_size: float = 0.2,
    pca_dims: int = DEFAULT_PCA_DIMS,
    seed: int = 42,
    iterations: int = 300,
    target_mode: str = "raw",
    use_cyclical_time: bool = True,
    use_target_priors: bool = False,
) -> dict[str, Any]:
    """Single author-grouped holdout for the selected model."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(np.arange(len(data.y)), data.y, data.groups))
    mats = make_fold_matrices(
        data,
        tr,
        te,
        pca_dims=pca_dims,
        use_cyclical_time=use_cyclical_time,
        use_target_priors=use_target_priors,
    )
    y_tr_t = transform_y(data.y[tr], target_mode)
    pred, fi, _ = _fit_predict_regressor(
        model_name,
        mats.X_train,
        y_tr_t,
        mats.X_test,
        mats.cat_indices,
        seed=seed,
        iterations=iterations,
        target_mode=target_mode,
    )

    metrics = _metrics(data.y[te], pred, data.brands[te], mats.brand_baseline_train)
    bri_pred = wer_to_bri(pred, data.brands[te], mats.brand_baseline_train)
    imp = (
        pd.DataFrame({"feature": mats.feature_names, "importance": fi})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "metrics": metrics,
        "baselines": mats.brand_baseline_train,
        "y_true": data.y[te],
        "y_pred": pred,
        "bri_pred": bri_pred,
        "brands": data.brands[te],
        "importances": imp,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }


def run_two_stage_cv(
    data: ModelData,
    *,
    n_splits: int = 5,
    pca_dims: int = DEFAULT_PCA_DIMS,
    seed: int = 42,
    iterations: int = 200,
    top_frac: float = 0.10,
    use_target_priors: bool = True,
) -> CVResult:
    """Stage1: top-frac WER classifier; Stage2: separate regressors for low/high."""
    from catboost import CatBoostClassifier

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(data.y), np.nan, dtype=float)
    rows = []
    for fold, (tr, te) in enumerate(gkf.split(np.arange(len(data.y)), data.y, data.groups)):
        mats = make_fold_matrices(
            data, tr, te, pca_dims=pca_dims, use_cyclical_time=True, use_target_priors=use_target_priors
        )
        thr = float(np.quantile(data.y[tr], 1.0 - top_frac))
        y_hi = (data.y[tr] >= thr).astype(int)

        n_features = mats.X_train.shape[1]
        cols = [f"f{i}" for i in range(n_features)]
        train_df = pd.DataFrame(mats.X_train, columns=cols)
        test_df = pd.DataFrame(mats.X_test, columns=cols)
        for i in mats.cat_indices:
            train_df[cols[i]] = train_df[cols[i]].round().astype(int)
            test_df[cols[i]] = test_df[cols[i]].round().astype(int)
        cat_names = [cols[i] for i in mats.cat_indices]

        clf = CatBoostClassifier(
            iterations=iterations,
            depth=5,
            learning_rate=0.06,
            random_seed=seed + fold,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
        )
        clf.fit(train_df, y_hi, cat_features=cat_names if cat_names else None)
        p_hi = clf.predict_proba(test_df)[:, 1]
        pred_hi_flag = p_hi >= 0.5

        # Regressors on train subsets
        lo_mask = y_hi == 0
        hi_mask = y_hi == 1
        reg_lo = fit_catboost_simple(
            mats.X_train[lo_mask],
            data.y[tr][lo_mask],
            mats.cat_indices,
            seed=seed + fold,
            iterations=iterations,
        )
        if hi_mask.sum() >= 30:
            reg_hi = fit_catboost_simple(
                mats.X_train[hi_mask],
                data.y[tr][hi_mask],
                mats.cat_indices,
                seed=seed + fold + 17,
                iterations=iterations,
            )
        else:
            reg_hi = reg_lo

        pred = np.empty(len(te), dtype=float)
        pred[~pred_hi_flag] = _catboost_predict(reg_lo, mats.X_test[~pred_hi_flag])
        pred[pred_hi_flag] = _catboost_predict(reg_hi, mats.X_test[pred_hi_flag])
        oof[te] = pred
        m = _metrics(data.y[te], pred, data.brands[te], mats.brand_baseline_train)
        rows.append({"model": "two_stage", "fold": fold, **m, "n_test": int(len(te))})

    return CVResult(
        scores=pd.DataFrame(rows),
        oof_pred=oof,
        oof_true=data.y.copy(),
        oof_brand=data.brands.copy(),
    )


def _quantile_edges(y: np.ndarray, n_bins: int) -> np.ndarray:
    """Train-only quantile cut edges; drops duplicate edges if needed."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y, qs))
    if len(edges) < 3:
        raise ValueError(f"Not enough unique quantile edges for n_bins={n_bins}")
    return edges


def _digitize_quantiles(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # edges include min/max; classes are intervals between consecutive edges
    return np.clip(np.digitize(y, edges[1:-1], right=True), 0, len(edges) - 2).astype(int)


def _class_metrics(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    proba: np.ndarray,
    y_true_wer: np.ndarray,
    class_ids: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
    )

    n_classes = proba.shape[1]
    expected = proba @ np.arange(n_classes, dtype=float)
    top_true = (y_true_cls == (n_classes - 1)).astype(int)
    top_score = proba[:, -1]
    out = {
        "accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
        "macro_f1": float(f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)),
        "spearman_expected": _spearman(y_true_wer, expected),
        "top_class_recall": float(
            np.sum((y_pred_cls == n_classes - 1) & (y_true_cls == n_classes - 1))
            / max(1, np.sum(y_true_cls == n_classes - 1))
        ),
        "top_class_precision": float(
            np.sum((y_pred_cls == n_classes - 1) & (y_true_cls == n_classes - 1))
            / max(1, np.sum(y_pred_cls == n_classes - 1))
        ),
    }
    try:
        # One-vs-rest multiclass AUC
        out["roc_auc_ovr"] = float(
            roc_auc_score(y_true_cls, proba, multi_class="ovr", average="macro", labels=class_ids)
        )
    except ValueError:
        out["roc_auc_ovr"] = float("nan")
    try:
        out["roc_auc_top"] = float(roc_auc_score(top_true, top_score))
    except ValueError:
        out["roc_auc_top"] = float("nan")
    return out


def _lift_at_k(y_true_top: np.ndarray, score: np.ndarray, k: float = 0.10) -> float:
    y_true_top = np.asarray(y_true_top).astype(int)
    score = np.asarray(score, dtype=float)
    n = len(score)
    m = max(1, int(round(n * k)))
    top_idx = np.argsort(score)[-m:]
    hit = float(np.mean(y_true_top[top_idx]))
    base = float(np.mean(y_true_top))
    if base <= 0:
        return float("nan")
    return hit / base


def _precision_at_k(y_true_top: np.ndarray, score: np.ndarray, k: float = 0.10) -> float:
    """Precision among the top-k% scored rows (hit = positive / Top-Q)."""
    y_true_top = np.asarray(y_true_top).astype(int)
    score = np.asarray(score, dtype=float)
    n = len(score)
    m = max(1, int(round(n * k)))
    top_idx = np.argsort(score)[-m:]
    return float(np.mean(y_true_top[top_idx]))


def rank_models_by_protocol(table: pd.DataFrame) -> pd.DataFrame:
    """Sort a comparison table by the screening selection protocol.

    Expects columns when available: lift_at_10(_mean), auc(_mean)/roc_auc_top(_mean),
    precision_at_10(_mean), spearman(_mean), macro_f1(_mean), accuracy(_mean).
    """

    def _col(*names: str) -> Optional[str]:
        for n in names:
            if n in table.columns:
                return n
        return None

    out = table.copy()
    sort_cols: list[str] = []
    ascending: list[bool] = []
    mapping = [
        (_col("lift_at_10_mean", "lift_at_10", "lift_q4_at_10_mean"), False),
        (_col("auc_mean", "roc_auc_top_mean", "auc_top_mean", "roc_auc_top"), False),
        (_col("precision_at_10_mean", "precision_at_10", "prec_at_10_mean"), False),
        (_col("spearman_mean", "spearman", "spearman_expected_mean"), False),
        (_col("macro_f1_mean", "macro_f1"), False),
        (_col("accuracy_mean", "accuracy"), False),
    ]
    for col, asc in mapping:
        if col is not None:
            sort_cols.append(col)
            ascending.append(asc)
    if not sort_cols:
        return out.reset_index(drop=True)
    return out.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)


def _best_threshold_for_precision(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    min_precision: float = 0.50,
) -> dict[str, float]:
    from sklearn.metrics import precision_recall_curve

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {
            "threshold": 0.5,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
    prec, rec, thr = precision_recall_curve(y_true, scores)
    # thr length = len(prec)-1; align
    if len(thr) == 0:
        return {"threshold": 0.5, "precision": float(prec[0]), "recall": float(rec[0]), "f1": float("nan")}
    prec_t, rec_t = prec[:-1], rec[:-1]
    ok = prec_t >= min_precision
    if not np.any(ok):
        # fallback: max F1
        f1 = np.where((prec_t + rec_t) > 0, 2 * prec_t * rec_t / (prec_t + rec_t), 0.0)
        j = int(np.argmax(f1))
        return {
            "threshold": float(thr[j]),
            "precision": float(prec_t[j]),
            "recall": float(rec_t[j]),
            "f1": float(f1[j]),
            "constraint_met": 0.0,
        }
    # among ok, maximize recall
    idxs = np.where(ok)[0]
    j = int(idxs[np.argmax(rec_t[idxs])])
    p, r = float(prec_t[j]), float(rec_t[j])
    return {
        "threshold": float(thr[j]),
        "precision": p,
        "recall": r,
        "f1": float(2 * p * r / (p + r)) if (p + r) else 0.0,
        "constraint_met": 1.0,
    }


def _calibrate_proba_isotonic(
    proba_train: np.ndarray,
    y_train: np.ndarray,
    proba_test: np.ndarray,
) -> np.ndarray:
    """OvR isotonic calibration per class; renormalize rows."""
    from sklearn.isotonic import IsotonicRegression

    n_classes = proba_train.shape[1]
    calibrated = np.zeros_like(proba_test, dtype=float)
    for k in range(n_classes):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(proba_train[:, k], (y_train == k).astype(float))
        calibrated[:, k] = iso.transform(proba_test[:, k])
    row_sum = calibrated.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    return calibrated / row_sum


def _train_oof_proba_for_calibration(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cat_indices: list[int],
    *,
    seed: int,
    iterations: int,
    n_splits: int = 3,
) -> np.ndarray:
    """Nested author GroupKFold OOF probabilities on train (for calibrator fit)."""
    from catboost import CatBoostClassifier, Pool

    n = len(y)
    n_classes = int(np.max(y)) + 1
    oof = np.full((n, n_classes), np.nan, dtype=float)
    uniq_groups = np.unique(groups)
    n_splits_eff = min(n_splits, len(uniq_groups))
    if n_splits_eff < 2:
        # too few authors — fall back to in-sample proba (weak calibrator)
        cols = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        for i in cat_indices:
            df[cols[i]] = df[cols[i]].round().astype(int)
        cat_names = [cols[i] for i in cat_indices]
        clf = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=max(50, iterations // 2),
            depth=5,
            learning_rate=0.08,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
        )
        clf.fit(Pool(df, y, cat_features=cat_names if cat_names else None))
        return clf.predict_proba(df)

    gkf = GroupKFold(n_splits=n_splits_eff)
    cols = [f"f{i}" for i in range(X.shape[1])]
    for fold, (tr, te) in enumerate(gkf.split(np.arange(n), y, groups)):
        train_df = pd.DataFrame(X[tr], columns=cols)
        test_df = pd.DataFrame(X[te], columns=cols)
        for i in cat_indices:
            train_df[cols[i]] = train_df[cols[i]].round().astype(int)
            test_df[cols[i]] = test_df[cols[i]].round().astype(int)
        cat_names = [cols[i] for i in cat_indices]
        clf = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=max(50, iterations // 2),
            depth=5,
            learning_rate=0.08,
            random_seed=seed + fold,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
        )
        clf.fit(Pool(train_df, y[tr], cat_features=cat_names if cat_names else None))
        oof[te] = clf.predict_proba(test_df)
    # fill any missed rows
    if np.isnan(oof).any():
        miss = np.isnan(oof[:, 0])
        oof[miss] = 1.0 / n_classes
    return oof


def run_quantile_class_cv(
    data: ModelData,
    *,
    n_bins: int = 4,
    n_splits: int = 5,
    pca_dims: int = DEFAULT_PCA_DIMS,
    seed: int = 42,
    iterations: int = 250,
    use_target_priors: bool = False,
    use_shrinkage_priors: bool = False,
    calibrate: bool = False,
    shrinkage_m: float = 10.0,
    lift_ks: Sequence[float] = (0.10, 0.20),
    min_precision_q4: float = 0.50,
) -> dict[str, Any]:
    """Quantile multiclass (CatBoost). Rank via expected class from (calibrated) probs."""
    from catboost import CatBoostClassifier, Pool

    gkf = GroupKFold(n_splits=n_splits)
    oof_score = np.full(len(data.y), np.nan, dtype=float)
    oof_q4 = np.full(len(data.y), np.nan, dtype=float)
    oof_cls = np.full(len(data.y), -1, dtype=int)
    rows = []
    label = f"qclass_{n_bins}"
    if use_shrinkage_priors:
        label += "+shrink"
    if calibrate:
        label += "+iso"

    for fold, (tr, te) in enumerate(gkf.split(np.arange(len(data.y)), data.y, data.groups)):
        mats = make_fold_matrices(
            data,
            tr,
            te,
            pca_dims=pca_dims,
            use_cyclical_time=True,
            use_target_priors=use_target_priors,
            use_shrinkage_priors=use_shrinkage_priors,
            shrinkage_m=shrinkage_m,
        )
        edges = _quantile_edges(data.y[tr], n_bins)
        n_classes = len(edges) - 1
        y_tr = _digitize_quantiles(data.y[tr], edges)
        y_te = _digitize_quantiles(data.y[te], edges)
        class_ids = np.arange(n_classes)

        n_features = mats.X_train.shape[1]
        cols = [f"f{i}" for i in range(n_features)]
        train_df = pd.DataFrame(mats.X_train, columns=cols)
        test_df = pd.DataFrame(mats.X_test, columns=cols)
        for i in mats.cat_indices:
            train_df[cols[i]] = train_df[cols[i]].round().astype(int)
            test_df[cols[i]] = test_df[cols[i]].round().astype(int)
        cat_names = [cols[i] for i in mats.cat_indices]

        clf = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=iterations,
            depth=6,
            learning_rate=0.06,
            random_seed=seed + fold,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
        )
        pool = Pool(train_df, y_tr, cat_features=cat_names if cat_names else None)
        clf.fit(pool)
        proba_raw = clf.predict_proba(test_df)

        if calibrate:
            oof_tr = _train_oof_proba_for_calibration(
                mats.X_train,
                y_tr,
                data.groups[tr],
                mats.cat_indices,
                seed=seed + 100 + fold,
                iterations=iterations,
            )
            proba = _calibrate_proba_isotonic(oof_tr, y_tr, proba_raw)
            # also calibrate a train-view for threshold selection
            proba_tr_cal = _calibrate_proba_isotonic(oof_tr, y_tr, oof_tr)
            q4_tr = proba_tr_cal[:, -1]
        else:
            proba = proba_raw
            q4_tr = clf.predict_proba(train_df)[:, -1]

        pred_cls = np.argmax(proba, axis=1).astype(int)
        expected = proba @ np.arange(n_classes, dtype=float)
        q4_score = proba[:, -1]
        y_te_top = (y_te == (n_classes - 1)).astype(int)
        y_tr_top = (y_tr == (n_classes - 1)).astype(int)

        # threshold from TRAIN scores (no test leakage)
        thr_info = _best_threshold_for_precision(y_tr_top, q4_tr, min_precision=min_precision_q4)
        q4_hat = (q4_score >= thr_info["threshold"]).astype(int)
        thr_prec = float(
            np.sum((q4_hat == 1) & (y_te_top == 1)) / max(1, np.sum(q4_hat == 1))
        )
        thr_rec = float(
            np.sum((q4_hat == 1) & (y_te_top == 1)) / max(1, np.sum(y_te_top == 1))
        )

        oof_score[te] = expected
        oof_q4[te] = q4_score
        oof_cls[te] = pred_cls
        m = _class_metrics(y_te, pred_cls, proba, data.y[te], class_ids)
        bin_mid = np.array(
            [
                float(np.median(data.y[tr][y_tr == k])) if np.any(y_tr == k) else float(np.median(data.y[tr]))
                for k in range(n_classes)
            ]
        )
        wer_hat = proba @ bin_mid
        reg_m = _metrics(data.y[te], wer_hat, data.brands[te], mats.brand_baseline_train)
        row = {
            "model": label,
            "fold": fold,
            "n_classes": n_classes,
            **m,
            "mae": reg_m["mae"],
            "spearman_wer_mapped": reg_m["spearman"],
            "q4_thr": thr_info["threshold"],
            "q4_thr_precision": thr_prec,
            "q4_thr_recall": thr_rec,
            "q4_constraint_met_train": thr_info.get("constraint_met", float("nan")),
            "n_test": int(len(te)),
        }
        for k in lift_ks:
            row[f"lift_at_{int(k*100)}"] = _lift_at_k(y_te_top, expected, k=k)
            row[f"lift_q4_at_{int(k*100)}"] = _lift_at_k(y_te_top, q4_score, k=k)
            row[f"precision_at_{int(k*100)}"] = _precision_at_k(y_te_top, expected, k=k)
            row[f"precision_q4_at_{int(k*100)}"] = _precision_at_k(y_te_top, q4_score, k=k)
        rows.append(row)

    scores = pd.DataFrame(rows)
    summary = {
        "label": label,
        "n_bins": n_bins,
        "calibrate": calibrate,
        "use_shrinkage_priors": use_shrinkage_priors,
        "accuracy_mean": scores["accuracy"].mean(),
        "macro_f1_mean": scores["macro_f1"].mean(),
        "roc_auc_ovr_mean": scores["roc_auc_ovr"].mean(),
        "roc_auc_top_mean": scores["roc_auc_top"].mean(),
        "spearman_expected_mean": scores["spearman_expected"].mean(),
        "spearman_wer_mapped_mean": scores["spearman_wer_mapped"].mean(),
        "top_recall_mean": scores["top_class_recall"].mean(),
        "top_precision_mean": scores["top_class_precision"].mean(),
        "mae_mapped_mean": scores["mae"].mean(),
        "q4_thr_precision_mean": scores["q4_thr_precision"].mean(),
        "q4_thr_recall_mean": scores["q4_thr_recall"].mean(),
        # protocol aliases
        "auc_top_mean": scores["roc_auc_top"].mean(),
        "spearman_mean": scores["spearman_expected"].mean(),
    }
    for k in lift_ks:
        summary[f"lift_at_{int(k*100)}_mean"] = scores[f"lift_at_{int(k*100)}"].mean()
        summary[f"lift_q4_at_{int(k*100)}_mean"] = scores[f"lift_q4_at_{int(k*100)}"].mean()
        summary[f"precision_at_{int(k*100)}_mean"] = scores[f"precision_at_{int(k*100)}"].mean()
        summary[f"precision_q4_at_{int(k*100)}_mean"] = scores[f"precision_q4_at_{int(k*100)}"].mean()
    # Primary screening score = Q4-proba Lift@10 when available
    if "lift_q4_at_10_mean" in summary:
        summary["lift_at_10_primary_mean"] = summary["lift_q4_at_10_mean"]
        summary["precision_at_10_primary_mean"] = summary.get("precision_q4_at_10_mean")
    return {
        "scores": scores,
        "summary": summary,
        "oof_expected": oof_score,
        "oof_q4_proba": oof_q4,
        "oof_class": oof_cls,
        "oof_true": data.y.copy(),
        "selection_protocol": MODEL_SELECTION_PROTOCOL,
    }


def compare_decision_layer(
    data: ModelData,
    *,
    n_splits: int = 5,
    pca_dims: int = 24,
    iterations: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """Baseline qclass_4 vs shrinkage / isotonic / combined (decision metrics)."""
    specs = [
        {"label": "qclass_4", "use_shrinkage_priors": False, "calibrate": False},
        {"label": "qclass_4+shrink", "use_shrinkage_priors": True, "calibrate": False},
        {"label": "qclass_4+iso", "use_shrinkage_priors": False, "calibrate": True},
        {"label": "qclass_4+shrink+iso", "use_shrinkage_priors": True, "calibrate": True},
    ]
    rows = []
    for spec in specs:
        out = run_quantile_class_cv(
            data,
            n_bins=4,
            n_splits=n_splits,
            pca_dims=pca_dims,
            iterations=iterations,
            seed=seed,
            use_shrinkage_priors=spec["use_shrinkage_priors"],
            calibrate=spec["calibrate"],
        )
        sm = out["summary"]
        rows.append(
            {
                "label": sm["label"],
                "spearman_mean": sm["spearman_expected_mean"],
                "roc_auc_top_mean": sm["roc_auc_top_mean"],
                "auc_mean": sm["roc_auc_top_mean"],
                "lift_at_10_mean": sm.get("lift_q4_at_10_mean", sm.get("lift_at_10_mean")),
                "precision_at_10_mean": sm.get("precision_q4_at_10_mean"),
                "lift_q4_at_10_mean": sm.get("lift_q4_at_10_mean"),
                "q4_thr_precision_mean": sm["q4_thr_precision_mean"],
                "q4_thr_recall_mean": sm["q4_thr_recall_mean"],
                "macro_f1_mean": sm["macro_f1_mean"],
                "accuracy_mean": sm["accuracy_mean"],
            }
        )
    return rank_models_by_protocol(pd.DataFrame(rows))


def _fit_binary_proba(
    backend: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cat_indices: list[int],
    *,
    seed: int,
    iterations: int,
    also_train_proba: bool = False,
    catboost_params: Optional[dict[str, Any]] = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return P(positive) on test (and optionally train) for a binary classifier."""
    if backend == "catboost":
        from catboost import CatBoostClassifier, Pool

        cols = [f"f{i}" for i in range(X_train.shape[1])]
        train_df = pd.DataFrame(X_train, columns=cols)
        test_df = pd.DataFrame(X_test, columns=cols)
        for i in cat_indices:
            train_df[cols[i]] = train_df[cols[i]].round().astype(int)
            test_df[cols[i]] = test_df[cols[i]].round().astype(int)
        cat_names = [cols[i] for i in cat_indices]
        params = {
            "loss_function": "Logloss",
            "iterations": iterations,
            "depth": 6,
            "learning_rate": 0.06,
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": 4,
        }
        if catboost_params:
            params.update(catboost_params)
            params["random_seed"] = seed
            params["verbose"] = False
            params["allow_writing_files"] = False
            params.setdefault("thread_count", 4)
            params.setdefault("loss_function", "Logloss")
        clf = CatBoostClassifier(**params)
        clf.fit(Pool(train_df, y_train, cat_features=cat_names if cat_names else None))
        test_p = clf.predict_proba(test_df)[:, 1]
        if also_train_proba:
            return test_p, clf.predict_proba(train_df)[:, 1]
        return test_p

    if backend == "lgbm":
        import lightgbm as lgb

        clf = lgb.LGBMClassifier(
            n_estimators=iterations,
            learning_rate=0.06,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=4,
            verbosity=-1,
        )
        clf.fit(X_train, y_train)
        test_p = clf.predict_proba(X_test)[:, 1]
        if also_train_proba:
            return test_p, clf.predict_proba(X_train)[:, 1]
        return test_p

    if backend == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=iterations,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=4,
            tree_method="hist",
            eval_metric="logloss",
        )
        clf.fit(X_train, y_train)
        test_p = clf.predict_proba(X_test)[:, 1]
        if also_train_proba:
            return test_p, clf.predict_proba(X_train)[:, 1]
        return test_p

    raise ValueError(f"Unknown binary backend: {backend}")


def run_binary_top_q_cv(
    data: ModelData,
    *,
    top_q: float = 0.25,
    n_splits: int = 5,
    pca_dims: int = DEFAULT_PCA_DIMS,
    seed: int = 42,
    iterations: int = 250,
    backends: Sequence[str] = ("catboost", "lgbm", "xgboost"),
    lift_ks: Sequence[float] = (0.05, 0.10, 0.20, 0.25),
    min_precision: float = 0.50,
    use_interactions: bool = False,
    use_engineered_text: bool = True,
    use_text_embedding: bool = True,
    use_visual_embedding: bool = False,
    use_visual_axes: Optional[bool] = None,
    use_clip_alignment: bool = False,
    use_content_cluster: bool = True,
    visual_pca_dims: int = DEFAULT_VISUAL_PCA_DIMS,
    catboost_params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Binary Top-Q challenger: is_top = 1 if WER >= train-fold (1-top_q) quantile.

    Does **not** replace qclass_4; use alongside champion for screening Lift@K / AUC.
    Default top_q=0.25 → train-fold Q75 threshold (Top 25% / Q4-equivalent).
    """
    from sklearn.metrics import roc_auc_score

    if not 0.0 < top_q < 1.0:
        raise ValueError("top_q must be in (0, 1)")
    if use_visual_axes is None:
        use_visual_axes = use_visual_embedding

    gkf = GroupKFold(n_splits=n_splits)
    results: dict[str, Any] = {}
    q_pct = int(round((1.0 - top_q) * 100))
    feat_tag = ""
    if use_interactions:
        feat_tag += "+ix"
    if use_visual_embedding:
        feat_tag += "+clip"
    if use_clip_alignment:
        feat_tag += "+align"

    for backend in backends:
        label = f"binary_top_q{int(top_q*100)}_{backend}{feat_tag}"
        oof_score = np.full(len(data.y), np.nan, dtype=float)
        oof_true = np.full(len(data.y), -1, dtype=int)
        rows = []

        for fold, (tr, te) in enumerate(gkf.split(np.arange(len(data.y)), data.y, data.groups)):
            mats = make_fold_matrices(
                data,
                tr,
                te,
                pca_dims=pca_dims,
                use_cyclical_time=True,
                use_interactions=use_interactions,
                use_engineered_text=use_engineered_text,
                use_text_embedding=use_text_embedding,
                use_visual_embedding=use_visual_embedding,
                use_visual_axes=use_visual_axes,
                use_clip_alignment=use_clip_alignment,
                use_content_cluster=use_content_cluster,
                visual_pca_dims=visual_pca_dims,
            )
            thr = float(np.quantile(data.y[tr], 1.0 - top_q))
            y_tr = (data.y[tr] >= thr).astype(int)
            y_te = (data.y[te] >= thr).astype(int)

            # If catboost_params sets iterations, honor it over the outer iterations arg
            fit_iterations = iterations
            cb_params = catboost_params
            if backend == "catboost" and catboost_params and "iterations" in catboost_params:
                fit_iterations = int(catboost_params["iterations"])
                cb_params = {k: v for k, v in catboost_params.items() if k != "iterations"}
                cb_params["iterations"] = fit_iterations

            score, score_tr = _fit_binary_proba(
                backend,
                mats.X_train,
                y_tr,
                mats.X_test,
                mats.cat_indices,
                seed=seed + fold,
                iterations=fit_iterations,
                also_train_proba=True,
                catboost_params=cb_params if backend == "catboost" else None,
            )
            oof_score[te] = score
            oof_true[te] = y_te

            thr_info = _best_threshold_for_precision(y_tr, score_tr, min_precision=min_precision)
            y_hat = (score >= thr_info["threshold"]).astype(int)
            prec = float(np.sum((y_hat == 1) & (y_te == 1)) / max(1, np.sum(y_hat == 1)))
            rec = float(np.sum((y_hat == 1) & (y_te == 1)) / max(1, np.sum(y_te == 1)))

            try:
                auc = float(roc_auc_score(y_te, score))
            except ValueError:
                auc = float("nan")

            row = {
                "model": label,
                "fold": fold,
                "threshold_wer": thr,
                "auc": auc,
                "spearman": _spearman(data.y[te], score),
                "positive_rate_test": float(np.mean(y_te)),
                "thr_precision": prec,
                "thr_recall": rec,
                "n_features": int(mats.X_train.shape[1]),
                "n_test": int(len(te)),
            }
            for k in lift_ks:
                row[f"lift_at_{int(k * 100)}"] = _lift_at_k(y_te, score, k=k)
                row[f"precision_at_{int(k * 100)}"] = _precision_at_k(y_te, score, k=k)
            rows.append(row)

        scores = pd.DataFrame(rows)
        summary = {
            "label": label,
            "role": "challenger",
            "top_q": top_q,
            "backend": backend,
            "train_quantile": f"Q{q_pct}",
            "use_interactions": use_interactions,
            "use_visual_embedding": use_visual_embedding,
            "auc_mean": scores["auc"].mean(),
            "spearman_mean": scores["spearman"].mean(),
            "thr_precision_mean": scores["thr_precision"].mean(),
            "thr_recall_mean": scores["thr_recall"].mean(),
            "n_features_mean": scores["n_features"].mean(),
            "selection_protocol": "primary=Lift@10; secondary=AUC+Precision@10; diagnostic=Spearman",
        }
        for k in lift_ks:
            summary[f"lift_at_{int(k * 100)}_mean"] = scores[f"lift_at_{int(k * 100)}"].mean()
            summary[f"precision_at_{int(k * 100)}_mean"] = scores[f"precision_at_{int(k * 100)}"].mean()

        results[backend] = {
            "scores": scores,
            "summary": summary,
            "oof_score": oof_score,
            "oof_true": oof_true,
            "oof_wer": data.y.copy(),
        }

    return results


def compare_feature_ablation(
    data: ModelData,
    *,
    n_splits: int = 5,
    pca_dims: int = 24,
    iterations: int = 180,
    seed: int = 42,
    top_q: float = 0.25,
    backend: str = "catboost",
) -> pd.DataFrame:
    """Ablate interaction + CLIP visual features on binary Top-Q (Lift@10 protocol)."""
    specs = [
        {"label": "baseline", "use_interactions": False, "use_visual_embedding": False},
        {"label": "+interactions", "use_interactions": True, "use_visual_embedding": False},
        {"label": "+clip_visual", "use_interactions": False, "use_visual_embedding": True},
        {"label": "+interactions+clip", "use_interactions": True, "use_visual_embedding": True},
    ]
    rows = []
    for spec in specs:
        res = run_binary_top_q_cv(
            data,
            top_q=top_q,
            n_splits=n_splits,
            pca_dims=pca_dims,
            iterations=iterations,
            seed=seed,
            backends=(backend,),
            use_interactions=spec["use_interactions"],
            use_visual_embedding=spec["use_visual_embedding"],
        )
        sm = res[backend]["summary"]
        rows.append(
            {
                "label": spec["label"],
                "backend": backend,
                "lift_at_10_mean": sm["lift_at_10_mean"],
                "auc_mean": sm["auc_mean"],
                "precision_at_10_mean": sm["precision_at_10_mean"],
                "spearman_mean": sm["spearman_mean"],
                "n_features_mean": sm["n_features_mean"],
                "use_interactions": spec["use_interactions"],
                "use_visual_embedding": spec["use_visual_embedding"],
            }
        )
    return rank_models_by_protocol(pd.DataFrame(rows))


def compare_modality_ablation(
    data: ModelData,
    *,
    top_q: float = 0.10,
    n_splits: int = 5,
    pca_dims: int = 24,
    visual_pca_dims: int = DEFAULT_VISUAL_PCA_DIMS,
    iterations: int = 180,
    seed: int = 42,
    backend: str = "catboost",
    catboost_params: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Progressive modality ladder for Top-Q screening (Lift@10 primary).

    M0   metadata only
    M1   + engineered text / taxonomy labels
    M2   + text embeddings (SBERT/cluster PCA) + content_cluster
    M3a  + rule/CLIP-axis visual labels: format + setting (+ scores)
    M3b  + CLIP visual embedding PCA
    M4   + visual–text CLIP alignment
    """
    ladder = [
        {
            "model": "M0_metadata",
            "label": "Metadata only",
            "use_engineered_text": False,
            "use_text_embedding": False,
            "use_visual_embedding": False,
            "use_visual_axes": False,
            "use_clip_alignment": False,
            "use_content_cluster": False,
        },
        {
            "model": "M1_eng_text",
            "label": "Metadata + engineered text",
            "use_engineered_text": True,
            "use_text_embedding": False,
            "use_visual_embedding": False,
            "use_visual_axes": False,
            "use_clip_alignment": False,
            "use_content_cluster": False,
        },
        {
            "model": "M2_text_emb",
            "label": "+ text embeddings (SBERT PCA)",
            "use_engineered_text": True,
            "use_text_embedding": True,
            "use_visual_embedding": False,
            "use_visual_axes": False,
            "use_clip_alignment": False,
            "use_content_cluster": True,
        },
        {
            "model": "M3a_visual_axes",
            "label": "+ visual format/setting axes",
            "use_engineered_text": True,
            "use_text_embedding": True,
            "use_visual_embedding": False,
            "use_visual_axes": True,
            "use_clip_alignment": False,
            "use_content_cluster": True,
        },
        {
            "model": "M3b_clip_pca",
            "label": "+ CLIP visual PCA",
            "use_engineered_text": True,
            "use_text_embedding": True,
            "use_visual_embedding": True,
            "use_visual_axes": True,
            "use_clip_alignment": False,
            "use_content_cluster": True,
        },
        {
            "model": "M4_alignment",
            "label": "+ visual–text alignment",
            "use_engineered_text": True,
            "use_text_embedding": True,
            "use_visual_embedding": True,
            "use_visual_axes": True,
            "use_clip_alignment": True,
            "use_content_cluster": True,
        },
    ]
    rows = []
    prev_lift = None
    for spec in ladder:
        res = run_binary_top_q_cv(
            data,
            top_q=top_q,
            n_splits=n_splits,
            pca_dims=pca_dims,
            iterations=iterations,
            seed=seed,
            backends=(backend,),
            use_interactions=False,
            use_engineered_text=spec["use_engineered_text"],
            use_text_embedding=spec["use_text_embedding"],
            use_visual_embedding=spec["use_visual_embedding"],
            use_visual_axes=spec["use_visual_axes"],
            use_clip_alignment=spec["use_clip_alignment"],
            use_content_cluster=spec["use_content_cluster"],
            visual_pca_dims=visual_pca_dims,
            catboost_params=catboost_params,
        )
        sm = res[backend]["summary"]
        lift = float(sm["lift_at_10_mean"])
        delta = None if prev_lift is None else lift - prev_lift
        rows.append(
            {
                "model": spec["model"],
                "label": spec["label"],
                "lift_at_5_mean": float(sm["lift_at_5_mean"]),
                "lift_at_10_mean": lift,
                "lift_at_20_mean": float(sm["lift_at_20_mean"]),
                "auc_mean": float(sm["auc_mean"]),
                "precision_at_10_mean": float(sm["precision_at_10_mean"]),
                "spearman_mean": float(sm["spearman_mean"]),
                "n_features_mean": float(sm["n_features_mean"]),
                "delta_lift_at_10": delta,
            }
        )
        prev_lift = lift
    return pd.DataFrame(rows)


DEFAULT_CATBOOST_BINARY = {
    "depth": 6,
    "learning_rate": 0.06,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "iterations": 180,
}

# From tune_binary_top_q_catboost (40-trial random search, +CLIP, Lift@10 primary)
# Reused as-is for Top10 (Q90) clone pipeline — no re-tune unless needed.
TUNED_CATBOOST_BINARY_TOP_Q = {
    "depth": 6,
    "learning_rate": 0.05167351884571456,
    "l2_leaf_reg": 6.087124958330699,
    "random_strength": 1.6474982861240384,
    "iterations": 300,
}

# Kept screening recipes (author GroupKFold, +CLIP visual PCA, tuned CatBoost).
# Primary headline = top10_q90; top25_q75 = broader-pool / robustness (where HP were tuned).
SCREENING_RECIPES = {
    "top10_q90": {"top_q": 0.10, "label": "binary_top_q10+clip+tuned", "keep": True, "role": "primary"},
    "top25_q75": {"top_q": 0.25, "label": "binary_top_q25+clip+tuned", "keep": True, "role": "robustness"},
}

# Human-readable driver families for screening interpretation (SHAP aggregation).
# Order = preferred display order when magnitudes are close.
# Keep families semantically tight — do not bundle unrelated levers
# (e.g. sentiment ≠ language; duration ≠ music ≠ post timing).
DRIVER_FAMILY_ORDER = [
    "creator_audience",
    "content_type",
    "social_mechanic",
    "brand_style",
    "caption_structure",
    "sentiment",
    "language",
    "visual_format",
    "visual_text_alignment",
    "text_semantics",
    "visual_semantics",
    "duration",
    "music",
    "timing",
    "content_cluster",
    "brand_account",
    "other",
]

DRIVER_FAMILY_LABELS = {
    "creator_audience": "Creator audience / identity",
    "content_type": "Content type",
    "social_mechanic": "Social mechanic",
    "brand_style": "Brand style",
    "caption_structure": "Caption characteristics",
    "sentiment": "Sentiment",
    "language": "Caption language",
    "visual_format": "Visual format / setting",
    "visual_text_alignment": "Visual–text alignment (CLIP)",
    "text_semantics": "Text semantics (embedding)",
    "visual_semantics": "Visual semantics (CLIP)",
    "duration": "Video duration",
    "music": "Music / trending audio",
    "timing": "Post timing",
    "content_cluster": "Content cluster",
    "brand_account": "Brand / account flags",
    "other": "Other",
}


def feature_driver_family(name: str) -> str:
    """Map a matrix column name to a driver family for SHAP rollups."""
    n = str(name)
    if n.startswith("emb_pca_"):
        return "text_semantics"
    if n.startswith("vis_pca_") or n == "has_visual_embedding":
        return "visual_semantics"
    if n.startswith("content_type="):
        return "content_type"
    if n.startswith("social_mechanic="):
        return "social_mechanic"
    if n.startswith("brand_styles="):
        return "brand_style"
    if n in {
        "author_follower_count_log1p",
        "creator_tier",
        "creator_type",
        "author_verified",
    }:
        return "creator_audience"
    if n in {
        "caption_length_words",
        "hashtag_count",
        "mention_count",
        "emoji_count",
        "exclamation_count",
        "question_count",
        "has_hook_phrase",
        "has_interactive_question",
        "has_giveaway_promo_phrase",
        "has_cta",
        "has_purchase_cta",
        "has_discovery_traffic_cta",
        "has_promo_language",
    }:
        return "caption_structure"
    if n == "sentiment_score":
        return "sentiment"
    if n == "caption_lang":
        return "language"
    if n in {
        "visual_format",
        "visual_setting",
        "visual_format_score",
        "visual_setting_score",
        "vf_campaign_visual_score",
    }:
        return "visual_format"
    if n in {"clip_alignment_mean", "clip_alignment_max"}:
        return "visual_text_alignment"
    if n == "video_duration_sec":
        return "duration"
    if n in {"has_music", "is_sample_trending_audio"}:
        return "music"
    if n in {
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "post_hour",
        "post_weekday",
    }:
        return "timing"
    if n == "content_cluster_id_clean_k12":
        return "content_cluster"
    if n in {"brand", "is_official_brand"}:
        return "brand_account"
    if n.startswith("ix_"):
        return "other"
    return "other"


def _direction_label(effect: float, *, eps: float = 1e-4) -> str:
    if effect > eps:
        return "increases"
    if effect < -eps:
        return "decreases"
    return "near_zero"


def shap_directional_effects(
    X: np.ndarray,
    shap_vals: np.ndarray,
    feature_names: Sequence[str],
    *,
    cat_levels: Optional[dict[str, list[str]]] = None,
    min_n: int = 12,
) -> pd.DataFrame:
    """Signed SHAP effects: does activating / raising a feature push P(top) up or down?

    - Multi-hot / bool (``content_type=tutorial``, ``has_cta``, …):
      effect = mean SHAP among rows where the flag is on.
    - Continuous: effect = mean SHAP in top tercile of X minus mean SHAP in bottom tercile
      (positive ⇒ higher values associated with higher predicted top-tail probability).
    - Ordinal categoricals (``creator_tier``, …): one row per level = mean SHAP when
      that level is present.

    Associative only — not causal.
    """
    cat_levels = cat_levels or {}
    rows: list[dict[str, Any]] = []

    for j, name in enumerate(feature_names):
        x = np.asarray(X[:, j], dtype=float)
        s = np.asarray(shap_vals[:, j], dtype=float)
        family = feature_driver_family(name)

        # Multi-hot list labels or boolean flags
        if "=" in name or name.startswith("has_") or name.startswith("is_") or name in {
            "author_verified",
            "has_visual_embedding",
        }:
            on = x >= 0.5
            n_on = int(on.sum())
            if n_on < min_n:
                continue
            effect = float(np.mean(s[on]))
            label = name.split("=", 1)[1] if "=" in name else name
            rows.append(
                {
                    "feature": name,
                    "level": label,
                    "kind": "flag_on",
                    "driver_family": family,
                    "n": n_on,
                    "effect": effect,
                    "direction": _direction_label(effect),
                    "mean_abs_shap": float(np.mean(np.abs(s))),
                }
            )
            continue

        # Ordinal categoricals with known levels
        if name in cat_levels:
            from .present import format_cluster_label

            levels = cat_levels[name]
            for code, level in enumerate(levels):
                mask = np.isclose(x, float(code))
                n_m = int(mask.sum())
                if n_m < min_n:
                    continue
                effect = float(np.mean(s[mask]))
                level_label: str
                if name == "content_cluster_id_clean_k12":
                    level_label = format_cluster_label(level)
                else:
                    level_label = str(level)
                rows.append(
                    {
                        "feature": name,
                        "level": level_label,
                        "kind": "category",
                        "driver_family": family,
                        "n": n_m,
                        "effect": effect,
                        "direction": _direction_label(effect),
                        "mean_abs_shap": float(np.mean(np.abs(s[mask]))),
                    }
                )
            continue

        # Skip latent PCA dims for directional narrative (hard to interpret)
        if name.startswith("emb_pca_") or name.startswith("vis_pca_"):
            continue

        # Continuous / numeric: high vs low tercile of the feature value
        finite = np.isfinite(x) & np.isfinite(s)
        if finite.sum() < min_n * 2:
            continue
        xf, sf = x[finite], s[finite]
        q_lo, q_hi = np.quantile(xf, [1 / 3, 2 / 3])
        if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_lo == q_hi:
            continue
        lo, hi = xf <= q_lo, xf >= q_hi
        if lo.sum() < min_n or hi.sum() < min_n:
            continue
        effect = float(np.mean(sf[hi]) - np.mean(sf[lo]))
        rows.append(
            {
                "feature": name,
                "level": "high_vs_low",
                "kind": "continuous",
                "driver_family": family,
                "n": int(hi.sum() + lo.sum()),
                "effect": effect,
                "direction": _direction_label(effect),
                "mean_abs_shap": float(np.mean(np.abs(s))),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "level",
                "kind",
                "driver_family",
                "driver_label",
                "n",
                "effect",
                "direction",
                "mean_abs_shap",
            ]
        )

    out = pd.DataFrame(rows)
    out["driver_label"] = out["driver_family"].map(DRIVER_FAMILY_LABELS)
    out["abs_effect"] = out["effect"].abs()
    return out.sort_values("abs_effect", ascending=False).reset_index(drop=True)


def content_type_shap_directions(
    directional: pd.DataFrame,
    *,
    family: str = "content_type",
    min_n: int = 25,
) -> pd.DataFrame:
    """Convenience: signed effects for one multi-hot family (e.g. content_type)."""
    d = directional.loc[
        (directional["driver_family"] == family) & (directional["n"] >= min_n)
    ].copy()
    return d.sort_values("effect", ascending=False).reset_index(drop=True)


def feature_level_shap_directions(
    directional: pd.DataFrame,
    feature: str,
    *,
    min_n: int = 8,
) -> pd.DataFrame:
    """Signed effects for one ordinal/categorical column (e.g. visual_format)."""
    d = directional.loc[
        (directional["feature"] == feature) & (directional["n"] >= min_n)
    ].copy()
    return d.sort_values("effect", ascending=False).reset_index(drop=True)


def _fit_screening_catboost(
    mats: FoldMatrices,
    y_train: np.ndarray,
    *,
    seed: int,
    catboost_params: Optional[dict[str, Any]],
):
    """Fit locked-style CatBoost binary classifier; return model + train/test DataFrames."""
    from catboost import CatBoostClassifier, Pool

    cols = list(mats.feature_names)
    train_df = pd.DataFrame(mats.X_train, columns=cols)
    test_df = pd.DataFrame(mats.X_test, columns=cols)
    for i in mats.cat_indices:
        train_df[cols[i]] = train_df[cols[i]].round().astype(int)
        test_df[cols[i]] = test_df[cols[i]].round().astype(int)
    cat_names = [cols[i] for i in mats.cat_indices]
    params = {
        "loss_function": "Logloss",
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.06,
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": 4,
    }
    if catboost_params:
        params.update(catboost_params)
        params["random_seed"] = seed
        params["verbose"] = False
        params["allow_writing_files"] = False
        params.setdefault("thread_count", 4)
        params.setdefault("loss_function", "Logloss")
    clf = CatBoostClassifier(**params)
    clf.fit(Pool(train_df, y_train, cat_features=cat_names if cat_names else None))
    return clf, train_df, test_df, cat_names


def explain_screening_shap(
    data: ModelData,
    *,
    top_q: float = 0.10,
    n_splits_hint: int = 5,
    pca_dims: int = DEFAULT_PCA_DIMS,
    visual_pca_dims: int = DEFAULT_VISUAL_PCA_DIMS,
    use_visual_embedding: bool = True,
    catboost_params: Optional[dict[str, Any]] = None,
    test_size: float = 0.25,
    max_explain: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """SHAP interpretation layer for the locked binary Top-Q screening model.

    Fits tuned CatBoost on an author-held-out split (GroupShuffleSplit), then
    computes **Approximate** CatBoost TreeSHAP on a capped explain set. Rolls
    feature-level mean |SHAP| up to driver families for the brief / README narrative.

    Not a separate analysis chapter — explains *why* the screening ranker
    separates high-potential candidates.
    """
    if catboost_params is None:
        catboost_params = dict(TUNED_CATBOOST_BINARY_TOP_Q)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(np.arange(len(data.y)), data.y, data.groups))
    thr = float(np.quantile(data.y[tr], 1.0 - top_q))
    y_tr = (data.y[tr] >= thr).astype(int)
    y_te = (data.y[te] >= thr).astype(int)

    mats = make_fold_matrices(
        data,
        tr,
        te,
        pca_dims=pca_dims,
        use_cyclical_time=True,
        use_interactions=False,
        use_engineered_text=True,
        use_text_embedding=True,
        use_visual_embedding=use_visual_embedding,
        use_visual_axes=use_visual_embedding,
        use_clip_alignment=True,
        use_content_cluster=True,
        visual_pca_dims=visual_pca_dims,
    )
    clf, train_df, test_df, cat_names = _fit_screening_catboost(
        mats, y_tr, seed=seed, catboost_params=catboost_params
    )

    # Cap explain rows for speed; keep class balance roughly
    rng = np.random.default_rng(seed)
    n_te = len(test_df)
    if n_te > max_explain:
        pos = np.flatnonzero(y_te == 1)
        neg = np.flatnonzero(y_te == 0)
        n_pos = min(len(pos), max(1, int(max_explain * max(0.05, float(y_te.mean())))))
        n_neg = min(len(neg), max_explain - n_pos)
        take = np.concatenate(
            [
                rng.choice(pos, size=n_pos, replace=False) if len(pos) else np.array([], dtype=int),
                rng.choice(neg, size=n_neg, replace=False) if len(neg) else np.array([], dtype=int),
            ]
        )
        rng.shuffle(take)
    else:
        take = np.arange(n_te)

    from catboost import Pool

    explain_df = test_df.iloc[take]
    pool = Pool(explain_df, cat_features=cat_names if cat_names else None)
    # Approximate TreeSHAP — Exact is prohibitively slow with emb/vis PCA dims.
    shap_raw = np.asarray(
        clf.get_feature_importance(
            pool,
            type="ShapValues",
            shap_calc_type="Approximate",
        ),
        dtype=float,
    )
    shap_vals = shap_raw[:, :-1]
    bias = float(np.mean(shap_raw[:, -1]))
    X_explain = explain_df.to_numpy(dtype=float)

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    feat_df = (
        pd.DataFrame(
            {
                "feature": mats.feature_names,
                "mean_abs_shap": mean_abs,
                "mean_shap": np.mean(shap_vals, axis=0),
                "driver_family": [feature_driver_family(f) for f in mats.feature_names],
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    feat_df["driver_label"] = feat_df["driver_family"].map(DRIVER_FAMILY_LABELS)

    fam = (
        feat_df.groupby(["driver_family", "driver_label"], as_index=False)
        .agg(
            mean_abs_shap_sum=("mean_abs_shap", "sum"),
            mean_abs_shap_mean=("mean_abs_shap", "mean"),
            n_features=("feature", "count"),
        )
    )
    fam["share"] = fam["mean_abs_shap_sum"] / fam["mean_abs_shap_sum"].sum()
    fam["_ord"] = fam["driver_family"].map(
        {k: i for i, k in enumerate(DRIVER_FAMILY_ORDER)}
    ).fillna(99)
    fam = fam.sort_values(["mean_abs_shap_sum", "_ord"], ascending=[False, True]).drop(
        columns="_ord"
    ).reset_index(drop=True)

    # Actionable / structured levers (exclude latent embedding PCA pools)
    structured = fam.loc[
        ~fam["driver_family"].isin({"text_semantics", "visual_semantics", "other"})
    ].copy()
    structured["share_among_structured"] = (
        structured["mean_abs_shap_sum"] / structured["mean_abs_shap_sum"].sum()
    )
    structured = structured.sort_values("mean_abs_shap_sum", ascending=False).reset_index(
        drop=True
    )

    # Top drivers blurb: embeddings first if dominant, then structured levers
    top_drivers = fam.loc[fam["share"] >= 0.02, "driver_label"].tolist()
    top_structured = structured.loc[
        structured["share_among_structured"] >= 0.05, "driver_label"
    ].tolist()

    directional = shap_directional_effects(
        X_explain,
        shap_vals,
        mats.feature_names,
        cat_levels=mats.cat_levels,
        min_n=12,
    )
    ctype_dir = content_type_shap_directions(directional, family="content_type", min_n=12)
    mechanic_dir = content_type_shap_directions(
        directional, family="social_mechanic", min_n=12
    )
    style_dir = content_type_shap_directions(directional, family="brand_style", min_n=12)

    # Brand-conditional signed SHAP (same pooled model; explain rows split by brand)
    brands_te = np.asarray(data.brands[te])
    brands_explain = brands_te[take]
    by_brand: dict[str, Any] = {}
    for brand in ("adidas", "nike"):
        bmask = np.array([str(b).lower() == brand for b in brands_explain])
        n_b = int(bmask.sum())
        if n_b < 40:
            continue
        dir_b = shap_directional_effects(
            X_explain[bmask],
            shap_vals[bmask],
            mats.feature_names,
            cat_levels=mats.cat_levels,
            min_n=8,
        )
        by_brand[brand] = {
            "n_explain": n_b,
            "directional_shap": dir_b,
            "content_type_directions": content_type_shap_directions(
                dir_b, family="content_type", min_n=8
            ),
            "social_mechanic_directions": content_type_shap_directions(
                dir_b, family="social_mechanic", min_n=8
            ),
            "brand_style_directions": content_type_shap_directions(
                dir_b, family="brand_style", min_n=8
            ),
            "content_cluster_directions": content_type_shap_directions(
                dir_b, family="content_cluster", min_n=8
            ),
            "visual_format_directions": feature_level_shap_directions(
                dir_b, "visual_format", min_n=8
            ),
            "visual_setting_directions": feature_level_shap_directions(
                dir_b, "visual_setting", min_n=8
            ),
        }

    alignment_dir = directional.loc[
        directional["driver_family"] == "visual_text_alignment"
    ].copy()

    return {
        "top_q": top_q,
        "train_quantile": f"Q{int(round((1.0 - top_q) * 100))}",
        "threshold_wer": thr,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_explain": int(len(take)),
        "positive_rate_test": float(np.mean(y_te)),
        "shap_bias": bias,
        "feature_shap": feat_df,
        "driver_shap": fam,
        "structured_driver_shap": structured,
        "directional_shap": directional,
        "content_type_directions": ctype_dir,
        "social_mechanic_directions": mechanic_dir,
        "brand_style_directions": style_dir,
        "alignment_directions": alignment_dir,
        "by_brand": by_brand,
        "top_drivers": top_drivers,
        "top_structured_drivers": top_structured,
        "feature_names": list(mats.feature_names),
        "has_clip_alignment": any(
            f.startswith("clip_alignment") for f in mats.feature_names
        ),
        "shap_values": shap_vals,
        "X_explain": X_explain,
        "explain_brands": brands_explain,
        "explain_index": take,
        "model": clf,
        "note": (
            "Approximate CatBoost TreeSHAP on author-held-out split; "
            "driver families aggregate PCA dims and multi-hot labels. "
            "Signed effects: flag_on = mean SHAP when active; "
            "continuous = high−low tercile SHAP. "
            "by_brand = same pooled model, SHAP rows split by brand (aligns with 02). "
            "Associative only — not causal."
        ),
        "n_splits_hint": n_splits_hint,
    }



def _eval_binary_top_q_params(
    data: ModelData,
    params: dict[str, Any],
    *,
    top_q: float,
    n_splits: int,
    pca_dims: int,
    seed: int,
    use_visual_embedding: bool,
    use_interactions: bool,
    visual_pca_dims: int,
) -> dict[str, float]:
    """One CV evaluation; primary metric Lift@10 (not accuracy)."""
    res = run_binary_top_q_cv(
        data,
        top_q=top_q,
        n_splits=n_splits,
        pca_dims=pca_dims,
        seed=seed,
        iterations=int(params.get("iterations", 180)),
        backends=("catboost",),
        use_interactions=use_interactions,
        use_visual_embedding=use_visual_embedding,
        visual_pca_dims=visual_pca_dims,
        catboost_params=params,
    )
    sm = res["catboost"]["summary"]
    return {
        "lift_at_10": float(sm["lift_at_10_mean"]),
        "auc": float(sm["auc_mean"]),
        "precision_at_10": float(sm["precision_at_10_mean"]),
        "spearman": float(sm["spearman_mean"]),
    }


def tune_binary_top_q_catboost(
    data: ModelData,
    *,
    n_trials: int = 40,
    top_q: float = 0.25,
    n_splits: int = 5,
    pca_dims: int = 24,
    seed: int = 42,
    use_visual_embedding: bool = True,
    use_interactions: bool = False,
    visual_pca_dims: int = DEFAULT_VISUAL_PCA_DIMS,
    min_relative_gain: float = 0.05,
) -> dict[str, Any]:
    """Lightweight random search (30–50 trials) for CatBoost binary Top-Q.

    Objective: maximize OOF **Lift@10** under GroupKFold(author).
    Also logs Q4/Top-Q AUC, Precision@10, Spearman — never Accuracy.

    Stop rule: if best Lift@10 and AUC do not improve ≥ ``min_relative_gain``
    (default 5% relative) vs default params, recommend stopping further tuning.
    """
    rng = np.random.default_rng(seed)

    baseline_params = dict(DEFAULT_CATBOOST_BINARY)
    baseline = _eval_binary_top_q_params(
        data,
        baseline_params,
        top_q=top_q,
        n_splits=n_splits,
        pca_dims=pca_dims,
        seed=seed,
        use_visual_embedding=use_visual_embedding,
        use_interactions=use_interactions,
        visual_pca_dims=visual_pca_dims,
    )

    rows = [{"trial": 0, "role": "baseline", **baseline_params, **baseline}]
    best = dict(baseline)
    best_params = dict(baseline_params)

    for t in range(1, n_trials + 1):
        params = {
            "depth": int(rng.integers(4, 9)),  # 4..8
            "learning_rate": float(10 ** rng.uniform(np.log10(0.02), np.log10(0.15))),
            "l2_leaf_reg": float(rng.uniform(1.0, 10.0)),
            "random_strength": float(rng.uniform(0.5, 2.0)),
            "iterations": int(rng.choice([120, 160, 200, 240, 300, 360])),
        }
        metrics = _eval_binary_top_q_params(
            data,
            params,
            top_q=top_q,
            n_splits=n_splits,
            pca_dims=pca_dims,
            seed=seed,
            use_visual_embedding=use_visual_embedding,
            use_interactions=use_interactions,
            visual_pca_dims=visual_pca_dims,
        )
        rows.append({"trial": t, "role": "search", **params, **metrics})
        # primary: Lift@10; tie-break AUC
        if (metrics["lift_at_10"], metrics["auc"]) > (best["lift_at_10"], best["auc"]):
            best = dict(metrics)
            best_params = dict(params)

    trials = pd.DataFrame(rows).sort_values(
        ["lift_at_10", "auc", "spearman"], ascending=False
    ).reset_index(drop=True)

    lift_gain = (best["lift_at_10"] / baseline["lift_at_10"] - 1.0) if baseline["lift_at_10"] else 0.0
    auc_gain = (best["auc"] / baseline["auc"] - 1.0) if baseline["auc"] else 0.0
    meaningful = (lift_gain >= min_relative_gain) or (auc_gain >= min_relative_gain)
    # User heuristic: tiny bumps like 1.70→1.75 aren't worth chasing; 1.9×+ already useful.
    stop_tuning = not meaningful

    recommendation = (
        "KEEP tuned params — ≥5% relative gain on Lift@10 and/or AUC."
        if meaningful
        else "STOP tuning — no stable ≥5% relative gain vs default; keep baseline (+CLIP)."
    )

    return {
        "baseline_params": baseline_params,
        "baseline_metrics": baseline,
        "best_params": best_params,
        "best_metrics": best,
        "lift_relative_gain": float(lift_gain),
        "auc_relative_gain": float(auc_gain),
        "min_relative_gain": min_relative_gain,
        "meaningful_improvement": bool(meaningful),
        "stop_tuning": bool(stop_tuning),
        "recommendation": recommendation,
        "trials": trials,
        "n_trials": n_trials,
        "selection_protocol": {
            "primary": "Lift@10",
            "secondary": ["Top-Q AUC", "Precision@10"],
            "diagnostic": "Spearman",
            "not_used": ["Accuracy"],
        },
    }


def compare_champion_vs_binary_top_q(
    data: ModelData,
    *,
    n_splits: int = 5,
    pca_dims: int = 24,
    iterations: int = 180,
    seed: int = 42,
    top_q: float = 0.25,
    backends: Sequence[str] = ("catboost", "lgbm", "xgboost"),
) -> pd.DataFrame:
    """Keep qclass_4 as champion; score binary Top-Q challengers on the same protocol.

    Fair Lift/AUC use the binary Top-Q label (train-fold threshold) as the hit event.
    Champion contributes its Q4-class probability for the same hit definition.
    """
    from sklearn.metrics import roc_auc_score

    # --- Champion: qclass_4 Q4-class probability (recipe unchanged; scored fairly vs Top-Q hit) ---
    gkf = GroupKFold(n_splits=n_splits)
    champ_rows = []
    from catboost import CatBoostClassifier, Pool

    for fold, (tr, te) in enumerate(gkf.split(np.arange(len(data.y)), data.y, data.groups)):
        mats = make_fold_matrices(data, tr, te, pca_dims=pca_dims, use_cyclical_time=True)
        edges = _quantile_edges(data.y[tr], 4)
        y4_tr = _digitize_quantiles(data.y[tr], edges)
        thr = float(np.quantile(data.y[tr], 1.0 - top_q))
        y_te = (data.y[te] >= thr).astype(int)

        cols = [f"f{i}" for i in range(mats.X_train.shape[1])]
        train_df = pd.DataFrame(mats.X_train, columns=cols)
        test_df = pd.DataFrame(mats.X_test, columns=cols)
        for i in mats.cat_indices:
            train_df[cols[i]] = train_df[cols[i]].round().astype(int)
            test_df[cols[i]] = test_df[cols[i]].round().astype(int)
        cat_names = [cols[i] for i in mats.cat_indices]
        clf = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=iterations,
            depth=6,
            learning_rate=0.06,
            random_seed=seed + fold,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
        )
        clf.fit(Pool(train_df, y4_tr, cat_features=cat_names if cat_names else None))
        proba = clf.predict_proba(test_df)
        q4_score = proba[:, -1]
        expected = proba @ np.arange(proba.shape[1], dtype=float)
        try:
            auc = float(roc_auc_score(y_te, q4_score))
        except ValueError:
            auc = float("nan")
        champ_rows.append(
            {
                "auc": auc,
                "spearman_q4": _spearman(data.y[te], q4_score),
                "spearman_exp": _spearman(data.y[te], expected),
                "lift_at_5": _lift_at_k(y_te, q4_score, 0.05),
                "lift_at_10": _lift_at_k(y_te, q4_score, 0.10),
                "lift_at_20": _lift_at_k(y_te, q4_score, 0.20),
                "lift_at_25": _lift_at_k(y_te, q4_score, 0.25),
                "precision_at_10": _precision_at_k(y_te, q4_score, 0.10),
            }
        )
    champ_df = pd.DataFrame(champ_rows)

    rows = [
        {
            "label": "qclass_4 (champion)",
            "role": "champion",
            "backend": "catboost",
            "score_used": "P(Q4 class)",
            "lift_at_10_mean": champ_df["lift_at_10"].mean(),
            "auc_mean": champ_df["auc"].mean(),
            "precision_at_10_mean": champ_df["precision_at_10"].mean(),
            "spearman_mean": champ_df["spearman_q4"].mean(),
            "lift_at_5_mean": champ_df["lift_at_5"].mean(),
            "lift_at_20_mean": champ_df["lift_at_20"].mean(),
            "lift_at_25_mean": champ_df["lift_at_25"].mean(),
            "macro_f1_mean": np.nan,
            "accuracy_mean": np.nan,
            "notes": "kept for Low/Mid/High/Viral tiering; not replaced",
        }
    ]

    binary = run_binary_top_q_cv(
        data,
        top_q=top_q,
        n_splits=n_splits,
        pca_dims=pca_dims,
        iterations=iterations,
        seed=seed,
        backends=backends,
    )
    for backend, res in binary.items():
        sm = res["summary"]
        rows.append(
            {
                "label": sm["label"],
                "role": "challenger",
                "backend": backend,
                "score_used": f"P(top{int(top_q*100)}%)",
                "lift_at_10_mean": sm.get("lift_at_10_mean"),
                "auc_mean": sm["auc_mean"],
                "precision_at_10_mean": sm.get("precision_at_10_mean"),
                "spearman_mean": sm["spearman_mean"],
                "lift_at_5_mean": sm.get("lift_at_5_mean"),
                "lift_at_20_mean": sm.get("lift_at_20_mean"),
                "lift_at_25_mean": sm.get("lift_at_25_mean"),
                "macro_f1_mean": np.nan,
                "accuracy_mean": np.nan,
                "notes": f"train thr={sm['train_quantile']}; Prec@thr={sm['thr_precision_mean']:.2f}",
            }
        )

    out = rank_models_by_protocol(pd.DataFrame(rows))
    out.attrs["selection_protocol"] = {
        "primary": "Lift@10",
        "secondary": ["Top-Q AUC", "Precision@10"],
        "diagnostic": "Spearman",
        "supporting": ["Macro-F1", "Accuracy"],
    }
    return out


def _group_counts_ordered(group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort by group id and return sorted indices + contiguous group sizes (LTR format)."""
    order = np.argsort(group_ids, kind="mergesort")
    g_sorted = group_ids[order]
    _, counts = np.unique(g_sorted, return_counts=True)
    return order, counts.astype(int), g_sorted


def run_ranker_cv(
    data: ModelData,
    *,
    n_splits: int = 5,
    pca_dims: int = DEFAULT_PCA_DIMS,
    seed: int = 42,
    iterations: int = 200,
    use_target_priors: bool = False,
    group_col: str = "publish_week",
    backend: str = "lgbm",
    min_group_size: int = 3,
) -> dict[str, Any]:
    """Learning-to-rank with proper list grouping (week/month/cluster/brand).

    Primary metric: Spearman(rank_score, WER). MAE via quantile-map is secondary.
    """
    from scipy.stats import rankdata

    if group_col not in data.frame.columns:
        raise KeyError(f"Missing LTR group column: {group_col}")

    # ensure brand available as a column for brand-grouped LTR
    if group_col == "brand" and "brand" not in data.frame.columns:
        data.frame = data.frame.copy()
        data.frame["brand"] = data.brands

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(data.y), np.nan, dtype=float)
    rows = []
    label = f"ranker_{backend}_{group_col}"

    for fold, (tr, te) in enumerate(gkf.split(np.arange(len(data.y)), data.y, data.groups)):
        mats = make_fold_matrices(
            data,
            tr,
            te,
            pca_dims=pca_dims,
            use_cyclical_time=True,
            use_target_priors=use_target_priors,
        )
        g_tr = data.frame.iloc[tr][group_col].astype(str).to_numpy()
        # drop tiny groups from training
        vc = pd.Series(g_tr).value_counts()
        keep = np.array([vc.get(g, 0) >= min_group_size for g in g_tr])
        if keep.sum() < 50:
            keep = np.ones(len(g_tr), dtype=bool)

        X_tr_raw = mats.X_train[keep]
        y_tr_raw = data.y[tr][keep]
        g_tr_keep = g_tr[keep]
        order, group_counts, g_sorted = _group_counts_ordered(g_tr_keep)
        X_tr = X_tr_raw[order]
        y_tr_cont = y_tr_raw[order]
        # LambdaMART expects non-negative integer relevance; grade within each list
        y_tr = np.empty(len(y_tr_cont), dtype=np.int32)
        start = 0
        for gsz in group_counts:
            sl = slice(start, start + gsz)
            vals = y_tr_cont[sl]
            if gsz >= 5:
                try:
                    y_tr[sl] = pd.qcut(vals, q=5, labels=False, duplicates="drop").astype(int)
                except ValueError:
                    y_tr[sl] = rankdata(vals, method="dense").astype(int) - 1
            else:
                y_tr[sl] = rankdata(vals, method="dense").astype(int) - 1
            start += gsz

        if backend == "lgbm":
            import lightgbm as lgb

            ranker = lgb.LGBMRanker(
                objective="lambdarank",
                n_estimators=iterations,
                learning_rate=0.06,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed + fold,
                n_jobs=4,
                verbosity=-1,
            )
            ranker.fit(X_tr, y_tr, group=group_counts)
            score = ranker.predict(mats.X_test)
        elif backend == "xgb":
            from xgboost import XGBRanker

            ranker = XGBRanker(
                objective="rank:pairwise",
                n_estimators=iterations,
                max_depth=5,
                learning_rate=0.06,
                random_state=seed + fold,
                n_jobs=4,
                tree_method="hist",
            )
            ranker.fit(X_tr, y_tr, group=group_counts)
            score = ranker.predict(mats.X_test)
        else:
            raise ValueError(backend)

        r = rankdata(score) / (len(score) + 1.0)
        pred_wer = np.quantile(data.y[tr], r)
        oof[te] = score
        base = mats.brand_baseline_train
        reg_m = _metrics(data.y[te], pred_wer, data.brands[te], base)
        rows.append(
            {
                "model": label,
                "fold": fold,
                "spearman": _spearman(data.y[te], score),
                "mae": reg_m["mae"],
                "rmse": reg_m["rmse"],
                "mae_bri": reg_m["mae_bri"],
                "n_train_groups": int(len(group_counts)),
                "median_group_size": float(np.median(group_counts)),
                "n_test": int(len(te)),
            }
        )

    scores = pd.DataFrame(rows)
    summary = {
        "label": label,
        "group_col": group_col,
        "backend": backend,
        "spearman_mean": scores["spearman"].mean(),
        "mae_mean": scores["mae"].mean(),
        "rmse_mean": scores["rmse"].mean(),
        "mae_bri_mean": scores["mae_bri"].mean(),
        "median_group_size_mean": scores["median_group_size"].mean(),
    }
    return {"scores": scores, "summary": summary, "oof_score": oof, "oof_true": data.y.copy()}


def compare_class_and_rank(
    data: ModelData,
    *,
    n_splits: int = 5,
    pca_dims: int = 24,
    iterations: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Head-to-head: regression baseline vs quantile class vs fixed LTR groups."""
    rows: list[dict[str, Any]] = []

    # regression baseline (CatBoost raw MAE)
    reg = run_grouped_cv(
        data,
        n_splits=n_splits,
        pca_dims=pca_dims,
        iterations=iterations,
        seed=seed,
        models=("catboost",),
        target_mode="raw",
        use_target_priors=False,
    )
    s = summarize_cv(reg).iloc[0]
    rows.append(
        {
            "label": "reg_catboost_raw",
            "task": "regression",
            "spearman_mean": s["spearman_mean"],
            "mae_mean": s["mae_mean"],
            "accuracy_mean": np.nan,
            "macro_f1_mean": np.nan,
            "roc_auc_ovr_mean": np.nan,
            "roc_auc_top_mean": np.nan,
            "notes": "raw WER MAE",
        }
    )

    for n_bins in (4, 5):
        qc = run_quantile_class_cv(
            data,
            n_bins=n_bins,
            n_splits=n_splits,
            pca_dims=pca_dims,
            iterations=iterations,
            seed=seed,
        )
        sm = qc["summary"]
        rows.append(
            {
                "label": sm["label"],
                "task": "quantile_class",
                "spearman_mean": sm["spearman_expected_mean"],
                "mae_mean": sm["mae_mapped_mean"],
                "accuracy_mean": sm["accuracy_mean"],
                "macro_f1_mean": sm["macro_f1_mean"],
                "roc_auc_ovr_mean": sm["roc_auc_ovr_mean"],
                "roc_auc_top_mean": sm["roc_auc_top_mean"],
                "notes": f"top_recall={sm['top_recall_mean']:.3f}",
            }
        )

    for group_col, backend in [
        ("publish_week", "lgbm"),
        ("publish_month", "lgbm"),
        ("content_cluster_id_clean_k12", "lgbm"),
        ("brand", "lgbm"),
        ("publish_week", "xgb"),
    ]:
        if group_col not in data.frame.columns and group_col != "brand":
            continue
        # brand lives on data.brands — inject temporary column if needed
        if group_col == "brand" and "brand" not in data.frame.columns:
            data.frame = data.frame.copy()
            data.frame["brand"] = data.brands
        try:
            rk = run_ranker_cv(
                data,
                n_splits=n_splits,
                pca_dims=pca_dims,
                iterations=iterations,
                seed=seed,
                group_col=group_col if group_col != "brand" else "brand",
                backend=backend,
                use_target_priors=False,
            )
            sm = rk["summary"]
            rows.append(
                {
                    "label": sm["label"],
                    "task": "ltr",
                    "spearman_mean": sm["spearman_mean"],
                    "mae_mean": sm["mae_mean"],
                    "accuracy_mean": np.nan,
                    "macro_f1_mean": np.nan,
                    "roc_auc_ovr_mean": np.nan,
                    "roc_auc_top_mean": np.nan,
                    "notes": f"med_g={sm['median_group_size_mean']:.1f}",
                }
            )
        except Exception as exc:  # noqa: BLE001 — ablation should continue
            rows.append(
                {
                    "label": f"ranker_{backend}_{group_col}",
                    "task": "ltr",
                    "spearman_mean": np.nan,
                    "mae_mean": np.nan,
                    "accuracy_mean": np.nan,
                    "macro_f1_mean": np.nan,
                    "roc_auc_ovr_mean": np.nan,
                    "roc_auc_top_mean": np.nan,
                    "notes": f"FAILED: {exc}",
                }
            )

    out = pd.DataFrame(rows)
    # Prefer Lift@10 protocol when present; else fall back to Spearman
    if "lift_at_10_mean" in out.columns:
        return rank_models_by_protocol(out)
    return out.sort_values("spearman_mean", ascending=False).reset_index(drop=True)


def run_ablation(
    data: ModelData,
    *,
    n_splits: int = 5,
    pca_dims: int = DEFAULT_PCA_DIMS,
    iterations: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare recipe variants under the same GroupKFold protocol."""
    specs = [
        {"label": "baseline_raw_mae", "target_mode": "raw", "use_target_priors": False, "model": "catboost"},
        {"label": "log1p_target", "target_mode": "log1p", "use_target_priors": False, "model": "catboost"},
        {"label": "log_eps_target", "target_mode": "log", "use_target_priors": False, "model": "catboost"},
        {"label": "raw+author_brand_prior", "target_mode": "raw", "use_target_priors": True, "model": "catboost"},
        {
            "label": "log+author_brand_prior",
            "target_mode": "log",
            "use_target_priors": True,
            "model": "catboost",
        },
        {"label": "xgb_raw+prior", "target_mode": "raw", "use_target_priors": True, "model": "xgboost"},
    ]
    rows = []
    for spec in specs:
        res = run_grouped_cv(
            data,
            n_splits=n_splits,
            pca_dims=pca_dims,
            iterations=iterations,
            seed=seed,
            models=(spec["model"],),
            target_mode=spec["target_mode"],
            use_target_priors=spec["use_target_priors"],
        )
        s = summarize_cv(res).iloc[0].to_dict()
        s["label"] = spec["label"]
        s["target_mode"] = spec["target_mode"]
        s["use_target_priors"] = spec["use_target_priors"]
        rows.append(s)

    # two-stage + legacy brand-group xgb ranker (kept for history; prefer compare_class_and_rank)
    cv = run_two_stage_cv(data, n_splits=n_splits, pca_dims=pca_dims, iterations=iterations, seed=seed)
    s = summarize_cv({"two_stage": cv}).iloc[0].to_dict()
    s["label"] = "two_stage"
    s["target_mode"] = "raw"
    s["use_target_priors"] = True
    rows.append(s)

    rk = run_ranker_cv(
        data,
        n_splits=n_splits,
        pca_dims=pca_dims,
        iterations=iterations,
        seed=seed,
        group_col="brand",
        backend="xgb",
        use_target_priors=True,
    )
    rows.append(
        {
            "label": "xgb_ranker_brand",
            "model": "xgb_ranker",
            "mae_mean": rk["summary"]["mae_mean"],
            "mae_std": rk["scores"]["mae"].std(),
            "rmse_mean": rk["summary"]["rmse_mean"],
            "spearman_mean": rk["summary"]["spearman_mean"],
            "mae_bri_mean": rk["summary"]["mae_bri_mean"],
            "target_mode": "raw",
            "use_target_priors": True,
        }
    )

    # brand-median naive for reference
    gkf = GroupKFold(n_splits=n_splits)
    naive_rows = []
    for fold, (tr, te) in enumerate(gkf.split(np.arange(len(data.y)), data.y, data.groups)):
        base = brand_baselines(data.y[tr], data.brands[tr])
        pred = np.array([base[str(b)] for b in data.brands[te]])
        m = _metrics(data.y[te], pred, data.brands[te], base)
        naive_rows.append(m)
    nd = pd.DataFrame(naive_rows)
    rows.append(
        {
            "label": "naive_brand_median",
            "model": "naive",
            "mae_mean": nd["mae"].mean(),
            "mae_std": nd["mae"].std(),
            "rmse_mean": nd["rmse"].mean(),
            "spearman_mean": nd["spearman"].mean(),
            "mae_bri_mean": nd["mae_bri"].mean(),
            "rmsle_mean": nd["rmsle"].mean(),
            "mae_log_mean": nd["mae_log"].mean(),
            "smape_mean": nd["smape"].mean(),
            "target_mode": "—",
            "use_target_priors": False,
        }
    )

    out = pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)
    cols = [
        "label",
        "mae_mean",
        "rmse_mean",
        "rmsle_mean",
        "mae_log_mean",
        "smape_mean",
        "spearman_mean",
        "mae_bri_mean",
        "target_mode",
        "use_target_priors",
    ]
    return out[[c for c in cols if c in out.columns]]
