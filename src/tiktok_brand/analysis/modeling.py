"""Engagement prediction: pre-publish features → WER → BRI.

No post-publish engagement/view features. Compare CatBoost vs XGBoost
under author-grouped CV; select on raw-WER MAE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder

from .data import LIST_LABEL_COLS, WER_COL, as_list

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
    "post_hour",
    "post_weekday",
]

BOOL_COLS = [
    "author_verified",
    "is_official_brand",
    "has_cta",
    "has_purchase_cta",
    "has_discovery_traffic_cta",
    "has_promo_language",
    "has_music",
    "is_sample_trending_audio",
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

LIST_COLS = [
    "content_type",
    "brand_styles",
    "social_mechanic",
]

EMB_COL = "cluster_text_embedding"
GROUP_COL = "author_id"
BRAND_COL = "brand"
DEFAULT_PCA_DIMS = 32
MISSING = "__MISSING__"


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


@dataclass
class FoldMatrices:
    X_train: np.ndarray
    X_test: np.ndarray
    feature_names: list[str]
    cat_indices: list[int]  # for CatBoost (ordinal-coded cats live here as ints)
    brand_baseline_train: dict[str, float]


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


class FeatureBuilder:
    """Fit/transform tabular + PCA(embeddings) inside each CV fold."""

    def __init__(self, pca_dims: int = DEFAULT_PCA_DIMS, list_min_count: int = 5):
        self.pca_dims = pca_dims
        self.list_min_count = list_min_count
        self.list_vocabs: dict[str, list[str]] = {}
        self.ord_encoder: Optional[OrdinalEncoder] = None
        self.pca: Optional[PCA] = None
        self.num_cols: list[str] = []
        self.bool_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.feature_names: list[str] = []
        self.cat_indices: list[int] = []

    def fit(self, frame: pd.DataFrame) -> "FeatureBuilder":
        self.num_cols = [c for c in NUMERIC_COLS if c in frame.columns]
        self.bool_cols = [c for c in BOOL_COLS if c in frame.columns]
        self.cat_cols = [c for c in CAT_COLS if c in frame.columns]
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

        emb = _embedding_matrix(frame[EMB_COL])
        n_comp = min(self.pca_dims, emb.shape[0], emb.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=42)
        self.pca.fit(emb)

        # names / cat indices (cats placed after numeric+bool+multihot)
        names: list[str] = []
        for c in self.num_cols:
            names.append(f"{c}_log1p" if c == "author_follower_count" else c)
        names.extend(self.bool_cols)
        for c, vocab in self.list_vocabs.items():
            names.extend([f"{c}={v}" for v in vocab])
        cat_start = len(names)
        names.extend(self.cat_cols)
        self.cat_indices = list(range(cat_start, cat_start + len(self.cat_cols)))
        names.extend([f"emb_pca_{i}" for i in range(self.pca.n_components_)])
        self.feature_names = names
        assert_no_leakage(self.feature_names)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        n = len(frame)
        if self.num_cols:
            num = frame[self.num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            # log1p followers in first matching col
            for i, c in enumerate(self.num_cols):
                if c == "author_follower_count":
                    num[:, i] = np.log1p(np.clip(num[:, i], 0, None))
            num = np.nan_to_num(num, nan=-1.0)
            parts.append(num.astype(np.float32))
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
        assert self.pca is not None
        emb = _embedding_matrix(frame[EMB_COL])
        parts.append(self.pca.transform(emb).astype(np.float32))
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
) -> FoldMatrices:
    builder = FeatureBuilder(pca_dims=pca_dims)
    builder.fit(data.frame.iloc[train_idx])
    X_train = builder.transform(data.frame.iloc[train_idx])
    X_test = builder.transform(data.frame.iloc[test_idx])
    baselines = brand_baselines(data.y[train_idx], data.brands[train_idx])
    return FoldMatrices(
        X_train=X_train,
        X_test=X_test,
        feature_names=builder.feature_names,
        cat_indices=builder.cat_indices,
        brand_baseline_train=baselines,
    )


@dataclass
class FoldScore:
    model: str
    fold: int
    mae: float
    rmse: float
    spearman: float
    mae_bri: float
    n_test: int


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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, brands: np.ndarray, baselines: dict[str, float]) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bri_true = wer_to_bri(y_true, brands, baselines)
    bri_pred = wer_to_bri(y_pred, brands, baselines)
    mae_bri = float(np.mean(np.abs(bri_pred - bri_true)))
    return {
        "mae": mae,
        "rmse": rmse,
        "spearman": _spearman(y_true, y_pred),
        "mae_bri": mae_bri,
    }


def fit_catboost_simple(X_train, y_train, cat_indices: list[int], *, seed: int = 42, iterations: int = 300):
    from catboost import CatBoostRegressor, Pool

    # CatBoost rejects float ndarray + cat_features; cast cat cols to int in a DataFrame.
    n_features = X_train.shape[1]
    cols = [f"f{i}" for i in range(n_features)]
    train_df = pd.DataFrame(X_train, columns=cols)
    for i in cat_indices:
        train_df[cols[i]] = train_df[cols[i]].round().astype(int)
    model = CatBoostRegressor(
        loss_function="MAE",
        iterations=iterations,
        depth=6,
        learning_rate=0.06,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )
    pool = Pool(train_df, y_train, cat_features=[cols[i] for i in cat_indices] if cat_indices else None)
    model.fit(pool)
    model._tiktok_feature_cols = cols  # type: ignore[attr-defined]
    model._tiktok_cat_indices = list(cat_indices)  # type: ignore[attr-defined]
    return model


def _catboost_predict(model, X) -> np.ndarray:
    cols = getattr(model, "_tiktok_feature_cols", [f"f{i}" for i in range(X.shape[1])])
    cat_indices = getattr(model, "_tiktok_cat_indices", [])
    df = pd.DataFrame(X, columns=cols)
    for i in cat_indices:
        df[cols[i]] = df[cols[i]].round().astype(int)
    return model.predict(df)


def fit_xgboost(X_train, y_train, *, seed: int = 42, n_estimators: int = 300):
    from xgboost import XGBRegressor

    model = XGBRegressor(
        objective="reg:absoluteerror",
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


def run_grouped_cv(
    data: ModelData,
    *,
    n_splits: int = 5,
    pca_dims: int = DEFAULT_PCA_DIMS,
    seed: int = 42,
    models: Sequence[str] = ("catboost", "xgboost"),
    iterations: int = 300,
) -> dict[str, CVResult]:
    """Author-grouped K-fold CV for CatBoost and/or XGBoost."""
    gkf = GroupKFold(n_splits=n_splits)
    results: dict[str, CVResult] = {}

    for model_name in models:
        oof = np.full(len(data.y), np.nan, dtype=float)
        rows = []
        imp_acc: dict[str, list[float]] = {}
        for fold, (tr, te) in enumerate(gkf.split(np.arange(len(data.y)), data.y, data.groups)):
            mats = make_fold_matrices(data, tr, te, pca_dims=pca_dims)
            if model_name == "catboost":
                model = fit_catboost_simple(
                    mats.X_train, data.y[tr], mats.cat_indices, seed=seed + fold, iterations=iterations
                )
                pred = _catboost_predict(model, mats.X_test)
                fi = model.get_feature_importance()
                for name, val in zip(mats.feature_names, fi):
                    imp_acc.setdefault(name, []).append(float(val))
            elif model_name == "xgboost":
                model = fit_xgboost(
                    mats.X_train, data.y[tr], seed=seed + fold, n_estimators=iterations
                )
                pred = model.predict(mats.X_test)
                fi = model.feature_importances_
                for name, val in zip(mats.feature_names, fi):
                    imp_acc.setdefault(name, []).append(float(val))
            else:
                raise ValueError(model_name)

            oof[te] = pred
            m = _metrics(data.y[te], pred, data.brands[te], mats.brand_baseline_train)
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "mae": m["mae"],
                    "rmse": m["rmse"],
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
        rows.append(
            {
                "model": name,
                "mae_mean": s["mae"].mean(),
                "mae_std": s["mae"].std(),
                "rmse_mean": s["rmse"].mean(),
                "spearman_mean": s["spearman"].mean(),
                "mae_bri_mean": s["mae_bri"].mean(),
            }
        )
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
) -> dict[str, Any]:
    """Single author-grouped holdout for the selected model."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(np.arange(len(data.y)), data.y, data.groups))
    mats = make_fold_matrices(data, tr, te, pca_dims=pca_dims)
    if model_name == "catboost":
        model = fit_catboost_simple(
            mats.X_train, data.y[tr], mats.cat_indices, seed=seed, iterations=iterations
        )
        pred = _catboost_predict(model, mats.X_test)
        fi = model.get_feature_importance()
    elif model_name == "xgboost":
        model = fit_xgboost(mats.X_train, data.y[tr], seed=seed, n_estimators=iterations)
        pred = model.predict(mats.X_test)
        fi = model.feature_importances_
    else:
        raise ValueError(model_name)

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
        "model": model,
    }
