"""Theme 2: Social Commerce Signals.

Core variable
-------------
Commerce Intensity = Purchase CTA + Discovery CTA + Promo Language + Giveaway

Score (0-4) is binned as:
  0 → None, 1 → Low, 2 → Medium, 3+ → High

All Theme 2 tables hang off this single intensity axis (no separate "Direct
Commerce" construct).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .data import BRI_COL, LIST_LABEL_COLS, WER_COL, explode_list_col
from tiktok_brand.etl.cta_rules import detect_cta_flags

COMMERCE_COMPONENTS = (
    "has_purchase_cta",
    "has_discovery_traffic_cta",
    "has_promo_language",
    "has_giveaway",
)

INTENSITY_ORDER = ("None", "Low", "Medium", "High")
INTENSITY_CAT = pd.CategoricalDtype(categories=list(INTENSITY_ORDER), ordered=True)

CONTENT_DIMS = (
    "content_type",
    "content_cluster",
    "brand_styles",
    "visual_format",
    "visual_setting",
)
PRODUCT_DIMS = (
    "product_categories",
    "product_lines",
)


def _as_bool(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def ensure_giveaway_flag(
    df: pd.DataFrame,
    *,
    caption_col: str = "caption_raw",
) -> pd.DataFrame:
    """Add ``has_giveaway`` from caption if the feature table lacks it."""
    out = df
    if "has_giveaway" in out.columns:
        return out
    out = out.copy()
    if caption_col not in out.columns:
        out["has_giveaway"] = False
        return out
    out["has_giveaway"] = (
        out[caption_col]
        .fillna("")
        .map(lambda t: bool(detect_cta_flags(str(t)).get("has_giveaway")))
    )
    return out


def score_to_intensity(score: pd.Series) -> pd.Series:
    """Map 0-4 component sum → None / Low / Medium / High."""
    s = pd.to_numeric(score, errors="coerce").fillna(0).astype(int)
    labels = np.where(
        s <= 0,
        "None",
        np.where(s == 1, "Low", np.where(s == 2, "Medium", "High")),
    )
    return pd.Series(labels, index=score.index, dtype=object).astype(INTENSITY_CAT)


def add_commerce_intensity(
    df: pd.DataFrame,
    *,
    caption_col: str = "caption_raw",
) -> pd.DataFrame:
    """Attach commerce component flags, score (0-4), and intensity bin."""
    out = ensure_giveaway_flag(df, caption_col=caption_col).copy()
    missing = [c for c in COMMERCE_COMPONENTS if c not in out.columns]
    if missing:
        raise KeyError(f"Missing commerce components: {missing}")

    for c in COMMERCE_COMPONENTS:
        out[c] = _as_bool(out[c])

    out["commerce_intensity_score"] = (
        out[list(COMMERCE_COMPONENTS)].astype(int).sum(axis=1).astype(int)
    )
    out["commerce_intensity"] = score_to_intensity(out["commerce_intensity_score"])
    return out


def _prepare_dim(df: pd.DataFrame, label: str) -> pd.DataFrame:
    work = df
    if label in LIST_LABEL_COLS or label in {"content_type", "brand_styles", "product_lines", "product_categories"}:
        if label in work.columns:
            work = explode_list_col(work, label)
    return work


def commerce_by_dimension(
    df: pd.DataFrame,
    label: str,
    *,
    min_count: int = 15,
) -> pd.DataFrame:
    """Parallel cross-tab: annotation dimension × Commerce Intensity.

    Returns one row per label level with mean score, intensity mix, and
    component rates (purchase / discovery / promo / giveaway).
    """
    if "commerce_intensity_score" not in df.columns:
        raise KeyError("Run add_commerce_intensity() first")

    work = _prepare_dim(df, label)
    if label not in work.columns:
        raise KeyError(f"Column not found: {label}")

    g = work.groupby(label, dropna=False)
    out = g.size().rename("n").to_frame()
    out["mean_commerce_intensity"] = g["commerce_intensity_score"].mean()
    high = (work["commerce_intensity"].astype(str) == "High").groupby(
        work[label], dropna=False
    ).mean()
    out["high_commerce_share"] = high
    for c in COMMERCE_COMPONENTS:
        out[c.replace("has_", "") + "_rate"] = g[c].mean()

    # Intensity mix (row %)
    mix = (
        pd.crosstab(work[label], work["commerce_intensity"], normalize="index")
        .reindex(columns=list(INTENSITY_ORDER), fill_value=0.0)
    )
    for lvl in INTENSITY_ORDER:
        out[f"share_{lvl.lower()}"] = mix[lvl]

    out = out[out["n"] >= min_count].sort_values(
        "mean_commerce_intensity", ascending=False
    )
    out.attrs["coverage_n"] = len(work)
    out.attrs["source_n"] = len(df)
    out.attrs["label"] = label
    return out.reset_index().rename(columns={label: "level"})


def brand_commerce_summary(
    df: pd.DataFrame,
    *,
    brand_col: str = "brand",
    brands: Sequence[str] = ("nike", "adidas"),
) -> pd.DataFrame:
    """Brand-level commerce orientation: mean score, component rates, mix."""
    if "commerce_intensity_score" not in df.columns:
        raise KeyError("Run add_commerce_intensity() first")

    rows = []
    for b in brands:
        sub = df.loc[df[brand_col] == b]
        if sub.empty:
            continue
        row = {
            "brand": b,
            "n": int(len(sub)),
            "mean_commerce_intensity": float(sub["commerce_intensity_score"].mean()),
            "high_commerce_share": float((sub["commerce_intensity"].astype(str) == "High").mean()),
        }
        for c in COMMERCE_COMPONENTS:
            row[c.replace("has_", "") + "_rate"] = float(sub[c].mean())
        mix = sub["commerce_intensity"].astype(str).value_counts(normalize=True)
        for lvl in INTENSITY_ORDER:
            row[f"share_{lvl.lower()}"] = float(mix.get(lvl, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def brand_commerce_mix(
    df: pd.DataFrame,
    *,
    brand_col: str = "brand",
) -> pd.DataFrame:
    """Long counts + share for stacked brand × intensity bars."""
    if "commerce_intensity" not in df.columns:
        raise KeyError("Run add_commerce_intensity() first")
    counts = (
        df.groupby([brand_col, "commerce_intensity"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = counts.groupby(brand_col)["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    counts["commerce_intensity"] = counts["commerce_intensity"].astype(INTENSITY_CAT)
    return counts.sort_values([brand_col, "commerce_intensity"])


def commerce_engagement_trend(
    df: pd.DataFrame,
    *,
    bri_col: str = BRI_COL,
    wer_col: str = WER_COL,
    top_q: float = 0.90,
) -> pd.DataFrame:
    """Overall Commerce Intensity × engagement (median BRI / WER / Top-q rate)."""
    if "commerce_intensity" not in df.columns:
        raise KeyError("Run add_commerce_intensity() first")

    wer = pd.to_numeric(df[wer_col], errors="coerce")
    thr = float(wer.quantile(top_q))
    is_top = wer >= thr

    rows = []
    for lvl in INTENSITY_ORDER:
        mask = df["commerce_intensity"].astype(str) == lvl
        sub = df.loc[mask]
        n = int(mask.sum())
        rows.append(
            {
                "commerce_intensity": lvl,
                "n": n,
                "median_bri": float(pd.to_numeric(sub[bri_col], errors="coerce").median())
                if n
                else np.nan,
                "median_wer": float(wer.loc[mask].median()) if n else np.nan,
                "top10_rate": float(is_top.loc[mask].mean()) if n else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["commerce_intensity"] = out["commerce_intensity"].astype(INTENSITY_CAT)
    out.attrs["top_q"] = top_q
    out.attrs["wer_threshold"] = thr
    return out


def commerce_engagement_bubbles(
    df: pd.DataFrame,
    label: str,
    *,
    bri_col: str = BRI_COL,
    min_count: int = 20,
) -> pd.DataFrame:
    """Group bubbles: mean commerce score (x) × median BRI (y), size = n."""
    if "commerce_intensity_score" not in df.columns:
        raise KeyError("Run add_commerce_intensity() first")

    work = _prepare_dim(df, label)
    g = work.groupby(label, dropna=False)
    out = g.size().rename("n").to_frame()
    out["mean_commerce_intensity"] = g["commerce_intensity_score"].mean()
    out["median_bri"] = g[bri_col].median()
    out = out[out["n"] >= min_count].sort_values("n", ascending=False)
    out.attrs["label"] = label
    out.attrs["coverage_n"] = len(work)
    out.attrs["source_n"] = len(df)
    return out.reset_index().rename(columns={label: "level"})


def brand_driver_commerce(
    df: pd.DataFrame,
    label: str,
    *,
    brand_col: str = "brand",
    brands: Sequence[str] = ("nike", "adidas"),
    min_count: int = 10,
) -> pd.DataFrame:
    """Per-brand mean commerce intensity by annotation level (what drives brand gap)."""
    frames = []
    for b in brands:
        sub = df.loc[df[brand_col] == b]
        tab = commerce_by_dimension(sub, label, min_count=min_count)
        tab.insert(0, "brand", b)
        frames.append(tab)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
