"""Scalar metric summaries for EDA / Theme 1."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .data import BRI_COL, VIEWS_COL, WER_COL


def skewed_summary(series: pd.Series, name: Optional[str] = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out = {
        "n": int(len(s)),
        "mean": float(s.mean()) if len(s) else np.nan,
        "median": float(s.median()) if len(s) else np.nan,
        "std": float(s.std()) if len(s) else np.nan,
        "p90": float(s.quantile(0.9)) if len(s) else np.nan,
        "p99": float(s.quantile(0.99)) if len(s) else np.nan,
        "max": float(s.max()) if len(s) else np.nan,
        "skew": float(s.skew()) if len(s) else np.nan,
        "mean_over_median": float(s.mean() / s.median()) if len(s) and s.median() else np.nan,
    }
    return pd.Series(out, name=name or getattr(series, "name", "value"))


def outcome_health(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in (VIEWS_COL, WER_COL, BRI_COL):
        if col in df.columns:
            rows.append(skewed_summary(df[col], name=col))
    return pd.DataFrame(rows)


def leakage_checklist() -> list[str]:
    return [
        "Do not use absolute like/comment/share/collect counts as features to predict WER/BRI.",
        "Do not use BRI as the headline KPI for Nike vs Adidas brand strength (use median WER / views).",
        "Exploding multi-label columns inflates row counts — report coverage_n vs source_n.",
        "Sparse rules (content_type ~20%, social_mechanic ~12%): always show labeled subset size.",
        "Predictive models (notebook 03): predict raw WER from pre-publish features only; convert to BRI via train-fold brand median WER.",
        "No post-publish engagement/view counts or derived rates as model features.",
    ]
