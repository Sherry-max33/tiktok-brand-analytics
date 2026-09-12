"""Descriptive crosstab helpers for Theme 1 (Awareness / Engagement)."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from .data import LIST_LABEL_COLS, WER_COL, as_list, explode_list_col


def crosstab_performance(
    df: pd.DataFrame,
    row: str,
    col: Optional[str] = None,
    *,
    value_col: str = WER_COL,
    views_col: str = "view_count",
    include_views_median: bool = False,
    normalize: str = "index",
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Count + row% (or col%) + median performance.

    - One-way: ``col=None`` → label distribution + median ``value_col``.
    - Two-way: counts, normalize % along ``normalize`` ('index'|'columns'),
      and median ``value_col`` per cell.
    List-label columns are exploded first.
    """
    work = df.copy()
    dims = [d for d in (row, col) if d]
    for d in dims:
        if d in LIST_LABEL_COLS:
            work = explode_list_col(work, d)

    if col is None:
        g = work.groupby(row, dropna=False)
        out = g.size().rename("count").to_frame()
        out["pct"] = out["count"] / out["count"].sum()
        out["median_wer"] = g[value_col].median()
        if include_views_median and views_col in work.columns:
            out["median_views"] = g[views_col].median()
        out = out[out["count"] >= min_count].sort_values("count", ascending=False)
        out.attrs["coverage_n"] = len(work)
        out.attrs["source_n"] = len(df)
        return out

    counts = pd.crosstab(work[row], work[col], dropna=False)
    if normalize == "index":
        pct = counts.div(counts.sum(axis=1).replace(0, pd.NA), axis=0)
    elif normalize == "columns":
        pct = counts.div(counts.sum(axis=0).replace(0, pd.NA), axis=1)
    else:
        pct = counts / counts.values.sum()

    med = work.pivot_table(
        index=row, columns=col, values=value_col, aggfunc="median", dropna=False
    )
    long = (
        counts.stack()
        .rename("count")
        .to_frame()
        .join(pct.stack().rename("pct"))
        .join(med.stack().rename("median_wer"))
        .reset_index()
    )
    long = long[long["count"] >= min_count].sort_values("count", ascending=False)
    long.attrs["coverage_n"] = len(work)
    long.attrs["source_n"] = len(df)
    return long


def brand_vs_label(
    df: pd.DataFrame,
    label: str,
    *,
    brand_col: str = "brand",
    value_col: str = WER_COL,
    min_count: int = 5,
) -> pd.DataFrame:
    """Core Theme 1 table: brand × label with count / brand-row% / median WER."""
    return crosstab_performance(
        df,
        brand_col,
        label,
        value_col=value_col,
        normalize="index",
        min_count=min_count,
    )


def one_way_with_wer(
    df: pd.DataFrame,
    label: str,
    *,
    value_col: str = WER_COL,
    min_count: int = 5,
    include_views_median: bool = False,
) -> pd.DataFrame:
    return crosstab_performance(
        df,
        label,
        None,
        value_col=value_col,
        min_count=min_count,
        include_views_median=include_views_median,
    )


def mannwhitney_wer_by_brand(
    df: pd.DataFrame,
    *,
    brand_col: str = "brand",
    value_col: str = WER_COL,
    brands: Sequence[str] = ("nike", "adidas"),
) -> dict:
    """Two-brand Mann–Whitney on WER + medians."""
    from scipy import stats

    a, b = brands[0], brands[1]
    xa = df.loc[df[brand_col] == a, value_col].dropna()
    xb = df.loc[df[brand_col] == b, value_col].dropna()
    if len(xa) == 0 or len(xb) == 0:
        return {"error": "empty group"}
    stat, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
    return {
        f"median_{a}": float(xa.median()),
        f"median_{b}": float(xb.median()),
        f"n_{a}": int(len(xa)),
        f"n_{b}": int(len(xb)),
        "U": float(stat),
        "p_value": float(p),
    }
