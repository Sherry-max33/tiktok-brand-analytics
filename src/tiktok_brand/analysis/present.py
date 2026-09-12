"""Display-only formatting for Theme 1 crosstab notebooks.

Does not change statistical calculations — wraps helper outputs for presentation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Categories with count below this are visually de-emphasized.
SMALL_N = 15

BRAND_COLORS = {
    "adidas": "#1a1a1a",
    "nike": "#fa5400",
}

# Shared matplotlib rc for crosstab notebook figures
PLOT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "medium",
    "axes.labelsize": 10,
    "axes.labelcolor": "#222222",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#222222",
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

DEFAULT_CLUSTER_PROFILES = Path("data/processed/feature/content_cluster_profiles_clean_k12.json")


@lru_cache(maxsize=4)
def load_cluster_labels(profiles_path: str | None = None) -> dict[int, str]:
    """Map cluster_id → human_label (fallback: suggested_label) from profiles JSON."""
    path = Path(profiles_path) if profiles_path else DEFAULT_CLUSTER_PROFILES
    if not path.is_file():
        # try repo-relative from CWD parent (notebooks/)
        alt = Path("..") / path
        if alt.is_file():
            path = alt
        else:
            return {}
    try:
        profiles = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[int, str] = {}
    for row in profiles:
        try:
            cid = int(row["cluster_id"])
        except (KeyError, TypeError, ValueError):
            continue
        name = (row.get("human_label") or row.get("suggested_label") or "").strip()
        if name:
            out[cid] = name
    return out


def format_cluster_label(val: Any, *, profiles_path: str | None = None) -> str:
    """0 → '0 — OOTD, Outfit Inspiration & Streetwear'; NaN → Unclassified."""
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        return "Unclassified"
    try:
        cid = int(val)
    except (TypeError, ValueError):
        return humanize_label(val)
    name = load_cluster_labels(profiles_path).get(cid)
    if name:
        return f"{cid} — {name}"
    return f"Cluster {cid}"


def humanize_label(val: Any) -> str:
    """product_showcase → Product Showcase; NaN → Unclassified; cluster ids → id — name."""
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        return "Unclassified"
    if isinstance(val, (int, np.integer)) or (isinstance(val, float) and float(val).is_integer()):
        return format_cluster_label(val)
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return "Unclassified"
    # numeric string cluster ids
    if s.isdigit() or (s.replace(".", "", 1).isdigit() and float(s).is_integer()):
        return format_cluster_label(int(float(s)))
    acronyms = {"ootd", "grwm", "cta", "ugc", "pov", "ama"}
    parts = s.replace("_", " ").split()
    return " ".join(p.upper() if p.lower() in acronyms else p.title() for p in parts)


def humanize_series(s: pd.Series) -> pd.Series:
    return s.map(humanize_label)


def coverage_annotation(table: pd.DataFrame, *, unit: str = "videos") -> str:
    """Coverage: 976 / 4,457 videos (21.9%)."""
    cov = table.attrs.get("coverage_n")
    src = table.attrs.get("source_n")
    if cov is None or src is None or not src:
        return ""
    return f"Coverage: {int(cov):,} / {int(src):,} {unit} ({cov / src:.1%})"


def format_pct(x: Any, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return "—"
    return f"{100 * float(x):.{digits}f}%"


def format_wer_pct(x: Any, digits: int = 2) -> str:
    """Rate like 0.0075 → 0.75%."""
    return format_pct(x, digits=digits)


def format_int(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return "—"
    return f"{int(round(float(x))):,}"


def one_way_display_table(raw: pd.DataFrame, *, label_name: str = "Category") -> pd.DataFrame:
    """Compact supporting table for Part 1 / Part 4 one-way results."""
    t = raw.reset_index()
    label_col = t.columns[0]
    out = pd.DataFrame(
        {
            label_name: humanize_series(t[label_col]),
            "Share": t["pct"].map(lambda x: format_pct(x, 1)),
            "Median WER": t["median_wer"].map(lambda x: format_wer_pct(x, 2)),
            "n": t["count"].map(format_int),
        }
    )
    if "median_views" in t.columns:
        out["Median views"] = t["median_views"].map(format_int)
    out.attrs.update(raw.attrs)
    return out


def brand_pivot_display(raw: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Compact Part 2 pivot: categories as rows, brands as columns.
    Values: within-brand share (%) and median WER (%).
    """
    t = raw.copy()
    label_col = label if label in t.columns else [c for c in t.columns if c not in ("brand", "count", "pct", "median_wer")][0]
    t["_label"] = humanize_series(t[label_col])
    t["_brand"] = t["brand"].astype(str).str.lower()

    share = t.pivot_table(index="_label", columns="_brand", values="pct", aggfunc="first")
    wer = t.pivot_table(index="_label", columns="_brand", values="median_wer", aggfunc="first")
    n = t.pivot_table(index="_label", columns="_brand", values="count", aggfunc="first")

    # sort by total n
    order = n.sum(axis=1).sort_values(ascending=False).index
    brands = [b for b in ("adidas", "nike") if b in share.columns] or list(share.columns)

    cols = {}
    for b in brands:
        btitle = b.title()
        if b in share.columns:
            cols[f"{btitle} share"] = share.loc[order, b].map(lambda x: format_pct(x, 1))
        if b in wer.columns:
            cols[f"{btitle} WER"] = wer.loc[order, b].map(lambda x: format_wer_pct(x, 2))
        if b in n.columns:
            cols[f"{btitle} n"] = n.loc[order, b].map(format_int)

    pretty = {
        "content_type": "Content type",
        "visual_format": "Visual format",
        "social_mechanic": "Social mechanic",
        "brand_styles": "Brand style",
        "content_cluster": "Cluster",
    }
    out = pd.DataFrame(cols)
    out.index.name = pretty.get(label, humanize_label(label))
    out.attrs.update(raw.attrs)
    return out


def heatmap_matrices(raw: pd.DataFrame, row: str, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pct wide, count wide) with humanized index/columns for heatmaps."""
    t = raw.copy()
    t[row] = humanize_series(t[row])
    t[col] = humanize_series(t[col])
    pct = t.pivot_table(index=row, columns=col, values="pct", aggfunc="first")
    cnt = t.pivot_table(index=row, columns=col, values="count", aggfunc="first")
    # sort rows/cols by total count
    if cnt.size:
        row_order = cnt.sum(axis=1).sort_values(ascending=False).index
        col_order = cnt.sum(axis=0).sort_values(ascending=False).index
        pct = pct.reindex(index=row_order, columns=col_order)
        cnt = cnt.reindex(index=row_order, columns=col_order)
    return pct, cnt
