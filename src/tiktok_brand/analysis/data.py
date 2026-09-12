"""Load feature table and small prep helpers for analysis notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union

import pandas as pd
import yaml

LIST_LABEL_COLS = ("content_type", "social_mechanic", "brand_styles", "product_lines", "product_categories")
CLUSTER_COL = "content_cluster_id_clean_k12"
WER_COL = "weighted_engagement_rate"
BRI_COL = "brand_relative_engagement_index"
VIEWS_COL = "view_count"


def project_paths(project_yaml: str | Path = "configs/project.yaml") -> dict[str, Path]:
    cfg = yaml.safe_load(Path(project_yaml).read_text(encoding="utf-8"))
    out = cfg.get("output") or {}
    feature_dir = Path(out.get("feature_dir", "data/processed/feature"))
    return {
        "feature_dir": feature_dir,
        "feature_table": feature_dir / "feature_table.parquet",
        "feature_rules": Path(out.get("feature_rules_cfg", "configs/feature_rules.yaml")),
    }


def load_feature_table(
    path: str | Path | None = None,
    *,
    project_yaml: str | Path = "configs/project.yaml",
) -> pd.DataFrame:
    if path is None:
        path = project_paths(project_yaml)["feature_table"]
    return pd.read_parquet(path)


def as_list(val: Any) -> List[Any]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return val
    try:
        import numpy as np

        if isinstance(val, np.ndarray):
            return val.tolist()
    except ImportError:
        pass
    return []


def nonempty_list_mask(series: pd.Series) -> pd.Series:
    return series.map(lambda x: len(as_list(x)) > 0)


def explode_list_col(
    df: pd.DataFrame,
    col: str,
    *,
    value_name: Optional[str] = None,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """One row per list element (for multi-label crosstabs)."""
    name = value_name or col
    out = df.copy()
    out[name] = out[col].map(as_list) if name == col else out[col].map(as_list)
    if name != col:
        out = out.drop(columns=[col], errors="ignore")
    out = out.explode(name, ignore_index=True)
    if drop_empty:
        out = out[out[name].notna() & (out[name].astype(str).str.len() > 0)]
    return out.reset_index(drop=True)


def coverage_summary(
    df: pd.DataFrame,
    cols: Sequence[str],
) -> pd.DataFrame:
    """Per-column fill / non-empty rates (list-aware)."""
    rows = []
    n = len(df)
    for c in cols:
        if c not in df.columns:
            rows.append({"column": c, "present": False, "filled": 0, "rate": 0.0})
            continue
        s = df[c]
        if c in LIST_LABEL_COLS:
            filled = int(nonempty_list_mask(s).sum())
        else:
            filled = int(s.notna().sum())
        rows.append(
            {
                "column": c,
                "present": True,
                "filled": filled,
                "rate": filled / n if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def coverage_by_brand(
    df: pd.DataFrame,
    cols: Sequence[str],
    brand_col: str = "brand",
) -> pd.DataFrame:
    pieces = []
    for brand, g in df.groupby(brand_col, dropna=False):
        part = coverage_summary(g, cols)
        part.insert(0, "brand", brand)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
