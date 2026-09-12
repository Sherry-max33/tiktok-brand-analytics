"""Within-brand relative performance vs brand median WER baseline.

Observational only — relative_index is not a causal lift estimate.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .data import LIST_LABEL_COLS, WER_COL, as_list, explode_list_col
from .present import format_int, format_pct, format_wer_pct, humanize_label, humanize_series

RELIABLE_N = 20
SMALL_N_REL = 20


def brand_baseline_wer(
    df: pd.DataFrame,
    *,
    brand_col: str = "brand",
    value_col: str = WER_COL,
) -> pd.Series:
    """Median WER by brand (full brand sample, not category-filtered)."""
    return (
        df.groupby(brand_col, dropna=False)[value_col]
        .median()
        .rename("brand_baseline_wer")
    )


def relative_performance_by_category(
    df: pd.DataFrame,
    label: str,
    *,
    brand_col: str = "brand",
    value_col: str = WER_COL,
    min_count: int = 5,
    brands: Sequence[str] = ("adidas", "nike"),
) -> pd.DataFrame:
    """
    Per brand × category: n, median WER, brand baseline, relative index/uplift, share.

    ``relative_index = median_wer / brand_baseline_wer``
    ``relative_uplift = relative_index - 1``
    """
    baselines = brand_baseline_wer(df, brand_col=brand_col, value_col=value_col)
    brand_totals = df.groupby(brand_col, dropna=False).size().rename("brand_n")

    work = df.copy()
    if label in LIST_LABEL_COLS:
        work = explode_list_col(work, label)

    rows = []
    for brand in brands:
        g = work[work[brand_col] == brand]
        if g.empty:
            continue
        base = float(baselines.get(brand, np.nan))
        total = int(brand_totals.get(brand, 0))
        for cat, sub in g.groupby(label, dropna=False):
            n = int(len(sub))
            if n < min_count:
                continue
            med = float(sub[value_col].median())
            rel = med / base if base and np.isfinite(base) and base != 0 else np.nan
            rows.append(
                {
                    "brand": brand,
                    "category": cat,
                    "n": n,
                    "median_wer": med,
                    "brand_baseline_wer": base,
                    "relative_index": rel,
                    "relative_uplift": rel - 1.0 if np.isfinite(rel) else np.nan,
                    "share_within_brand": n / total if total else np.nan,
                    "brand_n": total,
                    "small_n": n < SMALL_N_REL,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["brand", "relative_uplift"], ascending=[True, False]).reset_index(drop=True)
    out.attrs["label"] = label
    out.attrs["baselines"] = baselines.to_dict()
    out.attrs["source_n"] = len(df)
    return out


def relative_display_table(raw: pd.DataFrame, *, reliable_only: bool = False) -> pd.DataFrame:
    """Compact presentation table for within-brand relative performance."""
    t = raw.copy()
    if reliable_only:
        t = t[~t["small_n"]].copy()
    t = t.sort_values(["brand", "relative_uplift"], ascending=[True, False])
    out = pd.DataFrame(
        {
            "Brand": t["brand"].astype(str).str.title(),
            "Category": humanize_series(t["category"]),
            "n": t["n"].map(format_int),
            "Median WER": t["median_wer"].map(lambda x: format_wer_pct(x, 2)),
            "Brand Baseline": t["brand_baseline_wer"].map(lambda x: format_wer_pct(x, 2)),
            "Relative Index": t["relative_index"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—"),
            "Relative Uplift": t["relative_uplift"].map(
                lambda x: f"{x:+.0%}" if pd.notna(x) else "—"
            ),
        }
    )
    out.attrs.update(raw.attrs)
    return out


def category_vs_rest_tests(
    df: pd.DataFrame,
    label: str,
    candidates: pd.DataFrame,
    *,
    brand_col: str = "brand",
    value_col: str = WER_COL,
    min_n: int = RELIABLE_N,
    fdr: bool = True,
) -> pd.DataFrame:
    """
    Mann–Whitney: category WER vs rest of same brand.

    ``candidates`` should be rows from relative_performance_by_category
    (typically reliable outperformers). BH-FDR applied when ``fdr=True``.
    """
    from scipy import stats

    rows = []
    work = df.copy()
    is_list = label in LIST_LABEL_COLS

    for _, row in candidates.iterrows():
        brand = row["brand"]
        cat = row["category"]
        brand_df = work[work[brand_col] == brand]
        if is_list:
            if pd.isna(cat):
                in_mask = brand_df[label].map(lambda v: len(as_list(v)) == 0)
            else:
                in_mask = brand_df[label].map(lambda v, c=cat: c in as_list(v))
        else:
            in_mask = brand_df[label].isna() if pd.isna(cat) else (brand_df[label] == cat)

        xa = pd.to_numeric(brand_df.loc[in_mask, value_col], errors="coerce").dropna()
        xb = pd.to_numeric(brand_df.loc[~in_mask, value_col], errors="coerce").dropna()
        if len(xa) < min_n or len(xb) < 5:
            continue
        u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        r_rb = 1.0 - (2.0 * u) / (len(xa) * len(xb))
        rows.append(
            {
                "brand": brand,
                "category": cat,
                "n_category": int(len(xa)),
                "n_other": int(len(xb)),
                "median_wer_category": float(xa.median()),
                "median_wer_other": float(xb.median()),
                "U": float(u),
                "p_value": float(p),
                "rank_biserial": float(r_rb),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if fdr and len(out) > 1:
        out["p_adj"] = _bh_fdr(out["p_value"].to_numpy())
    else:
        out["p_adj"] = out["p_value"]
    return out.sort_values("p_adj").reset_index(drop=True)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adj
    return out


def strategic_bucket(row: pd.Series, *, uplift_cut: float = 0.10, share_cut: float = 0.05) -> str:
    """A niche winner / B scalable winner / C high-volume underperformer / other."""
    up = row.get("relative_uplift", np.nan)
    share = row.get("share_within_brand", np.nan)
    n = row.get("n", 0)
    small = bool(row.get("small_n", n < SMALL_N_REL))
    if not np.isfinite(up):
        return "other"
    if up >= uplift_cut and small:
        return "A_niche_high"
    if up >= uplift_cut and n >= SMALL_N_REL and (share >= share_cut or n >= 50):
        return "B_scalable_winner"
    if up >= uplift_cut and n >= SMALL_N_REL:
        return "B_reliable_winner"
    if up < 0 and np.isfinite(share) and share >= share_cut and n >= SMALL_N_REL:
        return "C_high_volume_under"
    return "other"


def auto_findings(
    content_type_rel: pd.DataFrame,
    brand_styles_rel: pd.DataFrame,
    *,
    tests: Optional[pd.DataFrame] = None,
) -> list[str]:
    """Concise observational findings for section 4.2 summary."""
    lines: list[str] = []

    def _top(rel: pd.DataFrame, brand: str, k: int = 3) -> pd.DataFrame:
        sub = rel[(rel["brand"] == brand) & (~rel["small_n"]) & (rel["relative_uplift"] > 0)]
        return sub.sort_values("relative_uplift", ascending=False).head(k)

    def _fmt_row(r: pd.Series) -> str:
        return (
            f"{humanize_label(r['category'])} "
            f"({r['relative_uplift']:+.0%}, n={int(r['n'])})"
        )

    for brand, title in (("adidas", "Adidas"), ("nike", "Nike")):
        tops = _top(content_type_rel, brand)
        if len(tops):
            lines.append(
                f"{title} content types associated with above-baseline median WER: "
                + "; ".join(_fmt_row(r) for _, r in tops.iterrows())
                + "."
            )
        else:
            lines.append(f"{title}: no reliable content_type categories above brand baseline (n≥{SMALL_N_REL}).")

    for brand, title in (("adidas", "Adidas"), ("nike", "Nike")):
        tops = _top(brand_styles_rel, brand, k=2)
        if len(tops):
            best = tops.iloc[0]
            lines.append(
                f"{title} strongest brand style (relative to own baseline): {_fmt_row(best)}."
            )

    # territory overlap
    def _set(rel: pd.DataFrame, brand: str) -> set:
        sub = rel[(rel["brand"] == brand) & (~rel["small_n"]) & (rel["relative_uplift"] > 0.05)]
        return set(humanize_series(sub["category"]))

    a_ct, n_ct = _set(content_type_rel, "adidas"), _set(content_type_rel, "nike")
    if a_ct or n_ct:
        shared = a_ct & n_ct
        only_a, only_n = a_ct - n_ct, n_ct - a_ct
        if shared:
            lines.append(
                "Shared above-baseline content types: " + ", ".join(sorted(shared)) + "."
            )
        if only_a or only_n:
            bits = []
            if only_a:
                bits.append("Adidas-leaning: " + ", ".join(sorted(only_a)))
            if only_n:
                bits.append("Nike-leaning: " + ", ".join(sorted(only_n)))
            lines.append(
                "Brands appear to win through partly different content territories — "
                + "; ".join(bits)
                + "."
            )

    # small-n caution
    caution = pd.concat(
        [
            content_type_rel[content_type_rel["small_n"] & (content_type_rel["relative_uplift"] > 0.15)],
            brand_styles_rel[brand_styles_rel["small_n"] & (brand_styles_rel["relative_uplift"] > 0.15)],
        ],
        ignore_index=True,
    )
    if len(caution):
        examples = ", ".join(
            f"{r['brand'].title()} {humanize_label(r['category'])} (n={int(r['n'])})"
            for _, r in caution.head(4).iterrows()
        )
        lines.append(
            f"Caution: some high relative-uplift cells are small-n and should not be treated as strategy winners — e.g. {examples}."
        )

    # scalable winners
    for rel, dim in ((content_type_rel, "content_type"), (brand_styles_rel, "brand_styles")):
        tmp = rel.copy()
        tmp["bucket"] = tmp.apply(strategic_bucket, axis=1)
        scal = tmp[tmp["bucket"].isin(["B_scalable_winner", "B_reliable_winner"])]
        if len(scal):
            bits = [
                f"{r['brand'].title()} {humanize_label(r['category'])} ({r['relative_uplift']:+.0%}, n={int(r['n'])})"
                for _, r in scal.sort_values("relative_uplift", ascending=False).head(4).iterrows()
            ]
            lines.append(
                f"Higher-confidence {dim.replace('_', ' ')} territories (sufficient n and above baseline): "
                + "; ".join(bits)
                + "."
            )
            break

    lines.append(
        "These patterns are observational associations with median engagement vs each brand’s own baseline — not causal effects."
    )
    return lines


def tests_display_table(tests: pd.DataFrame) -> pd.DataFrame:
    if tests is None or tests.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Brand": tests["brand"].astype(str).str.title(),
            "Category": humanize_series(tests["category"]),
            "n (cat)": tests["n_category"].map(format_int),
            "n (other)": tests["n_other"].map(format_int),
            "Med WER cat": tests["median_wer_category"].map(lambda x: format_wer_pct(x, 2)),
            "Med WER other": tests["median_wer_other"].map(lambda x: format_wer_pct(x, 2)),
            "p": tests["p_value"].map(lambda x: f"{x:.2e}" if x < 1e-3 else f"{x:.3f}"),
            "p_adj (BH)": tests["p_adj"].map(lambda x: f"{x:.2e}" if x < 1e-3 else f"{x:.3f}"),
            "Effect (r)": tests["rank_biserial"].map(lambda x: f"{x:+.2f}"),
        }
    )
