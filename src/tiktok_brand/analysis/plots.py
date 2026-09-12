"""Plot helpers for EDA + Theme 1 crosstab presentation."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .data import BRI_COL, VIEWS_COL, WER_COL
from .present import (
    BRAND_COLORS,
    PLOT_RC,
    SMALL_N,
    coverage_annotation,
    format_pct,
    format_wer_pct,
    heatmap_matrices,
    humanize_label,
    humanize_series,
)


def _require_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def _apply_style(plt):
    plt.rcParams.update(PLOT_RC)


def sample_mix_bars(df: pd.DataFrame, *, brand_col: str = "brand", source_col: str = "source_type"):
    """Side-by-side bars: brand counts + source_type counts."""
    plt = _require_matplotlib()
    _apply_style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    df[brand_col].value_counts().sort_index().plot(
        kind="bar", ax=axes[0], color=["#1a1a1a", "#888"], rot=0
    )
    axes[0].set_title("Videos by brand")
    axes[0].set_ylabel("count")
    if source_col in df.columns:
        df[source_col].value_counts().sort_index().plot(kind="bar", ax=axes[1], color="#444", rot=0)
        axes[1].set_title("Videos by source_type")
    fig.tight_layout()
    return fig


def outcome_dist_charts(
    df: pd.DataFrame,
    *,
    brand_col: str = "brand",
    views_col: str = VIEWS_COL,
    wer_col: str = WER_COL,
    bri_col: str = BRI_COL,
):
    """log1p(views) hist, WER boxplot, BRI hist — all by brand."""
    plt = _require_matplotlib()
    _apply_style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    brands = sorted(df[brand_col].dropna().unique())
    colors = BRAND_COLORS

    ax = axes[0]
    for b in brands:
        x = np.log1p(pd.to_numeric(df.loc[df[brand_col] == b, views_col], errors="coerce").dropna())
        ax.hist(x, bins=40, alpha=0.45, label=b, color=colors.get(b, None), density=True)
    ax.set_title("log1p(view_count)")
    ax.set_xlabel("log1p(views)")
    ax.legend(frameon=False)

    ax = axes[1]
    data = [
        pd.to_numeric(df.loc[df[brand_col] == b, wer_col], errors="coerce").dropna().clip(upper=0.05)
        for b in brands
    ]
    bp = ax.boxplot(data, labels=list(brands), patch_artist=True, showfliers=False)
    for patch, b in zip(bp["boxes"], brands):
        patch.set_facecolor(colors.get(b, "#ccc"))
        patch.set_alpha(0.55)
    ax.set_title("WER (box, clipped @0.05)")
    ax.set_ylabel("weighted_engagement_rate")

    ax = axes[2]
    for b in brands:
        x = pd.to_numeric(df.loc[df[brand_col] == b, bri_col], errors="coerce").dropna().clip(upper=5)
        ax.hist(x, bins=40, alpha=0.45, label=b, color=colors.get(b, None), density=True)
    ax.axvline(1.0, color="#333", ls="--", lw=1, label="median=1")
    ax.set_title("BRI (clipped @5)")
    ax.set_xlabel("brand_relative_engagement_index")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


def coverage_bars(
    coverage_df: pd.DataFrame,
    *,
    rate_col: str = "rate",
    column_col: str = "column",
    title: str = "Feature coverage",
):
    """Horizontal bar of fill rates from coverage_summary()."""
    plt = _require_matplotlib()
    _apply_style(plt)
    plot_df = coverage_df.sort_values(rate_col)
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(plot_df))))
    ax.barh(plot_df[column_col], plot_df[rate_col], color="#444")
    ax.set_xlim(0, 1)
    ax.set_xlabel("fill rate")
    ax.set_title(title)
    for y, r in zip(plot_df[column_col], plot_df[rate_col]):
        ax.text(min(r + 0.02, 0.98), y, f"{r:.1%}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Theme 1 crosstab presentation charts
# ---------------------------------------------------------------------------


def landscape_hbar(
    raw: pd.DataFrame,
    *,
    title: str,
    small_n: int = SMALL_N,
):
    """Part 1: horizontal bars of category share, sorted descending."""
    plt = _require_matplotlib()
    _apply_style(plt)
    t = raw.reset_index()
    label_col = t.columns[0]
    t = t.sort_values("pct", ascending=True)  # barh draws bottom→top
    labels = humanize_series(t[label_col])
    heights = t["pct"].to_numpy(dtype=float)
    counts = t["count"].to_numpy(dtype=float)
    colors = ["#bbbbbb" if c < small_n else "#444444" for c in counts]

    fig_h = max(2.8, 0.38 * len(t) + 0.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.barh(labels, heights, color=colors)
    ax.set_xlabel("Share of labeled set")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
    cov = coverage_annotation(raw)
    ax.set_title(title if not cov else f"{title}\n{cov}", loc="left", pad=10)

    xmax = max(heights.max() * 1.18, 0.05) if len(heights) else 0.1
    ax.set_xlim(0, xmax)
    for y, p, c, n in zip(labels, heights, colors, counts):
        note = f"{format_pct(p, 1)}  (n={int(n):,})"
        if n < small_n:
            note += "  · small n"
        ax.text(p + xmax * 0.01, y, note, va="center", fontsize=8, color="#666" if n < small_n else "#222")

    # room for long cluster names on y-axis
    if labels.astype(str).str.len().max() > 28:
        fig.subplots_adjust(left=0.42)
        fig.tight_layout()
    else:
        fig.tight_layout()
    return fig


def brand_compare_hbar(
    raw: pd.DataFrame,
    label: str,
    *,
    title: str,
    brands: Sequence[str] = ("adidas", "nike"),
    small_n: int = SMALL_N,
):
    """Part 2: grouped horizontal bars of within-brand share."""
    plt = _require_matplotlib()
    _apply_style(plt)
    t = raw.copy()
    label_col = label if label in t.columns else [
        c for c in t.columns if c not in ("brand", "count", "pct", "median_wer")
    ][0]
    t["_label"] = humanize_series(t[label_col])
    t["_brand"] = t["brand"].astype(str).str.lower()

    # category order by total count desc (plot bottom=smallest)
    totals = t.groupby("_label")["count"].sum().sort_values(ascending=True)
    cats = list(totals.index)
    brand_list = [b for b in brands if b in set(t["_brand"])]

    fig_h = max(3.0, 0.42 * len(cats) + 1.0)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(len(cats))
    height = 0.35 if len(brand_list) == 2 else 0.6 / max(len(brand_list), 1)

    for i, b in enumerate(brand_list):
        sub = t[t["_brand"] == b].set_index("_label")
        vals = np.array([float(sub.loc[c, "pct"]) if c in sub.index else 0.0 for c in cats])
        ns = np.array([float(sub.loc[c, "count"]) if c in sub.index else 0.0 for c in cats])
        offset = (i - (len(brand_list) - 1) / 2) * height
        colors = []
        alphas = []
        base = BRAND_COLORS.get(b, "#888")
        for n in ns:
            colors.append(base)
            alphas.append(0.35 if n < small_n else 0.9)
        bars = ax.barh(y + offset, vals, height=height * 0.92, label=b.title(), color=colors)
        for bar, a in zip(bars, alphas):
            bar.set_alpha(a)

    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.set_xlabel("Within-brand share")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
    cov = coverage_annotation(raw)
    ax.set_title(title if not cov else f"{title}\n{cov}", loc="left", pad=10)
    ax.legend(frameon=False, loc="lower right")
    if cats and max(len(str(c)) for c in cats) > 28:
        fig.subplots_adjust(left=0.42)
    fig.tight_layout()
    return fig


def crosstab_heatmap(
    raw: pd.DataFrame,
    row: str,
    col: str,
    *,
    title: str,
    small_n: int = SMALL_N,
    annot_min: float = 0.05,
):
    """Part 3: annotated heatmap of row-normalized percentages.

    Small-n cells keep diagonal hatch; labels are plain auto black/white
    text (no bbox, no halo) with ``*`` for small samples.
    """
    plt = _require_matplotlib()
    _apply_style(plt)
    pct, cnt = heatmap_matrices(raw, row, col)
    if pct.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No cells above min_count", ha="center")
        ax.axis("off")
        return fig

    # cluster heatmaps: short y tick ids + key under chart
    import re

    row_labels = list(pct.index.astype(str))
    col_labels = list(pct.columns.astype(str))
    cluster_pat = re.compile(r"^(\d+)\s*—\s*(.+)$")
    row_parsed = [cluster_pat.match(lab) for lab in row_labels]
    use_cluster_key = sum(p is not None for p in row_parsed) >= max(3, len(row_labels) // 2)

    fig_w = max(7.2, 0.95 * pct.shape[1] + 3.2)
    fig_h = max(3.8, 0.48 * pct.shape[0] + (2.2 if use_cluster_key else 1.4))
    if use_cluster_key:
        from matplotlib import gridspec

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3.6, 1.0], hspace=0.32)
        ax = fig.add_subplot(gs[0])
        ax_key = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax_key = None

    data = pct.to_numpy(dtype=float)
    counts = cnt.to_numpy(dtype=float)
    vmax = max(0.35, float(np.nanmax(data)) if np.isfinite(data).any() else 0.35)
    cmap = plt.cm.Greys
    im = ax.imshow(np.nan_to_num(data, nan=0.0), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            n = counts[i, j]
            p = data[i, j]
            if np.isnan(n) or n <= 0 or np.isnan(p):
                continue
            is_small = n < small_n
            if is_small:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="#888888",
                        linewidth=0.0,
                        zorder=2,
                    )
                )
            if p < annot_min:
                continue
            txt = format_pct(p, 0) + ("*" if is_small else "")
            rgba = cmap(np.clip(p / vmax if vmax else 0.0, 0.0, 1.0))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "#ffffff" if lum < 0.55 else "#111111"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=color,
                zorder=5,
            )

    ytick = [m.group(1) if m is not None else ("U" if lab == "Unclassified" else lab) for lab, m in zip(row_labels, row_parsed)] if use_cluster_key else row_labels
    ax.set_xticks(range(pct.shape[1]))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(pct.shape[0]))
    ax.set_yticklabels(ytick)
    ax.tick_params(axis="both", length=0, pad=4)
    cov = coverage_annotation(raw)
    note = "Hatch / * = small n"
    head = f"{title}\n{cov}  ·  {note}" if cov else f"{title}\n{note}"
    ax.set_title(head, loc="left", pad=10, color="#1a1a1a")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Row share", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))

    if ax_key is not None:
        ax_key.axis("off")
        key_rows = []
        for lab, m in zip(row_labels, row_parsed):
            if m is not None:
                key_rows.append((m.group(1), m.group(2)))
            elif lab == "Unclassified":
                key_rows.append(("U", "Unclassified"))
        # de-dupe preserving order
        seen = set()
        uniq = []
        for cid, name in key_rows:
            if cid in seen:
                continue
            seen.add(cid)
            uniq.append((cid, name))

        def _sk(row):
            k = row[0]
            return (1, k) if k == "U" else (0, int(k) if k.isdigit() else k)

        uniq = sorted(uniq, key=_sk)
        ax_key.text(0.0, 1.0, "Cluster key", transform=ax_key.transAxes, va="top", ha="left", fontsize=9.5, color="#1a1a1a")
        ncols = 3
        for idx, (cid, name) in enumerate(uniq):
            r, c = divmod(idx, ncols)
            display = name if len(name) <= 42 else name[:41] + "…"
            ax_key.text(
                c / ncols,
                0.72 - r * 0.18,
                f"{cid}  {display}",
                transform=ax_key.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="#333333",
                clip_on=False,
            )
        fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.06, hspace=0.35)
    else:
        if max(len(s) for s in row_labels) > 28:
            fig.subplots_adjust(left=0.32)
        fig.tight_layout()
    return fig


def performance_bubble(
    raw: pd.DataFrame,
    *,
    title: str,
    small_n: int = SMALL_N,
):
    """Part 4: share (x) vs median WER (y); bubble size = count.

    Cluster labels (``0 — Name``) are plotted as id numbers only, with a
    3-column name key under the chart. Other dimensions keep short on-plot labels.
    """
    import re
    from matplotlib import gridspec

    plt = _require_matplotlib()
    _apply_style(plt)
    t = raw.reset_index()
    label_col = t.columns[0]
    labels = [str(x) for x in humanize_series(t[label_col])]
    x = t["pct"].to_numpy(dtype=float)
    y = t["median_wer"].to_numpy(dtype=float)
    n = t["count"].to_numpy(dtype=float)

    cluster_pat = re.compile(r"^(\d+)\s*—\s*(.+)$")
    parsed = [cluster_pat.match(lab) for lab in labels]
    use_id_legend = sum(p is not None for p in parsed) >= max(3, len(labels) // 2)

    if use_id_legend:
        fig = plt.figure(figsize=(9.2, 7.2))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3.4, 1.15], hspace=0.28)
        ax = fig.add_subplot(gs[0])
        ax_leg = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax_leg = None

    sizes = 40 + 400 * (n / n.max()) if n.max() else n
    colors = ["#bbbbbb" if c < small_n else "#2a2a2a" for c in n]
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.75, edgecolors="white", linewidths=0.6)

    legend_rows: list[tuple[str, str]] = []
    for xi, yi, lab, ni, m in zip(x, y, labels, n, parsed):
        if use_id_legend and m is not None:
            marker = m.group(1)
            legend_rows.append((marker, m.group(2)))
        elif use_id_legend and lab == "Unclassified":
            marker = "U"
            legend_rows.append(("U", "Unclassified"))
        else:
            marker = lab if len(lab) <= 28 else lab[:27] + "…"
        ax.annotate(
            f"{marker}*" if ni < small_n else marker,
            (xi, yi),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=9 if use_id_legend else 7.5,
            fontweight="bold" if use_id_legend else "normal",
            color="#666" if ni < small_n else "#111",
        )

    ax.set_xlabel("Category share")
    ax.set_ylabel("Median weighted engagement rate")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.2f}%"))
    cov = coverage_annotation(raw)
    subtitle = "Bubble size = n; gray / * = small sample"
    ax.set_title(
        f"{title}\n{cov}" + (f"\n{subtitle}" if cov else subtitle),
        loc="left",
        pad=10,
    )

    if ax_leg is not None:
        ax_leg.axis("off")

        def _sort_key(row: tuple[str, str]):
            k = row[0]
            return (1, k) if k == "U" else (0, int(k) if k.isdigit() else k)

        seen: set[str] = set()
        uniq: list[tuple[str, str]] = []
        for row in sorted(legend_rows, key=_sort_key):
            if row[0] in seen:
                continue
            seen.add(row[0])
            uniq.append(row)

        ncols = 3
        nrows = int(np.ceil(len(uniq) / ncols)) if uniq else 1
        ax_leg.text(
            0.0,
            1.0,
            "Cluster key",
            transform=ax_leg.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            fontweight="medium",
            color="#1a1a1a",
        )
        # 3-column grid under the title
        for idx, (cid, name) in enumerate(uniq):
            r, c = divmod(idx, ncols)
            # wrap long names lightly
            display = name if len(name) <= 42 else name[:41] + "…"
            ax_leg.text(
                c / ncols,
                0.78 - r * 0.16,
                f"{cid}  {display}",
                transform=ax_leg.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="#333333",
                clip_on=False,
            )

        fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.04, hspace=0.35)
    else:
        fig.tight_layout()
    return fig


def brand_wer_violin(
    df: pd.DataFrame,
    mw: dict,
    *,
    brand_col: str = "brand",
    value_col: str = WER_COL,
    brands: Sequence[str] = ("adidas", "nike"),
    clip_upper: float = 0.05,
):
    """Part 4: Nike vs Adidas WER violin/box with Mann–Whitney p annotated."""
    plt = _require_matplotlib()
    _apply_style(plt)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    data = []
    labels = []
    for b in brands:
        s = pd.to_numeric(df.loc[df[brand_col] == b, value_col], errors="coerce").dropna()
        data.append(s.clip(upper=clip_upper))
        labels.append(b.title())

    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        b = brands[i]
        body.set_facecolor(BRAND_COLORS.get(b, "#888"))
        body.set_alpha(0.40)
        body.set_edgecolor(BRAND_COLORS.get(b, "#888"))
    # overlay box
    bp = ax.boxplot(
        data,
        positions=range(1, len(data) + 1),
        widths=0.16,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#c45c26", "linewidth": 1.6},
        whiskerprops={"color": "#444"},
        capprops={"color": "#444"},
        boxprops={"linewidth": 1.1},
    )
    for patch, b in zip(bp["boxes"], brands):
        patch.set_facecolor("white")
        patch.set_edgecolor(BRAND_COLORS.get(b, "#888"))

    # annotate Q1 / median / Q3 beside each box
    from matplotlib import patheffects as pe

    for i, s in enumerate(data):
        q1, med, q3 = float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75))
        x = i + 1
        stats = [
            (q3, f"Q3 {format_wer_pct(q3, 2)}", "#444444"),
            (med, f"Med {format_wer_pct(med, 2)}", "#c45c26"),
            (q1, f"Q1 {format_wer_pct(q1, 2)}", "#444444"),
        ]
        for y, txt, color in stats:
            ax.annotate(
                txt,
                xy=(x, y),
                xytext=(12, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=8.5,
                color=color,
                fontweight="bold" if "Med" in txt else "normal",
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
            )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Weighted engagement rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.1f}%"))
    # room for side labels
    ax.set_xlim(0.5, len(labels) + 0.85)

    if "p_value" in mw:
        p = mw["p_value"]
        p_txt = f"p = {p:.2e}" if p < 1e-3 else f"p = {p:.3f}"
        ax.set_title(
            f"Nike vs Adidas engagement\nMann–Whitney U  ·  {p_txt}",
            loc="left",
            pad=10,
        )
    else:
        ax.set_title("Nike vs Adidas engagement", loc="left")

    ax.text(
        0.98,
        0.02,
        f"Display clipped at {format_wer_pct(clip_upper, 0)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#888",
    )
    fig.tight_layout()
    return fig


def relative_uplift_diverging(
    raw: pd.DataFrame,
    *,
    title: str,
    brands: Sequence[str] = ("adidas", "nike"),
    small_n: int = 20,
    reliable_only: bool = False,
):
    """Two-panel diverging bars: relative uplift vs brand median baseline."""
    plt = _require_matplotlib()
    _apply_style(plt)
    t = raw.copy()
    if reliable_only:
        t = t[~t["small_n"]].copy()
    brand_list = [b for b in brands if b in set(t["brand"].astype(str).str.lower())]
    if not brand_list:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No rows", ha="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(
        1,
        len(brand_list),
        figsize=(5.2 * len(brand_list), max(3.2, 0.36 * t.groupby("brand").size().max() + 1.5)),
        sharex=True,
    )
    if len(brand_list) == 1:
        axes = [axes]

    for ax, brand in zip(axes, brand_list):
        sub = t[t["brand"] == brand].sort_values("relative_uplift", ascending=True)
        labels = [
            humanize_label(c) + ("*" if sn else "")
            for c, sn in zip(sub["category"], sub["small_n"])
        ]
        vals = sub["relative_uplift"].to_numpy(dtype=float)
        ns = sub["n"].to_numpy(dtype=float)
        colors = []
        for v, sn in zip(vals, sub["small_n"]):
            if sn:
                colors.append("#bbbbbb")
            elif v >= 0:
                colors.append(BRAND_COLORS.get(brand, "#444"))
            else:
                colors.append("#888888")
        y = np.arange(len(sub))
        bars = ax.barh(y, vals, color=colors, height=0.72)
        for bar, sn in zip(bars, sub["small_n"]):
            if sn:
                bar.set_hatch("///")
                bar.set_alpha(0.55)
            else:
                bar.set_alpha(0.88)
        ax.axvline(0.0, color="#222", lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Relative uplift vs brand baseline")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
        base = sub["brand_baseline_wer"].iloc[0] if len(sub) else np.nan
        ax.set_title(
            f"{brand.title()}\nbaseline median WER {format_wer_pct(base, 2)}",
            loc="left",
            pad=8,
        )
        xmax = max(0.05, float(np.nanmax(np.abs(vals))) * 1.25) if len(vals) else 0.2
        ax.set_xlim(-xmax, xmax)
        for yi, v, n, sn in zip(y, vals, ns, sub["small_n"]):
            side = 1 if v >= 0 else -1
            ax.text(
                v + side * xmax * 0.02,
                yi,
                f"{v:+.0%}  (n={int(n):,})",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8,
                color="#666" if sn else "#222",
            )

    fig.suptitle(
        f"{title}\n0 = brand median WER  ·  hatch / * = small n (n<{small_n})  ·  observational",
        x=0.01,
        ha="left",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def scalable_winners_scatter(
    raw: pd.DataFrame,
    *,
    title: str,
    brands: Sequence[str] = ("adidas", "nike"),
    small_n: int = 20,
):
    """Share within brand (x) vs relative uplift (y); bubble = n."""
    plt = _require_matplotlib()
    _apply_style(plt)
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    t = raw.copy()
    ax.axhline(0.0, color="#222", lw=1.0, zorder=1)
    ax.axvline(0.05, color="#ccc", ls="--", lw=0.8, zorder=1)

    for brand in brands:
        sub = t[t["brand"] == brand]
        if sub.empty:
            continue
        x = sub["share_within_brand"].to_numpy(dtype=float)
        y = sub["relative_uplift"].to_numpy(dtype=float)
        n = sub["n"].to_numpy(dtype=float)
        sizes = 40 + 350 * (n / max(n.max(), 1))
        colors = [BRAND_COLORS.get(brand, "#444")] * len(sub)
        alphas = [0.35 if sn else 0.8 for sn in sub["small_n"]]
        for xi, yi, si, ci, ai in zip(x, y, sizes, colors, alphas):
            ax.scatter([xi], [yi], s=[si], c=[ci], alpha=ai, edgecolors="white", linewidths=0.5)
        # label notable points: high |uplift|, OR high share / large n (scalable territory)
        reliable = sub[~sub["small_n"]].copy()
        by_uplift = reliable[reliable["relative_uplift"].abs() >= 0.08]
        by_share = reliable[reliable["share_within_brand"] >= 0.05]
        by_n = reliable.nlargest(2, "n")
        notable = pd.concat([by_uplift, by_share, by_n], ignore_index=True)
        notable = notable.drop_duplicates(subset=["category"], keep="first")
        for _, r in notable.iterrows():
            ax.annotate(
                humanize_label(r["category"]),
                (r["share_within_brand"], r["relative_uplift"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7.5,
                color=BRAND_COLORS.get(brand, "#333"),
            )

    ax.set_xlabel("Category share within brand")
    ax.set_ylabel("Relative uplift vs brand baseline")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.set_title(
        f"{title}\nBubble size = n  ·  faded = small n  ·  dashed line ≈ 5% share",
        loc="left",
        pad=10,
    )
    # simple legend
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BRAND_COLORS["adidas"], markersize=8, label="Adidas"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BRAND_COLORS["nike"], markersize=8, label="Nike"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig
