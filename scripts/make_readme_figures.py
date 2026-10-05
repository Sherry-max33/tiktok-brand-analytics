"""Render the README charts into docs/assets/.

The content strategy map reads notebooks/executive_data/content_type_relative.csv (built by
scripts/build_executive_data.py from the 02 analysis, the same table behind 06's first chart).
Lift@K is fixed to the Top-10% screening model in notebooks/03_engagement_modeling.ipynb;
update those numbers here if that result changes.

    python scripts/make_readme_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
CONTENT_TABLE = ROOT / "notebooks" / "executive_data" / "content_type_relative.csv"

BG = "#faf8f5"
INK = "#1f1e1c"
SOFT = "#6f6a64"
LINE = "#e7e2da"
ADIDAS = "#1f1e1c"
NIKE = "#b4a898"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": LINE,
    "axes.labelcolor": SOFT,
    "xtick.color": SOFT,
    "ytick.color": INK,
    "text.color": INK,
})


def _frame(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(length=0)


def _title(fig, title: str, subtitle: str) -> None:
    fig.text(0.04, 0.95, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.04, 0.885, subtitle, fontsize=10.5, color=SOFT, va="top")


CONTENT_ROWS = {
    "vibe_ootd": "Vibe / OOTD",
    "collaboration": "Collaboration",
    "product_showcase": "Product Showcase",
    "tutorial_utility": "Tutorial / Utility",
    "product_review": "Product Review",
    "product_promo": "Product Promo",
}
STRATEGY_CMAP = LinearSegmentedColormap.from_list("strategy", ["#b25d3d", "#efe9e1", "#244a3b"])


def _pattern(adidas: float, nike: float) -> str:
    if adidas >= 1 and nike >= 1:
        return "Both above baseline"
    if adidas < 1 and nike < 1:
        return "Both below baseline"
    return "Adidas above, Nike below" if adidas >= 1 else "Nike above, Adidas below"


def strategy_map() -> None:
    table = pd.read_csv(CONTENT_TABLE)
    table = table[~table["small_n"] & table["category"].isin(CONTENT_ROWS)]
    bri = table.pivot(index="category", columns="brand", values="relative_index").loc[list(CONTENT_ROWS)]
    n = table.pivot(index="category", columns="brand", values="n").loc[list(CONTENT_ROWS)]
    norm = TwoSlopeNorm(vmin=0.55, vcenter=1.0, vmax=1.3)

    fig, ax = plt.subplots(figsize=(9, 5.9))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.8, bottom=0.1)
    _title(fig, "Where each brand wins: content strategy map",
           "Median engagement vs the brand's own baseline (BRI; 1.0 = brand median)")
    ax.set_xlim(0, 10)
    ax.set_ylim(len(CONTENT_ROWS), -0.75)
    ax.axis("off")

    columns = {"adidas": (2.45, "Adidas"), "nike": (4.85, "Nike")}
    for x, label in columns.values():
        ax.text(x + 1.1, -0.35, label, ha="center", fontsize=11, fontweight="bold")
    ax.text(7.4, -0.35, "Pattern", ha="left", fontsize=11, fontweight="bold", color=SOFT)

    for row, (category, label) in enumerate(CONTENT_ROWS.items()):
        ax.text(0.05, row + 0.5, label, va="center", fontsize=11)
        for brand, (x, _) in columns.items():
            value, count = bri.loc[category, brand], int(n.loc[category, brand])
            color = STRATEGY_CMAP(norm(value))
            strong = abs(value - 1.0) >= 0.2
            ax.add_patch(FancyBboxPatch((x, row + 0.1), 2.2, 0.8, boxstyle="round,pad=0,rounding_size=0.12",
                                        facecolor=color, edgecolor="none"))
            text_color = "white" if strong else INK
            ax.text(x + 1.1, row + 0.43, f"{value:.2f}×", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=text_color)
            ax.text(x + 1.1, row + 0.74, f"{value - 1:+.0%} · n={count}", ha="center", va="center",
                    fontsize=8, color=text_color, alpha=0.9)
        ax.text(7.4, row + 0.5, _pattern(bri.loc[category, "adidas"], bri.loc[category, "nike"]),
                va="center", fontsize=10, color=SOFT)

    fig.text(0.04, 0.035,
             "Content-type labelled videos (887 of 4,457); types with n ≥ 20 per brand. "
             "Associations in this sample, not causal lift.",
             fontsize=8, color=SOFT)
    fig.savefig(OUT / "brand-content-map.png", dpi=200)
    plt.close(fig)


def lift() -> None:
    ks = ["Top 5%", "Top 10%", "Top 20%"]
    values = [3.10, 2.39, 2.06]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.74, bottom=0.17)
    _title(fig, "Ranking finds top performers 2–3× more often than chance",
           "Share of actual top-10% videos among the model's highest-scored K%, relative to the base rate")

    bars = ax.bar(ks, values, width=0.5, color=NIKE)
    bars[1].set_color(INK)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.07, f"{value:.2f}×",
                ha="center", fontsize=13, fontweight="bold")
    ax.axhline(1.0, color=SOFT, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(-0.82, 1.06, "random = 1.0×", ha="left", fontsize=8.5, color=SOFT)
    ax.set_xlim(-0.85, 2.4)
    ax.set_ylim(0, 3.6)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=11)
    _frame(ax)
    fig.text(0.04, 0.035,
             "CatBoost + CLIP visual features · creator-grouped cross-validation · ROC-AUC 0.686 · Precision@10% 0.241",
             fontsize=8, color=SOFT)
    fig.savefig(OUT / "lift-at-k.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    strategy_map()
    lift()
    print(f"wrote {OUT}")
