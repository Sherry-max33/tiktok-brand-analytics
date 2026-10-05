"""Render the README charts into docs/assets/.

Figures are fixed to the locked results: Brand x Content BRI from
notebooks/06_executive_analysis.ipynb (recomputed from the feature table) and Lift@K from the
Top-10% screening model in notebooks/03_engagement_modeling.ipynb. Update the numbers here if
those results change.

    python scripts/make_readme_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"

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


def brand_content() -> None:
    territories = ["Vibe / OOTD", "Tutorial / Utility", "Collaboration", "Product Promo"]
    adidas = [(1.21, 188), (0.77, 45), (1.12, 77), (0.75, 35)]
    nike = [(1.03, 113), (1.11, 40), (1.13, 69), (0.92, 88)]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.76, bottom=0.14)
    _title(fig, "Each brand wins in a different content territory",
           "Median engagement relative to the brand's own baseline (BRI; 1.0 = brand median)")

    h = 0.36
    for i, (name, (a_v, a_n), (n_v, n_n)) in enumerate(zip(territories, adidas, nike)):
        y = len(territories) - 1 - i
        for offset, value, n, color in ((h / 2, a_v, a_n, ADIDAS), (-h / 2, n_v, n_n, NIKE)):
            ax.barh(y + offset, value, height=h * 0.92, color=color)
            ax.text(value + 0.015, y + offset, f"{value:.2f}×  n={n}", va="center",
                    fontsize=8.5, color=SOFT)
    ax.set_yticks(range(len(territories)))
    ax.set_yticklabels(territories[::-1], fontsize=10.5)
    ax.axvline(1.0, color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xlim(0, 1.45)
    ax.set_xlabel("BRI", fontsize=9)
    _frame(ax)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ADIDAS), plt.Rectangle((0, 0), 1, 1, color=NIKE)],
               labels=["Adidas", "Nike"], frameon=False, ncol=2, fontsize=9.5,
               loc="upper right", bbox_to_anchor=(0.96, 0.84))
    fig.text(0.04, 0.03, "Content-type labelled videos only (887 of 4,457). Associations in this sample, not causal lift.",
             fontsize=8, color=SOFT)
    fig.savefig(OUT / "brand-content-bri.png", dpi=200)
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
    brand_content()
    lift()
    print(f"wrote {OUT}")
