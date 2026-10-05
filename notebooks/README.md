# Analysis notebooks

Nike vs Adidas TikTok analytics, in two layers.

## Executive layer

**[`06_executive_analysis.ipynb`](06_executive_analysis.ipynb)**: the 5–10 minute business report for stakeholders and hiring managers. Its charts use the same plotting functions as `02`–`04d` and read the grouped aggregates in [`executive_data/`](executive_data/) (medians, shares and counts per group; no video- or account-level rows), so it runs without the local feature tables. Rebuild those aggregates with `python scripts/build_executive_data.py`.

1. Earn attention: content type vs each brand's baseline
2. Carry commercial intent: commerce cues vs engagement
3. Choose the messenger: official vs UGC, creator tier × content type
4. Read the response: topic and Product Review comment sentiment
5. Prioritize: pre-publish screening (Lift@K)
6. Insight to action, and what not to claim

## Supporting evidence

Full analysis behind each headline, for technical review: distributions, every crosstab, statistical tests, model diagnostics, and robustness checks.

| Notebook | Evidence for |
|----------|--------------|
| [`01_eda.ipynb`](01_eda.ipynb) | Distributions and coverage (no strategy crosstabs) |
| [`02_descriptive_crosstabs.ipynb`](02_descriptive_crosstabs.ipynb) | Awareness & Engagement: landscape, brand mix, within-brand BRI |
| [`03_engagement_modeling.ipynb`](03_engagement_modeling.ipynb) | Pre-publish screening: tiering, Top-Q screening, modality ablation, SHAP |
| [`04b_theme_commerce.ipynb`](04b_theme_commerce.ipynb) | Social Commerce: commerce intensity and engagement |
| [`04c_theme_influencer.ipynb`](04c_theme_influencer.ipynb) | Creator Strategy: official vs UGC, creator tier, collaboration |
| [`04d_theme_sentiment.ipynb`](04d_theme_sentiment.ipynb) | Sentiment & Topics: comment sentiment and topic taxonomy |

## Working notebooks

| Notebook | Role |
|----------|------|
| `00_shared_setup.ipynb` | Paths, feature table, leakage checklist (run first) |
| `04a_theme_awareness.ipynb` | Awareness write-up drawing on `02`–`03` |
| `05_summary.ipynb` | Earlier summary of locked conclusions; superseded by `06` |

Run `00` → `01` → `02` → `03` → `04a`–`04d` to reproduce from `data/processed/feature/feature_table.parquet`. Code: `src/tiktok_brand/analysis/`.

Keep full analysis in `02`–`04d`; when a conclusion is signed off, update the matching section of `06`.

## Outcomes

| Question | Primary measure |
|----------|-----------------|
| Brand-to-brand engagement | Median `weighted_engagement_rate` (WER); views for reach |
| Within-brand content effect | `brand_relative_engagement_index` (BRI) = WER / brand-median WER |
| Screening | Lift@K on top-decile / top-quartile labels |

BRI > 1 means above that brand’s typical WER. Do not use BRI as the headline KPI for Nike vs Adidas brand strength.

## Theme notes

**Theme 1 (`02`–`03`, write-up `04a`).** Crosstabs for strategy; models for tiering and shortlist screening only. Author-grouped CV; pre-publish features.

**Theme 2 (`04b`).** Caption commerce cues (Purchase / Discovery / Promo / Giveaway) → intensity bins. Not sales or conversion.

**Theme 3 (`04c`).** Official vs UGC; UGC `creator_tier` × content → BRI. Collaboration = caption language, not verified deals.

**Theme 4 (`04d`).** Comment subsample (~105 videos / ~10.4k comments). **n = comments** here (Themes 1–3: n = videos unless noted). Overall comment sentiment; content context via `content_type`, `brand_styles`, and text `content_cluster`; comment-level topic taxonomy. Volume vs sentiment denominators differ (~10.4k labeled vs ~8.7k scored).

All chapters are descriptive or predictive screening unless stated otherwise. They do not establish causal lift.
