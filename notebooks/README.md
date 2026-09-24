# Analysis notebooks

Nike vs Adidas TikTok analytics. Run in order unless noted.

| Notebook | Role |
|----------|------|
| `00_shared_setup.ipynb` | Paths, feature table, leakage checklist |
| `01_eda.ipynb` | Distributions and coverage (no strategy crosstabs) |
| `02_descriptive_crosstabs.ipynb` | Theme 1 strategy: landscape, brand mix, engagement |
| `03_engagement_modeling.ipynb` | Theme 1 predictive: tiering, Top-Q screening, ablation, SHAP |
| `04a_theme_awareness.ipynb` | Theme 1 write-up (fold locked bullets into `05` when signed off) |
| `04b_theme_commerce.ipynb` | Theme 2: commerce intensity and engagement |
| `04c_theme_influencer.ipynb` | Theme 3: creator scale and collaboration |
| `04d_theme_sentiment.ipynb` | Theme 4: comment sentiment and topics |
| `05_summary.ipynb` | Report-facing summary of locked commercial conclusions |

Code: `src/tiktok_brand/analysis/`.

No separate per-theme "brief" notebooks. Keep full analysis in `02`–`04d`; put signed-off takeaways in `05_summary.ipynb`.

## Outcomes

| Question | Primary measure |
|----------|-----------------|
| Brand-to-brand engagement | Median `weighted_engagement_rate` (WER); views for reach |
| Within-brand content effect | `brand_relative_engagement_index` (BRI) = WER / brand-median WER |
| Screening | Lift@K on top-decile / top-quartile labels |

BRI > 1 means above that brand’s typical WER. Do not use BRI as the headline KPI for Nike vs Adidas brand strength.

## Theme notes

**Theme 1 (`02`–`03`, write-up `04a`).** Crosstabs for strategy; models for tiering and shortlist screening only. Author-grouped CV; pre-publish features.

**Theme 2 (`04b`).** Caption commerce cues (Purchase / Discovery / Promo / Giveaway) → intensity bins. Not sales or conversion. Locked summary in `05`.

**Theme 3 (`04c`).** Official vs UGC; UGC `creator_tier` × content → BRI. Collaboration = caption language, not verified deals.

**Theme 4 (`04d`).** Comment subsample (~105 videos / ~10.4k comments). **n = comments** here (Themes 1–3: n = videos unless noted). Overall comment sentiment; content context via `content_type`, `brand_styles`, and text `content_cluster`; comment-level topic taxonomy. Volume vs sentiment denominators differ (~10.4k labeled vs ~8.7k scored).

All chapters are descriptive or predictive screening unless stated otherwise. They do not establish causal lift.
