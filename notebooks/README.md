# Analysis notebooks (run order)

Finalized pipeline for Nike vs Adidas TikTok analytics.

```
notebooks/
  00_shared_setup.ipynb              # load feature table, paths, leakage checklist, shared imports
  01_eda.ipynb                       # distributions, missingness, coverage, outliers
  02_descriptive_crosstabs.ipynb     # brand × content_type × cluster × social_mechanic
  03_engagement_modeling.ipynb       # prediction + feature importance + ablation (M0–M4)
  04a_theme_awareness.ipynb          # Theme 1: awareness / engagement (1–2 screen brief)
  04b_theme_commerce.ipynb           # Theme 2: social commerce signals
  04c_theme_influencer.ipynb         # Theme 3: creator / brand collaboration
  04d_theme_sentiment.ipynb          # Theme 4: sentiment & topics
  05_llm_content_analyst.ipynb       # AI Content Analyst (after findings → choose prompt features)
```

Reusable code lives in `src/tiktok_brand/analysis/` (plots, crosstabs, train/importance/ablation, retrieval, analyst prompts).

## Why this order

| Step | Role |
|------|------|
| 00–01 | Data readiness + EDA |
| 02 | Descriptive evidence **before** modeling |
| 03 | Predictive evidence (importance / ablation inform later feature choices) |
| 04a–04d | Short theme narratives citing 01–03 |
| 05 | **Primary LLM product**: structured features + Top-K similar videos → report; feature set chosen from 03/04 findings |

## LLM Content Analyst (notebook 05)

Not raw-video understanding. Flow:

1. Select `video_id`
2. Read structured features (brand, caption, sentiment, `content_type`, `social_mechanic`, cluster, CTA, engagement + brand percentiles)
3. Retrieve Top-K similar videos (text / cluster embeddings; visual optional)
4. Compare on Brand / Theme / Visual / CTA / Performance
5. LLM writes a data-driven report from that context only

Secondary LLM uses (hard-to-rule signals, comment topics) are optional and separate from this main path.

## Four business themes

1. **Awareness / engagement** — what drives views/likes/comments/shares  
2. **Social commerce** — purchase / conversion-oriented content  
3. **Influencer strategy** — creator types & collaboration performance  
4. **Sentiment & topics** — audience feeling / brand perception  
