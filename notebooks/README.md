# Analysis notebooks (run order)

Finalized pipeline for Nike vs Adidas TikTok analytics.

```
notebooks/
  00_shared_setup.ipynb              # load feature table, paths, leakage checklist, shared imports
  01_eda.ipynb                       # distributions, missingness, coverage — NO crosstabs
  02_descriptive_crosstabs.ipynb     # Theme 1 (Awareness / Engagement) descriptive chapter
  03_engagement_modeling.ipynb       # Theme 1: predict WER → BRI; CatBoost vs XGBoost
  04a_theme_awareness.ipynb          # Theme 1 brief (synthesizes 02–03)
  04b_theme_commerce.ipynb           # Theme 2: social commerce signals
  04c_theme_influencer.ipynb         # Theme 3: creator / brand collaboration
  04d_theme_sentiment.ipynb           # Theme 4: sentiment & topics
  05_llm_content_analyst.ipynb       # AI Content Analyst (after findings → choose prompt features)
```

Reusable code: `src/tiktok_brand/analysis/` (`data`, `crosstabs`, `metrics`, `plots`, …).

## Outcome metrics (do not force one Y)

| Question | Primary Y | Notes |
|----------|-----------|--------|
| Brand-to-brand performance | **median `weighted_engagement_rate`** (+ **`view_count`** for awareness) | Distributions, Mann–Whitney / bootstrap CI |
| Within-brand content effectiveness / prediction | **`brand_relative_engagement_index`** | “What outperforms this brand’s typical performance?” |
| Awareness volume | **`view_count`** | Separate from engagement quality |

### `brand_relative_engagement_index` (code)

```
BRIi = weighted_engagement_rate_i / median(weighted_engagement_rate | brand_i)
```

- Denominator is **brand median** (WER is right-skewed; median = “typical”).
- `BRI > 1` → above that brand’s typical WER; median(BRI) ≈ 1 within brand.
- Brand comparison chapters use **median WER / views**, not BRI as the headline KPI.

## Why this order

| Step | Role |
|------|------|
| 00–01 | Data readiness + **short EDA** (no strategy crosstabs) |
| 02–03 | **Theme 1 Awareness / Engagement** evidence (crosstab → model) |
| 04a | Theme 1 1–2 screen brief |
| 04b–04d | Other themes (may reuse 02 tables lightly) |
| 05 | LLM Content Analyst |

## Notebook 01 — EDA (keep short)

Do:

1. Sample mix (brand, source_type, date span)  
2. WER / views / BRI skew, extremes, missing  
3. Coverage matrix: `content_type`, `social_mechanic`, `brand_styles`, visual axes, `caption_en`, alignment  
4. Coverage by brand (avoid false strategy gaps)  
5. Leakage checklist  

Do **not**: strategy crosstabs → **02 / Theme 1**.

## Theme 1 — Awareness / Engagement

**Crosstabs live here** (notebook `02`).

Narrative: **What exists → What each brand uses → How strategies combine → What performs best.**

Every core table: **count + row%/col% + median `weighted_engagement_rate`**.  
Sparse labels: report coverage.  
Cluster id: **`content_cluster_id_clean_k12`**.  
List fields (`content_type`, `social_mechanic`, `brand_styles`): explode for group-bys (one video can appear in multiple rows).

### Part 1 — Content landscape

`content_type` / `visual_format` / `social_mechanic` / `brand_styles` / `content_cluster`

### Part 2 — Nike vs Adidas strategy

**Core tables:**

1. `brand × content_type`  
2. `brand × visual_format`  
3. `brand × social_mechanic`  
4. `brand × brand_styles`  
5. `brand × content_cluster`  

### Part 3 — Strategy combinations

- `content_type × visual_format`  
- `content_type × social_mechanic`  
- `content_cluster × content_type`  
- optional: `content_type × brand_styles`  

Brand drill-down only if Part 2 shows a clear gap.

### Part 4 — What works

- Each label axis → median WER (+ optional views)  
- `brand ×` each axis → median WER  
- Brand chapter: Nike vs Adidas median WER, distributions, Mann–Whitney / bootstrap CI; awareness via views  

### 4.2 Within-brand outperformance

Relative to each brand’s **own median WER** (not cross-brand raw WER):

- `relative_index = category median WER / brand baseline`  
- Priority charts: `content_type`, `brand_styles`  
- Flag small-n (n < 20); optional Mann–Whitney vs rest of brand with BH-FDR  
- Distinguish niche winners vs scalable winners (share × relative uplift)  
- Closing **Key Findings** synthesizes the full descriptive chapter  

Observational only — not causal lift.
### Modeling (`03`) then brief (`04a`)

**Target:** predict raw **`weighted_engagement_rate`** from **pre-publish** features only.  
**Decision metric:** MAE on WER (RMSE auxiliary; Spearman optional).  
**BRI conversion:** `pred_BRI = pred_WER / median(WER | brand)` using **train-fold** brand medians (never test-fold).  
**Models:** CatBoost vs XGBoost, same feature matrix.  
**Validation:** `GroupKFold` by `author_id` (no author leakage across folds).  
**Features:** creator / strategy / text-CTA / video / time / brand + `cluster_text_embedding` → PCA (fit inside fold).  
**Leakage:** no views/likes/shares/collects or derived rates / BRI as inputs.

## Four business themes

1. **Awareness / engagement** — 02 crosstabs + 03 WER→BRI modeling + 04a brief  
2. **Social commerce**  
3. **Influencer strategy**  
4. **Sentiment & topics**  
