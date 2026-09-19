# Analysis notebooks (run order)

Finalized pipeline for Nike vs Adidas TikTok analytics.

```
notebooks/
  00_shared_setup.ipynb              # load feature table, paths, leakage checklist, shared imports
  01_eda.ipynb                       # distributions, missingness, coverage — NO crosstabs
  02_descriptive_crosstabs.ipynb     # Theme 1 (Awareness / Engagement) descriptive chapter
  03_engagement_modeling.ipynb       # Theme 1: tiering + Top-Q screening + ablation/SHAP
  04a_theme_awareness.ipynb          # Theme 1 brief (synthesizes 02-03)
  04b_theme_commerce.ipynb           # Theme 2: Social Commerce Signals
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

Stakeholder brief: [`04a_theme_awareness.ipynb`](./04a_theme_awareness.ipynb) (strategy → tiering → screening → ablation → SHAP → operating loop).

Focus of `03`: pre-publish features for **tiering + top-tail screening**, not exact WER regression. Strategy narrative still comes mainly from `02` within-brand uplift.

**Locked architecture**

| Layer | Method | Question |
|-------|--------|----------|
| Strategy | Crosstab + within-brand BRI (`02`) | What appears to work for Nike vs Adidas? |
| Tiering | `qclass_4` CatBoost | Where does each video sit in the WER spectrum? |
| Screening (primary) | `binary_top_q10` + CLIP + tuned CatBoost | Which candidates are most likely in the top-decile tail? |

Top-25 (`binary_top_q25`) is the broader-pool / robustness check. Tuning happens on Top-25; those params are reused for Top-10 (no separate search).

**Validation:** `GroupKFold(author_id)`. Binary thresholds use **that fold’s** `quantile(y_train, …)` only, then label train/valid (no global quantile).

**Kept methods**

| Role | Method | Use |
|------|--------|-----|
| Tiering | `qclass_4` CatBoost | Low/Mid/High/Top; expected score for library ranking |
| Screening (primary) | `binary_top_q10` + CLIP PCA + `TUNED_CATBOOST_BINARY_TOP_Q` | Top-10% tail screen; Lift@10 ≈ 2.39×, Lift@5 ≈ 3.10×, AUC ≈ 0.686 |
| Screening (robust / wider pool) | `binary_top_q25` (same stack, train Q75) | Top-25% pool; Lift@10 ≈ 2.33×, AUC ≈ 0.676 |

**Model selection:** primary Lift@10; secondary Top-Q AUC / Prec@10; diagnostic Spearman; Acc/F1 supporting only (not for selection).

**Interpretation (not a separate chapter):** `explain_screening_shap` (Approximate CatBoost TreeSHAP on the Top-10% model).
- Features include `clip_alignment_mean` / `clip_alignment_max` (visual-text CLIP similarity).
- Importance: mean |SHAP| by driver family (fine in conclusions if labeled as **screening drivers**, not mixed with the `02` strategy list).
- Direction: signed SHAP; §10c `by_brand` = same model, Adidas vs Nike rows (aligned with `02` within-brand). Associative only, not causal.

**Vs crosstab:** `02` = brand-median uplift (“what looks effective”); `03` SHAP = pre-publish screening (“what moves P(top-decile)”). Both talk about high performance, but one is relative to baseline and the other is extreme-tail plus multivariate control.

**Storyline**
1. `02` crosstab → strategy (brand-median uplift)
2. `03` `qclass_4` → tiering / coarse library ranking
3. `03` Top-10 screening → extreme-tail priority (Lift@10)
4. §9b modality ablation → how much Lift SBERT / visual axes / CLIP PCA / alignment buy
5. §10b/10c SHAP → full-model importance and direction by brand

**Ablation ladder (cleanest test):** M0 metadata → M1 eng-text → M2 SBERT → M3a format/setting → M3b CLIP PCA → M4 alignment.

This run’s Top-10% Lift@10: M0 1.97 → M1 2.32 → M2 2.12 → M3a 2.03 → M3b 2.59 → M4 2.35. Lift gains come mainly from M1 engineered text and M3b CLIP PCA; M3a axis labels barely help; alignment is more useful as interpretation.

**Key modeling takeaway:** exact WER regression only improves brand-median MAE by ~3%, so skip point forecasting. The same pre-publish signals are more useful for top-tail screening (Top-10 Lift@10 2.39×).

**Not on the main path:** WER regression, LTR, ensembles, more large-scale tuning.

Details: `03` §11 Modeling conclusion.

## Theme 2 — Social Commerce Signals (`04b`)

Notebook: [`04b_theme_commerce.ipynb`](./04b_theme_commerce.ipynb).

**Core variable:** Commerce Intensity = Purchase CTA + Discovery CTA + Promo Language + Giveaway  
Bins: 0 None / 1 Low / 2 Medium / 3+ High. No separate "Direct Commerce" layer.

**Chapter flow**
1. Intensity definition + component rates
2. Parallel cross-tabs: content_type, content cluster, brand_styles, visual_format, visual_setting, product_categories, product_lines
3. Nike vs Adidas intensity mix + what drives the gap
4. Intensity → median BRI / WER / Top-10% rate (do not assume a penalty)
5. Commerce × BRI group bubbles (content panel + product panel)

Helpers: `tiktok_brand.analysis.commerce` (`add_commerce_intensity`, `commerce_by_dimension`, …).

## Four business themes

1. **Awareness / engagement:** 02 crosstabs + 03 modeling + 04a brief  
2. **Social commerce:** 04b (Commerce Intensity × content/product × engagement)  
3. **Influencer strategy**  
4. **Sentiment & topics**  
