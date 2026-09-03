# Architecture

## Purpose

Sample-based analytics comparing **Nike** and **Adidas** TikTok content across:

1. Awareness / engagement
2. Social commerce signals
3. Influencer strategy (official vs UGC)
4. Sentiment & topics (NLP; embeddings partially placeholder)

Hashtag feeds are ranked streams, not a complete archive. Each run records `crawled_at` for reproducibility.

## Medallion layout

```
crawl_exports/     Bronze — crawler-native CSV (Apify export, unmapped)
raw/               Bronze — mapped JSONL (VideoRecord / CommentRecord)
processed/clean/   Silver — harmonized video fact table
processed/feature/ Gold   — analysis / modeling features
```

| Layer | Path | Writer | Notes |
|-------|------|--------|-------|
| Crawl export | `data/crawl_exports/` | Apify / manual | Optional; gitignored |
| Raw videos | `data/raw/*.jsonl` | `scripts/crawl_*.py` | Append-oriented; one JSON object per line |
| Raw comments | `data/raw/*comment*.jsonl` | `scripts/crawl_comments.py` | Skipped by video clean ETL |
| Clean | `data/processed/clean/` | `build_clean_table` | Deduped by `video_id` |
| Feature | `data/processed/feature/` | `build_feature_table` | Partitioned prod table or test parquet |

## Pipeline flow

```
Hashtag seeds + official accounts
        │
        ▼
  Apify actors ──► crawl_exports (optional CSV)
        │
        ▼
  Mapper ──► data/raw/*.jsonl  (VideoRecord)
        │
        ▼
  build_clean_table
        │  caption/hashtag cleanup, normalize_tags, brand,
        │  seed_hashtag, is_official_brand, create_time, dedupe
        │  NO taxonomy, NO engagement rates
        ▼
  data/processed/clean/*.parquet
        │
        ▼
  build_feature_table
        │  engagement metrics, CTA flags, content_type,
        │  creator_type/tier, taxonomy multi-labels, embeddings stubs
        ▼
  data/processed/feature/...
```

## Layer responsibilities

### Clean (silver)

- Harmonize field names (`collect_count`; legacy `save_count` accepted)
- Strip caption; clean hashtag lists
- `normalized_hashtags` via `configs/hashtags.yaml` → `normalize_tags`
- Infer `brand` when missing (from nike*/adidas* tags)
- Set `seed_hashtag`, `is_official_brand`, `create_time`
- Drop `raw_payload`; dedupe on `video_id` (keep latest `crawled_at_ts`)

### Feature (gold)

- Engagement count / rate / weighted metrics / brand-relative index
- CTA multi-flags (`has_cta`, purchase / engagement / discovery / promo)
- `content_type` (P0–P6 keyword rules)
- Creator type & tier
- Taxonomy: `brand_styles` → `product_lines` → `product_categories` (cascaded)
- Placeholders: `sentiment_score`, `text_embedding`, `visual_embedding`, `appearance_type`, `content_cluster_id`

## Config ownership

| Concern | Config |
|---------|--------|
| Seed hashtags + alias normalize | `hashtags.yaml` |
| Official usernames | `accounts.yaml` |
| Sample sizes, paths, actors | `project.yaml` |
| Style / line / category maps | `taxonomy.yaml` |
| Weights, content_type, tiers, trending audio | `feature_rules.yaml` |
| CTA regex | `src/tiktok_brand/etl/cta_rules.py` |

## Official brand label

`is_official_brand = author_username ∈ official_accounts` from `accounts.yaml`:

- Nike: `nike`, `jumpman23`
- Adidas: `adidas`

Platform verification (`author_verified`) is stored separately; **verified ≠ official**.
