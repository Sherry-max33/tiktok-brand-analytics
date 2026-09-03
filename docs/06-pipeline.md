# Pipeline runbook

## Setup

```bash
uv sync
# or: python -m venv .venv && source .venv/bin/activate && pip install -e .
```

Copy `.env.example` → `.env` and set `APIFY_API_TOKEN` for live Apify crawls. Without a token, smoke/mock paths use Apify-shaped fixtures.

## End-to-end (prod-style)

```bash
# 1) Crawl
python -m scripts.crawl_hashtags
python -m scripts.crawl_users
python -m scripts.crawl_comments   # optional; after videos exist

# 2) Clean + feature
python -m scripts.build_dataset
```

Outputs:

- `data/processed/clean/tiktok_videos.parquet`
- `data/processed/feature/feature_table/` (partitioned)

## Small / test pipeline

```bash
python -m scripts.run_small_pipeline
# or
python scripts/smoke_test.py
```

Smoke writes test parquet under `data/processed/clean/` and `data/processed/feature/`, then validates required columns and value domains.

## Useful paths

| Artifact | Path |
|----------|------|
| Project params | `configs/project.yaml` |
| Seeds | `configs/hashtags.yaml`, `configs/accounts.yaml` |
| Taxonomy / feature rules | `configs/taxonomy.yaml`, `configs/feature_rules.yaml` |
| Unit tests | `pytest` (`tests/`) |

## Quality checks

```bash
PYTHONPATH=src pytest tests/ -q
```

Feature contract highlights (enforced in tests):

- Taxonomy columns are plural lists: `brand_styles`, `product_lines`, `product_categories`
- Collect field is `collect_count`
- Engagement includes weighted metrics and `brand_relative_engagement_index`
- CTA uses `has_cta` plus granular purchase / engagement / discovery / promo flags

## Ethics

- Public data only; respect TikTok ToS and local law.
- Sample-based hashtag feeds; results depend on crawl time and location.
- Prefer not to commit large `crawl_exports/` or raw payloads with PII beyond what analysis needs.
