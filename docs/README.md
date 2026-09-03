# Documentation

English documentation for the TikTok Brand Analytics pipeline (Nike vs Adidas).

Read in order:

| # | Doc | Audience | Contents |
|---|-----|----------|----------|
| 01 | [01-architecture.md](./01-architecture.md) | Engineers | Medallion layers, data flow, config map |
| 02 | [02-crawl.md](./02-crawl.md) | Crawl operators | Seeds, VideoRecord fields, Apify mapping |
| 03 | [03-data-dictionary.md](./03-data-dictionary.md) | Engineers / analysts | Schema contracts (raw → clean → feature) |
| 04 | [04-taxonomy.md](./04-taxonomy.md) | Engineers / analysts | Multi-label style / line / category cascade |
| 05 | [05-feature-engineering.md](./05-feature-engineering.md) | Analysts | Engagement, CTA, content type, analysis dims |
| 06 | [06-pipeline.md](./06-pipeline.md) | Engineers | How to run crawl + ETL + smoke tests |

**Source of truth (when docs and code disagree):** code + tests + YAML under `configs/`.

Configs:

| File | Role |
|------|------|
| `configs/hashtags.yaml` | Crawl seed tags + `normalize_tags` (clean) |
| `configs/accounts.yaml` | Official accounts |
| `configs/project.yaml` | Sampling, paths, Apify actors |
| `configs/taxonomy.yaml` | Product/style/category maps (feature) |
| `configs/feature_rules.yaml` | Engagement weights, content_type, tiers |
| CTA regex | `src/tiktok_brand/etl/cta_rules.py` |
