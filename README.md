# TikTok Brand Analytics: Nike vs Adidas

A reproducible, sample-based analytics pipeline comparing **Nike** and **Adidas** TikTok content across:

1. Awareness / engagement  
2. Social commerce signals  
3. Influencer strategy (official vs UGC)  
4. Sentiment & topics (NLP)

> **Scope:** Hashtag collection is **sample-based** (ranked feeds, not a full archive). Each crawl records `crawled_at` for reproducibility.

Full documentation: **[docs/README.md](docs/README.md)**.

---

## Repository structure

```
tiktok-brand-analytics/
├─ configs/                 # Crawl seeds + ETL rules
│  ├─ hashtags.yaml         # seed tags + normalize_tags (clean)
│  ├─ accounts.yaml         # official accounts
│  ├─ project.yaml          # sampling / paths / Apify
│  ├─ taxonomy.yaml         # style / line / category (feature)
│  └─ feature_rules.yaml    # engagement / content_type / tiers
├─ data/
│  ├─ crawl_exports/        # Bronze: crawler-native CSV (optional)
│  ├─ raw/                  # Bronze: VideoRecord / CommentRecord JSONL
│  └─ processed/
│     ├─ clean/             # Silver fact parquet
│     └─ feature/           # Gold feature table / test parquet
├─ docs/                    # Architecture, schemas, runbook
├─ src/tiktok_brand/        # Python package
├─ scripts/                 # Crawl + ETL entrypoints
├─ notebooks/               # Analysis / storytelling
└─ tests/
```

---

## Data layers (summary)

| Layer | Responsibility |
|-------|----------------|
| **Raw** | Mapped crawl JSONL (`VideoRecord`) |
| **Clean** | Caption/hashtag cleanup, `normalized_hashtags`, `brand`, `is_official_brand`, dedupe — **no** taxonomy, **no** engagement rates |
| **Feature** | Engagement metrics, CTA flags, `content_type`, creator tier/type, taxonomy multi-labels |

Official accounts (Phase 1): `@nike`, `@jumpman23`, `@adidas` → `is_official_brand`. Verification ≠ official.

Stats field for bookmarks/favorites: **`collect_count`** (legacy `save_count` accepted in clean).

Taxonomy (feature): **`brand_styles` → `product_lines` → `product_categories`** (cascaded). Details: [docs/04-taxonomy.md](docs/04-taxonomy.md).

---

## Quickstart

### 1) Setup

```bash
uv sync
# or: pip install -e .
```

### 2) Crawl

Requires `APIFY_API_TOKEN` for live Apify runs (see `.env.example`):

```bash
python -m scripts.crawl_hashtags
python -m scripts.crawl_users
python -m scripts.crawl_comments   # optional
```

### 3) Build clean + feature

```bash
python -m scripts.build_dataset
```

Outputs:

- `data/processed/clean/tiktok_videos.parquet`
- `data/processed/feature/feature_table/`

Smoke / small run: `python scripts/smoke_test.py` or `python -m scripts.run_small_pipeline`.

---

## Schema pointers

- Clean & feature contracts: [docs/03-data-dictionary.md](docs/03-data-dictionary.md)
- Crawl fields & seeds: [docs/02-crawl.md](docs/02-crawl.md)

---

## Ethics & limitations

- Respect TikTok Terms of Service and local laws.
- Designed for **public**, **sample-based** analysis.
- Hashtag feeds are algorithmically ranked; results depend on crawl time and location.

---

## Roadmap

- Comment clean/feature ETL and deeper NLP
- Sentence-BERT / CLIP embeddings (beyond placeholders)
- Retrieval demo (caption + visual similarity)
- Quasi-experimental “A/B-like” template effect analysis
