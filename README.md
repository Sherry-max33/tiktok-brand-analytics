# TikTok Brand Analytics: Nike vs Adidas

A reproducible, sample-based analytics pipeline comparing **Nike** and **Adidas** TikTok content across:

1. Awareness / engagement  
2. Social commerce signals  
3. Creator & Collaboration Strategy (official vs UGC; scale × content × BRI)  
4. Sentiment & topics (NLP)

> **Scope:** Collection is **sample-based**: TikTok search results for brand/product seed terms plus official accounts (algorithmically ranked, not a full archive). Each crawl records `crawled_at` for reproducibility.

Full documentation: **[docs/README.md](docs/README.md)**.

---

## Repository structure

```
tiktok-brand-analytics/
├─ configs/                 # Crawl seeds + ETL rules
│  ├─ hashtags.yaml         # seed search terms + normalize_tags (clean)
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

### 4) Run the app

```bash
streamlit run app/streamlit_app.py
```

Reads `data/processed/feature/feature_table.parquet` (falls back to mock cards if missing). Video visuals are placeholders; the app links to the original TikTok post and does not host TikTok media. An optional hero asset of our own can be added at `app/assets/hero.mp4` or `app/assets/hero.jpg`.

---

## Schema pointers

- Clean & feature contracts: [docs/03-data-dictionary.md](docs/03-data-dictionary.md)
- Crawl fields & seeds: [docs/02-crawl.md](docs/02-crawl.md)

---

## Data Source & Disclaimer

This is an independent, non-commercial portfolio project developed for educational, research, and data science demonstration purposes.

- **Data Source:** The analysis is based on publicly accessible TikTok content and engagement metadata collected using Apify. The dataset is used to study content, engagement, and brand-related patterns.

- **Data & Media Handling:** The public repository does not host or redistribute original TikTok video files. Non-essential personal identifiers are excluded from the analytical workflow where possible, and the project focuses on content-level and aggregated analytical insights.

- **Independent Analysis:** Findings reflect patterns observed in the collected sample and should not be interpreted as official platform statistics, brand performance reports, or representative measurements of the broader TikTok population.

- **Affiliation:** This project is not affiliated with, sponsored by, or endorsed by TikTok, ByteDance, Nike, or Adidas. All trademarks and brand names belong to their respective owners.

---

## Roadmap

- Visual embeddings / `appearance_type` (CV)
- `content_cluster_id` from text embeddings
- Quasi-experimental “A/B-like” template effect analysis
