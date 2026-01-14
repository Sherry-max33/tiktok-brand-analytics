# TikTok Brand Analytics: Nike vs Adidas

A reproducible, sample-based analytics pipeline comparing **Nike** and **Adidas** TikTok content across:
1) **Awareness / Engagement**
2) **Social Commerce Signals**
3) **Influencer Strategy (Official vs UGC)**
4) **Sentiment & Topics (NLP)**

> **Scope note (important):** Hashtag collection is **sample-based** (TikTok hashtag feeds are ranked streams, not a complete historical archive). Each crawl run records `crawled_at` to make analyses reproducible.

---

## Repository structure

```
tiktok-brand-analytics/
├─ configs/                # Frozen “data contract” (accounts/hashtags/mappings/sampling)
├─ data/
│  ├─ raw/                 # append-only JSONL outputs from crawlers
│  ├─ interim/             # optional intermediate artifacts (dedup / light normalize)
│  └─ processed/           # analysis-ready parquet tables
├─ src/tiktok_brand/       # production-style python package
├─ scripts/                # runnable entrypoints (no notebooks required)
├─ notebooks/              # storytelling + visualization (thin layer)
└─ tests/                  # schema + transforms unit tests
```

---

## Part 1 — Data collection (frozen v1)

### A) Official accounts (user crawl)

We only treat the two main accounts as “official brand accounts” in Phase 1:

- Nike: `@nike`
- Adidas: `@adidas`

We also store the platform verification flag (`author_verified`) when available, but **verification ≠ official**.

Official label (Phase 1):
- `is_official_brand = author_username in {nike, adidas}`

### B) Hashtag crawl (seed hashtags)

We crawl a **seed list** of hashtags for each brand (see `configs/hashtags.yaml`).  
Each record keeps:
- `source_type`: `"user"` or `"hashtag"`
- `source_query`: username or seed hashtag
- `seed_hashtag`: (hashtag crawl only)

### C) Stats fields (MVP)

We store core engagement stats:
- `view_count`, `like_count`, `comment_count`, `share_count`
- `save_count` (bookmark/favorite) **if present**; otherwise null.

### D) Time fields

We persist:
- `create_time_ts` (int, seconds)
- `crawled_at` (ISO-8601 with timezone, e.g. `2026-01-06T22:30:12-05:00`)
- `crawled_at_ts` (int, seconds)

### E) Normalization & semantic fields

In the cleaned dataset we add:
- `normalized_hashtags` (list) — e.g. `adidassambas → adidassamba`
- `brand_style`: `originals` or null (from `adidasoriginals`)
- `product_line`: one of `{samba, gazelle, spezial, tech, null}`

**Rationale:** `product_line` is a *content-semantic line* (minimal explainable unit).  
`niketech` behaves like a concrete collection line on TikTok (Tech Fleece), while `nikeair` is a broad family/tech prefix and is kept as a seed hashtag but **not** mapped to `product_line`.

---

## Quickstart (local)

### 1) Setup
Using `uv` (recommended):

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Run crawlers (placeholders)
The crawler scripts are scaffolded. Add credentials / session configuration as needed:

```bash
python -m scripts.crawl_users
python -m scripts.crawl_hashtags
```

### 3) Build clean dataset

```bash
python -m scripts.build_dataset
```

Outputs:
- `data/processed/tiktok_videos.parquet`

---

## Data contract (clean table schema)

Minimum columns expected in `data/processed/tiktok_videos.parquet`:

- ids: `video_id`, `author_id`, `author_username`
- source: `platform`, `source_type`, `source_query`, `seed_hashtag`
- time: `create_time_ts`, `create_time`, `crawled_at`, `crawled_at_ts`
- text: `caption_raw`, `hashtags`, `normalized_hashtags`
- stats: `view_count`, `like_count`, `comment_count`, `share_count`, `save_count`
- labels: `is_official_brand`, `author_verified`, `brand`, `brand_style`, `product_line`
- derived: `engagement`, `engagement_rate`

---

## Ethics & limitations

- Respect TikTok Terms of Service and local laws.
- This repo is designed for **public data** and **sample-based** analyses.
- Hashtag feeds are algorithmically ranked; results depend on crawl time and location.

---

## Next milestones
- Part 2: feature table + caption embeddings
- Retrieval demo: content-based similarity (caption + visual)
- BA add-ons: quasi-experimental “A/B-like” template effect analysis
