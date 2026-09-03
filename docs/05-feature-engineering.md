# Feature engineering & analysis dimensions

Config: `configs/feature_rules.yaml`  
Builder: `src/tiktok_brand/etl/feature_table.py`

For column types and formulas, see [03-data-dictionary.md](./03-data-dictionary.md). Taxonomy details: [04-taxonomy.md](./04-taxonomy.md).

## Engagement metrics

Weights (`engagement_weights`):

| Signal | Weight |
|--------|--------|
| like | 0.10 |
| comment | 0.25 |
| share | 0.30 |
| collect | 0.35 |

Derived columns:

- `engagement_count` — unweighted sum
- `engagement_rate` / `engagement_to_view_rate`
- `weighted_engagement_count` / `weighted_engagement_rate`
- `brand_relative_engagement_index` — weighted rate ÷ brand mean

## CTA flags

Multi-label (regex in `cta_rules.py`):

- `has_purchase_cta`
- `has_engagement_cta`
- `has_discovery_traffic_cta`
- `has_promo_language` (orthogonal; not OR’d into `has_cta`)
- `has_cta` = purchase ∨ engagement ∨ discovery

## Content type (P0–P6)

Match order in `content_type_rules` (classification priority, not quality ranking):

1. `social_viral`
2. `official_campaign`
3. `story_heritage`
4. `tutorial_utility`
5. `shopping_haul`
6. `vibe_ootd`
7. `community_collab`

## Creator

| Field | Logic |
|-------|-------|
| `creator_type` | Official → `brand`; else keywords on `author_signature` |
| `creator_tier` | Follower thresholds: nano &lt; 10k &lt; micro &lt; 100k &lt; mid &lt; 1M &lt; macro &lt; 10M ≤ mega |
| `is_sample_trending_audio` | High-frequency `music_id` in this sample (not global TikTok trend) |

## Analysis dimensions (feature table)

### 1. Brand & source

| Dimension | Field | Example uses |
|-----------|-------|--------------|
| Brand | `brand` | Nike vs Adidas splits |
| Source | `source_type` | Hashtag explore vs profile crawl |
| Seed | `source_query` / `seed_hashtag` | Tag/account contribution |

### 2. Time

| Dimension | Field |
|-----------|-------|
| Date | `post_date` |
| Hour | `post_hour` |
| Weekday | `post_weekday` |
| Crawl batch | `crawl_at` / `crawl_batch_id` |

### 3. Content form

| Dimension | Field |
|-----------|-------|
| Rule type | `content_type` |
| Cluster (experimental) | `content_cluster_id` |
| Appearance (CV stub) | `appearance_type` |
| Duration / music | `video_duration_sec`, `has_music`, `music_id` |
| Sample trending audio | `is_sample_trending_audio` |

### 4. Brand semantics (multi-label)

| Dimension | Field | Notes |
|-----------|-------|-------|
| Styles | `brand_styles` | list; `[]` = unrecognized |
| Lines | `product_lines` | list; `[]` = unrecognized |
| Categories | `product_categories` | list; always includes at least one label |

### 5. Creator

| Dimension | Field |
|-----------|-------|
| Official | `is_official_brand` |
| Type | `creator_type` |
| Tier | `creator_tier` |
| Followers / verified | `author_follower_count`, `author_verified` |

### 6. Caption & intent

| Dimension | Field |
|-----------|-------|
| Length / tags / mentions | `caption_length_words`, `hashtag_count`, `mention_count` |
| CTA | `has_cta`, `has_purchase_cta`, … |

### 7. Outcomes

Absolute counts (`view_count`, `like_count`, …, `collect_count`) plus rates and weighted / brand-relative metrics above.

### 8. Model stubs

`sentiment_score`, `text_embedding`, `visual_embedding`, `content_cluster_id`

## Example cuts

- Brand × `product_categories` × `weighted_engagement_rate`
- Brand × `content_type` × `brand_relative_engagement_index`
- `post_date` × brand volume and engagement trends
- `creator_type` × brand (official vs UGC)
- `product_lines` × engagement (e.g. jordan, samba, tech_fleece)

When exploding list columns for group-bys, one video can appear in multiple style/line/category buckets.
