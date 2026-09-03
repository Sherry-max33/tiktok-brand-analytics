# Data dictionary

Canonical field names and layer ownership. Types are logical (Parquet/JSON may coerce).

## Raw — `VideoRecord`

Written by crawl mappers to `data/raw/*.jsonl`. See also [02-crawl.md](./02-crawl.md).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `platform` | str | yes | `"tiktok"` |
| `source_type` | str | yes | `"hashtag"` \| `"user"` |
| `source_query` | str | yes | Seed hashtag or username |
| `brand` | str \| null | no | `nike` / `adidas` if known |
| `video_id` | str | yes | Unique video id |
| `create_time_ts` | int | yes | Publish time, Unix seconds |
| `caption_raw` | str | yes | Full caption |
| `hashtags` | list[str] | yes | Lowercase, no `#` |
| `author_id` | str | yes | Author id |
| `author_username` | str | yes | Author handle |
| `author_verified` | bool \| null | recommended | Platform verified flag |
| `author_follower_count` | int \| null | recommended | Follower count |
| `author_signature` | str \| null | recommended | Bio / signature |
| `video_duration_sec` | float \| null | recommended | Duration in seconds |
| `music_id` | str \| null | recommended | Music id |
| `has_music` | bool | recommended | Music present |
| `view_count` | int | yes | Plays |
| `like_count` | int | yes | Likes |
| `comment_count` | int | yes | Comments |
| `share_count` | int | yes | Shares |
| `collect_count` | int \| null | yes* | Favorites/bookmarks (Apify `collectCount`) |
| `crawled_at` | str | yes | ISO-8601 crawl time |
| `crawled_at_ts` | int | yes | Crawl time, Unix seconds |
| `raw_payload` | object | optional | Full API payload (dropped in clean) |

\*Legacy CSV may use `save_count`; clean aliases it to `collect_count`.

## Clean — silver fact table

Path: `data/processed/clean/tiktok_videos.parquet` (prod) or `clean_test.parquet` (smoke).

**Clean does not add taxonomy or engagement rates.**

| Field | Source | Notes |
|-------|--------|-------|
| All raw video fields except `raw_payload` | pass-through / cleanup | |
| `normalized_hashtags` | clean | Alias map from `hashtags.yaml` |
| `brand` | raw or inferred | `nike` / `adidas` / `both` / null |
| `seed_hashtag` | clean | `source_query` when `source_type=hashtag`, else null |
| `is_official_brand` | clean | Username in `accounts.yaml` |
| `create_time` | clean | ISO from `create_time_ts` + project timezone |
| `collect_count` | raw / alias | Prefer this name over `save_count` |

Dedup key: `video_id` (keep row with latest `crawled_at_ts`).

## Feature — gold table

Path: `data/processed/feature/feature_table/` (partitioned by `brand`, `post_date`) or `feature_test.parquet`.

### Identity & links

| Field | Type | Notes |
|-------|------|-------|
| `video_id` | str | |
| `video_url` | str | `https://www.tiktok.com/video/{id}` |
| `page_url` | str \| null | `@user/video/{id}` when username present |

### Time

| Field | Type | Notes |
|-------|------|-------|
| `post_date` | str | Local date from `create_time_ts` |
| `post_hour` | int | 0–23 |
| `post_weekday` | int | 0–6 (Mon–Sun) |
| `crawl_at` | datetime | Parsed `crawled_at` |
| `crawl_batch_id` | str | From `crawled_at_ts` |

### Text

| Field | Type | Notes |
|-------|------|-------|
| `caption_text` | str | Caption as string |
| `caption_clean` | str | Cleaned for NLP |
| `embedding_text` | str | Caption + hashtags for embeddings |
| `caption_length_words` | int | |
| `hashtag_list` | list[str] | Same as cleaned hashtags |
| `hashtag_count` | int | |
| `mention_count` | int | `@` mentions in caption |

### Engagement

| Field | Formula / notes |
|-------|-----------------|
| `engagement_count` | like + comment + share + collect |
| `engagement_rate` | `engagement_count / view_count` (null if views ≤ 0) |
| `weighted_engagement_count` | `0.10*like + 0.25*comment + 0.30*share + 0.35*collect` (see `feature_rules.yaml`) |
| `weighted_engagement_rate` | weighted count / views |
| `brand_relative_engagement_index` | weighted rate / brand mean weighted rate |
| `like_to_view_rate` | like / views |
| `comment_to_view_rate` | comment / views |
| `share_to_view_rate` | share / views |
| `engagement_to_view_rate` | same as engagement_rate (explicit alias) |
| `comment_to_like_ratio` | comment / like |

### CTA (multi-flag)

| Field | Meaning |
|-------|---------|
| `has_purchase_cta` | Purchase / shop intent |
| `has_engagement_cta` | Engage / comment / duet-style ask |
| `has_discovery_traffic_cta` | Follow / link-in-bio / discover |
| `has_promo_language` | Promo wording (kept separate) |
| `has_cta` | purchase OR engagement OR discovery |

Regex lives in `cta_rules.py`, not YAML.

### Content & creator

| Field | Values / notes |
|-------|----------------|
| `content_type` | `social_viral`, `official_campaign`, `story_heritage`, `tutorial_utility`, `shopping_haul`, `vibe_ootd`, `community_collab` (P0–P6 priority) |
| `creator_type` | `brand` if official; else signature keywords → sports / lifestyle / fashion / beauty / other |
| `creator_tier` | nano / micro / mid / macro / mega from follower thresholds |
| `is_sample_trending_audio` | Sample-relative: frequent `music_id` above percentile |
| `appearance_type` | Placeholder (CV later) |

### Taxonomy (multi-label lists)

Compute order: **`brand_styles` → `product_lines` → `product_categories`**.

| Field | Empty meaning |
|-------|---------------|
| `brand_styles` | `[]` = unrecognized |
| `product_lines` | `[]` = unrecognized |
| `product_categories` | Always ≥1 label; fallback `["uncategorized"]` |

See [04-taxonomy.md](./04-taxonomy.md).

### Model placeholders

| Field | Status |
|-------|--------|
| `sentiment_score` | Stub / lightweight |
| `text_embedding` | Placeholder string |
| `visual_embedding` | Id stub from video_id |
| `content_cluster_id` | Filled later (clustering) |
| `raw_payload_path` | Path stub |

## Comment records

Comment crawl writes separate JSONL (`CommentRecord`). Comment clean/feature ETL is not the primary video dictionary above; see crawl scripts and `comment_schemas.py`.
