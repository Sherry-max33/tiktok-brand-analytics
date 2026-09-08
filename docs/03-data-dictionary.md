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

Path: `data/processed/feature/feature_table/` (partitioned by `brand`) and flat `feature_table.parquet`.

The feature table **keeps clean columns** and **adds** derived fields below. Rules / formulas: [05-feature-engineering.md](./05-feature-engineering.md). Taxonomy: [04-taxonomy.md](./04-taxonomy.md).

### Full column inventory

#### A) Pass-through from clean (when present on input)

| Field | Notes |
|-------|-------|
| `platform` | Usually `tiktok` |
| `video_id` | Primary key |
| `author_id` | |
| `author_username` | |
| `author_verified` | |
| `author_follower_count` | Used for `creator_tier` |
| `author_signature` | Used for `creator_type` |
| `source_type` | `hashtag` \| `user` |
| `source_query` | Seed tag or username |
| `seed_hashtag` | Set in clean when `source_type=hashtag` |
| `brand` | `nike` / `adidas` / `both` / null |
| `is_official_brand` | From `accounts.yaml` |
| `caption_raw` | Original caption |
| `hashtags` | Cleaned tag list |
| `normalized_hashtags` | After `normalize_tags` |
| `create_time_ts` | Unix publish time |
| `create_time` | ISO publish time (clean) |
| `crawled_at` | ISO crawl time |
| `crawled_at_ts` | Unix crawl time |
| `view_count` | |
| `like_count` | |
| `comment_count` | |
| `share_count` | |
| `collect_count` | |
| `video_duration_sec` | |
| `music_id` | |
| `has_music` | |

#### B) Identity & links (derived)

| Field | Type | Notes |
|-------|------|-------|
| `video_url` | str | `https://www.tiktok.com/video/{id}` |
| `page_url` | str \| null | `@user/video/{id}` when username present |

#### C) Time (derived)

| Field | Type | Notes |
|-------|------|-------|
| `post_date` | str | Local date from `create_time_ts` |
| `post_hour` | int | 0–23 |
| `post_weekday` | int | 0–6 (Mon–Sun) |
| `crawl_at` | datetime | Parsed `crawled_at` |
| `crawl_batch_id` | str | From `crawled_at_ts` |

#### D) Text (derived)

| Field | Type | Notes |
|-------|------|-------|
| `caption_text` | str | Caption as string |
| `caption_clean` | str | URLs/hashtags removed; lowercased |
| `embedding_text` | str | Caption + hashtags for embeddings |
| `caption_length_words` | int | From `caption_clean` |
| `hashtag_list` | list[str] | Same as cleaned `hashtags` |
| `hashtag_count` | int | |
| `mention_count` | int | `@` mentions in caption |

#### E) Engagement (derived)

| Field | Formula / notes |
|-------|-----------------|
| `engagement_count` | like + comment + share + collect |
| `engagement_rate` | `engagement_count / view_count` (null if views ≤ 0) |
| `weighted_engagement_count` | `0.10*like + 0.25*comment + 0.30*share + 0.35*collect` |
| `weighted_engagement_rate` | weighted count / views |
| `brand_relative_engagement_index` | weighted rate / brand mean weighted rate |
| `like_to_view_rate` | like / views |
| `comment_to_view_rate` | comment / views |
| `share_to_view_rate` | share / views |
| `engagement_to_view_rate` | alias of `engagement_rate` |
| `comment_to_like_ratio` | comment / like |

#### F) CTA (derived)

| Field | Meaning |
|-------|---------|
| `has_purchase_cta` | Purchase / shop intent |
| `has_engagement_cta` | Engage / comment / duet-style ask |
| `has_discovery_traffic_cta` | Follow / link-in-bio / discover |
| `has_promo_language` | Promo wording (kept separate) |
| `has_cta` | purchase OR engagement OR discovery |

Regex lives in `cta_rules.py`, not YAML.

#### G) Content & creator (derived)

| Field | Values / notes |
|-------|----------------|
| `content_type` | Multi-label list of rule-defined categories: `official_campaign`, `collaboration`, `community_impact`, `story_heritage`, `tutorial_utility`, `product_review`, `product_promo`, `product_showcase`, `vibe_ootd` (empty list = no hit) |
| `social_mechanic` | Multi-label TikTok propagation/format cues: `trend`, `challenge`, `pov`, `duet_stitch`, `template_remix`, `audio_driven`, `bts`, `grwm_ootd_format` (empty = no hit); parallel to `content_type` |
| `creator_type` | `brand` if official; else sports / lifestyle / fashion / beauty / other |
| `creator_tier` | nano / micro / mid / macro / mega |
| `is_sample_trending_audio` | Sample-relative frequent `music_id` |
| `appearance_type` | Placeholder (CV later) |

#### H) Taxonomy (derived, multi-label)

Compute order: **`brand_styles` → `product_lines` → `product_categories`**.

| Field | Empty meaning |
|-------|---------------|
| `brand_styles` | Multi-label positioning: `seed_style_map` ∪ `style_keywords` (not product SKU→style); `[]` = unrecognized |
| `product_lines` | `[]` = unrecognized |
| `product_categories` | Always ≥1 label; fallback `["uncategorized"]` |

#### I) Sentiment & language (derived, v1)

| Field | Type | Notes |
|-------|------|-------|
| `caption_original` | str | Raw caption |
| `caption_lang` | str | `en` / other ISO / `mixed` / `und` |
| `caption_lang_confidence` | float | Detector confidence |
| `has_mixed_language` | bool | e.g. EN body + non-Latin tags |
| `is_emoji_only` | bool | Essentially emoji caption |
| `caption_en` | str \| null | Native EN text or MT output fed to VADER |
| `translation_status` | str | `not_applicable` \| `done` \| `failed` \| `skipped` \| `pending` |
| `sentiment_score` | float \| null | VADER compound, or null |
| `sentiment_method` | str | `vader_en` \| `vader_via_mt` \| `not_scored_non_en` \| `not_scored` |

#### J) Text embeddings (derived)

Multilingual Sentence-BERT on `embedding_text` (**no MT** — unlike sentiment).  
Model default: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

| Field | Type | Notes |
|-------|------|-------|
| `text_embedding` | list[float] \| null | L2-normalized vector; null if skipped/disabled |
| `embedding_method` | str | `sbert_multilingual` \| `not_embedded` |
| `embedding_model` | str \| null | HuggingFace / ST model id when embedded |

Config: `feature_rules.yaml` → `text_embedding` (`enabled`, `model_name`, `min_chars`, `batch_size`).

#### K) Visual embeddings (derived) — shared frames

**One media pipeline per video** (no separate cover crawl required):

1. Assemble TikTok URL (`page_url` / `video_url` / `@user/video/{id}`)
2. Download MP4 once with **yt-dlp** → `data/processed/frames/{video_id}/video.mp4`
3. Extract **4 frames at 20% / 40% / 60% / 80%** of duration (OpenCV)
4. Same frames → CLIP (`visual_embedding`) + zero-shot `visual_format` / `visual_setting`

| Field | Type | Notes |
|-------|------|-------|
| `visual_embedding` | list[float] \| null | Mean-pooled CLIP vector (L2-normalized) |
| `visual_embedding_method` | str | `clip_visual` \| `not_embedded` |
| `visual_embedding_model` | str \| null | e.g. `sentence-transformers/clip-ViT-B-32` |
| `visual_format` | CLIP zero-shot: product_closeup / on_body_styling / sports_action / talking_head / campaign_visual / archival_retro / other / unknown |
| `visual_setting` | CLIP zero-shot: studio / outdoor / gym_training / sports_venue / event_crowd / home_indoor / retail_store / other / unknown |
| `visual_format_score` / `visual_setting_score` | Top-1 aggregated cosine score |
| `visual_format_margin` / `visual_setting_margin` | Top-1 − top-2 score |
| `visual_format_second*` / `visual_setting_second*` | Runner-up label + score |
| `visual_valid_frames` | Usable frame count (of 4) |
| `visual_classification_status` | `ok` / `partial` / `unknown` |
| `vf_*_score` / `vs_*_score` | Per-class video-level scores |
| `appearance_type` | **Removed** (was detector stub → almost always `other`) |
| `frame_paths` | list[str] | Cached `frame_000.jpg`…`frame_003.jpg` |

Config (`visual_embedding`): `frame_fractions: [0.2, 0.4, 0.6, 0.8]`, `download_video: false` by default (enable with `true` or `VISUAL_DOWNLOAD=1`). Cover is only a last-resort fallback.

#### L) Model / lineage placeholders

| Field | Status |
|-------|--------|
| `content_cluster_id` | Filled later (clustering on `text_embedding`) |
| `raw_payload_path` | Path stub |

## Comment records

Comment crawl writes `CommentRecord` JSONL (`data/raw/*comment*.jsonl`).

### Clean — `data/processed/clean/tiktok_comments.parquet`

| Field | Notes |
|-------|-------|
| CommentRecord fields except `raw_payload` | Pass-through / strip `comment_text` |
| `brand`, `source_type`, `source_query`, `seed_hashtag`, `is_official_brand`, `author_username` | Joined from video clean on `video_id` (nullable if video missing) |

Dedup key: `comment_id` (keep latest `crawled_at_ts`).

### Feature — `data/processed/feature/comment_feature_table.parquet`

| Field | Notes |
|-------|-------|
| Clean columns | Pass-through |
| `comment_text_clean` | Same cleaner as captions (URLs/hashtags stripped) |
| `comment_length_words` | |
| `mention_count` | `@` mentions |
| Sentiment / language fields | Same contract as video: `caption_lang`, `sentiment_score`, `sentiment_method`, … |
| `text_embedding`, `embedding_method`, `embedding_model` | Multilingual SBERT on cleaned comment text |
| `crawl_at`, `crawl_batch_id` | Crawl lineage |

Builder: `build_clean_comments` + `build_comment_feature_table` via `scripts/build_dataset.py`.
