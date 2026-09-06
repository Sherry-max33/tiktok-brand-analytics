# Feature engineering & analysis dimensions

Config: `configs/feature_rules.yaml`  
Builder: `src/tiktok_brand/etl/feature_table.py`

For **full column inventory** (pass-through + derived), see [03-data-dictionary.md](./03-data-dictionary.md) § Feature. Taxonomy details: [04-taxonomy.md](./04-taxonomy.md). This doc covers **how** derived fields are computed and how to analyze them.

### Feature outputs — what each file is

| Path | Entity | Meaning |
|------|--------|---------|
| `feature/feature_table.parquet` | **Video** | Flat full video feature table (easiest to load) |
| `feature/feature_table/brand=nike/` … `brand=adidas/` | **Video** | Same video features, Hive-partitioned by `brand` for scalable reads |
| `feature/comment_feature_table.parquet` | **Comment** | Comment-level features (sentiment, embeddings, …) |
| `feature/feature_test.parquet` | Video (smoke) | Small test run only |

`brand=nike` is a **folder name** (partition key=value), not a missing filename. Inside are parquet parts for that brand.

## Comment feature dimensions

Comment builder: `comment_feature_table.py` → `comment_feature_table.parquet`.

| Group | Fields |
|-------|--------|
| Identity | `comment_id`, `video_id`, `platform` |
| Text | `comment_text`, `comment_text_clean`, `comment_length_words`, `mention_count` |
| Engagement | `comment_like_count`, `comment_reply_count` |
| Commenter | `comment_author_id`, `comment_author_username`, `comment_author_nickname`, `region` |
| Video lineage (join) | `brand`, `source_type`, `source_query`, `seed_hashtag`, `is_official_brand`, `author_username` |
| Time / crawl | `comment_create_time`, `crawled_at`, `crawled_at_ts`, `crawl_at`, `crawl_batch_id` |
| Sentiment / lang | same contract as video: `caption_lang`, `sentiment_score`, `sentiment_method`, … |
| Text embedding | `text_embedding`, `embedding_method`, `embedding_model` |

No video taxonomy / CTA / content_type on comments (those stay on the video table).

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

## Sentiment (`sentiment_score`) — language routing + MT

**Model:** VADER (`vaderSentiment`) compound ≈ **[-1, 1]**.  
**Routing:** detect language first. High-confidence **English** → native VADER. **Other languages** → machine-translate to English, then VADER. Do **not** pool `vader_en` and `vader_via_mt` as one indistinguishable mean.

### Fields

| Field | Behavior |
|-------|----------|
| `caption_original` | Raw caption string |
| `caption_lang` | `en` \| `es`/`vi`/`ru`/`ko`/… \| `mixed` \| `und` |
| `caption_lang_confidence` | Detector probability in `[0, 1]` |
| `has_mixed_language` | e.g. English body + non-Latin hashtags → `True` (still may score if `en`) |
| `is_emoji_only` | Caption is essentially emoji (+ noise) |
| `caption_en` | Native cleaned EN when `vader_en`; MT English when `vader_via_mt` |
| `translation_status` | `not_applicable` \| `done` \| `failed` \| `skipped` \| `pending` (MT off) |
| `sentiment_score` | VADER compound, or null if unscored |
| `sentiment_method` | `vader_en` \| `vader_via_mt` \| `not_scored_non_en` \| `not_scored` |

Language detection runs on text **after** removing URLs, `@mentions`, hashtags, and brand proper nouns (`Nike`, `Adidas`, …) so brand names do not bias “English”.

Config (`configs/feature_rules.yaml` → `sentiment`):

| Key | Default | Meaning |
|-----|---------|---------|
| `en_confidence_threshold` | `0.80` | EN + conf ≥ this → `vader_en` |
| `mt_enabled` | `true` | Non-EN MT path on/off |
| `mt_min_chars` | `8` | Below this → `skipped`, no score |

```
if caption_lang == "en" and confidence >= 0.80:
    sentiment_method = vader_en
else if und | mixed | emoji-only | low-conf en:
    sentiment_method = not_scored
else if mt_enabled:
    translate → caption_en
    if ok: sentiment_method = vader_via_mt, translation_status = done
    if fail: not_scored_non_en, translation_status = failed
else:
    not_scored_non_en, translation_status = pending
```

- **English body + Korean hashtags:** classify by body → often `en` + `has_mixed_language=True`; still eligible for native VADER.
- **Emoji-only / hashtag-only / too short:** `und` (or MT `skipped`); not scored.
- **MT provider:** `deep-translator` GoogleTranslator (network). Needs connectivity for live feature builds; tests inject a fake `translate_fn`.

### Limitations (required interpretation)

VADER has limited ability to handle **irony/sarcasm**, **complex context**, **brand-specific jargon**, and **some emerging TikTok slang**. **MT adds further error** (slang, sarcasm, short TikTok lines). Therefore:

1. Plan a **spot-check sample** (caption + `caption_en` + score + lang/method) before trusting aggregates.
2. Do **not** interpret the score as true consumer attitude toward the brand — it measures the **emotional tone expressed in the (possibly translated) caption text**.
3. Always analyze **`vader_en` and `vader_via_mt` separately** (or report both), never as a single unlabeled mean.

### Reporting (avoid selection bias)

Always show alongside mean sentiment:

- total captions
- scored count by method (`vader_en` vs `vader_via_mt`)
- **sentiment coverage rate** = scored / total (and by method)
- **language distribution by brand**
- MT failure / skip rates (`translation_status`)

Nike vs Adidas English share and MT mix may differ; comparing pooled means without coverage/lang/method breakdown is misleading.

## Text embeddings (`text_embedding`)

**Model:** multilingual Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2` by default).  
**Input:** `embedding_text` (cleaned caption + hashtags).  
**No MT:** non-English captions are encoded directly in a shared multilingual space (different from VADER’s English-only lexicon).

| Field | Meaning |
|-------|---------|
| `text_embedding` | `list[float]` (normalized) or null |
| `embedding_method` | `sbert_multilingual` \| `not_embedded` |
| `embedding_model` | Model id when embedded |

Skip when `enabled: false` or text shorter than `min_chars`. First live run downloads the model (needs network / disk cache).

Use `caption_lang` for coverage checks; short / hashtag-heavy rows may still embed but are noisier. Keep sentiment (`vader_en` / `vader_via_mt`) and embeddings as separate features.

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

### 8. Sentiment & language / embeddings

Sentiment: `caption_lang`, `caption_lang_confidence`, `has_mixed_language`, `is_emoji_only`, `caption_original`, `caption_en`, `translation_status`, `sentiment_score`, `sentiment_method`

Text embeddings: `text_embedding`, `embedding_method`, `embedding_model`

## Visual features (unified frames)

Two consumers, **one** extraction:

| Consumer | Output |
|----------|--------|
| CLIP | `visual_embedding` (+ method/model) |
| Appearance | `appearance_type` |

Pipeline:

```
page_url / video_url / @user/video/{id}
        → yt-dlp download (VISUAL_DOWNLOAD=1 or download_video: true)
        → 4 frames at 20% / 40% / 60% / 80%
        → CLIP + appearance_type
```

Cache: `data/processed/frames/{video_id}/video.mp4` + `frame_000.jpg`…`frame_003.jpg`.  
Bulk feature builds keep `download_video: false` unless you opt in (downloads are slow / ToS-sensitive).

Still stub: `content_cluster_id`

## Example cuts

- Brand × `product_categories` × `weighted_engagement_rate`
- Brand × `content_type` × `brand_relative_engagement_index`
- `post_date` × brand volume and engagement trends
- `creator_type` × brand (official vs UGC)
- `product_lines` × engagement (e.g. jordan, samba, tech_fleece)

When exploding list columns for group-bys, one video can appear in multiple style/line/category buckets.
