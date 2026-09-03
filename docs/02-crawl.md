# Crawl

Operators and field mapping for video (and comment) collection.

## Sources

1. **Hashtag crawl** — `source_type = "hashtag"`, `source_query = <seed tag>`
2. **Official account crawl** — `source_type = "user"`, `source_query = <username>`

Both share the same `VideoRecord` schema. Downstream dedupes by `video_id`.

## Seed hashtags

Canonical list: `configs/hashtags.yaml` (Nike **19** + Adidas **20** = **39**).

### Nike

`nike`, `nikeshoes`, `nikesneakers`, `nikeair`, `nikestyle`, `nikeoutfit`, `nikefit`, `nikelook`, `nikerunning`, `nikebasketball`, `niketraining`, `niketiktok`, `nikereview`, `niketech`, `jordan`, `airmax`, `af1`, `nikedunk`, `airforce1`

### Adidas

`adidas`, `adidasshoes`, `adidasoriginals`, `adidasstyle`, `adidasoutfit`, `adidaslook`, `adidasfootball`, `adidasrunning`, `adidastraining`, `adidastiktok`, `adidasreview`, `adidassamba`, `adidassambas`, `adidasgazelle`, `adidasgazelles`, `adidasspezial`, `adidasspezials`, `ultraboost`, `adidasforum`, `tracksuit`

### Alias normalization (clean layer)

```yaml
normalize_tags:
  adidassambas: adidassamba
  adidasgazelles: adidasgazelle
  adidasspezials: adidasspezial
```

Crawlers may return either form; clean maps plurals to canonical tags before feature taxonomy.

Product/style maps are in `configs/taxonomy.yaml`, not `hashtags.yaml`.

## Official accounts

From `configs/accounts.yaml`:

| Brand | Usernames |
|-------|-----------|
| Nike | `nike`, `jumpman23` |
| Adidas | `adidas` |

## Sampling (parameters)

From `configs/project.yaml` (pipeline parameters, not achieved totals):

| Parameter | Typical value |
|-----------|---------------|
| `videos_per_hashtag` | 100 |
| `videos_per_account` | 500 |
| Comment sort | `like_count` desc |
| `target_videos_for_comments` | 105 |
| `candidate_videos_for_comments` | 120 |
| `comments_per_video` | 100 |

Comment algorithm: rank videos → candidate pool → crawl in order → skip failures → stop at target successes.

## Required video fields

| Field | Type | Notes |
|-------|------|-------|
| `video_id` | str | Unique id |
| `create_time_ts` | int | Unix seconds |
| `caption_raw` | str | Full caption |
| `hashtags` | list[str] | Prefer API list; else parse `#` from caption |
| `author_id` | str | |
| `author_username` | str | Official detection |
| `view_count` | int | Always `view_count` (never `play_count` in our schema) |
| `like_count` | int | |
| `comment_count` | int | |
| `share_count` | int | |
| `collect_count` | int | Apify `collectCount`; legacy alias `save_count` |
| `source_type` | str | `hashtag` \| `user` |
| `source_query` | str | Seed |
| `crawled_at` | str | ISO-8601 |
| `crawled_at_ts` | int | Unix seconds |

## Recommended video fields

| Field | Type | Downstream use |
|-------|------|----------------|
| `author_signature` | str | `creator_type` |
| `author_follower_count` | int | `creator_tier` |
| `author_verified` | bool | Stored as-is |
| `video_duration_sec` | float | Convert ms → sec if needed |
| `music_id` | str | `is_sample_trending_audio` |
| `has_music` | bool | |
| `brand` | str \| null | Else inferred in clean |

## Optional

- `platform`: `"tiktok"`
- `raw_payload`: full API object (dropped in clean)

## Not produced by crawl

These are **feature** (or clean) derived — do not invent them in the crawler:

| Field | Layer |
|-------|-------|
| `normalized_hashtags` | clean |
| `brand_styles`, `product_lines`, `product_categories` | feature |
| `content_type`, CTA flags, engagement rates | feature |
| `appearance_type`, embeddings, `content_cluster_id` | feature placeholders |

## Output format

- **Format:** JSONL, UTF-8, one `VideoRecord` JSON object per line
- **Suggested names:**
  - Hashtag: `tiktok_hashtag_{brand}_{hashtag}_{timestamp}.jsonl`
  - User: `tiktok_user_{brand}_{username}_{timestamp}.jsonl`
- **Directory:** `data/raw/`

Dedup is **not** required at crawl time; clean keeps the latest `crawled_at_ts` per `video_id`.

## Apify mapping

Actors (see `project.yaml`):

- Videos: `clockworks/tiktok-scraper`
- Comments: `clockworks/tiktok-comments-scraper`

Mapper: `src/tiktok_brand/crawl/apify_video_mapper.py`

Typical mappings:

| Apify | VideoRecord |
|-------|-------------|
| `id` / `idStr` | `video_id` |
| `text` / `desc` | `caption_raw` |
| `playCount` | `view_count` |
| `diggCount` | `like_count` |
| `commentCount` | `comment_count` |
| `shareCount` | `share_count` |
| `collectCount` | `collect_count` |
| `videoMeta.duration` | `video_duration_sec` |
| `musicMeta.musicId` | `music_id` |

Crawler injects `source_type`, `source_query`, `brand`, `crawled_at`, `crawled_at_ts`, `platform`, `raw_payload`.
