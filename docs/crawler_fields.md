# 爬虫需获取字段（基于研究目标与 Feature Table）

研究目标：Nike vs Adidas 品牌热度对比、内容策略验证、互动与创作者分析。  
下游：clean table（语义事实）→ feature table（分析/建模特征）。

以下为 **Apify clockworks/tiktok-scraper** 经 `apify_mapper` 映射后必须或建议提供的字段（`VideoRecord` / feature table 一一对应）。  
Raw 层落盘格式：**JSONL**（`data/raw/*.jsonl`，每行一个 JSON 对象）。

---

## 一、必须提供（缺一不可）

| 字段 | 类型 | 用途 | 对应 Feature 列 |
|------|------|------|-----------------|
| **video_id** | str | 视频唯一标识、去重、video_url | video_id, video_url |
| **create_time_ts** | int (Unix 秒) | 发布时间、post_date/post_hour/post_weekday | post_date, post_hour, post_weekday |
| **caption_raw** | str | 文案 → content_type / 情感 / CTA / product_category 启发式 | caption_text, content_type, has_call_to_action, has_product_link, sentiment_score |
| **hashtags** | list[str] | 或从 caption 抽取；→ normalized_hashtags → brand_style / product_line / product_category | normalized_hashtags, brand_style, product_line, product_category, hashtag_list, hashtag_count |
| **author_username** | str | 官方账号识别 (is_official_brand)、创作者维度 | is_official_brand, creator_type(品牌) |
| **author_id** | str | 创作者唯一标识 | author_id |
| **view_count** | int | 播放量（全流程统一用 view_count，无 play_count） | view_count, 各类 *_to_view_rate, norm_engagement_score |
| **like_count** | int | 点赞 | like_count, engagement_count, comment_to_like_ratio |
| **comment_count** | int | 评论 | comment_count, engagement_count |
| **share_count** | int | 分享 | share_count, engagement_count |
| **save_count** | int | 收藏，参与 engagement 计算 | save_count, engagement_count |
| **source_type** | str | `hashtag` \| `user`，数据来源 | source_type, seed_hashtag |
| **source_query** | str | 种子 hashtag 或 username | source_query, seed_hashtag |
| **crawled_at** | str (ISO) | 抓取时间 | crawled_at, crawl_at |
| **crawled_at_ts** | int | 抓取时间戳、crawl_batch_id | crawled_at_ts, crawl_batch_id |

---

## 二、强烈建议（支撑分析与研究目标）

| 字段 | 类型 | 用途 | 对应 Feature 列 |
|------|------|------|-----------------|
| **author_signature** | str | KOL 主页描述 → creator_type (sports/lifestyle/fashion/beauty/other) | creator_type |
| **author_follower_count** | int | 创作者影响力、分层分析 | author_follower_count |
| **author_verified** | bool | 认证标识 | author_verified |
| **video_duration_sec** | float | 视频时长（秒），若 API 给 ms 需除以 1000 | video_duration_sec |
| **music_id** | str | 音乐 ID（或 idStr） | music_id, has_music |
| **has_music** | bool | 是否有音乐（可由 music 非空推断） | has_music |
| **brand** | str \| null | 若 API 能带则保留，否则由 hashtags 推断 | brand |

---

## 三、可选（有则保留，无则占位）

| 字段 | 类型 | 用途 | 说明 |
|------|------|------|------|
| **raw_payload** | dict | 原始 API 响应，审计/回溯 | clean 表会 drop；可不落库或只存路径 |
| **platform** | str | 固定 `tiktok` | 爬虫可写死 |

---

## 四、不在爬虫层、由下游补齐的字段

| 字段 | 来源 | 说明 |
|------|------|------|
| **brand_style** | clean (hashtags.yaml) | 标签→风格映射 |
| **product_line** | clean (hashtags.yaml) | 标签→产品线映射 |
| **product_category** | clean (三层逻辑) | line/category_map/caption 启发式 |
| **content_type** | feature (关键词规则) | P0–P6 基于 caption_text |
| **appearance_type** | feature (占位) | 后续 CV 模型填充 |
| **content_cluster_id** | feature (占位) | notebook K-Means 聚类后填 |
| **sentiment_score** | feature (占位/模型) | 基于 caption_raw |
| **visual_embedding** | feature (占位/模型) | 基于 video_id |

---

## 五、Apify 字段映射

Actor 输出经 `apify_mapper.py` 映射到 VideoRecord：

- `id` → **video_id**
- `createTime` → **create_time_ts**
- `text` → **caption_raw**
- `hashtags[].name` → **hashtags**（list[str]）
- `authorMeta.id` / `authorMeta.name` → **author_id** / **author_username**
- `authorMeta.signature` → **author_signature**
- `authorMeta.verified` → **author_verified**
- `authorMeta.fans` → **author_follower_count**
- `playCount` / `diggCount` / `commentCount` / `shareCount` / `collectCount` → 互动指标
- `videoMeta.duration` → **video_duration_sec**（秒）
- `musicMeta.musicId` → **music_id**；非空 → **has_music**

抓取时由 crawler 写入 **source_type**、**source_query**、**brand**、**crawled_at** / **crawled_at_ts**、**platform**、**raw_payload**。
