# 爬虫需求 Brief（给执行方）

## 一、数据来源（两路都要爬）

1. **按 Hashtag 爬**：每个 seed hashtag 下抓视频，`source_type = "hashtag"`，`source_query = 该 hashtag`（如 nike, jordan, adidassamba）。
2. **按官方账号爬**：每个官方账号主页抓其发布的视频，`source_type = "user"`，`source_query = 该账号 username`（如 nike, jumpman23, adidas）。

两路数据都进同一套字段，用 `source_type` / `source_query` 区分即可。

---

## 二、每条视频需返回字段

### 必填（缺一不可）
| 字段 | 类型 | 说明 |
|------|------|------|
| video_id | string | 视频唯一 ID |
| create_time_ts | int | 发布时间，Unix 秒 |
| caption_raw | string | 文案/描述全文 |
| hashtags | array of string | 若 API 有结构化标签列表直接给；否则从 caption 里解析 #xxx，统一小写、去掉 # |
| author_id | string | 作者 ID |
| author_username | string | 作者用户名（用于区分是否官方账号） |
| view_count | int | 播放量 |
| like_count | int | 点赞 |
| comment_count | int | 评论 |
| share_count | int | 分享 |
| save_count | int | 收藏（若无则 0 或 null） |
| source_type | string | 固定为 `"hashtag"` 或 `"user"` |
| source_query | string | 本条的种子：hashtag 名 或 用户名 |
| crawled_at | string | 抓取时间，建议 ISO（如 2026-03-10T12:00:00Z） |
| crawled_at_ts | int | 抓取时间 Unix 秒 |

### 强烈建议（用于分析）
| 字段 | 类型 | 说明 |
|------|------|------|
| author_signature | string | 作者主页简介/签名 |
| author_follower_count | int | 粉丝数 |
| author_verified | bool | 是否认证 |
| video_duration_sec | float | 视频时长（秒）；若 API 给毫秒请除以 1000 |
| music_id | string | 音乐 ID（或 idStr） |
| has_music | bool | 是否有音乐 |
| brand | string / null | 若接口能带品牌则填 nike / adidas，否则可留空，下游按 hashtags 推断 |

### 可选
- platform：可写死 `"tiktok"`。

---

## 三、Seed 列表

### 1）按 Hashtag 爬（source_type = "hashtag"）

**Nike 品牌相关（19 个）**  
nike, nikeshoes, nikesneakers, nikeair, nikestyle, nikeoutfit, nikefit, nikelook, nikerunning, nikebasketball, niketraining, niketiktok, nikereview, niketech, jordan, airmax, af1, nikedunk, airforce1  

**Adidas 品牌相关（20 个）**  
adidas, adidasshoes, adidasoriginals, adidasstyle, adidasoutfit, adidaslook, adidasfootball, adidasrunning, adidastraining, adidastiktok, adidasreview, adidassamba, adidassambas, adidasgazelle, adidasgazelles, adidasspezial, adidasspezials, ultraboost, adidasforum, tracksuit  

### 2）按官方账号爬（source_type = "user"）

**Nike**：nike, jumpman23  
**Adidas**：adidas  

---

## 四、输出格式与命名

- **格式**：JSONL，每行一个 JSON 对象，字段名与上表一致（英文、下划线）。
- **命名建议**（便于我们接 pipeline）：  
  - 按 hashtag：`tiktok_hashtag_{brand}_{hashtag}_{timestamp}.jsonl`（如 tiktok_hashtag_nike_jordan_1773123456.jsonl）  
  - 按 user：`tiktok_user_{brand}_{username}_{timestamp}.jsonl`（如 tiktok_user_nike_jumpman23_1773123456.jsonl）  
- **编码**：UTF-8。

---

## 五、规模与去重说明（参考）

- 目标总视频量约 **6,000**，约 **30** 个关键词（hashtag + user 合计），平均每个约 **200** 条（可按 API 限流/配额调整）。
- 同一视频可能出现在多个 hashtag 或同时出现在 hashtag + user；**不需要**在爬虫侧去重，下游会按 `video_id` 去重保留最新一条。

---

## 六、Hashtags 字段约定

- 若 API 返回结构化 hashtag 列表，**优先直接使用**，并保证为字符串数组、小写、无 # 前缀（如 `["nike", "jordan", "fashion"]`）。
- 若无结构化数据，则从 `caption_raw` 中解析所有 `#xxx`，去 #、转小写后放入 `hashtags` 数组。
- 下游会对 adidassambas / adidasgazelles / adidasspezials 等做复数归一化，爬虫按 API 原样返回即可。

---

以上 brief 对应 Apify `clockworks/tiktok-scraper` 输出；crawler 经 `apify_mapper` 写入 `data/raw/*.jsonl`。
