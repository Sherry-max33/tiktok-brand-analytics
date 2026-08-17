# Feature Table 分析维度一览

基于当前 feature table 的字段，按「可用来分组、筛选、对比」的维度整理，便于做报表和建模。

---

## 1. 品牌与来源（Who / 从哪来）

| 维度 | 字段 | 取值示例 | 分析用途 |
|------|------|----------|----------|
| 品牌 | `brand` | nike, adidas, both, null | 品牌对比、分品牌看热度/内容分布 |
| 数据来源 | `source_type` | hashtag, user | 区分 hashtag 探索 vs 官方/用户主页 |
| 种子标签 | `source_query` / `seed_hashtag` | nike, jordan, adidassamba, nike (user) | 看哪个 hashtag/账号贡献了多少内容与互动 |

---

## 2. 时间（When）

| 维度 | 字段 | 取值示例 | 分析用途 |
|------|------|----------|----------|
| 发布日期 | `post_date` | 2026-03-01 | 趋势、按日/周/月聚合 |
| 发布小时 | `post_hour` | 0–23 | 发布时间偏好、高峰时段 |
| 星期几 | `post_weekday` | 0–6 (Mon–Sun) | 周末 vs 工作日 |
| 抓取批次 | `crawl_at` / `crawl_batch_id` | datetime / 1772930516 | 区分爬取批次、增量更新 |

---

## 3. 内容形态（What 内容）

| 维度 | 字段 | 取值示例 | 分析用途 |
|------|------|----------|----------|
| 内容类型（规则） | `content_type` | social_viral, official_campaign, story_heritage, tutorial_utility, shopping_haul, vibe_ootd, community_collab | 验证策略执行、按内容类型看互动 |
| 内容聚类（实验） | `content_cluster_id` | 0, 1, 2, … (K-Means) | 发现未定义的内容模式、探索分析 |
| 出镜类型（预留） | `appearance_type` | creator, model, product_only, mixed, null | 谁出镜对互动的影响（需 CV 填充） |
| 视频时长 | `video_duration_sec` | 15.5 | 时长分布、与互动关系 |
| 是否有音乐 | `has_music` | True/False | 音乐使用与互动 |
| 音乐 ID | `music_id` | 7xxxxx | 爆款 BGM、is_trending_audio 扩展 |

---

## 4. 品牌语义（产品/风格假设）

| 维度 | 字段 | 取值示例 | 分析用途 |
|------|------|----------|----------|
| 风格 | `brand_style` | performance, lifestyle, technical, retro | 按风格看热度、品牌调性对比 |
| 产品线 | `product_line` | tech_fleece, jordan, samba, dunk, … | 具体系列热度、鞋 vs 服装 |
| 品类 | `product_category` | shoes, apparel, uncategorized | 鞋类 vs 服装类对比、占比 |

---

## 5. 创作者（Who 发布）

| 维度 | 字段 | 取值示例 | 分析用途 |
|------|------|----------|----------|
| 是否官方 | `is_official_brand` | True/False | 官方 vs UGC 对比 |
| 创作者类型 | `creator_type` | brand, sports, lifestyle, fashion, beauty, other | KOL 类型分布、哪种类型互动高 |
| 粉丝数 | `author_follower_count` | 10000 | 影响力分层、大号 vs 小号 |
| 是否认证 | `author_verified` | True/False | 认证与互动/可信度 |

---

## 6. 文案与互动意图

| 维度 | 字段 | 取值示例 | 分析用途 |
|------|------|----------|----------|
| 文案词数 | `caption_length_words` | 12 | 文案长度与互动 |
| 话题数 | `hashtag_count` | 5 | 打标数量与曝光/互动 |
| 是否 CTA | `has_call_to_action` | True/False | 转化向内容占比 |
| 是否带产品链接意向 | `has_product_link` | True/False | 带货向内容占比 |
| @ 提及数 | `mention_count` | 2 | 合作/互动意图 |

---

## 7. 互动指标（Outcome / 可聚合）

| 维度 | 字段 | 说明 | 分析用途 |
|------|------|------|----------|
| 播放 | `view_count` | 绝对播放量 | 热度、爆款筛选 |
| 点赞 | `like_count` | 绝对点赞 | 喜好度 |
| 评论 | `comment_count` | 绝对评论 | 讨论度 |
| 分享 | `share_count` | 绝对分享 | 传播度 |
| 收藏 | `save_count` | 绝对收藏 | 留存/种草 |
| 总互动 | `engagement_count` | like+comment+share+save | 总互动量 |
| 互动率 | `engagement_rate` | engagement_count / view_count | 内容效率 |
| 点赞率 | `like_to_view_rate` | like_count / view_count | 点赞效率 |
| 评论率 | `comment_to_view_rate` | comment_count / view_count | 评论效率 |
| 分享率 | `share_to_view_rate` | share_count / view_count | 分享效率 |
| 综合得分 | `engagement_score` | 加权组合 (like/comment/share) | 单视频综合表现 |
| 相对得分 | `norm_engagement_score` | 该视频互动率 / 品牌均值 | 跨品牌可比、超常表现 |
| 评论/点赞比 | `comment_to_like_ratio` | comment / like | 讨论深度 |

---

## 8. 实验/模型产出（占位或后续填）

| 维度 | 字段 | 说明 |
|------|------|------|
| 情感 | `sentiment_score` | 文案情感（占位/模型） |
| 视觉向量 | `visual_embedding` | 视频视觉表征（占位/模型） |
| 内容簇 | `content_cluster_id` | K-Means 等聚类 ID（notebook 填） |

---

## 9. 常用分析组合示例

- **品牌 × 品类 × 互动率**：Nike 鞋 vs 服装 的 engagement_to_view_rate 对比  
- **品牌 × content_type**：各内容类型（tutorial / shopping_haul 等）的占比与 norm_engagement_score  
- **时间 × brand**：按 post_date 看两品牌发稿量与互动趋势  
- **creator_type × brand**：官方 vs 各 KOL 类型的互动表现  
- **product_line × engagement**：各产品线（jordan / samba / tech_fleece）的热度与互动  

---

## 10. 标识与元数据（不单独做维度，用于关联）

- `video_id`, `video_url`：唯一标识与回链  
- `platform`, `crawled_at`, `crawl_batch_id`, `raw_payload_path`：数据 lineage 与回溯  
