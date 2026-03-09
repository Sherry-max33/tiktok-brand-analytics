"""
Feature table: 在 clean table 基础上派生分析/建模特征。

命名规范：
- *_at = datetime, *_ts = Unix int；*_count = 计数；*_rate = 比率；is_/has_ = bool。
- 全流程无 play_count，仅使用 view_count；无 engagement 别名，统一使用 engagement_count（含 save）。
- 无 post_create_time，只保留 post_date/post_hour/post_weekday；无 author_is_verified，使用 author_verified。

继承规则：
- clean table 中的大部分列（platform/source_type/source_query/brand/video_id/...）在 feature table 中原样保留。
- 本模块只新增或重算下面这些特征列。

时间类：
- post_date/post_hour/post_weekday
  - 若存在 create_time_ts：按 tz（默认 America/New_York）将 Unix 秒转为带时区 datetime，再拆出日期/小时/星期几。
  - 否则：设为 None / NA。

互动与比率：
- engagement_count
  - 若 clean 已有：直接保留。
  - 若没有：按 like+comment+share+save 重算（缺失视为 0）。
- engagement_rate
  - 若 clean 已有：直接保留。
  - 若没有：engagement_count / view_count（view_count<=0 时设为 NA）。
- like_to_view_rate/comment_to_view_rate/share_to_view_rate/engagement_to_view_rate
  - 各自计数 ÷ view_count（view_count==0 记为 NA）。
- engagement_score
  - 0.2 * view_count * like_to_view_rate
    + 0.5 * view_count * comment_to_view_rate
    + 0.3 * view_count * share_to_view_rate（权重可调）。
- norm_engagement_score
  - 对每个品牌 brand，计算其 engagement_to_view_rate 的历史均值；
  - norm_engagement_score = 当前视频的 engagement_to_view_rate / 品牌均值（分母为 0 时记为 NA）。
- comment_to_like_ratio
  - comment_count / like_count（like_count==0 或缺失记为 NA）。

文本 & hashtag：
- caption_text
  - 等于 clean.caption_raw 的字符串版本。
- caption_length_words
  - caption_text 按空格拆分后的词数。
- hashtag_list/hashtag_count
  - hashtag_list = clean.hashtags（保证为 list）；hashtag_count 为其长度。
- mention_count
  - caption_text 中出现 @username 的次数（正则 @\\w+）。
- has_call_to_action
  - caption_text.lower() 中是否包含 CTA 关键词：{shop,buy,link in bio,linkinbio,get it,order,check out,sale}。
- has_product_link
  - caption_text.lower() 中是否包含电商关键词：{shop,buy,order,sale,link,product}。

账号 & 创作者：
- creator_type
  - 若 is_official_brand 为 True：creator_type = \"brand\"。
  - 否则：根据 author_signature.lower() 中是否包含 sports/lifestyle/fashion/beauty 关键词映射；
    若均未命中，则为 \"other\"。
- author_follower_count
  - 若 clean 中存在：直接保留；否则为 NA。

视频 & 音乐：
- video_url
  - 由 video_id 拼接为 \"https://www.tiktok.com/video/{video_id}\"，video_id 为空时为 None。
- video_duration_sec
  - 若 clean 中存在：保留；否则 NA。
- music_id/has_music
  - 若 clean 中存在：保留；否则 music_id=None, has_music=False。
- is_trending_audio
  - 预留占位，目前统一为 False。
- content_type
  - 基于 caption_text.lower() 的关键词规则 _CONTENT_TYPE_KEYWORDS，自上而下匹配：
    - unboxing/unbox/开箱 → \"unboxing\"
    - review/测评/试穿/试穿分享 → \"review\"
    - tutorial/how to/教学/教程/穿搭技巧 → \"tutorial\"
    - ootd/outfit of the day/今日穿搭 → \"ootd\"
    - haul/haul video/购物分享 → \"haul\"
  - 未命中则为 None，用于验证已知内容策略执行情况。
- appearance_type
  - 预留占位，由后续 CV 流水线根据画面中谁出镜（creator/model/product_only/mixed 等）填充；当前为 None。

品牌语义：
- brand_style
  - 直接沿用 clean.brand_style，它由 configs/hashtags.yaml.brand_style_map 对 normalized_hashtags 映射而来，
    属于业务假设层（Hypothesis）。
- product_line/product_category
  - 直接沿用 clean 中的 product_line/product_category：
    - product_line 源自 product_line_map。
    - product_category 基于 product_line / product_category_map / caption_raw 关键词的三层逻辑推断。

Embedding & 聚类：
- sentiment_score
  - 对 caption_raw 调用 compute_sentiment_score（目前为占位实现，可替换为真实情感模型）。
- visual_embedding
  - 对 video_id 调用 compute_visual_embedding_id（目前为占位 ID，未来可替换为真实视觉向量）。
- content_cluster_id
  - 占位字段：默认 NA，计划由 notebook/analysis 基于 visual_embedding + 文本向量，
    使用 K-Means 等无监督聚类结果填充，用于发现未预先定义的内容模式（Experiment）。

抓取元数据：
- crawl_at
  - 从 crawled_at 解析得到的 datetime（带时区），便于时间排序和时间差计算。
- crawl_batch_id
  - 由 crawled_at_ts 转成字符串，用来标记本次 crawl 批次。
- raw_payload_path
  - 预留列，目前为 \"\"；未来可写入指向原始 JSON/截图等原始资产的存储路径，实现从特征表回溯到 raw 数据。
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..embeddings.caption_embed import compute_sentiment_score
from ..embeddings.visual_embed import compute_visual_embedding_id

_CTA_KEYWORDS = frozenset(
    {"shop", "buy", "link in bio", "linkinbio", "get it", "order", "check out", "sale"}
)

# caption_text 关键词 → content_type（自上而下，验证已知；顺序优先，先匹配先返回）
# P0–P6 优先级依次降低：social_viral → official_campaign → story_heritage → tutorial_utility
# → shopping_haul → vibe_ootd → community_collab
_CONTENT_TYPE_KEYWORDS = {
    # P0 社交爆款 / 流量引擎
    "social_viral": [
        "trend",
        "trending",
        "meme",
        "pov",
        "capcut",
        "challenge",
        "funny",
        "transition",
        "joke",
    ],
    # P1 官方 Campaign / 上新大片
    "official_campaign": [
        "official",
        "collection",
        "launch",
        "announcing",
        "global",
        "season",
        "exclusive",
    ],
    # P2 品牌故事 / 传承
    "story_heritage": [
        "history",
        "legacy",
        "archive",
        "founder",
        "story",
        "interview",
        "behind the scenes",
        "bts",
        "making of",
    ],
    # P3 教学 / 功能指导
    "tutorial_utility": [
        "tutorial",
        "how to",
        "hack",
        "tips",
        "guide",
        "technique",
        "steps",
        "diy",
        "教学",
        "教程",
    ],
    # P4 种草 / 购物分享
    "shopping_haul": [
        "unboxing",
        "unbox",
        "haul",
        "shopping",
        "pickups",
        "wishlist",
        "try on",
        "review",
        "测评",
        "开箱",
    ],
    # P5 日常穿搭 / 氛围
    "vibe_ootd": [
        "ootd",
        "grwm",
        "get ready with me",
        "outfit of the day",
        "style with me",
        "fit check",
        "穿搭",
    ],
    # P6 社群/合作
    "community_collab": [
        "collab",
        "collaboration",
        "partnership",
        "partner",
        "tagged",
        "repost",
        "creator",
        "ambassador",
    ],
}

# KOL 主页描述关键词 → creator_type（顺序优先，先匹配先返回）
_CREATOR_TYPE_KEYWORDS = {
    # sports 不包含 nike/adidas，避免仅因品牌词误判
    "sports": ["sport", "fitness", "athlete", "gym", "running", "training", "workout"],
    "lifestyle": ["lifestyle", "life", "daily", "vlog", "living", "creator", "content"],
    "fashion": ["fashion", "style", "outfit", "ootd", "wear", "streetwear", "sneaker"],
    "beauty": ["beauty", "makeup", "skincare", "cosmetic", "glam"],
}


def _infer_content_type_from_caption(caption: Optional[str]) -> Optional[str]:
    """
    基于 caption_text 关键词推断 content_type（自上而下，Rule-based）。
    用于验证已知：确认品牌是否按既定策略（如多发教学类视频）执行。
    """
    if not caption or not str(caption).strip():
        return None
    text = str(caption).lower()
    for content_type, keywords in _CONTENT_TYPE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return content_type
    return None


def _infer_creator_type_from_signature(signature: Optional[str]) -> str:
    """
    Infer creator_type from KOL 主页描述 (author_signature).
    Returns one of: sports, lifestyle, fashion, beauty, other.
    """
    if not signature or not str(signature).strip():
        return "other"
    text = str(signature).lower()
    for creator_type, keywords in _CREATOR_TYPE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return creator_type
    return "other"


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engagement_count (like+comment+share+save) and engagement_rate.
    Naming: _count for counts, _rate for ratios.
    """
    df = df.copy()
    df = _ensure_numeric(
        df, ["view_count", "like_count", "comment_count", "share_count", "save_count"]
    )
    df["engagement_count"] = (
        df["like_count"].fillna(0)
        + df["comment_count"].fillna(0)
        + df["share_count"].fillna(0)
        + df["save_count"].fillna(0)
    )
    df["engagement_rate"] = df["engagement_count"] / df["view_count"]
    df.loc[df["view_count"].fillna(0) <= 0, "engagement_rate"] = pd.NA
    return df


def build_feature_table(df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    """
    Add all feature columns. Schema columns (video_id, view_count, like_count, etc.)
    are preserved. Feature outputs use correct mapping from schema (see module doc).
    """
    df = df.copy()

    # --- Basic: only add video_url (schema has video_id) ---
    if "video_id" not in df.columns:
        df["video_id"] = df.get("id", pd.Series(dtype=object))
    df["video_url"] = df["video_id"].apply(
        lambda x: f"https://www.tiktok.com/video/{x}" if pd.notna(x) and x else None
    )

    # --- Time: from create_time_ts; only post_date, post_hour, post_weekday (no post_create_time) ---
    if "create_time_ts" in df.columns:
        create_at = pd.to_datetime(df["create_time_ts"], unit="s", utc=True).dt.tz_convert(tz)
        df["post_date"] = create_at.dt.date.astype(str)
        df["post_hour"] = create_at.dt.hour
        df["post_weekday"] = create_at.dt.weekday
    else:
        df["post_date"] = None
        df["post_hour"] = pd.NA
        df["post_weekday"] = pd.NA

    # --- Engagement: view_count only (no play_count); engagement_count from clean or recompute ---
    df = _ensure_numeric(
        df, ["view_count", "like_count", "comment_count", "share_count", "save_count"]
    )
    if "engagement_count" not in df.columns:
        df["engagement_count"] = (
            df["like_count"].fillna(0)
            + df["comment_count"].fillna(0)
            + df["share_count"].fillna(0)
            + df["save_count"].fillna(0)
        )
    if "engagement_rate" not in df.columns:
        df["engagement_rate"] = df["engagement_count"] / df["view_count"].replace(0, pd.NA)
        df.loc[df["view_count"].fillna(0) <= 0, "engagement_rate"] = pd.NA
    views = df["view_count"].replace(0, pd.NA)
    df["like_to_view_rate"] = df["like_count"] / views
    df["comment_to_view_rate"] = df["comment_count"] / views
    df["share_to_view_rate"] = df["share_count"] / views
    df["engagement_to_view_rate"] = df["engagement_count"] / views
    df["engagement_score"] = (
        0.2 * df["view_count"].fillna(0) * df["like_to_view_rate"].fillna(0)
        + 0.5 * df["view_count"].fillna(0) * df["comment_to_view_rate"].fillna(0)
        + 0.3 * df["view_count"].fillna(0) * df["share_to_view_rate"].fillna(0)
    )
    # norm_engagement_score = 该视频互动率 / 该品牌历史平均互动率
    brand_mean_rate = df.groupby("brand", dropna=False)["engagement_to_view_rate"].transform("mean")
    df["norm_engagement_score"] = df["engagement_to_view_rate"] / brand_mean_rate.replace(0, pd.NA)
    likes = df["like_count"].replace(0, pd.NA)
    df["comment_to_like_ratio"] = df["comment_count"] / likes

    # --- Caption: caption_raw->caption_text; hashtags->hashtag_list ---
    caption = df.get("caption_raw", pd.Series(dtype=object)).fillna("")
    df["caption_text"] = caption.astype(str)
    df["caption_length_words"] = caption.str.split().str.len().fillna(0).astype(int)
    hashtags = df.get("hashtags", pd.Series(dtype=object)).apply(
        lambda x: x if isinstance(x, list) else []
    )
    df["hashtag_list"] = hashtags
    df["hashtag_count"] = hashtags.apply(len)
    df["mention_count"] = caption.str.count(r"@\w+").fillna(0).astype(int)
    cap_lower = caption.str.lower()
    df["has_call_to_action"] = cap_lower.apply(
        lambda s: any(kw in (s or "") for kw in _CTA_KEYWORDS)
    )

    # --- Author: creator_type from is_official_brand or author_signature (KOL 主页描述) ---
    official = df.get("is_official_brand", False)
    signature = df.get("author_signature", pd.Series(dtype=object)).fillna("")
    df["creator_type"] = official.map(lambda x: "brand" if x else None)
    df["creator_type"] = df["creator_type"].fillna(signature.apply(_infer_creator_type_from_signature))
    # author_follower_count: from clean (crawler authorStats.followerCount); fallback NA
    if "author_follower_count" not in df.columns:
        df["author_follower_count"] = pd.NA

    # --- Video/music: from clean (crawler video.duration, music.id/idStr); fallback placeholders ---
    if "video_duration_sec" not in df.columns:
        df["video_duration_sec"] = pd.NA
    if "music_id" not in df.columns:
        df["music_id"] = None
    if "has_music" not in df.columns:
        df["has_music"] = False
    df["is_trending_audio"] = False
    df["content_type"] = caption.apply(_infer_content_type_from_caption)
    df["appearance_type"] = None
    if "product_category" not in df.columns:
        df["product_category"] = None

    # --- Commerce: from caption_raw ---
    commerce_keywords = {"shop", "buy", "order", "sale", "link", "product"}
    df["has_product_link"] = cap_lower.apply(
        lambda s: any(kw in (s or "") for kw in commerce_keywords)
    )

    # --- Crawl metadata: _at = datetime, _ts = Unix int ---
    df["crawl_at"] = pd.to_datetime(df.get("crawled_at", pd.NaT), errors="coerce")
    df["crawl_batch_id"] = df.get("crawled_at_ts", pd.NA).astype(str)
    df["raw_payload_path"] = ""

    # --- Embeddings: pipeline interface (input from schema: caption_raw → sentiment; video_id → visual) ---
    df["sentiment_score"] = caption.apply(compute_sentiment_score)
    df["visual_embedding"] = df["video_id"].apply(
        lambda vid: compute_visual_embedding_id(str(vid) if pd.notna(vid) else "", None)
    )

    # --- Analysis / experiment: content_cluster_id from notebook (K-Means 聚类), placeholder here ---
    if "content_cluster_id" not in df.columns:
        df["content_cluster_id"] = pd.NA

    return df


def write_partitioned_parquet(
    df: pd.DataFrame,
    base_path: str | Path,
    partition_cols: Optional[List[str]] = None,
) -> Path:
    """Write feature table as partitioned Parquet (e.g. by brand, post_date)."""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    partition_cols = partition_cols or ["brand", "post_date"]
    out = df.copy()
    for c in partition_cols:
        if c not in out.columns:
            out[c] = "__unknown__"
        else:
            out[c] = out[c].fillna("__unknown__").astype(str)
    out.to_parquet(base_path, partition_cols=partition_cols, index=False)
    return base_path
