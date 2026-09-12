"""
Feature table: analysis/modeling features derived from the clean table.

Engagement metrics:
- engagement_count: like + comment + share + collect
- weighted_engagement_count: 0.10*like + 0.25*comment + 0.30*share + 0.35*collect
- weighted_engagement_rate: weighted_engagement_count / view_count
- brand_relative_engagement_index: weighted_engagement_rate / brand median

Taxonomy (configs/taxonomy.yaml) — feature-layer multi-label; compute order:
- brand_styles → product_lines → product_categories (cascade; see yaml)

CTA:
- has_purchase_cta / has_engagement_cta / has_discovery_traffic_cta / has_promo_language
- has_cta = purchase OR engagement OR discovery (promo kept separate)

content_type: multi-label list from configs/feature_rules.yaml
  (all matching categories; order is stable, not exclusive)
social_mechanic: multi-label TikTok propagation/format cues (parallel axis)

Sentiment:
- detect caption_lang + confidence; high-conf English → vader_en
- other languages → MT to English then vader_via_mt (configurable; keep methods separate)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..embeddings.caption_embed import compute_text_embeddings
from ..embeddings.visual_embed import compute_visual_features
from .content_type_rules import infer_content_types
from .social_mechanic_rules import infer_social_mechanics
from .cta_rules import detect_cta_flags
from .rule_config import get_engagement_weights, load_feature_rules
from .sentiment_rules import score_caption_sentiment
from .taxonomy_rules import (
    infer_brand_styles,
    infer_product_categories,
    infer_product_lines,
)
from .text_prep import build_embedding_text, clean_caption

_CREATOR_TYPE_KEYWORDS = {
    "sports": ["sport", "fitness", "athlete", "gym", "running", "training", "workout"],
    "lifestyle": ["lifestyle", "life", "daily", "vlog", "living", "creator", "content"],
    "fashion": ["fashion", "style", "outfit", "ootd", "wear", "streetwear", "sneaker"],
    "beauty": ["beauty", "makeup", "skincare", "cosmetic", "glam"],
}


def _infer_creator_type_from_signature(signature: Optional[str]) -> str:
    if not signature or not str(signature).strip():
        return "other"
    text = str(signature).lower()
    for creator_type, keywords in _CREATOR_TYPE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return creator_type
    return "other"


def _infer_creator_tier(followers: Optional[float], rules_path: str = "configs/feature_rules.yaml") -> Optional[str]:
    if followers is None or pd.isna(followers):
        return None
    thresholds = (load_feature_rules(rules_path).get("creator_tier_thresholds") or {})
    f = float(followers)
    if f < float(thresholds.get("nano", 10_000)):
        return "nano"
    if f < float(thresholds.get("micro", 100_000)):
        return "micro"
    if f < float(thresholds.get("mid", 1_000_000)):
        return "mid"
    if f < float(thresholds.get("macro", 10_000_000)):
        return "macro"
    return "mega"


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add engagement_count, engagement_rate, and weighted engagement metrics."""
    df = df.copy()
    df = _ensure_numeric(
        df, ["view_count", "like_count", "comment_count", "share_count", "collect_count"]
    )
    w_like, w_comment, w_share, w_collect = get_engagement_weights()

    df["engagement_count"] = (
        df["like_count"].fillna(0)
        + df["comment_count"].fillna(0)
        + df["share_count"].fillna(0)
        + df["collect_count"].fillna(0)
    )
    df["engagement_rate"] = df["engagement_count"] / df["view_count"]
    df.loc[df["view_count"].fillna(0) <= 0, "engagement_rate"] = pd.NA

    df["weighted_engagement_count"] = (
        w_like * df["like_count"].fillna(0)
        + w_comment * df["comment_count"].fillna(0)
        + w_share * df["share_count"].fillna(0)
        + w_collect * df["collect_count"].fillna(0)
    )
    views = df["view_count"].replace(0, pd.NA)
    df["weighted_engagement_rate"] = df["weighted_engagement_count"] / views

    brand_typical = df.groupby("brand", dropna=False)["weighted_engagement_rate"].transform(
        "median"
    )
    df["brand_relative_engagement_index"] = df["weighted_engagement_rate"] / brand_typical.replace(
        0, pd.NA
    )
    return df


def _mark_sample_trending_audio(df: pd.DataFrame) -> pd.Series:
    cfg = load_feature_rules().get("trending_audio") or {}
    top_pct = float(cfg.get("top_percentile", 0.9))
    min_count = int(cfg.get("min_video_count", 3))

    if "music_id" not in df.columns:
        return pd.Series(False, index=df.index)

    counts = df["music_id"].dropna().astype(str).value_counts()
    eligible = counts[counts >= min_count]
    if eligible.empty:
        return pd.Series(False, index=df.index)

    threshold = eligible.quantile(top_pct)
    trending_ids = set(eligible[eligible >= threshold].index)
    return df["music_id"].fillna("").astype(str).isin(trending_ids)


def build_feature_table(
    df: pd.DataFrame,
    tz: str = "America/New_York",
    taxonomy_cfg_path: str = "configs/taxonomy.yaml",
    feature_rules_path: str = "configs/feature_rules.yaml",
    encode_batch_fn=None,
) -> pd.DataFrame:
    df = df.copy()
    tax_path = str(taxonomy_cfg_path)

    if "video_id" not in df.columns:
        df["video_id"] = df.get("id", pd.Series(dtype=object))

    df["video_url"] = df["video_id"].apply(
        lambda x: f"https://www.tiktok.com/video/{x}" if pd.notna(x) and x else None
    )
    if "author_username" in df.columns:
        df["page_url"] = df.apply(
            lambda r: (
                f"https://www.tiktok.com/@{str(r['author_username']).lstrip('@')}/video/{r['video_id']}"
                if pd.notna(r.get("video_id")) and r.get("author_username")
                else None
            ),
            axis=1,
        )
    else:
        df["page_url"] = None

    if "create_time_ts" in df.columns:
        create_at = pd.to_datetime(df["create_time_ts"], unit="s", utc=True).dt.tz_convert(tz)
        df["post_date"] = create_at.dt.date.astype(str)
        df["post_hour"] = create_at.dt.hour
        df["post_weekday"] = create_at.dt.weekday
    else:
        df["post_date"] = None
        df["post_hour"] = pd.NA
        df["post_weekday"] = pd.NA

    df = _ensure_numeric(
        df, ["view_count", "like_count", "comment_count", "share_count", "collect_count"]
    )
    # Always (re)compute engagement features in feature layer
    df = add_derived_metrics(df)
    views = df["view_count"].replace(0, pd.NA)
    df["like_to_view_rate"] = df["like_count"] / views
    df["comment_to_view_rate"] = df["comment_count"] / views
    df["share_to_view_rate"] = df["share_count"] / views
    df["engagement_to_view_rate"] = df["engagement_count"] / views
    likes = df["like_count"].replace(0, pd.NA)
    df["comment_to_like_ratio"] = df["comment_count"] / likes

    caption = df.get("caption_raw", pd.Series(dtype=object)).fillna("")
    hashtags = df.get("hashtags", pd.Series(dtype=object)).apply(
        lambda x: x if isinstance(x, list) else []
    )
    df["caption_text"] = caption.astype(str)
    df["caption_clean"] = caption.apply(clean_caption)
    df["embedding_text"] = [
        build_embedding_text(cap, tags) for cap, tags in zip(caption.tolist(), hashtags.tolist())
    ]
    df["caption_length_words"] = df["caption_clean"].str.split().str.len().fillna(0).astype(int)
    df["hashtag_list"] = hashtags
    df["hashtag_count"] = hashtags.apply(len)
    df["mention_count"] = caption.str.count(r"@\w+").fillna(0).astype(int)

    cta_df = pd.DataFrame(caption.apply(detect_cta_flags).tolist(), index=df.index)
    for col in [
        "has_purchase_cta",
        "has_engagement_cta",
        "has_discovery_traffic_cta",
        "has_promo_language",
        "has_cta",
    ]:
        df[col] = cta_df[col]

    official = df.get("is_official_brand", False)
    signature = df.get("author_signature", pd.Series(dtype=object)).fillna("")
    df["creator_type"] = official.map(lambda x: "brand" if x else None)
    df["creator_type"] = df["creator_type"].fillna(signature.apply(_infer_creator_type_from_signature))
    if "author_follower_count" not in df.columns:
        df["author_follower_count"] = pd.NA
    df["creator_tier"] = df["author_follower_count"].apply(_infer_creator_tier)

    if "video_duration_sec" not in df.columns:
        df["video_duration_sec"] = pd.NA
    if "music_id" not in df.columns:
        df["music_id"] = None
    if "has_music" not in df.columns:
        df["has_music"] = False
    df["is_sample_trending_audio"] = _mark_sample_trending_audio(df)
    df["content_type"] = caption.apply(infer_content_types)
    df["social_mechanic"] = caption.apply(infer_social_mechanics)

    # Taxonomy: brand_styles → product_lines → product_categories (cascade in yaml)
    tags_series = df.get("normalized_hashtags", hashtags)
    df["brand_styles"] = [
        infer_brand_styles(
            tags if isinstance(tags, list) else [],
            tax_path,
            caption=cap,
        )
        for tags, cap in zip(tags_series.tolist(), caption.tolist())
    ]
    df["product_lines"] = tags_series.apply(lambda tags: infer_product_lines(tags, tax_path))
    df["product_categories"] = [
        infer_product_categories(
            product_lines=lines if isinstance(lines, list) else [],
            tags=tags if isinstance(tags, list) else [],
            caption=cap,
            path=tax_path,
        )
        for lines, tags, cap in zip(
            df["product_lines"].tolist(),
            tags_series.tolist(),
            caption.tolist(),
        )
    ]

    df["crawl_at"] = pd.to_datetime(df.get("crawled_at", pd.NaT), errors="coerce")
    df["crawl_batch_id"] = df.get("crawled_at_ts", pd.NA).astype(str)
    df["raw_payload_path"] = ""

    sentiment_rows = [
        score_caption_sentiment(
            cap,
            tags if isinstance(tags, list) else [],
            rules_path=feature_rules_path,
        )
        for cap, tags in zip(caption.tolist(), hashtags.tolist())
    ]
    sent_df = pd.DataFrame(sentiment_rows, index=df.index)
    for col in [
        "caption_original",
        "caption_lang",
        "caption_lang_confidence",
        "has_mixed_language",
        "is_emoji_only",
        "caption_en",
        "translation_status",
        "sentiment_score",
        "sentiment_method",
    ]:
        df[col] = sent_df[col]

    emb_rows = compute_text_embeddings(
        df["embedding_text"].tolist(),
        rules_path=feature_rules_path,
        encode_batch_fn=encode_batch_fn,
    )
    emb_df = pd.DataFrame(emb_rows, index=df.index)
    df["text_embedding"] = emb_df["text_embedding"]
    df["embedding_method"] = emb_df["embedding_method"]
    df["embedding_model"] = emb_df["embedding_model"]

    # Shared frame resolve: URL → download → frames → CLIP + zero-shot axes
    cover_urls = df["cover_url"].tolist() if "cover_url" in df.columns else [None] * len(df)
    cover_paths = df["cover_path"].tolist() if "cover_path" in df.columns else [None] * len(df)
    video_paths = df["video_path"].tolist() if "video_path" in df.columns else [None] * len(df)
    video_urls = df["video_url"].tolist() if "video_url" in df.columns else [None] * len(df)
    page_urls = df["page_url"].tolist() if "page_url" in df.columns else [None] * len(df)
    author_usernames = (
        df["author_username"].tolist() if "author_username" in df.columns else [None] * len(df)
    )
    vis_rows = compute_visual_features(
        df["video_id"].tolist() if "video_id" in df.columns else [None] * len(df),
        video_urls=video_urls,
        page_urls=page_urls,
        author_usernames=author_usernames,
        cover_urls=cover_urls,
        cover_paths=cover_paths,
        video_paths=video_paths,
        rules_path=feature_rules_path,
    )
    vis_df = pd.DataFrame(vis_rows, index=df.index)
    df["visual_embedding"] = vis_df["visual_embedding"]
    df["visual_embedding_method"] = vis_df["visual_embedding_method"]
    df["visual_embedding_model"] = vis_df["visual_embedding_model"]
    df["frame_paths"] = vis_df["frame_paths"]
    from tiktok_brand.embeddings.visual_classify import classification_column_names

    for col in classification_column_names(feature_rules_path):
        if col in vis_df.columns:
            df[col] = vis_df[col]
    # Drop deprecated detector-based appearance_type if present from older builds
    if "appearance_type" in df.columns:
        df = df.drop(columns=["appearance_type"])

    if "content_cluster_id" not in df.columns:
        df["content_cluster_id"] = pd.NA

    return df


def write_partitioned_parquet(
    df: pd.DataFrame,
    base_path: str | Path,
    partition_cols: Optional[List[str]] = None,
) -> Path:
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
