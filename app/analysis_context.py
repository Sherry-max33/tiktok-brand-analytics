"""Evidence layer for the AI Content Analyst (Phase 1 of docs/07-ai-layer.md).

build_analysis_context() packages what the app already knows about a video, plus the similar
high performers that passed relevance validation (relevance.py), into one JSON-serializable
object. It is the only input the Analyst and Brief generators will see. Nothing here is
generated, inferred or repaired:

- None  = the value is not available for this video (not collected / not computed).
- []    = the rule-based classifier ran and detected nothing.

Rejected retrieval candidates are not included; only their count is.
"""

from __future__ import annotations

import math

import pandas as pd

import analysis_data
import catalog
import relevance

SCHEMA_VERSION = "analysis_context/v1"

CONVENTIONS = {
    "null": "Not available for this video; do not infer it.",
    "empty_list": "The rule-based classifier ran on the caption/hashtags and detected nothing.",
    "caption_features": "content_type, brand_style, social_mechanic and cta_signals come from "
    "the caption and hashtags, not from the video frames.",
    "frame_features": "visual_format and visual_setting come from sampled video frames "
    "(CLIP zero-shot, one label each). A label is given only when the classifier clearly "
    "preferred it; otherwise it is null (unknown), like a label that fit no defined category.",
    "performance": "Engagement counts are a snapshot from the crawl date. top_percent is the "
    "WER percentile across the library (1 = best). WER = (0.10*likes + 0.25*comments + "
    "0.30*shares + 0.35*saves) / views. BRI = WER / the brand's median WER.",
    "product": "Product lines come from a hashtag taxonomy. 'unresolved' means the taxonomy "
    "matched only a broad or conflicting tag, so no specific line is asserted.",
    "similarity": "Retrieval scores in [0, 1] (text = caption/hashtag embedding cosine, "
    "visual = frame embedding cosine, strategy_overlap = smoothed overlap of caption-derived "
    "labels). null = not computable because one side lacks that signal. High scores alone do "
    "not establish substantive relevance. shared_labels = labels both videos carry.",
    "relevance": "similar_high_performers lists only retrieved candidates judged substantively "
    "relevant (aligned on at least two dimensions, including subject_product or "
    "narrative_intent). retrieval_assessment.cross_content_synthesis_allowed is false when "
    "fewer than two relevant comparables exist; cross-content patterns must not be stated then.",
}


def _num(value, digits: int | None = None):
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    value = float(value)
    return round(value, digits) if digits is not None else value


def _int(value):
    number = _num(value)
    return int(number) if number is not None else None


def _row(video_id: str) -> pd.Series | None:
    df = catalog.load_library()
    match = df[df["video_id"] == str(video_id)]
    return None if match.empty else match.iloc[0]


def _creator(row: pd.Series) -> dict:
    return {
        "username": row["author_username"] or None,
        "account_type": "official_brand_account" if row["is_official_brand"] else "creator",
        "creator_tier": row["creator_tier"] or None,
        "followers": _int(row["author_follower_count"]),
    }


def _product(row: pd.Series) -> dict:
    lines = [catalog.product_label(p) for p in row["display_lines"]]
    if row["product_unresolved"]:
        status = "unresolved"
    elif lines:
        status = "resolved"
    else:
        status = "not_identified"
    return {
        "status": status,
        "product_lines": lines,
        "categories": [
            catalog.CATEGORY_LABELS[c] for c in row["display_categories"] if c in catalog.CATEGORY_LABELS
        ],
    }


def _performance(row: pd.Series, *, full: bool = True) -> dict:
    out = {
        "top_percent": _int(row["top_pct"]),
        "wer": _num(row["wer"], 5),
        "bri": _num(row["bri"], 3),
        "views": _int(row["view_count"]),
    }
    if full:
        out.update(
            likes=_int(row["like_count"]),
            comments=_int(row["comment_count"]),
            shares=_int(row["share_count"]),
            saves=_int(row["collect_count"]),
            collected_on=row["crawl_at"].date().isoformat() if pd.notna(row["crawl_at"]) else None,
        )
    return out


def _content_profile(row: pd.Series) -> dict:
    return {
        "frames": {
            "visual_format": analysis_data.VISUAL_FORMAT_LABELS.get(row["visual_format"])
            if row["visual_format"]
            else None,
            "visual_setting": analysis_data.VISUAL_SETTING_LABELS.get(row["visual_setting"])
            if row["visual_setting"]
            else None,
        },
        "caption_and_hashtags": {
            "content_type": [catalog.content_type_label(c) for c in row["content_type"]],
            "brand_style": analysis_data._labels(
                row["brand_styles"], analysis_data.BRAND_STYLE_LABELS
            ),
            "social_mechanic": analysis_data._labels(
                row["social_mechanic"], analysis_data.SOCIAL_MECHANIC_LABELS
            ),
            "cta_signals": [
                label for col, label in analysis_data.CTA_LABELS.items() if row[col]
            ],
        },
    }


def _audience(video_id: str) -> dict | None:
    sentiment = analysis_data.audience_sentiment(video_id)
    if not sentiment:
        return None
    return {
        "comments_analyzed": sentiment["comments"],
        "positive_pct": round(sentiment["positive"] * 100, 1),
        "negative_pct": round(sentiment["negative"] * 100, 1),
        "net_sentiment": round((sentiment["positive"] - sentiment["negative"]) * 100),
        "method": "VADER on collected comments; non-English comments machine-translated first",
    }


def _caption(row: pd.Series) -> dict:
    return {
        "text": catalog.display_caption(row["caption_raw"]) or None,
        "language": row["caption_lang"] or None,
        "english_translation": row["caption_en"]
        if row["translation_status"] == "done"
        else None,
    }


def _comparable(card: dict) -> dict:
    row = _row(card["video_id"])
    return {
        "video_id": card["video_id"],
        "brand": catalog.brand_label(row["brand"]),
        "creator": _creator(row),
        "caption": _caption(row),
        "hashtags": list(row["hashtags"]),
        "product": _product(row),
        "duration_sec": _num(row["video_duration_sec"], 1),
        "performance": _performance(row, full=False),
        "content_profile": _content_profile(row),
        "audience_signals": _audience(card["video_id"]),
        "similarity": {**card["similarity"], "shared_labels": card["shared_labels"]},
    }


def build_analysis_context(video_id: str) -> dict | None:
    row = _row(video_id)
    if row is None:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "conventions": CONVENTIONS,
        "selected_video": {
            "video_id": row["video_id"],
            "brand": catalog.brand_label(row["brand"]),
            "creator": _creator(row),
            "caption": _caption(row),
            "hashtags": list(row["hashtags"]),
            "product": _product(row),
            "duration_sec": _num(row["video_duration_sec"], 1),
            "posted_on": row["post_date"].date().isoformat() if pd.notna(row["post_date"]) else None,
        },
        "performance": _performance(row),
        "content_profile": _content_profile(row),
        "audience_signals": _audience(row["video_id"]),
        **_validated_comparables(row["video_id"]),
    }


def _validated_comparables(video_id: str) -> dict:
    assessment = relevance.assess(video_id)
    judged = {c["video_id"]: c for c in assessment["candidates"]}
    comparables = []
    for card in relevance.similar_high_performers(video_id):
        item = _comparable(card)
        item["relevance"] = {
            "aligned_dimensions": judged[card["video_id"]]["aligned_dimensions"],
            "rationale": judged[card["video_id"]]["rationale"],
        }
        comparables.append(item)
    return {
        "similar_high_performers": comparables,
        "retrieval_assessment": assessment["retrieval_assessment"],
    }
