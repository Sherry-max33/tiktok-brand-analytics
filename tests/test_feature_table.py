import pandas as pd
from tiktok_brand.etl.cta_rules import detect_cta_flags
from tiktok_brand.etl.content_type_rules import infer_content_type
from tiktok_brand.etl.feature_table import add_derived_metrics, build_feature_table
from tiktok_brand.etl.text_prep import build_embedding_text, clean_caption


def test_add_derived_metrics():
    df = pd.DataFrame(
        [{"view_count": 100, "like_count": 10, "comment_count": 2, "share_count": 1, "collect_count": 0, "brand": "nike"}]
    )
    out = add_derived_metrics(df)
    assert out.loc[0, "engagement_count"] == 13
    assert abs(out.loc[0, "engagement_rate"] - 0.13) < 1e-9
    assert out.loc[0, "weighted_engagement_count"] == 0.10 * 10 + 0.25 * 2 + 0.30 * 1 + 0.35 * 0
    assert abs(out.loc[0, "weighted_engagement_rate"] - 1.8 / 100) < 1e-9
    assert out.loc[0, "brand_relative_engagement_index"] == 1.0


def test_build_feature_table():
    clean = pd.DataFrame(
        [
            {
                "video_id": "v1",
                "create_time_ts": 1700000000,
                "caption_raw": "Check this out #nike #fashion shop now",
                "hashtags": ["nike", "fashion"],
                "normalized_hashtags": ["nike", "niketech"],
                "view_count": 1000,
                "like_count": 100,
                "comment_count": 20,
                "share_count": 10,
                "collect_count": 5,
                "brand": "nike",
                "source_type": "hashtag",
                "source_query": "nike",
                "crawled_at": "2026-01-01T00:00:00-05:00",
                "crawled_at_ts": 1704067200,
                "is_official_brand": False,
                "author_username": "nike",
                "author_follower_count": 50000,
                "music_id": "m1",
            }
        ]
    )
    out = build_feature_table(clean)
    assert "video_url" in out.columns
    assert out.loc[0, "page_url"] == "https://www.tiktok.com/@nike/video/v1"
    assert "weighted_engagement_count" in out.columns
    assert "brand_relative_engagement_index" in out.columns
    assert "has_cta" in out.columns
    assert "has_purchase_cta" in out.columns
    assert "has_product_link" not in out.columns
    assert "engagement_score" not in out.columns
    assert "caption_clean" in out.columns
    assert "[HASHTAGS]" in out.loc[0, "embedding_text"]
    assert "creator_tier" in out.columns
    assert out.loc[0, "creator_tier"] == "micro"
    assert out.loc[0, "has_purchase_cta"] == True
    assert out.loc[0, "has_cta"] == True
    assert out.loc[0, "brand_styles"] == ["technical"]
    assert out.loc[0, "product_lines"] == ["tech_fleece"]
    assert out.loc[0, "product_categories"] == ["apparel"]


def test_taxonomy_multi_label():
    from tiktok_brand.etl.taxonomy_rules import (
        infer_brand_styles,
        infer_product_categories,
        infer_product_lines,
    )

    tags = ["adidasoriginals", "adidassamba", "adidasstyle"]
    assert infer_brand_styles(tags) == ["lifestyle", "retro"]
    assert infer_product_lines(tags) == ["originals_apparel", "samba"]
    # (1) lines present → only line_to_category (not tag apparel from adidasstyle)
    assert infer_product_categories(
        product_lines=["originals_apparel", "samba"],
        tags=tags,
        caption="",
    ) == ["shoes", "apparel"]

    # (2) no lines → hashtag category map
    assert infer_product_categories(
        product_lines=[],
        tags=["nikeshoes", "nikeoutfit"],
        caption="",
    ) == ["shoes", "apparel"]

    # (3) caption heuristic
    assert infer_product_categories(
        product_lines=[],
        tags=["nike"],
        caption="love this sneaker pair",
    ) == ["shoes"]

    # (4) uncategorized
    assert infer_product_categories(product_lines=[], tags=["nike"], caption="") == [
        "uncategorized"
    ]
    assert infer_brand_styles(["nike"]) == []



def test_detect_cta_flags_promo_not_cta():
    flags = detect_cta_flags("50% off today only")
    assert flags["has_promo_language"] is True
    assert flags["has_cta"] is False


def test_infer_content_type_priority():
    assert infer_content_type("join the challenge #viral") == "social_viral"
    assert infer_content_type("how to style these samba") == "tutorial_utility"


def test_build_embedding_text_dedupes_hashtags_in_caption():
    text = build_embedding_text("New Samba #adidas #samba", ["adidas", "samba"])
    assert "#adidas" not in text
    assert "[HASHTAGS]" in text
    assert "samba" in text


def test_clean_caption():
    assert clean_caption("  Hello   #nike  https://x.com/a  ") == "hello"
