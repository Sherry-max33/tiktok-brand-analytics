import pandas as pd
from tiktok_brand.etl.cta_rules import detect_cta_flags
from tiktok_brand.etl.content_type_rules import infer_content_types
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
    out = build_feature_table(
        clean,
        encode_batch_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
    )
    assert "video_url" in out.columns
    assert out.loc[0, "page_url"] == "https://www.tiktok.com/@nike/video/v1"
    assert "weighted_engagement_count" in out.columns
    assert "brand_relative_engagement_index" in out.columns
    assert "has_cta" in out.columns
    assert "has_purchase_cta" in out.columns
    assert "has_giveaway" in out.columns
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
    assert "sentiment_score" in out.columns
    assert "caption_lang" in out.columns
    assert "sentiment_method" in out.columns
    assert out.loc[0, "sentiment_method"] in {
        "vader_en",
        "not_scored",
        "not_scored_non_en",
        "vader_via_mt",
    }
    assert out.loc[0, "embedding_method"] == "sbert_multilingual"
    assert out.loc[0, "text_embedding"] == [0.1, 0.2, 0.3]
    assert out.loc[0, "visual_embedding_method"] == "not_embedded"
    assert out.loc[0, "visual_format"] == "unknown"
    assert out.loc[0, "visual_classification_status"] == "unknown"
    assert "appearance_type" not in out.columns


def test_taxonomy_multi_label():
    from tiktok_brand.etl.taxonomy_rules import (
        brand_styles_to_flags,
        infer_brand_styles,
        infer_product_categories,
        infer_product_lines,
    )

    tags = ["adidasoriginals", "adidassamba", "adidasstyle"]
    # Seed-hashtag priors only (samba is a product line, not a style)
    assert infer_brand_styles(tags) == ["lifestyle"]
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

    # Seed-hashtag prior ∪ keyword evidence (multi-label); no SKU→style
    assert infer_brand_styles(
        ["nikerunning"], caption="new foam cushioning technology"
    ) == ["performance", "technical"]
    assert infer_brand_styles(
        ["adidassamba"], caption="retro samba with streetwear ootd fit check"
    ) == ["lifestyle", "retro"]
    assert infer_brand_styles(["adidassamba"], caption="just posted") == []
    flags = brand_styles_to_flags(["performance", "technical"])
    assert flags == {
        "brand_style_performance": True,
        "brand_style_technical": True,
        "brand_style_lifestyle": False,
        "brand_style_retro": False,
    }
    # Alias / new line tags from unmapped-hashtag audit
    assert infer_product_lines(["jordanbrand", "sambas", "dunks", "nikeairmax"]) == [
        "air_max",
        "dunk",
        "jordan",
        "samba",
    ]
    assert infer_product_lines(["f50", "predator", "adidassuperstar", "adizero"]) == [
        "adizero",
        "f50",
        "predator",
        "superstar",
    ]
    assert infer_product_categories(
        product_lines=["f50", "superstar"], tags=[], caption=""
    ) == ["shoes"]
    # Generic footwear tags (no product_line) → category map layer
    assert infer_product_categories(
        product_lines=[],
        tags=["sneakers", "sneakerhead", "cleats"],
        caption="",
    ) == ["shoes"]

    # Weak fashion alone ≠ apparel; pair with product term / shoes tag
    assert infer_product_categories(
        product_lines=[], tags=["ootd", "fashion", "streetwear"], caption=""
    ) == ["uncategorized"]
    assert infer_product_categories(
        product_lines=[], tags=["ootd", "jacket"], caption=""
    ) == ["apparel"]
    assert infer_product_categories(
        product_lines=[], tags=["ootd", "sneakers"], caption=""
    ) == ["shoes"]
    assert infer_product_categories(
        product_lines=[], tags=["nikeoutfit"], caption=""
    ) == ["apparel"]
    # Bare caption "outfit" / "fit" no longer forces apparel
    assert infer_product_categories(
        product_lines=[], tags=["nike"], caption="today's outfit look"
    ) == ["uncategorized"]
    assert infer_product_categories(
        product_lines=[], tags=["nike"], caption="love this jacket"
    ) == ["apparel"]


def test_detect_cta_flags_promo_not_cta():
    flags = detect_cta_flags("50% off today only")
    assert flags["has_promo_language"] is True
    assert flags["has_cta"] is False


def test_infer_content_types_multilabel():
    assert infer_content_types("how to style these samba") == ["tutorial_utility"]
    assert infer_content_types("honest review of these runners first impression") == [
        "product_review"
    ]
    assert infer_content_types("shopping haul unboxing what i bought") == [
        "product_showcase"
    ]
    assert infer_content_types("shop now back in stock link in bio") == ["product_promo"]
    assert infer_content_types("ootd fit check outfit inspo") == ["vibe_ootd"]
    assert infer_content_types("teamed up with adidas in collaboration with") == [
        "collaboration"
    ]
    assert infer_content_types("giving back charity fundraiser community event") == [
        "community_impact"
    ]
    # bare BTS alone should not force story_heritage
    assert infer_content_types("bts on set today") == []
    assert infer_content_types("bts workshop craftsmanship how it's made") == [
        "story_heritage"
    ]
    # viral/challenge alone is no longer a type
    assert infer_content_types("join the challenge #viral") == []
    # multi-label: review + ootd can co-occur
    both = infer_content_types(
        "honest review first impression and ootd fit check outfit inspo"
    )
    assert "product_review" in both
    assert "vibe_ootd" in both
    assert both.index("product_review") < both.index("vibe_ootd")
    # hashtag-compact forms from audit
    assert "collaboration" in infer_content_types("essentials #createdwithadidas @adidas")
    assert "product_review" in infer_content_types("hyperboost #runningshoereviews")
    assert "product_showcase" in infer_content_types("gazelles #thrifthaul #shoppingvlog")
    # LLM round: conversion / experience / collab-x / non-how-to tutorial
    assert infer_content_types("on sale now available grab yours") == ["product_promo"]
    assert "product_review" in infer_content_types("fit true to size and worth it")
    assert "collaboration" in infer_content_types("adidas x diesel capsule drop")
    assert "collaboration" not in infer_content_types("shoutout @jamesharden only")
    assert "collaboration" not in infer_content_types("box is 10 x 10 cm")
    assert "tutorial_utility" in infer_content_types("master the fundamentals watch & learn")
    assert "story_heritage" in infer_content_types("pays homage to sneaker history")
    assert "official_campaign" in infer_content_types("nike presents the new chapter")


def test_build_embedding_text_dedupes_hashtags_in_caption():
    text = build_embedding_text("New Samba #adidas #samba", ["adidas", "samba"])
    assert "#adidas" not in text
    assert "[HASHTAGS]" in text
    assert "samba" in text


def test_clean_caption():
    assert clean_caption("  Hello   #nike  https://x.com/a  ") == "hello"
