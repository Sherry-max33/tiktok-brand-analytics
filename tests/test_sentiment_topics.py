"""Theme 4 sentiment / topic aggregation + lexicon classification."""

from __future__ import annotations

import pandas as pd

from tiktok_brand.analysis.sentiment_topics import (
    add_polarity_bin,
    apply_lexicon_topics,
    brand_perception_summary,
    brand_style_sentiment,
    classify_comment_text,
    content_type_sentiment,
    content_types_both_brands,
    topic_sentiment_by_brand,
    topic_sentiment_gap,
    topic_volume_by_brand,
)


def test_topic_lexicon_examples():
    from tiktok_brand.analysis.sentiment_topics import topic_lexicon_examples

    lex = topic_lexicon_examples(layers=["substantive"], max_examples=5)
    assert len(lex) == 9
    assert {"topic", "example_signals"}.issubset(lex.columns)
    assert lex["example_signals"].str.len().min() > 0
    row = lex.loc[lex["topic"] == "Product / Model Discussion"].iloc[0]
    assert "samba" in row["example_signals"] or "jordan" in row["example_signals"] or "air force" in row["example_signals"]
    df = pd.DataFrame({"sentiment_score": [-0.5, 0.0, 0.5, None]})
    out = add_polarity_bin(df)
    assert list(out["sentiment_polarity"].astype(str)) == [
        "negative",
        "neutral",
        "positive",
        "nan",
    ]


def test_brand_and_content_sentiment():
    comments = pd.DataFrame(
        {
            "brand": ["nike", "nike", "adidas", "adidas"],
            "video_id": ["v1", "v1", "v2", "v2"],
            "sentiment_score": [0.6, -0.2, 0.1, 0.0],
        }
    )
    summary = brand_perception_summary(comments)
    assert set(summary["brand"]) == {"nike", "adidas"}
    assert int(summary.loc[summary["brand"] == "nike", "n_comments"].iloc[0]) == 2

    videos = pd.DataFrame(
        {
            "video_id": ["v1", "v2"],
            "brand": ["nike", "adidas"],
            "content_type": [["product_review"], ["vibe_ootd", "collaboration"]],
        }
    )
    tab = content_type_sentiment(comments, videos, min_count=1)
    assert "product_review" in set(tab["level"])
    assert "vibe_ootd" in set(tab["level"])
    both = content_types_both_brands(tab)
    assert both == []  # each type only one brand in this toy set

    videos2 = videos.copy()
    videos2["brand_styles"] = [["lifestyle"], ["lifestyle", "performance"]]
    styles = brand_style_sentiment(comments, videos2, min_count=1)
    assert "lifestyle" in set(styles["level"])

    videos3 = videos.copy()
    videos3["content_cluster_id_clean_k12"] = [0, 1]
    from tiktok_brand.analysis.sentiment_topics import content_cluster_sentiment

    clusters = content_cluster_sentiment(comments, videos3, min_count=1)
    assert set(clusters["level"]) >= {0, 1}


def test_classify_comment_text_priority():
    primary, matched = classify_comment_text("these sambas look fire but too expensive")
    assert primary == "price_value"
    assert "design_aesthetics" in matched or "product_model" in matched

    primary2, _ = classify_comment_text("lamine yamal is insane")
    assert primary2 == "athlete_celebrity"

    primary3, _ = classify_comment_text("🔥🔥🔥")
    assert primary3 == "generic_product_reaction"

    primary4, _ = classify_comment_text("[stickers]")
    assert primary4 == "meme_chatter"

    primary5, _ = classify_comment_text("adidas is better than nike")
    assert primary5 == "brand_comparison"

    # commercial attribute > product entity
    assert classify_comment_text("nike tech is so ugly")[0] == "design_aesthetics"
    assert classify_comment_text("sambas are super comfy")[0] == "fit_comfort"
    assert classify_comment_text("jordan didn’t play football")[0] == "athlete_celebrity"

    # product-adjacent slang is product_reaction, not Design
    assert classify_comment_text("these are fire 🔥")[0] == "generic_product_reaction"
    assert classify_comment_text("hard")[0] == "generic_product_reaction"
    assert classify_comment_text("where can i find them")[0] == "availability"
    assert classify_comment_text("name?")[0] == "product_model"


def test_lexicon_topics_and_volume_layers():
    comments = pd.DataFrame(
        {
            "brand": ["nike", "adidas", "nike", "adidas", "nike"],
            "comment_text": [
                "need these so bad",
                "beautiful colorway",
                "lol bro",
                "sold out already",
                "xyz qqq",
            ],
            "sentiment_score": [0.4, 0.5, 0.0, -0.1, None],
        }
    )
    labeled = apply_lexicon_topics(comments)
    assert "comment_topic" in labeled.columns
    assert labeled.loc[0, "comment_topic"] == "product_desire"
    assert labeled.loc[2, "topic_layer"] == "social"

    vol_all = topic_volume_by_brand(labeled, min_count=1)
    assert vol_all["n"].sum() == 5
    vol_sub = topic_volume_by_brand(labeled, min_count=1, layers=["substantive"])
    assert "Generic Positive Reaction" not in set(vol_sub["topic"].astype(str))
    assert "Meme / Social Chatter" not in set(vol_sub["topic"].astype(str))

    sent = topic_sentiment_by_brand(labeled, min_count=1, layers=["substantive"])
    assert int(sent["n"].sum()) == 3  # scored substantive only (excl None score + meme)
    gap = topic_sentiment_gap(sent)
    assert "gap" in gap.columns
