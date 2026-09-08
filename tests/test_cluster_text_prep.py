from tiktok_brand.etl.text_prep import (
    build_cluster_embedding_text,
    build_embedding_text,
    hashtag_split_for_cluster,
    scrub_caption_for_cluster,
)


def test_build_embedding_text_unchanged_contract():
    text = build_embedding_text("New Samba #adidas #samba", ["adidas", "samba"])
    assert "[HASHTAGS]" in text
    assert "adidas adidas" in text or "adidas" in text
    assert "samba" in text


def test_cluster_text_no_marker_no_raw_dup_no_brand_traffic():
    text = build_cluster_embedding_text(
        "Love these @nike Air Max with @kylian #fyp #Nike #AirMax #AirMax #samba",
        ["fyp", "Nike", "AirMax", "AirMax", "samba", "viral"],
        caption_clean="love these @nike air max with @kylian",
    )
    assert "[HASHTAGS]" not in text
    assert "nike nike" not in text
    assert "@nike" not in text
    assert "@kylian" in text
    assert "air max" in text  # from caption and/or AirMax split
    assert "samba" in text
    assert "fyp" not in text.split()
    assert "viral" not in text.split()
    # split-only: AirMax → "air max", not "... airmax"
    assert "airmax" not in text.replace("air max", "")


def test_hashtag_split_drops_brand_token_in_camel():
    # NikeFootball → nike football → football
    assert hashtag_split_for_cluster("NikeFootball") == "football"
    assert hashtag_split_for_cluster("fyp") == ""
    assert hashtag_split_for_cluster("Jordan") == "jordan"


def test_scrub_keeps_other_mentions():
    assert scrub_caption_for_cluster("@adidas drop with @creator fyp nike shoes") == (
        "drop with @creator shoes"
    )
