from tiktok_brand.crawl.apify_mapper import (
    apify_item_to_video_record,
    is_apify_item,
    normalize_duration_sec,
)


SAMPLE_APIFY_ITEM = {
    "id": "7534061113365859586",
    "text": "Nike outfit #nike #fashion",
    "createTime": 1754157903,
    "authorMeta": {
        "id": "6733984297591636998",
        "name": "nike",
        "verified": True,
        "signature": "Just Do It.",
        "fans": 7600000,
    },
    "videoMeta": {"duration": 16},
    "musicMeta": {"musicId": "7529403355681147665"},
    "playCount": 55700,
    "diggCount": 5344,
    "commentCount": 24,
    "shareCount": 701,
    "collectCount": 291,
    "hashtags": [{"name": "nike"}, {"name": "fashion"}],
    "webVideoUrl": "https://www.tiktok.com/@nike/video/7534061113365859586",
}


def test_is_apify_item():
    assert is_apify_item(SAMPLE_APIFY_ITEM) is True
    assert is_apify_item({"id": "1", "desc": "x", "author": {}}) is False


def test_normalize_duration_sec_keeps_seconds():
    assert normalize_duration_sec(16) == 16.0
    assert normalize_duration_sec(None) is None


def test_apify_item_to_video_record():
    rec = apify_item_to_video_record(
        SAMPLE_APIFY_ITEM,
        source_type="hashtag",
        source_query="nike",
        brand="nike",
        tz_name="America/New_York",
    )
    d = rec.to_dict()
    assert d["video_id"] == "7534061113365859586"
    assert d["caption_raw"] == "Nike outfit #nike #fashion"
    assert d["hashtags"] == ["nike", "fashion"]
    assert d["author_username"] == "nike"
    assert d["author_follower_count"] == 7600000
    assert d["author_verified"] is True
    assert d["video_duration_sec"] == 16.0
    assert d["view_count"] == 55700
    assert d["like_count"] == 5344
    assert d["save_count"] == 291
    assert d["music_id"] == "7529403355681147665"
    assert d["has_music"] is True
    assert d["source_type"] == "hashtag"
    assert d["source_query"] == "nike"
    assert d["brand"] == "nike"
    assert d["raw_payload"]["webVideoUrl"].endswith("7534061113365859586")
