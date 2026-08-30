from tiktok_brand.crawl.apify_video_mapper import (
    apify_video_item_to_record,
    is_apify_video_item,
)
from tiktok_brand.crawl.video_schemas import VideoRecord


SAMPLE_APIFY_VIDEO_ITEM = {
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


def test_is_apify_video_item():
    assert is_apify_video_item(SAMPLE_APIFY_VIDEO_ITEM) is True
    assert is_apify_video_item({"id": "1", "desc": "x", "author": {}}) is False


def test_apify_video_item_to_record():
    rec = apify_video_item_to_record(
        SAMPLE_APIFY_VIDEO_ITEM,
        source_type="hashtag",
        source_query="nike",
        brand="nike",
        tz_name="America/New_York",
    )
    d = rec.to_dict()
    assert d["video_id"] == "7534061113365859586"
    assert d["caption_raw"] == "Nike outfit #nike #fashion"
    assert d["hashtags"] == ["nike", "fashion"]
    assert d["collect_count"] == 291
    assert d["author_id"] == "6733984297591636998"
    assert isinstance(rec, VideoRecord)
