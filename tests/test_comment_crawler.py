from tiktok_brand.crawl.apify_comment_mapper import apify_comment_item_to_record
from tiktok_brand.crawl.comment_crawler import build_tiktok_video_url, video_id_from_url


def test_build_tiktok_video_url_from_id():
    assert build_tiktok_video_url("7391028884474547489") == (
        "https://www.tiktok.com/video/7391028884474547489"
    )


def test_video_id_from_url():
    url = "https://www.tiktok.com/@nike/video/7391028884474547489"
    assert video_id_from_url(url) == "7391028884474547489"


def test_apify_comment_item_to_record_from_apify_shape():
    rec = apify_comment_item_to_record(
        {
            "cid": "7527391834584711938",
            "text": "Great shoes",
            "diggCount": 42,
            "replyCommentTotal": 3,
            "createTimeISO": "2025-07-16T03:24:56.000Z",
            "uid": "7099133983321998342",
            "uniqueId": "hubertoloo",
            "nickName": "Huberto",
            "videoWebUrl": "https://www.tiktok.com/@nike/video/7391028884474547489",
            "region": "PL",
        },
        tz_name="America/New_York",
    )
    d = rec.to_dict()
    assert d["video_id"] == "7391028884474547489"
    assert d["comment_text"] == "Great shoes"
    assert d["comment_like_count"] == 42
    assert d["comment_reply_count"] == 3
    assert d["comment_author_id"] == "7099133983321998342"
