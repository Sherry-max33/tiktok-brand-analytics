from tiktok_brand.crawl.apify_comment_mapper import apify_comment_item_to_record
from tiktok_brand.crawl.comment_schemas import CommentRecord


def test_apify_comment_item_to_record_from_csv_shape():
    rec = apify_comment_item_to_record(
        {
            "video_id": "7391028884474547489",
            "comment_id": "7527391834584711938",
            "text": "I miss this kid…",
            "digg_count": "9712",
            "reply_total": "20",
            "create_time": "2025-07-16 03:24:56",
            "uid": "7099133983321998342",
            "nickname": "Huberto🇵🇱",
            "unique_id": "hubertoloo",
            "region": "PL",
        },
        tz_name="America/New_York",
    )
    d = rec.to_dict()
    assert isinstance(rec, CommentRecord)
    assert d["comment_text"] == "I miss this kid…"
    assert d["comment_author_id"] == "7099133983321998342"
    assert d["comment_author_username"] == "hubertoloo"
    assert d["comment_author_nickname"] == "Huberto🇵🇱"
    assert d["comment_reply_count"] == 20
    assert d["comment_like_count"] == 9712
