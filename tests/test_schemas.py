from tiktok_brand.crawl.schemas import VideoRecord

def test_video_record_to_dict():
    vr = VideoRecord(
        platform="tiktok", source_type="user", source_query="nike", seed_hashtag=None,
        video_id="1", create_time_ts=None, caption_raw=None,
        author_id=None, author_username=None, author_verified=None,
        view_count=None, like_count=None, comment_count=None, share_count=None, save_count=None,
        crawled_at="2026-01-06T00:00:00-05:00", crawled_at_ts=0,
        extra={}
    )
    d = vr.to_dict()
    assert d["platform"] == "tiktok"
