import pandas as pd
from tiktok_brand.etl.feature_table import add_derived_metrics, build_feature_table

def test_add_derived_metrics():
    df = pd.DataFrame([{"view_count": 100, "like_count": 10, "comment_count": 2, "share_count": 1, "save_count": 0}])
    out = add_derived_metrics(df)
    assert out.loc[0, "engagement_count"] == 13
    assert abs(out.loc[0, "engagement_rate"] - 0.13) < 1e-9


def test_build_feature_table():
    clean = pd.DataFrame([{
        "video_id": "v1",
        "create_time_ts": 1700000000,
        "caption_raw": "Check this out #nike #fashion shop now",
        "hashtags": ["nike", "fashion"],
        "view_count": 1000,
        "like_count": 100,
        "comment_count": 20,
        "share_count": 10,
        "save_count": 5,
        "brand": "nike",
        "source_type": "hashtag",
        "source_query": "nike",
        "crawled_at": "2026-01-01T00:00:00-05:00",
        "crawled_at_ts": 1704067200,
        "is_official_brand": False,
    }])
    out = build_feature_table(clean)
    assert "video_url" in out.columns
    assert "sentiment_score" in out.columns
    assert "visual_embedding" in out.columns
    assert "norm_engagement_score" in out.columns
    assert "post_date" in out.columns
    assert "post_hour" in out.columns
    assert "post_weekday" in out.columns
    assert "crawl_at" in out.columns
    assert "post_create_time" not in out.columns
    assert "play_count" not in out.columns
    assert "author_is_verified" not in out.columns
    assert out.loc[0, "engagement_count"] == 135
    assert out.loc[0, "has_call_to_action"] == True
