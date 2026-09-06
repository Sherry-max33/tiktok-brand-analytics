"""Tests for comment clean + feature ETL."""

import json
from pathlib import Path

import pandas as pd

from tiktok_brand.etl.build_clean_comments import build_clean_comments
from tiktok_brand.etl.comment_feature_table import build_comment_feature_table


def _write_comment_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_build_clean_comments_dedupe_and_join(tmp_path: Path):
    raw = tmp_path / "tiktok_comments_sample.jsonl"
    _write_comment_jsonl(
        raw,
        [
            {
                "platform": "tiktok",
                "video_id": "v1",
                "comment_id": "c1",
                "comment_text": " love it ",
                "comment_like_count": 3,
                "comment_reply_count": 1,
                "comment_create_time": "2025-01-01",
                "comment_author_id": "u1",
                "comment_author_username": "alice",
                "comment_author_nickname": "Alice",
                "region": "US",
                "crawled_at": "2026-01-01T00:00:00Z",
                "crawled_at_ts": 100,
                "raw_payload": {"x": 1},
            },
            {
                "platform": "tiktok",
                "video_id": "v1",
                "comment_id": "c1",
                "comment_text": "love it updated",
                "comment_like_count": 5,
                "comment_reply_count": 2,
                "comment_create_time": "2025-01-01",
                "comment_author_id": "u1",
                "comment_author_username": "alice",
                "comment_author_nickname": "Alice",
                "region": "US",
                "crawled_at": "2026-01-02T00:00:00Z",
                "crawled_at_ts": 200,
                "raw_payload": {"x": 2},
            },
        ],
    )
    videos = pd.DataFrame(
        [{"video_id": "v1", "brand": "nike", "source_type": "hashtag", "is_official_brand": False}]
    )
    clean = build_clean_comments([raw], video_clean_df=videos)
    assert len(clean) == 1
    assert clean.loc[0, "comment_text"] == "love it updated"
    assert clean.loc[0, "brand"] == "nike"
    assert "raw_payload" not in clean.columns


def test_build_comment_feature_table_sentiment_and_embedding():
    clean = pd.DataFrame(
        [
            {
                "video_id": "v1",
                "comment_id": "c1",
                "comment_text": "I absolutely love these sneakers!",
                "comment_like_count": 10,
                "comment_reply_count": 2,
                "brand": "nike",
                "crawled_at": "2026-01-01T00:00:00-05:00",
                "crawled_at_ts": 1704067200,
            }
        ]
    )
    out = build_comment_feature_table(
        clean,
        encode_batch_fn=lambda texts: [[0.1, 0.2] for _ in texts],
    )
    assert out.loc[0, "comment_text_clean"]
    assert "sentiment_method" in out.columns
    assert out.loc[0, "embedding_method"] == "sbert_multilingual"
    assert out.loc[0, "text_embedding"] == [0.1, 0.2]
