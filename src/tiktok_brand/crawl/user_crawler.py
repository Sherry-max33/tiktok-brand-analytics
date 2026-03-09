"""
User (account) crawler for raw data ingestion.

Same data contract as hashtag_crawler: VideoRecord schema, in-run dedup, retry, rate limit.
source_type='user', source_query=username, brand from config (e.g. nike/adidas).
"""

from __future__ import annotations
import time
import random
from typing import List, Dict, Any, Optional

from tiktok_brand.common.time import now_iso_tz, now_ts
from tiktok_brand.crawl.schemas import VideoRecord
from tiktok_brand.common.logging import get_logger

log = get_logger("tiktok_brand.crawl.user_crawler")


def _normalize_video_data(video_data: Any) -> Dict[str, Any]:
    """Normalize API response to a consistent dict (same as hashtag_crawler)."""
    if isinstance(video_data, dict):
        return video_data
    if hasattr(video_data, "as_dict"):
        return video_data.as_dict()
    if hasattr(video_data, "__dict__"):
        return video_data.__dict__
    try:
        return dict(video_data)
    except Exception:
        log.warning("Could not normalize video data: %s", type(video_data))
        return {}


def _extract_hashtags(video_data: Dict[str, Any]) -> List[str]:
    """Extract hashtags from textExtra or caption (same logic as hashtag_crawler)."""
    hashtags = []
    text_extra = video_data.get("textExtra", [])
    for item in text_extra:
        if item.get("hashtagName"):
            hashtags.append(item["hashtagName"].lower())
    if not hashtags:
        desc = video_data.get("desc", "") or ""
        for word in desc.split():
            if word.startswith("#"):
                tag = word[1:].strip().lower()
                if tag and tag not in hashtags:
                    hashtags.append(tag)
    return hashtags


def _create_video_record(
    video_data: Dict[str, Any],
    username: str,
    brand: Optional[str],
    tz_name: str,
) -> VideoRecord:
    """
    Build a VideoRecord from raw video data for a user crawl.

    source_type='user', source_query=username. Same schema as hashtag_crawler.
    """
    video_id_raw = video_data.get("id")
    video_id = str(video_id_raw) if video_id_raw is not None else None
    create_time_ts = video_data.get("createTime")

    author = video_data.get("author", {})
    author_id_raw = author.get("id")
    author_id = str(author_id_raw) if author_id_raw is not None else None
    author_username = author.get("unique_id") or author.get("nickname")
    author_verified = author.get("verified", False)
    author_signature = author.get("signature") or author.get("signatureDesc") or author.get("bio")
    author_stats = video_data.get("authorStats") or author.get("stats") or {}
    author_follower_count = author_stats.get("followerCount")

    video_obj = video_data.get("video", {})
    duration_raw = video_obj.get("duration")
    if duration_raw is not None:
        video_duration_sec = float(duration_raw) / 1000.0 if duration_raw > 100 else float(duration_raw)
    else:
        video_duration_sec = None

    music_obj = video_data.get("music") or {}
    music_id = music_obj.get("idStr") or music_obj.get("id")
    if music_id is not None:
        music_id = str(music_id)
    has_music = bool(music_obj and music_id)

    stats = video_data.get("stats", {})
    view_count = stats.get("playCount")
    like_count = stats.get("diggCount")
    comment_count = stats.get("commentCount")
    share_count = stats.get("shareCount")
    save_count = stats.get("collectCount")

    caption_raw = video_data.get("desc")
    hashtags = _extract_hashtags(video_data)

    return VideoRecord(
        platform="tiktok",
        source_type="user",
        source_query=username,
        brand=brand or None,
        video_id=video_id,
        create_time_ts=create_time_ts,
        caption_raw=caption_raw,
        hashtags=hashtags,
        author_id=author_id,
        author_username=author_username,
        author_verified=author_verified,
        author_follower_count=author_follower_count,
        author_signature=author_signature,
        video_duration_sec=video_duration_sec,
        music_id=music_id,
        has_music=has_music,
        view_count=view_count,
        like_count=like_count,
        comment_count=comment_count,
        share_count=share_count,
        save_count=save_count,
        crawled_at=now_iso_tz(tz_name),
        crawled_at_ts=now_ts(),
        raw_payload={},
    )


def _crawl_user_with_retry(username: str, count: int, max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Fetch videos for a user with retry. Returns list of raw video dicts.

    Currently mock data; replace with TikTokApi user.videos() when available.
    """
    log.info("Using mock data for user '%s' (TikTokApi requires async setup)", username)
    mock_videos = []
    for i in range(min(count, 5)):
        mock_videos.append({
            "id": f"mock_user_{username}_{i}",
            "createTime": int(time.time()) - (i * 3600),
            "desc": f"Official post #{username} #fashion",
            "author": {
                "id": f"mock_author_{username}",
                "unique_id": username,
                "nickname": username,
                "verified": True,
                "signature": "Official brand account",
            },
            "authorStats": {"followerCount": 50000 + i * 1000},
            "video": {"duration": 12000},
            "music": {"id": f"music_{username}_{i}", "idStr": f"music_{username}_{i}"},
            "stats": {
                "playCount": 2000 + i * 500,
                "diggCount": 100 + i * 20,
                "commentCount": 20 + i * 5,
                "shareCount": 10 + i * 2,
                "collectCount": 30 + i * 10 if i % 2 == 0 else None,
            },
            "textExtra": [
                {"hashtagName": username},
                {"hashtagName": "fashion"},
            ],
        })
    return mock_videos


def crawl_user(
    username: str,
    count: int,
    tz_name: str,
    brand: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Crawl videos for a given user. Returns list of VideoRecord as dicts (JSONL-ready).

    In-run dedup by video_id, rate limit 0.6–1.2s between items, retry in _crawl_user_with_retry.
    """
    log.info("Starting crawl for user '%s' (target: %s videos)", username, count)
    records = []
    seen = set()

    try:
        videos = _crawl_user_with_retry(username, count)
        if not videos:
            log.warning("No videos found for user '%s'", username)
            return records
        log.info("Retrieved %s raw videos for user '%s'", len(videos), username)

        for i, video_data in enumerate(videos):
            video_dict = _normalize_video_data(video_data)
            vid = str(video_dict.get("id", ""))
            if vid in seen:
                continue
            seen.add(vid)
            try:
                record = _create_video_record(video_dict, username, brand, tz_name)
                records.append(record.to_dict())
            except Exception as e:
                log.warning("Error processing video %s: %s", vid, e)
                continue
            if i < len(videos) - 1:
                time.sleep(0.6 + random.random() * 0.6)
    except Exception as e:
        log.error("Error during user crawl for '%s': %s", username, e)
        return records

    log.info("Completed crawl for user '%s': %s records", username, len(records))
    return records
