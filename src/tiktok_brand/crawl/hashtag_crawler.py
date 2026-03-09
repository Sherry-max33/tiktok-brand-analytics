"""
Simple TikTok hashtag crawler for raw data ingestion.

This module handles Stage 1 of the pipeline: collecting sample-based TikTok videos
from seed hashtags. Each crawl creates a new JSONL file with raw video records.

NOTE: Currently uses mock data for demonstration. To use real TikTokApi:
1. The TikTokApi library requires async/await and session management
2. Replace _crawl_hashtag_with_retry with actual async API calls
3. Handle session creation and browser automation requirements
"""

import time
import random
from typing import List, Dict, Any, Optional
from TikTokApi import TikTokApi

from tiktok_brand.common.time import now_iso_tz, now_ts
from tiktok_brand.crawl.schemas import VideoRecord
from tiktok_brand.common.logging import get_logger

log = get_logger("tiktok_brand.crawl.hashtag_crawler")


def _normalize_video_data(video_data: Any) -> Dict[str, Any]:
    """
    Normalize TikTok API response to a consistent dict format.

    TikTokApi may return different object types depending on version.
    This adapter ensures we always work with dictionaries.
    """
    if isinstance(video_data, dict):
        return video_data
    elif hasattr(video_data, 'as_dict'):
        return video_data.as_dict()
    elif hasattr(video_data, '__dict__'):
        return video_data.__dict__
    else:
        # Fallback: try to convert to dict
        try:
            return dict(video_data)
        except:
            log.warning(f"Could not normalize video data: {type(video_data)}")
            return {}


def _extract_hashtags(video_data: Dict[str, Any]) -> List[str]:
    """
    Extract hashtags from video data.

    First tries textExtra (structured hashtag data), then falls back to parsing caption.
    """
    hashtags = []

    # Try textExtra first (structured hashtag data)
    text_extra = video_data.get('textExtra', [])
    for item in text_extra:
        if item.get('hashtagName'):
            hashtags.append(item['hashtagName'].lower())

    # If no textExtra or it's empty, parse from caption
    if not hashtags:
        desc = video_data.get('desc', '')
        if desc:
            words = desc.split()
            for word in words:
                if word.startswith('#'):
                    hashtag = word[1:].strip().lower()
                    if hashtag and hashtag not in hashtags:
                        hashtags.append(hashtag)

    return hashtags


def _create_video_record(video_data: Dict[str, Any], seed_hashtag: str, brand: str, tz_name: str) -> VideoRecord:
    """
    Create a standardized video record from TikTok API data.

    This is the core mapping function that transforms raw API responses
    into our internal schema (VideoRecord) for the raw data layer.
    """
    # Extract basic video info
    video_id_raw = video_data.get('id')
    video_id = str(video_id_raw) if video_id_raw is not None else None
    create_time_ts = video_data.get('createTime')

    # Extract author info
    author = video_data.get('author', {})
    author_id_raw = author.get('id')
    author_id = str(author_id_raw) if author_id_raw is not None else None
    author_username = author.get('unique_id') or author.get('nickname')
    author_verified = author.get('verified', False)
    author_signature = author.get('signature') or author.get('signatureDesc') or author.get('bio')
    author_stats = video_data.get('authorStats') or author.get('stats') or {}
    author_follower_count = author_stats.get('followerCount')

    # Video duration: video.duration (often in ms)
    video_obj = video_data.get('video', {})
    duration_raw = video_obj.get('duration')
    if duration_raw is not None:
        video_duration_sec = float(duration_raw) / 1000.0 if duration_raw > 100 else float(duration_raw)
    else:
        video_duration_sec = None

    # Music: music.id or music.idStr; has_music = non-empty music with valid id
    music_obj = video_data.get('music') or {}
    music_id = music_obj.get('idStr') or music_obj.get('id')
    if music_id is not None:
        music_id = str(music_id)
    has_music = bool(music_obj and music_id)
    stats = video_data.get('stats', {})
    view_count = stats.get('playCount')
    like_count = stats.get('diggCount')
    comment_count = stats.get('commentCount')
    share_count = stats.get('shareCount')
    save_count = stats.get('collectCount')  # Bookmark/save count

    # Extract text content
    caption_raw = video_data.get('desc')
    hashtags = _extract_hashtags(video_data)

    return VideoRecord(
        platform='tiktok',
        source_type='hashtag',
        source_query=seed_hashtag,
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


def _crawl_hashtag_with_retry(api: TikTokApi, hashtag: str, count: int, max_retries: int = 3) -> List[Any]:
    """
    Crawl videos for a hashtag with simple retry logic.

    Returns raw video data objects from TikTokApi.
    """
    # For now, create mock data since TikTokApi requires complex async setup
    # This demonstrates the pipeline structure and can be replaced with real API calls
    log.info(f"Using mock data for hashtag '{hashtag}' (TikTokApi requires async setup)")

    mock_videos = []
    for i in range(min(count, 5)):  # Mock up to 5 videos for testing
        mock_video = {
            'id': f'mock_video_{hashtag}_{i}',
            'createTime': int(time.time()) - (i * 3600),  # Hours ago
            'desc': f'Mock video about {hashtag} #{hashtag} #fashion',
            'author': {
                'id': f'mock_author_{i}',
                'unique_id': f'user_{i}',
                'nickname': f'User {i}',
                'verified': i % 3 == 0,
                'signature': 'Fitness & sports content creator' if i % 3 == 0 else 'Lifestyle | fashion',
            },
            'authorStats': {'followerCount': 10000 + (i * 1000)},
            'video': {'duration': 15000 + (i * 1000)},  # ms
            'music': {'id': f'mock_music_{i}', 'idStr': f'mock_music_{i}'} if i % 2 == 0 else {},
            'stats': {
                'playCount': 1000 + (i * 500),
                'diggCount': 50 + (i * 20),
                'commentCount': 10 + (i * 5),
                'shareCount': 5 + (i * 2),
                'collectCount': 20 + (i * 8) if i % 2 == 0 else None
            },
            'textExtra': [
                {'hashtagName': hashtag},
                {'hashtagName': 'fashion'},
                {'hashtagName': 'style'}
            ]
        }
        mock_videos.append(mock_video)

    return mock_videos


def crawl_hashtag(seed_hashtag: str, count: int, tz_name: str, brand: str = "") -> List[Dict[str, Any]]:
    """
    Crawl videos for a given hashtag and return standardized records.

    This is the main entry point for hashtag-based data collection.
    It handles rate limiting, deduplication, and data transformation.

    Args:
        seed_hashtag: The hashtag to crawl (without #)
        count: Maximum number of videos to collect
        tz_name: Timezone name for crawled_at timestamp
        brand: Brand name (nike/adidas) for record metadata

    Returns:
        List of video records as dictionaries, ready for JSONL output
    """
    log.info(f"Starting crawl for hashtag '{seed_hashtag}' (target: {count} videos)")

    records = []
    seen_video_ids = set()

    try:
        api = TikTokApi()
        # Get videos from TikTok API with retry logic
        videos = _crawl_hashtag_with_retry(api, seed_hashtag, count)

        if not videos:
            log.warning(f"No videos found for hashtag '{seed_hashtag}'")
            return records

        log.info(f"Retrieved {len(videos)} raw videos for hashtag '{seed_hashtag}'")

        # Process each video
        for i, video_data in enumerate(videos):
            # Normalize the API response to a dict
            video_dict = _normalize_video_data(video_data)

            # Extract video ID for deduplication
            video_id = str(video_dict.get('id', ''))

            # Skip duplicates within this run
            if video_id in seen_video_ids:
                log.debug(f"Skipping duplicate video {video_id}")
                continue

            seen_video_ids.add(video_id)

            try:
                # Convert to our standardized record format (VideoRecord schema)
                record = _create_video_record(video_dict, seed_hashtag, brand, tz_name)
                records.append(record.to_dict())

                log.debug(f"Processed video {i+1}/{len(videos)}: {video_id}")

            except Exception as e:
                log.warning(f"Error processing video {video_id}: {e}")
                continue

            # Rate limiting: random sleep between 0.6-1.2 seconds
            if i < len(videos) - 1:  # Don't sleep after the last video
                sleep_time = 0.6 + random.random() * 0.6
                time.sleep(sleep_time)

    except Exception as e:
        log.error(f"Error during hashtag crawl for '{seed_hashtag}': {e}")
        return records

    log.info(f"Completed crawl for hashtag '{seed_hashtag}': {len(records)} records collected")
    return records


def crawl_hashtags_from_config(per_hashtag: int = 200) -> None:
    """
    Crawl hashtags from config files and save to raw data directory.

    This is a convenience function for testing and quick crawls.
    For production use, call crawl_hashtag directly with specific parameters.
    """
    import yaml
    from pathlib import Path
    from tiktok_brand.common.io import write_jsonl

    log.info("Starting hashtag crawl from config")

    # Load configs
    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    hashtags_cfg = yaml.safe_load(Path("configs/hashtags.yaml").read_text(encoding="utf-8"))

    tz = project_cfg["time"]["timezone"]
    raw_dir = Path(project_cfg["output"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    total_records = 0

    # Crawl each brand's hashtags
    for brand in ["nike", "adidas"]:
        brand_tags = hashtags_cfg.get(brand, [])
        log.info(f"Crawling {len(brand_tags)} hashtags for brand '{brand}'")

        for tag in brand_tags:
            try:
                records = crawl_hashtag(seed_hashtag=tag, count=per_hashtag, tz_name=tz, brand=brand)

                if records:
                    # Create unique filename with timestamp
                    timestamp = now_ts()
                    out_path = raw_dir / f"tiktok_hashtag_{brand}_{tag}_{timestamp}.jsonl"
                    write_jsonl(out_path, records)

                    log.info(f"Wrote {len(records)} records to {out_path}")
                    total_records += len(records)
                else:
                    log.warning(f"No records collected for hashtag '{tag}'")

            except Exception as e:
                log.error(f"Failed to crawl hashtag '{tag}': {e}")
                continue

    log.info(f"Hashtag crawl completed: {total_records} total records across all hashtags")