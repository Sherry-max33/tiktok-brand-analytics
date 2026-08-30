"""
Video user crawler: Apify → VideoRecord → JSONL.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

from tiktok_brand.common.logging import get_logger
from tiktok_brand.crawl.apify_video_client import fetch_user_videos, get_apify_token
from tiktok_brand.crawl.apify_video_mapper import apify_video_item_to_record
from tiktok_brand.crawl.mock_video_items import mock_user_video_items

log = get_logger("tiktok_brand.crawl.video_user_crawler")


def _fetch_user_items(username: str, count: int, max_retries: int = 3) -> List[Dict[str, Any]]:
    if get_apify_token():
        log.info("Using Apify TikTok Scraper for user '%s'", username)
        for attempt in range(1, max_retries + 1):
            try:
                return fetch_user_videos(username, count)
            except Exception as exc:
                log.warning(
                    "Apify user crawl failed for '%s' (attempt %s/%s): %s",
                    username,
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt == max_retries:
                    raise
                time.sleep(1.5 * attempt)
        return []

    log.info(
        "Using mock Apify video items for user '%s' (set APIFY_API_TOKEN for live crawl)",
        username,
    )
    return mock_user_video_items(username, count)


def crawl_user(
    username: str,
    count: int,
    tz_name: str,
    brand: Optional[str] = None,
) -> List[Dict[str, Any]]:
    log.info("Starting video crawl for user '%s' (target: %s)", username, count)

    records: List[Dict[str, Any]] = []
    seen: set[str] = set()

    try:
        items = _fetch_user_items(username, count)
        if not items:
            return records

        for i, item in enumerate(items):
            video_id = str(item.get("id", ""))
            if not video_id or video_id in seen:
                continue
            seen.add(video_id)

            try:
                record = apify_video_item_to_record(
                    item,
                    source_type="user",
                    source_query=username,
                    brand=brand,
                    tz_name=tz_name,
                )
                records.append(record.to_dict())
            except Exception as exc:
                log.warning("Error processing video %s: %s", video_id, exc)
                continue

            if i < len(items) - 1:
                time.sleep(0.6 + random.random() * 0.6)

    except Exception as exc:
        log.error("Error during user video crawl for '%s': %s", username, exc)

    log.info("Completed user video crawl for '%s': %s records", username, len(records))
    return records
