"""Apify clockworks/tiktok-scraper client for hashtag and profile crawls."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from tiktok_brand.common.logging import get_logger

log = get_logger("tiktok_brand.crawl.apify_client")

DEFAULT_ACTOR_ID = "clockworks/tiktok-scraper"
DEFAULT_RESULTS_PER_PAGE = 200


def get_apify_token() -> Optional[str]:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("APIFY_API_TOKEN") or os.environ.get("APIFY_TOKEN")


def _base_run_input(scrape_additional_author_meta: bool = True) -> Dict[str, Any]:
    return {
        "shouldDownloadVideos": False,
        "shouldDownloadCovers": False,
        "shouldDownloadSlideshowImages": False,
        "shouldDownloadAvatars": False,
        "shouldDownloadMusicCovers": False,
        "commentsPerPost": 0,
        "topLevelCommentsPerPost": 0,
        "maxRepliesPerComment": 0,
        "scrapeAdditionalAuthorMeta": scrape_additional_author_meta,
        "scrapeRelatedVideos": False,
        "downloadSubtitlesOptions": "NEVER_DOWNLOAD_SUBTITLES",
        "proxyCountryCode": "None",
    }


def _run_actor(
    run_input: Dict[str, Any],
    *,
    actor_id: str = DEFAULT_ACTOR_ID,
    api_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    token = api_token or get_apify_token()
    if not token:
        raise RuntimeError(
            "APIFY_API_TOKEN is not set. Add it to your environment or .env file."
        )

    try:
        from apify_client import ApifyClient
    except ImportError as exc:
        raise RuntimeError(
            "apify-client is not installed. Run: pip install apify-client"
        ) from exc

    client = ApifyClient(token)
    log.info("Starting Apify actor %s", actor_id)
    run = client.actor(actor_id).call(run_input=run_input)

    items: List[Dict[str, Any]] = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        if isinstance(item, dict):
            items.append(item)
    log.info("Apify actor finished: %s items", len(items))
    return items


def fetch_hashtag_videos(
    hashtag: str,
    count: int,
    *,
    actor_id: str = DEFAULT_ACTOR_ID,
    api_token: Optional[str] = None,
    scrape_additional_author_meta: bool = True,
) -> List[Dict[str, Any]]:
    tag = hashtag.lstrip("#")
    run_input = {
        **_base_run_input(scrape_additional_author_meta),
        "hashtags": [tag],
        "resultsPerPage": min(count, DEFAULT_RESULTS_PER_PAGE),
    }
    return _run_actor(run_input, actor_id=actor_id, api_token=api_token)


def fetch_user_videos(
    username: str,
    count: int,
    *,
    actor_id: str = DEFAULT_ACTOR_ID,
    api_token: Optional[str] = None,
    scrape_additional_author_meta: bool = True,
) -> List[Dict[str, Any]]:
    handle = username.lstrip("@")
    run_input = {
        **_base_run_input(scrape_additional_author_meta),
        "profiles": [handle],
        "resultsPerPage": min(count, DEFAULT_RESULTS_PER_PAGE),
        "profileScrapeSections": ["videos"],
        "profileSorting": "latest",
        "excludePinnedPosts": False,
    }
    return _run_actor(run_input, actor_id=actor_id, api_token=api_token)
