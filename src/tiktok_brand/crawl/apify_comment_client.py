"""Apify clockworks/tiktok-comments-scraper client for comment crawls."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tiktok_brand.common.logging import get_logger
from tiktok_brand.crawl.apify_video_client import get_apify_token

log = get_logger("tiktok_brand.crawl.apify_comment_client")

DEFAULT_ACTOR_ID = "clockworks/tiktok-comments-scraper"


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
    log.info("Starting Apify actor %s for %s post URL(s)", actor_id, len(run_input.get("postURLs") or []))
    run = client.actor(actor_id).call(run_input=run_input)

    items: List[Dict[str, Any]] = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        if isinstance(item, dict):
            items.append(item)
    log.info("Apify comment actor finished: %s items", len(items))
    return items


def fetch_comments_for_post_urls(
    post_urls: List[str],
    comments_per_post: int,
    *,
    actor_id: str = DEFAULT_ACTOR_ID,
    api_token: Optional[str] = None,
    max_replies_per_comment: int = 0,
) -> List[Dict[str, Any]]:
    urls = [u.strip() for u in post_urls if u and str(u).strip()]
    if not urls:
        return []

    run_input = {
        "postURLs": urls,
        "commentsPerPost": max(1, comments_per_post),
        "maxRepliesPerComment": max(0, max_replies_per_comment),
    }
    return _run_actor(run_input, actor_id=actor_id, api_token=api_token)
