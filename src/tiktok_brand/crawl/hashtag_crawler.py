"""Hashtag crawler scaffold (sample-based).

Hashtag feeds are ranked streams. Always record `crawled_at` and `seed_hashtag`.
"""
from __future__ import annotations
from typing import List, Dict, Any
from ..common.logging import get_logger
from ..common.time import now_iso_tz, now_ts

log = get_logger(__name__)

def crawl_hashtag(seed_hashtag: str, count: int, tz_name: str) -> List[Dict[str, Any]]:
    """Return a list of raw dict rows (JSON-serializable) for a given seed hashtag.

    TODO: Implement using your chosen TikTok access method.
    """
    log.info("Scaffold crawl_hashtag called: seed_hashtag=%s count=%s", seed_hashtag, count)
    return []
