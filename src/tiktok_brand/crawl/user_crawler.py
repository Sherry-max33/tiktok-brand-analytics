"""User (account) crawler scaffold.

This module is intentionally a scaffold: different TikTok access methods vary by environment.
Keep the *data contract* stable and swap the backend implementation as needed.
"""
from __future__ import annotations
from typing import Iterable, Dict, Any, List, Optional
from .schemas import VideoRecord
from ..common.time import now_iso_tz, now_ts
from ..common.logging import get_logger

log = get_logger(__name__)

def crawl_user(username: str, count: int, tz_name: str) -> List[Dict[str, Any]]:
    """Return a list of raw dict rows (JSON-serializable) for a given username.

    TODO: Implement using your chosen TikTok access method.
    """
    crawled_at = now_iso_tz(tz_name)
    crawled_at_ts = now_ts()
    log.info("Scaffold crawl_user called: username=%s count=%s", username, count)

    # Placeholder empty result
    return []

