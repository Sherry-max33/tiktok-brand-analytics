"""Map Apify / external comment rows to CommentRecord (field rename only)."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from tiktok_brand.common.time import now_iso_tz, now_ts
from tiktok_brand.crawl.comment_schemas import CommentRecord

_VIDEO_ID_RE = re.compile(r"/video/(\d+)")


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _video_id_from_item(item: Dict[str, Any]) -> Optional[str]:
    direct = _to_str(item.get("video_id") or item.get("videoId"))
    if direct:
        return direct
    url = _to_str(item.get("videoWebUrl") or item.get("video_web_url"))
    if not url:
        return None
    match = _VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


def apify_comment_item_to_record(item: Dict[str, Any], *, tz_name: str) -> CommentRecord:
    """
    Apify clockworks/tiktok-comments-scraper style fields:
    cid, text, diggCount, replyCommentTotal, createTimeISO, uid, uniqueId, ...
    """
    video_id = _video_id_from_item(item)
    comment_id = _to_str(item.get("comment_id") or item.get("cid"))
    comment_text = _to_str(item.get("comment_text") or item.get("text")) or ""

    return CommentRecord(
        platform="tiktok",
        video_id=video_id or "",
        comment_id=comment_id or "",
        comment_text=comment_text,
        comment_like_count=_to_int(
            item.get("comment_like_count") or item.get("diggCount") or item.get("digg_count")
        ),
        comment_reply_count=_to_int(
            item.get("comment_reply_count")
            or item.get("replyCommentTotal")
            or item.get("reply_total")
        ),
        comment_create_time=_to_str(
            item.get("comment_create_time") or item.get("createTimeISO") or item.get("create_time")
        ),
        comment_author_id=_to_str(item.get("comment_author_id") or item.get("uid")),
        comment_author_username=_to_str(
            item.get("comment_author_username") or item.get("uniqueId") or item.get("unique_id")
        ),
        comment_author_nickname=_to_str(
            item.get("comment_author_nickname") or item.get("nickName") or item.get("nickname")
        ),
        region=_to_str(item.get("region")),
        crawled_at=now_iso_tz(tz_name),
        crawled_at_ts=now_ts(),
        raw_payload={
            k: v
            for k, v in {
                "videoWebUrl": item.get("videoWebUrl"),
                "avatarThumbnail": item.get("avatarThumbnail"),
            }.items()
            if v
        },
    )
