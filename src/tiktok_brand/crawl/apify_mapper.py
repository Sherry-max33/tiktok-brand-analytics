"""Map Apify clockworks/tiktok-scraper dataset items to VideoRecord."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tiktok_brand.common.time import now_iso_tz, now_ts
from tiktok_brand.crawl.schemas import VideoRecord


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _extract_hashtags(item: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    seen: set[str] = set()

    for entry in item.get("hashtags") or []:
        if isinstance(entry, dict):
            name = entry.get("name")
        else:
            name = entry
        if not name:
            continue
        tag = str(name).strip().lstrip("#").lower()
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)

    if tags:
        return tags

    text = item.get("text") or ""
    for word in str(text).split():
        if word.startswith("#"):
            tag = word[1:].strip().lower()
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
    return tags


def normalize_duration_sec(raw: Any) -> Optional[float]:
    """Apify videoMeta.duration is seconds; keep as-is."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def is_apify_item(item: Dict[str, Any]) -> bool:
    return "authorMeta" in item or "playCount" in item or "videoMeta" in item


def apify_item_to_video_record(
    item: Dict[str, Any],
    *,
    source_type: str,
    source_query: str,
    brand: Optional[str],
    tz_name: str,
) -> VideoRecord:
    author = _as_dict(item.get("authorMeta"))
    video_meta = _as_dict(item.get("videoMeta"))
    music_meta = _as_dict(item.get("musicMeta"))

    video_id_raw = item.get("id")
    video_id = str(video_id_raw) if video_id_raw is not None else None

    author_id_raw = author.get("id")
    author_id = str(author_id_raw) if author_id_raw is not None else None

    music_id_raw = music_meta.get("musicId")
    music_id = str(music_id_raw) if music_id_raw is not None else None

    return VideoRecord(
        platform="tiktok",
        source_type=source_type,
        source_query=source_query,
        brand=brand or None,
        video_id=video_id,
        create_time_ts=item.get("createTime"),
        caption_raw=item.get("text"),
        hashtags=_extract_hashtags(item),
        author_id=author_id,
        author_username=author.get("name") or author.get("nickName"),
        author_verified=author.get("verified"),
        author_follower_count=author.get("fans"),
        author_signature=author.get("signature"),
        video_duration_sec=normalize_duration_sec(video_meta.get("duration")),
        music_id=music_id,
        has_music=bool(music_id),
        view_count=item.get("playCount"),
        like_count=item.get("diggCount"),
        comment_count=item.get("commentCount"),
        share_count=item.get("shareCount"),
        save_count=item.get("collectCount"),
        crawled_at=now_iso_tz(tz_name),
        crawled_at_ts=now_ts(),
        raw_payload={"webVideoUrl": item.get("webVideoUrl")},
    )
