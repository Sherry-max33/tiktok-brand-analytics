"""
Comment crawler: Apify → CommentRecord → JSONL.

Raw layer: field-name mapping only; text cleaning happens in comment ETL.
"""

from __future__ import annotations

import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tiktok_brand.common.logging import get_logger
from tiktok_brand.common.time import now_ts
from tiktok_brand.crawl.apify_comment_client import fetch_comments_for_post_urls
from tiktok_brand.crawl.apify_comment_mapper import apify_comment_item_to_record
from tiktok_brand.crawl.apify_video_client import get_apify_token
from tiktok_brand.crawl.mock_comment_items import mock_comment_items

log = get_logger("tiktok_brand.crawl.comment_crawler")

_VIDEO_ID_RE = re.compile(r"/video/(\d+)")


def build_tiktok_video_url(video_id_or_url: str) -> str:
    value = str(video_id_or_url).strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"https://www.tiktok.com/video/{value.lstrip('/')}"


def video_id_from_url(url: str) -> Optional[str]:
    match = _VIDEO_ID_RE.search(str(url))
    return match.group(1) if match else None


def _can_use_apify_live() -> bool:
    if not get_apify_token():
        return False
    try:
        import apify_client  # noqa: F401

        return True
    except ImportError:
        log.info("apify-client not installed; using mock comment items")
        return False


def _fetch_comment_items(
    video_ids_or_urls: Sequence[str],
    comments_per_video: int,
    *,
    actor_id: Optional[str] = None,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    urls = [build_tiktok_video_url(v) for v in video_ids_or_urls]
    if not urls:
        return []

    if _can_use_apify_live():
        log.info("Using Apify TikTok Comments Scraper for %s video(s)", len(urls))
        for attempt in range(1, max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "post_urls": urls,
                    "comments_per_post": comments_per_video,
                }
                if actor_id:
                    kwargs["actor_id"] = actor_id
                return fetch_comments_for_post_urls(**kwargs)
            except Exception as exc:
                log.warning(
                    "Apify comment crawl failed (attempt %s/%s): %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt == max_retries:
                    raise
                time.sleep(1.5 * attempt)
        return []

    log.info(
        "Using mock Apify comment items for %s video(s) (set APIFY_API_TOKEN for live crawl)",
        len(urls),
    )
    items: List[Dict[str, Any]] = []
    for url in urls:
        vid = video_id_from_url(url) or url.rsplit("/", 1)[-1]
        items.extend(mock_comment_items(vid, comments_per_video))
    return items


def _items_to_records(
    items: List[Dict[str, Any]],
    *,
    tz_name: str,
    fallback_video_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_comment_ids: set[str] = set()

    for item in items:
        comment_id = str(item.get("comment_id") or item.get("cid") or "").strip()
        if comment_id and comment_id in seen_comment_ids:
            continue
        if comment_id:
            seen_comment_ids.add(comment_id)

        if fallback_video_id and not (
            item.get("video_id") or item.get("videoId") or item.get("videoWebUrl")
        ):
            item = {**item, "video_id": fallback_video_id}

        try:
            record = apify_comment_item_to_record(item, tz_name=tz_name)
            if not record.comment_id:
                continue
            records.append(record.to_dict())
        except Exception as exc:
            log.warning("Error processing comment %s: %s", comment_id or "?", exc)

    return records


def crawl_video_comments(
    video_id_or_url: str,
    comments_per_video: int,
    tz_name: str,
    *,
    actor_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    video_ref = str(video_id_or_url).strip()
    log.info(
        "Starting comment crawl for video '%s' (target: %s comments)",
        video_ref,
        comments_per_video,
    )

    try:
        items = _fetch_comment_items(
            [video_ref],
            comments_per_video,
            actor_id=actor_id,
        )
        fallback_video_id = video_id_from_url(build_tiktok_video_url(video_ref)) or video_ref
        records = _items_to_records(items, tz_name=tz_name, fallback_video_id=fallback_video_id)
    except Exception as exc:
        log.error("Error during comment crawl for '%s': %s", video_ref, exc)
        return []

    log.info("Completed comment crawl for '%s': %s records", video_ref, len(records))
    return records


def crawl_comments(
    video_ids_or_urls: Sequence[str],
    comments_per_video: int,
    tz_name: str,
    *,
    actor_id: Optional[str] = None,
    batch_size: int = 25,
) -> List[Dict[str, Any]]:
    refs = [str(v).strip() for v in video_ids_or_urls if str(v).strip()]
    if not refs:
        return []

    log.info(
        "Starting batch comment crawl for %s video(s) (target: %s comments each)",
        len(refs),
        comments_per_video,
    )

    all_records: List[Dict[str, Any]] = []
    seen_comment_ids: set[str] = set()
    chunk_size = max(1, batch_size)

    for start in range(0, len(refs), chunk_size):
        chunk = refs[start : start + chunk_size]
        try:
            items = _fetch_comment_items(chunk, comments_per_video, actor_id=actor_id)
            chunk_records = _items_to_records(items, tz_name=tz_name)
            for record in chunk_records:
                comment_id = str(record.get("comment_id") or "")
                if comment_id and comment_id in seen_comment_ids:
                    continue
                if comment_id:
                    seen_comment_ids.add(comment_id)
                all_records.append(record)
        except Exception as exc:
            log.error(
                "Error during comment crawl batch %s-%s: %s",
                start + 1,
                start + len(chunk),
                exc,
            )

        if start + chunk_size < len(refs):
            time.sleep(0.8 + random.random() * 0.8)

    log.info("Completed batch comment crawl: %s records from %s video(s)", len(all_records), len(refs))
    return all_records


def load_top_video_ids_from_feature_table(
    feature_table_path: Path,
    *,
    limit: int,
    sort_by: str = "engagement",
) -> List[str]:
    import pandas as pd

    path = Path(feature_table_path)
    if path.is_dir():
        df = pd.read_parquet(path)
    else:
        df = pd.read_parquet(path)

    if df.empty or "video_id" not in df.columns:
        return []

    sort_col = {
        "engagement": "engagement_count",
        "views": "view_count",
        "likes": "like_count",
    }.get(sort_by, "engagement_count")

    if sort_col not in df.columns:
        sort_col = "view_count" if "view_count" in df.columns else "video_id"

    ordered = df.sort_values(sort_col, ascending=False, na_position="last")
    ids = ordered["video_id"].dropna().astype(str).str.strip()
    ids = ids[ids != ""].drop_duplicates()
    return ids.head(limit).tolist()


def crawl_comments_from_config(
    *,
    video_ids: Optional[Sequence[str]] = None,
    feature_table_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> None:
    import yaml
    from tiktok_brand.common.io import write_jsonl

    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    sampling = project_cfg.get("sampling") or {}
    apify_cfg = project_cfg.get("apify") or {}

    tz = project_cfg["time"]["timezone"]
    raw_dir = Path(project_cfg["output"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    comments_per_video = int(sampling.get("comments_per_video", 100))
    target_videos = int(limit or sampling.get("target_videos_for_comments", 1000))
    sort_by = str(sampling.get("sort_by", "engagement"))
    actor_id = apify_cfg.get("comment_actor_id")

    refs: List[str]
    if video_ids:
        refs = [str(v).strip() for v in video_ids if str(v).strip()]
    else:
        processed_dir = Path(project_cfg["output"]["processed_dir"])
        feature_path = feature_table_path or processed_dir / "feature_table"
        if not feature_path.exists() and (processed_dir / "tiktok_videos.parquet").exists():
            feature_path = processed_dir / "tiktok_videos.parquet"
        refs = load_top_video_ids_from_feature_table(
            feature_path,
            limit=target_videos,
            sort_by=sort_by,
        )

    if not refs:
        log.warning("No video IDs found for comment crawl")
        return

    records = crawl_comments(
        refs,
        comments_per_video=comments_per_video,
        tz_name=tz,
        actor_id=actor_id,
    )
    if not records:
        log.warning("Comment crawl returned no records")
        return

    out_path = raw_dir / f"tiktok_top{len(refs)}_comments_{now_ts()}.jsonl"
    write_jsonl(out_path, records)
    log.info("Wrote %s comment records to %s", len(records), out_path)
