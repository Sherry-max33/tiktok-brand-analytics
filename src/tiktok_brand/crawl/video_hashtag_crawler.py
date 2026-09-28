"""
Video search crawler: TikTok search results for seed terms (Apify) → VideoRecord → JSONL.

Seed terms live in configs/hashtags.yaml. Records keep source_type="hashtag" and the
tiktok_hashtag_* file names for compatibility with existing data; "hashtag" here means
"seed-term search", not the TikTok hashtag page.

Raw layer: field-name mapping only; normalization happens in clean ETL.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List

from tiktok_brand.common.logging import get_logger
from tiktok_brand.common.time import now_ts
from tiktok_brand.crawl.apify_video_client import fetch_search_videos, get_apify_token
from tiktok_brand.crawl.apify_video_mapper import apify_video_item_to_record
from tiktok_brand.crawl.mock_video_items import mock_hashtag_video_items

log = get_logger("tiktok_brand.crawl.video_hashtag_crawler")


def _fetch_search_items(term: str, count: int, max_retries: int = 3) -> List[Dict[str, Any]]:
    if get_apify_token():
        log.info("Using Apify TikTok Scraper search for '%s'", term)
        for attempt in range(1, max_retries + 1):
            try:
                return fetch_search_videos(term, count)
            except Exception as exc:
                log.warning(
                    "Apify search crawl failed for '%s' (attempt %s/%s): %s",
                    term,
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt == max_retries:
                    raise
                time.sleep(1.5 * attempt)
        return []

    log.info(
        "Using mock Apify video items for search '%s' (set APIFY_API_TOKEN for live crawl)",
        term,
    )
    return mock_hashtag_video_items(term, count)


def crawl_hashtag(seed_hashtag: str, count: int, tz_name: str, brand: str = "") -> List[Dict[str, Any]]:
    """Search TikTok for one seed term and map results to records."""
    log.info("Starting video search crawl for '%s' (target: %s)", seed_hashtag, count)

    records: List[Dict[str, Any]] = []
    seen_video_ids: set[str] = set()

    try:
        items = _fetch_search_items(seed_hashtag, count)
        if not items:
            log.warning("No videos found for search '%s'", seed_hashtag)
            return records

        for i, item in enumerate(items):
            video_id = str(item.get("id", ""))
            if not video_id or video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)

            try:
                record = apify_video_item_to_record(
                    item,
                    source_type="hashtag",
                    source_query=seed_hashtag,
                    brand=brand or None,
                    tz_name=tz_name,
                )
                records.append(record.to_dict())
            except Exception as exc:
                log.warning("Error processing video %s: %s", video_id, exc)
                continue

            if i < len(items) - 1:
                time.sleep(0.6 + random.random() * 0.6)

    except Exception as exc:
        log.error("Error during search video crawl for '%s': %s", seed_hashtag, exc)
        return records

    log.info("Completed search video crawl for '%s': %s records", seed_hashtag, len(records))
    return records


def crawl_hashtags_from_config(per_hashtag: int = 200) -> None:
    import yaml
    from pathlib import Path

    from tiktok_brand.common.io import write_jsonl

    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    hashtags_cfg = yaml.safe_load(Path("configs/hashtags.yaml").read_text(encoding="utf-8"))

    tz = project_cfg["time"]["timezone"]
    raw_dir = Path(project_cfg["output"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    total_records = 0
    for brand in ["nike", "adidas"]:
        for tag in hashtags_cfg.get(brand, []):
            try:
                records = crawl_hashtag(seed_hashtag=tag, count=per_hashtag, tz_name=tz, brand=brand)
                if records:
                    out_path = raw_dir / f"tiktok_hashtag_{brand}_{tag}_{now_ts()}.jsonl"
                    write_jsonl(out_path, records)
                    log.info("Wrote %s records to %s", len(records), out_path)
                    total_records += len(records)
            except Exception as exc:
                log.error("Failed to crawl search term '%s': %s", tag, exc)

    log.info("Search video crawl completed: %s total records", total_records)
