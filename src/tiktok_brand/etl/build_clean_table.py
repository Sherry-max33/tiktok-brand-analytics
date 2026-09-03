"""
Clean table: raw VideoRecord → harmonized fact table.

Clean does NOT compute engagement rates or product taxonomy
(those are feature-layer derived fields).

Fields:
- caption_raw strip; hashtags list cleanup
- normalized_hashtags via configs/hashtags.yaml normalize_tags
- brand: prefer raw brand; else infer from nike*/adidas* tags
- seed_hashtag, is_official_brand, create_time
- dedupe by video_id (keep latest crawled_at_ts); drop raw_payload
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from ..common.text import extract_hashtags
from ..common.time import ts_to_iso
from .normalize_tags import normalize_hashtags


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_clean_table(
    raw_paths: List[str | Path],
    accounts_cfg_path: str | Path,
    project_cfg_path: str | Path,
    hashtags_cfg_path: str | Path | None = None,
    taxonomy_cfg_path: str | Path | None = None,  # unused; call-site compat
) -> pd.DataFrame:
    accounts_cfg = load_yaml(accounts_cfg_path)
    project_cfg = load_yaml(project_cfg_path)
    tz = project_cfg["time"]["timezone"]

    tags_path = hashtags_cfg_path or (project_cfg.get("output") or {}).get(
        "hashtags_cfg", "configs/hashtags.yaml"
    )
    hashtags_cfg = load_yaml(tags_path)
    normalize_map = {
        str(k).lower(): str(v).lower()
        for k, v in (hashtags_cfg.get("normalize_tags") or {}).items()
    }

    official_usernames = set()
    for _brand, users in (accounts_cfg.get("official_accounts") or {}).items():
        for u in users:
            official_usernames.add(u.lower())

    rows: List[Dict[str, Any]] = []
    for p in raw_paths:
        name = Path(p).name.lower()
        if "comments" in name or "comment_" in name:
            continue
        rows.extend(read_jsonl(p))

    df = pd.DataFrame(rows)

    if "collect_count" not in df.columns and "save_count" in df.columns:
        df["collect_count"] = df["save_count"]

    if "caption_raw" not in df.columns and "caption" in df.columns:
        df["caption_raw"] = df["caption"]

    if "caption_raw" in df.columns:
        df["caption_raw"] = df["caption_raw"].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    if "hashtags" not in df.columns:
        df["hashtags"] = df["caption_raw"].fillna("").map(extract_hashtags)

    def _clean_hashtag_list(tags: Any) -> List[str]:
        if not isinstance(tags, list):
            return []
        seen: set[str] = set()
        out: List[str] = []
        for item in tags:
            t = str(item).strip().lstrip("#").lower()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    df["hashtags"] = df["hashtags"].apply(_clean_hashtag_list)
    df["normalized_hashtags"] = df["hashtags"].apply(
        lambda tags: normalize_hashtags(tags, normalize_map)
    )

    def infer_brand(tags):
        tags = [t.lower() for t in (tags or [])]
        has_nike = any(t.startswith("nike") for t in tags)
        has_adidas = any(t.startswith("adidas") for t in tags)
        if has_nike and not has_adidas:
            return "nike"
        if has_adidas and not has_nike:
            return "adidas"
        if has_nike and has_adidas:
            return "both"
        return None

    inferred = df["normalized_hashtags"].apply(infer_brand)
    if "brand" in df.columns:
        df["brand"] = df["brand"].apply(
            lambda x: str(x).strip().lower() if pd.notna(x) and str(x).strip() else pd.NA
        )
        df["brand"] = df["brand"].fillna(inferred)
    else:
        df["brand"] = inferred

    df["seed_hashtag"] = df.apply(
        lambda r: r.get("source_query") if r.get("source_type") == "hashtag" else None,
        axis=1,
    )

    df["author_username"] = df.get("author_username")
    df["is_official_brand"] = df["author_username"].fillna("").str.lower().isin(official_usernames)

    if "create_time_ts" in df.columns:
        df["create_time"] = df["create_time_ts"].apply(
            lambda x: ts_to_iso(int(x), tz) if pd.notna(x) else pd.NA
        )

    if "video_id" in df.columns:
        df = df.sort_values(by=["crawled_at_ts"], ascending=True)
        df = df.drop_duplicates(subset=["video_id"], keep="last")

    if "raw_payload" in df.columns:
        df = df.drop(columns=["raw_payload"])

    return df
