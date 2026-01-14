from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import yaml

from ..common.text import extract_hashtags
from ..common.time import ts_to_iso
from .normalize_tags import normalize_hashtags
from .feature_table import add_derived_metrics

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
    hashtags_cfg_path: str | Path,
    accounts_cfg_path: str | Path,
    project_cfg_path: str | Path,
) -> pd.DataFrame:
    hashtags_cfg = load_yaml(hashtags_cfg_path)
    accounts_cfg = load_yaml(accounts_cfg_path)
    project_cfg = load_yaml(project_cfg_path)
    tz = project_cfg["time"]["timezone"]

    normalize_map = {k.lower(): v.lower() for k, v in (hashtags_cfg.get("normalize_tags") or {}).items()}
    product_map = {k.lower(): v for k, v in (hashtags_cfg.get("product_line_map") or {}).items()}
    style_map = {k.lower(): v for k, v in (hashtags_cfg.get("brand_style_map") or {}).items()}

    official_usernames = set()
    for brand, users in (accounts_cfg.get("official_accounts") or {}).items():
        for u in users:
            official_usernames.add(u.lower())

    rows: List[Dict[str, Any]] = []
    for p in raw_paths:
        rows.extend(read_jsonl(p))

    df = pd.DataFrame(rows)

    # Basic harmonization
    if "caption_raw" not in df.columns and "caption" in df.columns:
        df["caption_raw"] = df["caption"]

    # Extract hashtags if missing
    if "hashtags" not in df.columns:
        df["hashtags"] = df["caption_raw"].fillna("").map(extract_hashtags)

    df["hashtags"] = df["hashtags"].apply(lambda x: x if isinstance(x, list) else [])
    df["normalized_hashtags"] = df["hashtags"].apply(lambda tags: normalize_hashtags(tags, normalize_map))

    # Brand label (simple: if any nike* tag then nike; if any adidas* tag then adidas; else null)
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

    df["brand"] = df["normalized_hashtags"].apply(infer_brand)

    # brand_style
    def infer_style(tags):
        for t in (tags or []):
            if t.lower() in style_map:
                return style_map[t.lower()]
        return None
    df["brand_style"] = df["normalized_hashtags"].apply(infer_style)

    # product_line
    def infer_product_line(tags):
        for t in (tags or []):
            t = t.lower()
            if t in product_map:
                return product_map[t]
        return None
    df["product_line"] = df["normalized_hashtags"].apply(infer_product_line)

    # official flag
    df["author_username"] = df.get("author_username")
    df["is_official_brand"] = df["author_username"].fillna("").str.lower().isin(official_usernames)

    # create_time ISO
    if "create_time_ts" in df.columns:
        df["create_time"] = df["create_time_ts"].apply(lambda x: ts_to_iso(int(x), tz) if pd.notna(x) else pd.NA)

    # Derived metrics
    df = add_derived_metrics(df)

    # De-dup by video_id, keep last (assumes later crawl has newer stats)
    if "video_id" in df.columns:
        df = df.sort_values(by=["crawled_at_ts"], ascending=True)
        df = df.drop_duplicates(subset=["video_id"], keep="last")

    return df
