"""Clean table: raw CommentRecord JSONL → harmonized comment fact table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_comment_file(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return "comment" in name


def build_clean_comments(
    raw_paths: Sequence[str | Path],
    *,
    video_clean_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build silver comment table.

    - Reads only comment JSONL files
    - Dedupes by comment_id (keep latest crawled_at_ts)
    - Drops raw_payload
    - Optionally joins video clean for brand / source lineage
    """
    rows: List[Dict[str, Any]] = []
    for p in raw_paths:
        if not _is_comment_file(p):
            continue
        rows.extend(read_jsonl(p))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "comment_text" in df.columns:
        df["comment_text"] = df["comment_text"].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    for col in ["comment_like_count", "comment_reply_count", "crawled_at_ts"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "comment_id" in df.columns:
        sort_cols = ["crawled_at_ts"] if "crawled_at_ts" in df.columns else []
        if sort_cols:
            df = df.sort_values(by=sort_cols, ascending=True)
        df = df.drop_duplicates(subset=["comment_id"], keep="last")

    if "raw_payload" in df.columns:
        df = df.drop(columns=["raw_payload"])

    if video_clean_df is not None and not video_clean_df.empty and "video_id" in df.columns:
        video_cols = [
            c
            for c in [
                "video_id",
                "brand",
                "source_type",
                "source_query",
                "seed_hashtag",
                "is_official_brand",
                "author_username",
            ]
            if c in video_clean_df.columns
        ]
        video_dim = video_clean_df[video_cols].drop_duplicates(subset=["video_id"], keep="last")
        df = df.merge(video_dim, on="video_id", how="left", suffixes=("", "_video"))

    return df.reset_index(drop=True)
