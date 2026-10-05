"""Export the slim, committed data the Streamlit app reads (app/data/).

The research tables under data/processed/ stay local (gitignored). The app gets only what it
uses:
- videos.parquet: the catalog columns (catalog.COLUMNS) plus the caption and visual
  embeddings used for similar-video retrieval. Public post metadata only: no account IDs,
  bios or source URLs (post links are rebuilt from the handle and video ID), and the
  collection time is reduced to its date.
- comment_sentiment.parquet: one row per video with its scored-comment count and positive /
  negative shares. No comment text, comment IDs, per-comment rows or commenter identities.

    python scripts/export_app_data.py

Re-run after rebuilding the feature tables.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
logging.getLogger("streamlit").setLevel(logging.ERROR)

import pandas as pd  # noqa: E402

import analysis_data  # noqa: E402
import catalog  # noqa: E402

SOURCE_DIR = ROOT / "data" / "processed" / "feature"
VIDEO_SOURCE = SOURCE_DIR / "feature_table.parquet"
COMMENT_SOURCE = SOURCE_DIR / "comment_feature_table.parquet"
EMBEDDINGS = ["text_embedding", "visual_embedding"]


def main() -> None:
    catalog.APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    source = pd.read_parquet(VIDEO_SOURCE)
    keep = [c for c in dict.fromkeys(catalog.COLUMNS + EMBEDDINGS) if c in source.columns]
    videos = source[keep].copy()
    videos["crawl_at"] = pd.to_datetime(videos["crawl_at"], errors="coerce").dt.normalize()
    videos.to_parquet(catalog.FEATURE_TABLE, index=False, compression="zstd")

    comments = pd.read_parquet(COMMENT_SOURCE, columns=["video_id", "sentiment_score"])
    sentiment = analysis_data.sentiment_by_video(comments)
    sentiment.to_parquet(catalog.COMMENT_SENTIMENT, index=False, compression="zstd")

    for path, df in ((catalog.FEATURE_TABLE, videos), (catalog.COMMENT_SENTIMENT, sentiment)):
        size = path.stat().st_size / 1e6
        print(f"{path.relative_to(ROOT)}: {len(df):,} rows, {list(df.columns)}, {size:.1f} MB")


if __name__ == "__main__":
    main()
