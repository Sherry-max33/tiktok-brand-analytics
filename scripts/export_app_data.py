"""Export the slim, committed data the Streamlit app reads (app/data/).

The research tables under data/processed/ stay local (gitignored). The app needs only:
- videos.parquet: the catalog columns (catalog.COLUMNS) plus the caption and visual
  embeddings used for similar-video retrieval.
- comment_sentiment.parquet: one row per scored comment with only its video_id and VADER
  score, so per-video audience sentiment can be computed. No comment text, comment IDs or
  commenter identities are exported.

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

import catalog  # noqa: E402

SOURCE_DIR = ROOT / "data" / "processed" / "feature"
VIDEO_SOURCE = SOURCE_DIR / "feature_table.parquet"
COMMENT_SOURCE = SOURCE_DIR / "comment_feature_table.parquet"
EMBEDDINGS = ["text_embedding", "visual_embedding"]


def main() -> None:
    catalog.APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    videos = pd.read_parquet(VIDEO_SOURCE)
    keep = [c for c in dict.fromkeys(catalog.COLUMNS + EMBEDDINGS) if c in videos.columns]
    videos[keep].to_parquet(catalog.FEATURE_TABLE, index=False, compression="zstd")

    comments = pd.read_parquet(COMMENT_SOURCE, columns=["video_id", "sentiment_score"])
    comments = comments.dropna(subset=["sentiment_score"])
    comments.to_parquet(catalog.COMMENT_SENTIMENT, index=False, compression="zstd")

    for path, df in ((catalog.FEATURE_TABLE, videos[keep]), (catalog.COMMENT_SENTIMENT, comments)):
        size = path.stat().st_size / 1e6
        print(f"{path.relative_to(ROOT)}: {len(df):,} rows, {len(df.columns)} columns, {size:.1f} MB")


if __name__ == "__main__":
    main()
