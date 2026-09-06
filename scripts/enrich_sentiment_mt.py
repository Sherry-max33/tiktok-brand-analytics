"""Enrich feature tables: MT + VADER for previously unscored non-English rows.

Only updates rows with sentiment_method == not_scored_non_en (and pending MT).
Does not recompute embeddings.

Usage (repo root):
  PYTHONPATH=src python -m scripts.enrich_sentiment_mt
  PYTHONPATH=src python -m scripts.enrich_sentiment_mt --videos-only
  PYTHONPATH=src python -m scripts.enrich_sentiment_mt --comments-only --limit 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from tiktok_brand.etl.feature_table import write_partitioned_parquet
from tiktok_brand.etl.sentiment_rules import (
    METHOD_NOT_SCORED_NON_EN,
    METHOD_VADER_VIA_MT,
    score_caption_sentiment,
)

SENT_COLS = [
    "caption_original",
    "caption_lang",
    "caption_lang_confidence",
    "has_mixed_language",
    "is_emoji_only",
    "caption_en",
    "translation_status",
    "sentiment_score",
    "sentiment_method",
]


def _enrich_rows(
    df: pd.DataFrame,
    text_col: str,
    *,
    rules_path: str,
    sleep_sec: float,
    limit: int | None,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    mask = out["sentiment_method"].astype(str) == METHOD_NOT_SCORED_NON_EN
    idxs = out.index[mask].tolist()
    if limit is not None:
        idxs = idxs[: int(limit)]

    stats = {
        "candidates": int(mask.sum()),
        "attempted": len(idxs),
        "vader_via_mt": 0,
        "failed": 0,
        "other": 0,
    }

    for n, idx in enumerate(idxs, start=1):
        text = out.at[idx, text_col] if text_col in out.columns else ""
        result = score_caption_sentiment(text, rules_path=rules_path)
        for col in SENT_COLS:
            if col in out.columns:
                out.at[idx, col] = result.get(col)
            elif col in result:
                out.at[idx, col] = result.get(col)

        method = result.get("sentiment_method")
        if method == METHOD_VADER_VIA_MT:
            stats["vader_via_mt"] += 1
        elif result.get("translation_status") == "failed" or method == METHOD_NOT_SCORED_NON_EN:
            stats["failed"] += 1
        else:
            stats["other"] += 1

        if n % 25 == 0 or n == len(idxs):
            print(
                f"  … {n}/{len(idxs)} (mt_ok={stats['vader_via_mt']}, fail={stats['failed']})",
                flush=True,
            )
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich sentiment via MT for non-EN rows")
    parser.add_argument("--videos-only", action="store_true")
    parser.add_argument("--comments-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to enrich per table")
    parser.add_argument("--sleep", type=float, default=0.35, help="Seconds between MT calls")
    args = parser.parse_args()

    do_videos = not args.comments_only
    do_comments = not args.videos_only

    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    feature_dir = Path(project_cfg["output"].get("feature_dir", "data/processed/feature"))
    rules_path = project_cfg["output"].get("feature_rules_cfg", "configs/feature_rules.yaml")

    # Force MT on for this enrich pass
    cfg = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8"))
    sent = dict(cfg.get("sentiment") or {})
    if not sent.get("mt_enabled", True):
        print("Note: enabling mt_enabled for this enrich run (config had it false).")
    # score_caption_sentiment reads yaml; ensure true in file for this session via temp
    import tempfile

    sent["mt_enabled"] = True
    cfg = dict(cfg)
    cfg["sentiment"] = sent
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    rules_path = tmp.name

    try:
        if do_videos:
            video_path = feature_dir / "feature_table.parquet"
            print(f"Videos: {video_path}", flush=True)
            vdf = pd.read_parquet(video_path)
            text_col = "caption_original" if "caption_original" in vdf.columns else "caption_raw"
            vdf, stats = _enrich_rows(
                vdf, text_col, rules_path=rules_path, sleep_sec=args.sleep, limit=args.limit
            )
            vdf.to_parquet(video_path, index=False)
            write_partitioned_parquet(vdf, feature_dir / "feature_table", partition_cols=["brand"])
            print(f"Videos done: {stats}")

        if do_comments:
            comment_path = feature_dir / "comment_feature_table.parquet"
            print(f"Comments: {comment_path}")
            cdf = pd.read_parquet(comment_path)
            cdf, stats = _enrich_rows(
                cdf,
                "comment_text",
                rules_path=rules_path,
                sleep_sec=args.sleep,
                limit=args.limit,
            )
            cdf.to_parquet(comment_path, index=False)
            print(f"Comments done: {stats}")
    finally:
        Path(rules_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
