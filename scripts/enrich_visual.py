"""Enrich video feature table: download MP4 → extract frames → CLIP + zero-shot axes.

Resumable: skips rows that already have visual_embedding_method == clip_visual,
and reuses cached frames under data/processed/frames/{video_id}/.

Usage (repo root):
  PYTHONPATH=src python -m scripts.enrich_visual --from-cache
  PYTHONPATH=src python -m scripts.enrich_visual --from-cache --classify-only
  PYTHONPATH=src VISUAL_DOWNLOAD=1 python -m scripts.enrich_visual --limit 5
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import yaml

from tiktok_brand.embeddings.frames import list_cached_frames, resolve_video_frames
from tiktok_brand.embeddings.visual_classify import classification_column_names
from tiktok_brand.embeddings.visual_embed import (
    METHOD_CLIP,
    compute_visual_features,
)
from tiktok_brand.etl.feature_table import write_partitioned_parquet

BASE_VIS_COLS = [
    "visual_embedding",
    "visual_embedding_method",
    "visual_embedding_model",
    "frame_paths",
    "clip_alignment_mean",
    "clip_alignment_max",
    "clip_alignment_min",
    "clip_alignment_std",
    "clip_text_embedding",
]


def _vis_cols(rules_path: str) -> list[str]:
    return BASE_VIS_COLS + classification_column_names(rules_path)


def _has_clip_embedding(row: pd.Series) -> bool:
    method = row.get("visual_embedding_method")
    if method is not None and not (isinstance(method, float) and pd.isna(method)):
        return str(method) == METHOD_CLIP
    emb = row.get("visual_embedding")
    if emb is None or (isinstance(emb, float) and pd.isna(emb)):
        return False
    if isinstance(emb, str):
        return bool(emb.strip())
    if isinstance(emb, (list, tuple)):
        return len(emb) > 0
    try:
        import numpy as np

        if isinstance(emb, np.ndarray):
            return emb.size > 0
    except Exception:
        pass
    return False


def _needs_visual(row: pd.Series, *, classify_only: bool) -> bool:
    if classify_only:
        status = row.get("visual_classification_status")
        if status is None or (isinstance(status, float) and pd.isna(status)):
            return True
        return str(status) == "unknown"
    return not _has_clip_embedding(row)


def _has_local_frames(video_id: str, frames_root: Path, n_needed: int = 1) -> bool:
    paths = list_cached_frames(str(video_id), frames_root)
    return len(paths) >= n_needed


def _ensure_cols(df: pd.DataFrame, rules_path: str) -> pd.DataFrame:
    out = df.copy()
    for col in _vis_cols(rules_path):
        if col not in out.columns:
            if col == "frame_paths":
                out[col] = [[] for _ in range(len(out))]
            else:
                out[col] = None
    return out


def _normalize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _emb(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, str):
            return None if not x.strip() else x
        if isinstance(x, (list, tuple)) and len(x) == 0:
            return None
        return x

    def _paths(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []
        if isinstance(x, str):
            return [] if not x.strip() else [x]
        if isinstance(x, (list, tuple)):
            return list(x)
        return []

    if "visual_embedding" in out.columns:
        out["visual_embedding"] = out["visual_embedding"].map(_emb)
    if "clip_text_embedding" in out.columns:
        out["clip_text_embedding"] = out["clip_text_embedding"].map(_emb)
    if "frame_paths" in out.columns:
        out["frame_paths"] = out["frame_paths"].map(_paths)
    if "visual_embedding_method" in out.columns:
        out["visual_embedding_method"] = out["visual_embedding_method"].where(
            out["visual_embedding_method"].notna(), None
        )
    return out


def _save(df: pd.DataFrame, video_path: Path, feature_dir: Path) -> None:
    clean = _normalize_for_parquet(df)
    if "appearance_type" in clean.columns:
        clean = clean.drop(columns=["appearance_type"])
    clean.to_parquet(video_path, index=False)
    write_partitioned_parquet(clean, feature_dir / "feature_table", partition_cols=["brand"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download videos, extract frames, CLIP embed + zero-shot classify"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--frames-only", action="store_true")
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Recompute visual_format/setting from cached frames (skip download)",
    )
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--cookies-from-browser", default=None)
    args = parser.parse_args()

    from_cache = bool(args.from_cache) or bool(args.classify_only)
    if from_cache:
        os.environ["VISUAL_DOWNLOAD"] = "0"
    else:
        os.environ.setdefault("VISUAL_DOWNLOAD", "1")

    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    feature_dir = Path(project_cfg["output"].get("feature_dir", "data/processed/feature"))
    rules_path = project_cfg["output"].get("feature_rules_cfg", "configs/feature_rules.yaml")
    video_path = feature_dir / "feature_table.parquet"
    vis_cols = _vis_cols(rules_path)

    print(f"Videos: {video_path}", flush=True)
    df = _ensure_cols(pd.read_parquet(video_path), rules_path)

    vis_cfg = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8")).get(
        "visual_embedding", {}
    )
    frames_root = Path(vis_cfg.get("frames_dir", "data/processed/frames"))
    fractions = tuple(vis_cfg.get("frame_fractions") or [0.2, 0.4, 0.6, 0.8])
    max_height = int(vis_cfg.get("download_max_height", 480))

    idxs = [
        i
        for i in df.index
        if _needs_visual(df.loc[i], classify_only=bool(args.classify_only))
    ]
    if from_cache:
        before = len(idxs)
        idxs = [
            i
            for i in idxs
            if _has_local_frames(str(df.at[i, "video_id"]), frames_root, 1)
        ]
        print(f"From-cache filter: {len(idxs)}/{before} have local frames", flush=True)
    if args.limit is not None:
        idxs = idxs[: int(args.limit)]

    print(
        f"Candidates: {len(idxs)} (of {len(df)}; from_cache={from_cache}; "
        f"classify_only={bool(args.classify_only)})",
        flush=True,
    )
    if not idxs:
        print("Nothing to do.")
        return

    cookies_from_browser = (
        args.cookies_from_browser
        or os.environ.get("YTDLP_COOKIES_FROM_BROWSER")
        or vis_cfg.get("cookies_from_browser")
        or None
    )
    if cookies_from_browser and not from_cache:
        print(f"Using cookies-from-browser: {cookies_from_browser}", flush=True)

    stats = {"ok_frames": 0, "fail_frames": 0, "clip_ok": 0, "clip_skip": 0}
    batch_size = max(1, int(args.batch_size))
    checkpoint_every = max(1, int(args.checkpoint_every))
    processed = 0
    download_video = not from_cache

    for start in range(0, len(idxs), batch_size):
        batch_idxs = idxs[start : start + batch_size]
        frame_paths_batch: list[list[str]] = []

        for i, idx in enumerate(batch_idxs, start=1):
            row = df.loc[idx]
            vid = str(row.get("video_id") or "")
            n = start + i
            try:
                cached = [str(p) for p in list_cached_frames(vid, frames_root)]
                if from_cache or cached:
                    paths = cached
                else:
                    paths = resolve_video_frames(
                        vid,
                        video_url=row.get("video_url"),
                        page_url=row.get("page_url"),
                        author_username=row.get("author_username"),
                        video_path=row.get("video_path") if "video_path" in df.columns else None,
                        cover_url=row.get("cover_url") if "cover_url" in df.columns else None,
                        cover_path=row.get("cover_path") if "cover_path" in df.columns else None,
                        frames_root=frames_root,
                        fractions=fractions,
                        download_video=download_video,
                        max_height=max_height,
                        cookies_from_browser=cookies_from_browser,
                    )
            except Exception as exc:
                print(f"  ! {vid}: {exc}", flush=True)
                paths = []

            frame_paths_batch.append(paths)
            if paths:
                stats["ok_frames"] += 1
            else:
                stats["fail_frames"] += 1

            if n % 25 == 0 or n == len(idxs):
                print(
                    f"  … frames {n}/{len(idxs)} "
                    f"(ok={stats['ok_frames']}, fail={stats['fail_frames']})",
                    flush=True,
                )
            if (not from_cache) and args.sleep > 0:
                time.sleep(args.sleep)

        if args.frames_only:
            for idx, paths in zip(batch_idxs, frame_paths_batch):
                df.at[idx, "frame_paths"] = paths
            processed += len(batch_idxs)
            continue

        captions = [
            df.at[idx, "caption_en"] if "caption_en" in df.columns else None
            for idx in batch_idxs
        ]
        rows = compute_visual_features(
            [str(df.at[idx, "video_id"]) for idx in batch_idxs],
            frame_paths_per_video=frame_paths_batch,
            captions_en=captions,
            rules_path=rules_path,
        )
        for idx, result in zip(batch_idxs, rows):
            for col in vis_cols:
                if col in result:
                    df.at[idx, col] = result.get(col)
            if result.get("visual_embedding_method") == METHOD_CLIP:
                stats["clip_ok"] += 1
            else:
                stats["clip_skip"] += 1

        processed += len(batch_idxs)
        if processed % checkpoint_every < batch_size or processed >= len(idxs):
            print(f"  checkpoint @ {processed}/{len(idxs)} …", flush=True)
            _save(df, video_path, feature_dir)

    if args.frames_only:
        _save(df, video_path, feature_dir)

    print(f"Done: {stats}", flush=True)


if __name__ == "__main__":
    main()
