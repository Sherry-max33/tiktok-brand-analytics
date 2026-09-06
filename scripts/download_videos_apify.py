"""Download TikTok MP4s via Apify postURLs (budget-limited).

Uses clockworks/tiktok-video-scraper with shouldDownloadVideos=true,
then copies media into data/processed/frames/{video_id}/video.mp4.
Optionally extracts 20/40/60/80% frames locally.

Skips:
  - videos that already have a local MP4
  - video_ids listed in the fail skip file (no retry by default)

Estimated Free-tier cost ≈ $0.0043 / video (result + video-download).

Usage (repo root):
  PYTHONPATH=src python -m scripts.download_videos_apify --limit 5
  PYTHONPATH=src python -m scripts.download_videos_apify --max-videos 900
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from tiktok_brand.crawl.apify_video_client import (
    ApifyQuotaExceeded,
    ApifyTokenPool,
    fetch_videos_by_post_urls,
    save_apify_video_item,
    video_id_from_apify_item,
)
from tiktok_brand.embeddings.frames import (
    DEFAULT_FRAME_FRACTIONS,
    extract_frames_at_fractions,
    list_cached_frames,
    video_frame_dir,
)

DEFAULT_FAIL_PATH = Path("data/processed/frames/_download_failed_ids.txt")
DEFAULT_SOFT_FAIL_PATH = Path("data/processed/frames/_download_failed_soft_ids.txt")
_FAIL_LINE_RE = re.compile(r"!\s+(?:no media|missing Apify item):\s*(\d+)")
_PRIVATE_RE = re.compile(r"Post not found or is private.*?/video/(\d+)")
_SCRAPE_OK_RE = re.compile(r"Post scraped successfully.*?/video/(\d+)")
_NO_MEDIA_RE = re.compile(r"!\s+no media:\s*(\d+)")
_MISSING_RE = re.compile(r"!\s+missing Apify item:\s*(\d+)")


def _already_have_video(video_id: str, frames_root: Path) -> bool:
    d = video_frame_dir(video_id, frames_root)
    for name in ("video.mp4", "video.webm", "video.mkv"):
        p = d / name
        if p.is_file() and p.stat().st_size > 0:
            return True
    return False


def _load_fail_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        vid = line.strip().split("#", 1)[0].strip()
        if vid:
            out.add(vid)
    return out


def _write_id_file(path: Path, ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{vid}\n" for vid in sorted(ids)), encoding="utf-8")


def _append_fail_id(path: Path, video_id: str, known: set[str]) -> None:
    if video_id in known:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{video_id}\n")
    known.add(video_id)


def _classify_fails_from_logs(log_paths: list[Path]) -> tuple[set[str], set[str]]:
    """Return (hard_fail_ids, soft_fail_ids).

    Soft = scraped successfully once but local media save failed (worth a small retry).
    Hard = private/not found / missing item / never scraped OK.
    """
    private: set[str] = set()
    scraped_ok: set[str] = set()
    no_media: set[str] = set()
    missing: set[str] = set()
    for log_path in log_paths:
        if not log_path.is_file():
            continue
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        private |= set(_PRIVATE_RE.findall(text))
        scraped_ok |= set(_SCRAPE_OK_RE.findall(text))
        no_media |= set(_NO_MEDIA_RE.findall(text))
        missing |= set(_MISSING_RE.findall(text))
        for m in _FAIL_LINE_RE.finditer(text):
            # already covered by no_media/missing patterns
            pass

    soft = (no_media - private) & scraped_ok
    hard = private | missing | (no_media - soft)
    # any remaining script failures without classification stay hard via backfill
    return hard, soft


def _backfill_fails_from_logs(
    fail_path: Path,
    soft_path: Path,
    log_paths: list[Path],
) -> tuple[int, int]:
    """Merge failure ids from prior download logs into hard/soft skip files."""
    hard, soft = _classify_fails_from_logs(log_paths)
    # also keep any previously recorded hard fails
    hard |= _load_fail_ids(fail_path) - soft
    soft |= _load_fail_ids(soft_path)
    # soft should not duplicate into hard
    hard -= soft
    before_h, before_s = len(_load_fail_ids(fail_path)), len(_load_fail_ids(soft_path))
    _write_id_file(fail_path, hard)
    _write_id_file(soft_path, soft)
    return len(hard) - before_h, len(soft) - before_s


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TikTok videos via Apify")
    parser.add_argument("--limit", type=int, default=None, help="Alias for --max-videos")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=2000,
        help="Max videos to request (default 2000)",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="postURLs per Apify run")
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        default=True,
        help="Extract 4 frames after MP4 save (default on)",
    )
    parser.add_argument(
        "--no-extract-frames",
        action="store_true",
        help="Skip local frame extraction",
    )
    parser.add_argument(
        "--prefer-official",
        action="store_true",
        help="Sort official accounts (nike/adidas/jumpman23) first",
    )
    parser.add_argument(
        "--fail-file",
        type=Path,
        default=DEFAULT_FAIL_PATH,
        help="Persistent skip list for hard-failed video_ids (private/not found)",
    )
    parser.add_argument(
        "--soft-fail-file",
        type=Path,
        default=DEFAULT_SOFT_FAIL_PATH,
        help="Skip list for soft fails (scraped OK but media save failed)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Do NOT skip hard/soft fail files (retry everything missing)",
    )
    parser.add_argument(
        "--retry-soft-sample",
        type=int,
        default=None,
        help="Only retry N soft-fail ids (suspected transient). Ignores normal queue.",
    )
    parser.add_argument(
        "--then-retry-soft",
        action="store_true",
        help="After never-tried IDs, also queue soft-fail IDs (hard fails still skipped).",
    )
    parser.add_argument(
        "--backfill-fail-logs",
        action="store_true",
        default=True,
        help="Merge failure ids from /tmp/apify_video_dl*.log into fail files (default on)",
    )
    parser.add_argument(
        "--no-backfill-fail-logs",
        action="store_true",
        help="Skip merging failure ids from prior logs",
    )
    args = parser.parse_args()

    max_videos = int(args.limit if args.limit is not None else args.max_videos)
    extract_frames = bool(args.extract_frames) and not bool(args.no_extract_frames)
    batch_size = max(1, int(args.batch_size))
    fail_path: Path = args.fail_file
    soft_path: Path = args.soft_fail_file
    skip_failed = not bool(args.retry_failed)
    retry_soft_n = args.retry_soft_sample

    log_paths = [
        Path("/tmp/apify_video_dl.log"),
        Path("/tmp/apify_video_dl_round2.log"),
        Path("/tmp/apify_video_dl_round3.log"),
        Path("/tmp/apify_video_dl_round4.log"),
    ]
    if args.backfill_fail_logs and not args.no_backfill_fail_logs:
        dh, ds = _backfill_fails_from_logs(fail_path, soft_path, log_paths)
        print(
            f"Backfilled fails → hard Δ{dh} ({fail_path}), soft Δ{ds} ({soft_path})",
            flush=True,
        )

    try:
        token_pool = ApifyTokenPool()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Apify tokens loaded: {len(token_pool.tokens)} "
        f"(active={token_pool.fingerprint})",
        flush=True,
    )

    hard_ids = _load_fail_ids(fail_path)
    soft_ids = _load_fail_ids(soft_path)
    then_soft = bool(args.then_retry_soft) and retry_soft_n is None
    # Always skip hard when skip_failed. Soft is skipped unless --then-retry-soft
    # (appended after never-tried) or --retry-soft-sample / --retry-failed.
    if skip_failed:
        skip_ids = set(hard_ids) if then_soft else (hard_ids | soft_ids)
        print(
            f"Skip-failed enabled: hard={len(hard_ids)} "
            f"soft={'retry-after-fresh' if then_soft else f'skip({len(soft_ids)})'}",
            flush=True,
        )
    else:
        skip_ids = set()

    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    feature_dir = Path(project_cfg["output"].get("feature_dir", "data/processed/feature"))
    rules_path = Path(project_cfg["output"].get("feature_rules_cfg", "configs/feature_rules.yaml"))
    vis_cfg = yaml.safe_load(rules_path.read_text(encoding="utf-8")).get("visual_embedding", {})
    frames_root = Path(vis_cfg.get("frames_dir", "data/processed/frames"))
    fractions = tuple(vis_cfg.get("frame_fractions") or list(DEFAULT_FRAME_FRACTIONS))

    video_path = feature_dir / "feature_table.parquet"
    print(f"Feature table: {video_path}", flush=True)
    df = pd.read_parquet(video_path, columns=["video_id", "page_url", "author_username"])
    by_vid = {str(r.video_id): r for _, r in df.iterrows() if pd.notna(r.get("video_id"))}

    skipped_fail = 0
    rows = []

    if retry_soft_n is not None:
        # Small probe of suspected-transient failures only
        candidates = [
            vid
            for vid in sorted(soft_ids)
            if vid in by_vid and not _already_have_video(vid, frames_root)
        ][: max(0, int(retry_soft_n))]
        print(
            f"Soft-retry sample: {len(candidates)} / soft_pool={len(soft_ids)}",
            flush=True,
        )
        for vid in candidates:
            r = by_vid[vid]
            url = r.get("page_url")
            if not url or not str(url).strip():
                user = r.get("author_username")
                if pd.notna(user) and str(user).strip():
                    url = f"https://www.tiktok.com/@{str(user).lstrip('@')}/video/{vid}"
                else:
                    url = f"https://www.tiktok.com/video/{vid}"
            rows.append(
                {
                    "video_id": vid,
                    "page_url": str(url).strip(),
                    "author_username": r.get("author_username"),
                }
            )
        # allow these soft ids for this run
        skip_ids -= set(candidates)
        max_videos = len(rows)
    else:
        def _row_for(vid: str) -> dict | None:
            r = by_vid.get(vid)
            if r is None:
                return None
            url = r.get("page_url")
            if not url or not str(url).strip():
                user = r.get("author_username")
                if pd.notna(user) and str(user).strip():
                    url = f"https://www.tiktok.com/@{str(user).lstrip('@')}/video/{vid}"
                else:
                    url = f"https://www.tiktok.com/video/{vid}"
            return {
                "video_id": vid,
                "page_url": str(url).strip(),
                "author_username": r.get("author_username"),
            }

        fresh_rows: list[dict] = []
        soft_rows: list[dict] = []
        for _, r in df.iterrows():
            vid = str(r["video_id"]) if pd.notna(r.get("video_id")) else ""
            if not vid:
                continue
            if _already_have_video(vid, frames_root):
                continue
            if skip_failed and vid in hard_ids:
                skipped_fail += 1
                continue
            if skip_failed and (not then_soft) and vid in soft_ids:
                skipped_fail += 1
                continue
            item = _row_for(vid)
            if item is None:
                continue
            if then_soft and vid in soft_ids:
                soft_rows.append(item)
            else:
                fresh_rows.append(item)

        if args.prefer_official:
            official = {"nike", "adidas", "jumpman23"}

            def _key(row: dict) -> tuple:
                u = str(row.get("author_username") or "").lstrip("@").lower()
                return (0 if u in official else 1, u)

            fresh_rows.sort(key=_key)
            soft_rows.sort(key=_key)

        # Never-tried first; soft only after (and only with --then-retry-soft).
        rows = fresh_rows + soft_rows
        print(
            f"Queue order: never-tried={len(fresh_rows)} soft={len(soft_rows)} "
            f"(hard skipped={len(hard_ids) if skip_failed else 0})",
            flush=True,
        )
        rows = rows[:max_videos]
    print(
        f"To download: {len(rows)} "
        f"(skip existing MP4s; skip_failed={skipped_fail}; max_videos={max_videos})",
        flush=True,
    )
    if not rows:
        print("Nothing to do.")
        return

    est = len(rows) * 0.0043
    print(f"Estimated Apify Free-tier cost ≈ ${est:.2f}", flush=True)

    stats = {
        "apify_items": 0,
        "saved_mp4": 0,
        "frames_ok": 0,
        "fail": 0,
        "skip_failed": skipped_fail,
        "token_switches": 0,
    }
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        urls = [b["page_url"] for b in batch]
        print(
            f"\nApify batch {start // batch_size + 1}: "
            f"{len(urls)} URLs ({start + 1}–{start + len(urls)} / {len(rows)}) "
            f"[token={token_pool.fingerprint}]",
            flush=True,
        )

        items = None
        while True:
            try:
                items, _run = fetch_videos_by_post_urls(
                    urls,
                    download_videos=True,
                    download_covers=False,
                    api_token=token_pool.current,
                )
                break
            except ApifyQuotaExceeded as exc:
                print(f"  quota hit: {exc}", flush=True)
                try:
                    token_pool.advance()
                except ApifyQuotaExceeded as done:
                    print(f"\nStopped: {done}", flush=True)
                    print(f"Done: {stats}", flush=True)
                    return
                stats["token_switches"] += 1
                print(
                    f"  switched token → {token_pool.fingerprint}; "
                    f"retrying same batch",
                    flush=True,
                )

        assert items is not None
        stats["apify_items"] += len(items)
        active_token = token_pool.current

        by_id = {}
        for item in items:
            vid = video_id_from_apify_item(item)
            if vid:
                by_id[vid] = item

        for b in batch:
            vid = b["video_id"]
            out_dir = video_frame_dir(vid, frames_root)
            dest = out_dir / "video.mp4"
            item = by_id.get(vid)
            if item is None:
                stats["fail"] += 1
                print(f"  ! missing Apify item: {vid}", flush=True)
                if skip_failed:
                    _append_fail_id(fail_path, vid, hard_ids)
                    soft_ids.discard(vid)
                continue

            path = save_apify_video_item(item, dest, api_token=active_token)
            if path is None:
                stats["fail"] += 1
                print(f"  ! no media: {vid}", flush=True)
                if skip_failed:
                    # scraped but media missing → soft (eligible for sample retry)
                    _append_fail_id(soft_path, vid, soft_ids)
                    hard_ids.discard(vid)
                continue
            stats["saved_mp4"] += 1
            # success: drop from soft list if present
            if vid in soft_ids:
                soft_ids.discard(vid)
                _write_id_file(soft_path, soft_ids)

            if extract_frames:
                existing = list_cached_frames(vid, frames_root)
                if len(existing) >= len(fractions):
                    stats["frames_ok"] += 1
                else:
                    frames = extract_frames_at_fractions(path, out_dir, fractions=fractions)
                    if frames:
                        stats["frames_ok"] += 1

        print(
            f"  progress saved_mp4={stats['saved_mp4']} "
            f"frames={stats['frames_ok']} fail={stats['fail']}",
            flush=True,
        )

    print(f"\nDone: {stats}", flush=True)


if __name__ == "__main__":
    main()
