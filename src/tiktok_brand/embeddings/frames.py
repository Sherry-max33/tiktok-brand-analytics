"""Shared video frame resolution for visual features.

Pipeline (no cover required):
  1) Assemble TikTok URL from video_id (+ optional @username)
  2) Download MP4 once (yt-dlp), cache under frames/{video_id}/video.mp4
  3) Extract **4 frames at 20% / 40% / 60% / 80%** of duration (OpenCV)
  4) Reuse the same frame paths for CLIP + appearance_type

Cache layout:
  data/processed/frames/{video_id}/video.mp4
  data/processed/frames/{video_id}/frame_000.jpg … frame_003.jpg
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_FRAMES_DIR = Path("data/processed/frames")
DEFAULT_FRAME_FRACTIONS: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_video_id(video_id: str) -> str:
    cleaned = _SAFE_ID_RE.sub("_", str(video_id).strip())
    return cleaned or "unknown"


def video_frame_dir(video_id: str, frames_root: Path | str = DEFAULT_FRAMES_DIR) -> Path:
    return Path(frames_root) / _safe_video_id(video_id)


def build_tiktok_video_url(
    video_id: Optional[str],
    *,
    author_username: Optional[str] = None,
    video_url: Optional[str] = None,
    page_url: Optional[str] = None,
) -> Optional[str]:
    """Prefer existing page_url / video_url; else assemble from id (+ @user)."""
    if page_url and str(page_url).strip():
        return str(page_url).strip()
    if video_url and str(video_url).strip():
        return str(video_url).strip()
    if not video_id:
        return None
    vid = str(video_id).strip()
    if author_username and str(author_username).strip():
        user = str(author_username).strip().lstrip("@")
        return f"https://www.tiktok.com/@{user}/video/{vid}"
    return f"https://www.tiktok.com/video/{vid}"


def list_cached_frames(video_id: str, frames_root: Path | str = DEFAULT_FRAMES_DIR) -> List[Path]:
    d = video_frame_dir(video_id, frames_root)
    if not d.is_dir():
        return []
    return sorted(d.glob("frame_*.jpg")) + sorted(d.glob("frame_*.png"))


def _cached_video_path(video_id: str, frames_root: Path | str = DEFAULT_FRAMES_DIR) -> Optional[Path]:
    d = video_frame_dir(video_id, frames_root)
    for name in ("video.mp4", "video.webm", "video.mkv"):
        p = d / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def download_tiktok_video(
    url: str,
    dest_mp4: Path,
    *,
    timeout_sec: int = 180,
    max_height: int = 480,
    cookies_from_browser: Optional[str] = None,
) -> Path:
    """
    Download a TikTok page/video URL to ``dest_mp4`` via yt-dlp.

    Caps resolution at ``max_height`` (default 480p) — enough for frame
    sampling / CLIP, avoids HD downloads.
    Requires ``yt-dlp`` on PATH (or ``python -m yt_dlp``).

    ``cookies_from_browser``: yt-dlp browser name (e.g. ``chrome``, ``safari``)
    when the host IP is blocked without a logged-in session.
    """
    dest_mp4 = Path(dest_mp4)
    dest_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(dest_mp4.with_suffix("")) + ".%(ext)s"
    # Prefer <= max_height; fall back to worst acceptable progressive file
    fmt = (
        f"bv*[height<={max_height}]+ba/"
        f"b[height<={max_height}]/"
        f"wv*[height<={max_height}]+ba/"
        f"w[height<={max_height}]/b"
    )

    base_args = [
        "-f",
        fmt,
        "-S",
        f"res:{max_height},+size",
        "--merge-output-format",
        "mp4",
        "-o",
        out_tmpl,
        "--no-playlist",
        "--no-warnings",
        "--impersonate",
        "chrome",
    ]
    if cookies_from_browser and str(cookies_from_browser).strip():
        base_args.extend(["--cookies-from-browser", str(cookies_from_browser).strip()])
    base_args.append(url)

    def _run_logged(argv: List[str]) -> None:
        try:
            subprocess.run(
                argv,
                check=True,
                timeout=timeout_sec,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or exc.stdout or "").strip()
            if err:
                snippet = err[-800:] if len(err) > 800 else err
                raise RuntimeError(f"yt-dlp failed (exit {exc.returncode}): {snippet}") from exc
            raise

    try:
        _run_logged(["yt-dlp", *base_args])
    except FileNotFoundError:
        _run_logged([sys.executable, "-m", "yt_dlp", *base_args])
    except RuntimeError:
        # CLI may exist but fail; retry module entrypoint once
        _run_logged([sys.executable, "-m", "yt_dlp", *base_args])

    # yt-dlp may write .mp4 next to template
    if dest_mp4.is_file() and dest_mp4.stat().st_size > 0:
        return dest_mp4
    # find newest media in dir matching stem
    stem = dest_mp4.with_suffix("").name
    candidates = sorted(dest_mp4.parent.glob(stem + ".*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if c.suffix.lower() in {".mp4", ".webm", ".mkv"} and c.stat().st_size > 0:
            if c != dest_mp4:
                shutil.move(str(c), str(dest_mp4))
            return dest_mp4
    raise FileNotFoundError(f"yt-dlp did not produce a video for {url}")


def _frame_indices(total_frames: int, fractions: Sequence[float]) -> List[int]:
    if total_frames <= 0:
        return []
    if total_frames <= len(fractions):
        return list(range(total_frames))
    indices = []
    for frac in fractions:
        f = min(max(float(frac), 0.0), 0.999)
        idx = int(total_frames * f)
        idx = min(max(idx, 0), total_frames - 1)
        indices.append(idx)
    # dedupe while preserving order
    seen = set()
    out: List[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def extract_frames_at_fractions(
    video_path: str | Path,
    out_dir: Path,
    *,
    fractions: Sequence[float] = DEFAULT_FRAME_FRACTIONS,
) -> List[Path]:
    """
    Extract frames at relative positions (default 20/40/60/80%) via OpenCV.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        logger.warning("opencv-python (cv2) not installed; cannot extract frames")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = _frame_indices(total, fractions)
    paths: List[Path] = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        path = out_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    cap.release()
    return paths


def resolve_video_frames(
    video_id: str,
    *,
    video_url: Optional[str] = None,
    page_url: Optional[str] = None,
    author_username: Optional[str] = None,
    video_path: Optional[str] = None,
    cover_url: Optional[str] = None,
    cover_path: Optional[str] = None,
    frames_root: Path | str = DEFAULT_FRAMES_DIR,
    fractions: Sequence[float] = DEFAULT_FRAME_FRACTIONS,
    download_video: bool = False,
    max_height: int = 480,
    cookies_from_browser: Optional[str] = None,
    force: bool = False,
) -> List[str]:
    """
    Return local frame paths for one video (cached).

    Priority:
      1) Cached frame_*.jpg
      2) Local ``video_path`` → extract at fractions
      3) If ``download_video``: assemble/download TikTok URL (≤ max_height) → extract
      4) Optional cover fallback (single image) — last resort only
    """
    root = Path(frames_root)
    out_dir = video_frame_dir(video_id, root)

    if not force:
        cached = list_cached_frames(video_id, root)
        if len(cached) >= len(fractions):
            return [str(p) for p in cached[: len(fractions)]]
        if cached:
            return [str(p) for p in cached]

    local_video: Optional[Path] = None
    if video_path and Path(video_path).is_file():
        local_video = Path(video_path)
    else:
        local_video = _cached_video_path(video_id, root)

    if local_video is None and download_video:
        url = build_tiktok_video_url(
            video_id,
            author_username=author_username,
            video_url=video_url,
            page_url=page_url,
        )
        if url:
            dest = out_dir / "video.mp4"
            try:
                local_video = download_tiktok_video(
                    url,
                    dest,
                    max_height=max_height,
                    cookies_from_browser=cookies_from_browser,
                )
            except Exception as exc:
                logger.warning("video download failed for %s (%s): %s", video_id, url, exc)

    frames: List[Path] = []
    if local_video is not None:
        frames = extract_frames_at_fractions(local_video, out_dir, fractions=fractions)

    # Last resort: cover image as a single frame (not preferred)
    if not frames:
        out_dir.mkdir(parents=True, exist_ok=True)
        cover_dest = out_dir / "frame_000.jpg"
        try:
            if cover_path and Path(cover_path).is_file():
                shutil.copyfile(cover_path, cover_dest)
                frames = [cover_dest]
            elif cover_url and str(cover_url).startswith(("http://", "https://")):
                import urllib.request

                with urllib.request.urlopen(str(cover_url), timeout=30) as resp:
                    cover_dest.write_bytes(resp.read())
                frames = [cover_dest]
        except Exception as exc:
            logger.debug("cover fallback failed for %s: %s", video_id, exc)

    return [str(p) for p in frames]


def resolve_frames_batch(
    video_ids: Sequence[Optional[str]],
    *,
    video_urls: Optional[Sequence[Optional[str]]] = None,
    page_urls: Optional[Sequence[Optional[str]]] = None,
    author_usernames: Optional[Sequence[Optional[str]]] = None,
    video_paths: Optional[Sequence[Optional[str]]] = None,
    cover_urls: Optional[Sequence[Optional[str]]] = None,
    cover_paths: Optional[Sequence[Optional[str]]] = None,
    frames_root: Path | str = DEFAULT_FRAMES_DIR,
    fractions: Sequence[float] = DEFAULT_FRAME_FRACTIONS,
    download_video: bool = False,
    max_height: int = 480,
    cookies_from_browser: Optional[str] = None,
) -> List[List[str]]:
    n = len(video_ids)
    video_urls = video_urls or [None] * n
    page_urls = page_urls or [None] * n
    author_usernames = author_usernames or [None] * n
    video_paths = video_paths or [None] * n
    cover_urls = cover_urls or [None] * n
    cover_paths = cover_paths or [None] * n
    out: List[List[str]] = []
    for i in range(n):
        vid = video_ids[i]
        if not vid:
            out.append([])
            continue
        out.append(
            resolve_video_frames(
                str(vid),
                video_url=video_urls[i] if i < len(video_urls) else None,
                page_url=page_urls[i] if i < len(page_urls) else None,
                author_username=author_usernames[i] if i < len(author_usernames) else None,
                video_path=video_paths[i] if i < len(video_paths) else None,
                cover_url=cover_urls[i] if i < len(cover_urls) else None,
                cover_path=cover_paths[i] if i < len(cover_paths) else None,
                frames_root=frames_root,
                fractions=fractions,
                download_video=download_video,
                max_height=max_height,
                cookies_from_browser=cookies_from_browser,
            )
        )
    return out
