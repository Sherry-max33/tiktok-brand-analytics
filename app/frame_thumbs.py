"""Local frame thumbnails for list previews (Drive Desktop mount).

Reads ``{FRAMES_ROOT}/{video_id}/frame_000.jpg``. Resized JPEGs land under
``.cache/frame_thumbs/`` (gitignored). Tuned for sharp card previews (retina-friendly),
not tiny 480px scrapes. No Drive API, no oEmbed, no TikTok CDN.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Bump CACHE_TAG when encode settings change so stale soft thumbs are not reused.
CACHE_TAG = "hd1"
CACHE_DIR = REPO_ROOT / ".cache" / "frame_thumbs" / CACHE_TAG
# Card CSS width ≈ 300–360px; 2× retina → ~720. 1080 leaves headroom for WebP/R2 later.
THUMB_MAX_EDGE = 1080
THUMB_JPEG_QUALITY = 90


def _discover_drive_frames() -> Path:
    """Prefer ``FRAMES_ROOT``, else the first Google Drive for Desktop ``…/My Drive/frames``."""
    cloud = Path.home() / "Library" / "CloudStorage"
    if cloud.is_dir():
        matches = sorted(cloud.glob("GoogleDrive-*/My Drive/frames"))
        if matches:
            return matches[0]
    return REPO_ROOT / "data" / "processed" / "frames"


def frames_root() -> Path:
    override = (os.getenv("FRAMES_ROOT") or "").strip()
    if override:
        return Path(override).expanduser()
    return _discover_drive_frames()


def frame_path(video_id: str, *, root: Path | None = None) -> Path | None:
    """Return path to frame_000.jpg when present and non-empty."""
    base = root or frames_root()
    path = base / str(video_id) / "frame_000.jpg"
    if path.is_file() and path.stat().st_size > 0:
        return path
    return None


def _cache_path(video_id: str) -> Path:
    return CACHE_DIR / f"{video_id}.jpg"


def _ensure_thumb(video_id: str, *, root: Path | None = None, force: bool = False) -> Path | None:
    """Build or reuse a resized JPEG under .cache/frame_thumbs/{CACHE_TAG}/."""
    cached = _cache_path(video_id)
    if not force and cached.is_file() and cached.stat().st_size > 0:
        return cached
    src = frame_path(video_id, root=root)
    if src is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return src
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        # LANCZOS via default thumbnail resampling on modern Pillow.
        img.thumbnail((THUMB_MAX_EDGE, THUMB_MAX_EDGE))
        img.save(
            cached,
            format="JPEG",
            quality=THUMB_JPEG_QUALITY,
            optimize=True,
            progressive=True,
        )
    return cached if cached.is_file() else None


def thumb_data_uri(video_id: str, *, root: Path | None = None, force: bool = False) -> str:
    """data:image/jpeg;base64,… or empty string on miss/failure."""
    path = _ensure_thumb(video_id, root=root, force=force)
    if path is None:
        return ""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    if not raw:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


@dataclass
class ThumbStats:
    requested: int = 0
    hits: int = 0
    misses: int = 0
    cache_hits: int = 0
    disk_reads: int = 0
    drive_api_calls: int = 0  # always 0 for the Desktop-mount path
    elapsed_ms: float = 0.0
    per_id: dict[str, bool] = field(default_factory=dict)


def attach_frame_previews(
    cards: list[dict],
    *,
    root: Path | None = None,
    stats: ThumbStats | None = None,
    force: bool = False,
) -> list[dict]:
    """Set ``card['preview']`` from local frames. Leaves preview unset on failure
    (caller must show a placeholder — never an iframe)."""
    out_stats = stats if stats is not None else ThumbStats()
    t0 = time.perf_counter()
    base = root or frames_root()
    for card in cards:
        vid = str(card.get("video_id") or "")
        out_stats.requested += 1
        if not vid:
            out_stats.misses += 1
            out_stats.per_id[vid or "?"] = False
            continue
        cached = _cache_path(vid)
        if not force and cached.is_file() and cached.stat().st_size > 0:
            out_stats.cache_hits += 1
        else:
            out_stats.disk_reads += 1
        uri = thumb_data_uri(vid, root=base, force=force)
        if uri:
            card["preview"] = uri
            out_stats.hits += 1
            out_stats.per_id[vid] = True
        else:
            out_stats.misses += 1
            out_stats.per_id[vid] = False
    out_stats.elapsed_ms = (time.perf_counter() - t0) * 1000
    return cards


def mapping_report(video_ids: list[str], *, root: Path | None = None) -> dict:
    """Quick video_id → frame_000.jpg coverage for a sample."""
    base = root or frames_root()
    present = []
    missing = []
    for vid in video_ids:
        if frame_path(vid, root=base):
            present.append(vid)
        else:
            missing.append(vid)
    return {
        "frames_root": str(base),
        "root_exists": base.is_dir(),
        "n": len(video_ids),
        "with_frame_000": len(present),
        "missing": missing,
        "present": present,
        "cache_tag": CACHE_TAG,
        "max_edge": THUMB_MAX_EDGE,
        "jpeg_quality": THUMB_JPEG_QUALITY,
    }
