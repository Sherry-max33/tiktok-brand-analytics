"""Benchmark the 20-video frame-thumb prototype (no Streamlit UI).

  PYTHONPATH=app python scripts/bench_frame_thumbs_proto.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app"))

import catalog  # noqa: E402
from frame_thumbs import (  # noqa: E402
    CACHE_DIR,
    ThumbStats,
    attach_frame_previews,
    frames_root,
    mapping_report,
)

N = 20


def _sample_cards(n: int = N) -> list[dict]:
    root = frames_root()
    df = catalog.load_library()
    ids = []
    for vid in df["video_id"].astype(str):
        if (root / vid / "frame_000.jpg").is_file():
            ids.append(vid)
        if len(ids) >= n:
            break
    cards = []
    for vid in ids:
        card = catalog.get_video(vid)
        if card:
            cards.append({"video_id": card["video_id"], "title": card.get("title")})
    return cards


def main() -> None:
    root = frames_root()
    print("FRAMES_ROOT:", root)
    print("exists:", root.is_dir())
    cards = _sample_cards()
    report = mapping_report([c["video_id"] for c in cards])
    print(
        f"mapping: {report['with_frame_000']}/{report['n']} have frame_000.jpg"
        + (f"  missing={report['missing']}" if report["missing"] else "")
    )

    # Cold: wipe cache for these ids
    if CACHE_DIR.exists():
        for c in cards:
            p = CACHE_DIR / f"{c['video_id']}.jpg"
            if p.exists():
                p.unlink()

    cold = ThumbStats()
    t0 = time.perf_counter()
    attach_frame_previews([dict(c) for c in cards], stats=cold)
    cold_wall = (time.perf_counter() - t0) * 1000

    warm = ThumbStats()
    t1 = time.perf_counter()
    attach_frame_previews([dict(c) for c in cards], stats=warm)
    warm_wall = (time.perf_counter() - t1) * 1000

    print("--- cold (first load) ---")
    print(
        f"success {cold.hits}/{cold.requested} ({100 * cold.hits / max(cold.requested, 1):.0f}%)"
    )
    print(f"elapsed_ms {cold.elapsed_ms:.1f}  wall_ms {cold_wall:.1f}")
    print(f"cache_hits {cold.cache_hits}  disk_reads {cold.disk_reads}")
    print(f"drive_api_calls {cold.drive_api_calls}")
    print("--- warm (cached) ---")
    print(
        f"success {warm.hits}/{warm.requested} ({100 * warm.hits / max(warm.requested, 1):.0f}%)"
    )
    print(f"elapsed_ms {warm.elapsed_ms:.1f}  wall_ms {warm_wall:.1f}")
    print(f"cache_hits {warm.cache_hits}  disk_reads {warm.disk_reads}")
    print(f"drive_api_calls {warm.drive_api_calls}")
    print("cache_dir:", CACHE_DIR, "files:", len(list(CACHE_DIR.glob('*.jpg'))) if CACHE_DIR.exists() else 0)


if __name__ == "__main__":
    main()
