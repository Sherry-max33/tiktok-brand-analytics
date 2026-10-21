"""Batch-download official TikTok covers via Apify → local ``.cache/apify_covers/``.

Skips any ``video_id`` that already has a local ``.jpg`` (pilot or main folder).
Does not upload to R2/Supabase. Cover-only (no MP4).

  PYTHONPATH=src:app python scripts/fetch_apify_covers_batch.py --limit 150
  PYTHONPATH=src:app python scripts/fetch_apify_covers_batch.py --limit 150 --batch-size 50
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "app"))

import catalog  # noqa: E402
from tiktok_brand.crawl.apify_video_client import (  # noqa: E402
    ApifyQuotaExceeded,
    ApifyTokenPool,
    download_url_to_file,
    fetch_videos_by_post_urls,
    get_apify_token,
    video_id_from_apify_item,
)

OUT_DIR = REPO / ".cache" / "apify_covers"
PILOT_DIRS = (
    REPO / ".cache" / "apify_covers_pilot",
    REPO / ".cache" / "apify_covers_review" / "img",
)
MANIFEST = OUT_DIR / "manifest.jsonl"
SKIP_FOREVER = OUT_DIR / "skip_forever.json"

# These almost never succeed on retry; keep spending Apify money is wasteful.
_PERMANENT_ERROR_SNIPPETS = (
    "not found",
    "private",
    "sensitive",
    "unavailable",
)


def _existing_ids() -> set[str]:
    have: set[str] = set()
    for d in (OUT_DIR, *PILOT_DIRS):
        if not d.is_dir():
            continue
        for p in d.glob("*.jpg"):
            if p.stat().st_size > 500:
                have.add(p.stem)
    return have


def _is_permanent_error(message: str) -> bool:
    text = (message or "").lower()
    return any(s in text for s in _PERMANENT_ERROR_SNIPPETS)


def _load_skip_forever() -> dict[str, str]:
    """video_id → reason. Rebuilt from manifest + on-disk skip file."""
    skip: dict[str, str] = {}
    if SKIP_FOREVER.exists():
        try:
            raw = json.loads(SKIP_FOREVER.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                skip.update({str(k): str(v) for k, v in raw.items()})
        except json.JSONDecodeError:
            pass
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = str(row.get("video_id") or "")
            if not vid:
                continue
            status = row.get("status") or ""
            err = str(row.get("error") or "")
            if status == "apify_error" and _is_permanent_error(err):
                skip[vid] = err or status
            elif status in {"no_cover_url"} and _is_permanent_error(err):
                skip[vid] = err or status
    return skip


def _save_skip_forever(skip: dict[str, str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SKIP_FOREVER.write_text(json.dumps(skip, indent=2, ensure_ascii=False), encoding="utf-8")


def _mark_skip(skip: dict[str, str], video_id: str, reason: str) -> None:
    if video_id in skip:
        return
    skip[video_id] = reason
    _save_skip_forever(skip)


def _cover_urls(item: dict) -> list[str]:
    urls: list[str] = []
    for key in ("mediaUrls", "downloadedCovers", "covers"):
        val = item.get(key)
        if isinstance(val, list):
            urls.extend(str(u) for u in val if u)
        elif isinstance(val, str) and val.strip():
            urls.append(val.strip())
        elif isinstance(val, dict):
            urls.extend(str(u) for u in val.values() if u)
    video_meta = item.get("videoMeta") if isinstance(item.get("videoMeta"), dict) else {}
    for key in ("coverUrl", "originCover", "originalCover", "dynamicCover", "zoomCover"):
        val = item.get(key) or video_meta.get(key)
        if isinstance(val, str) and val.startswith("http"):
            urls.append(val)
        elif isinstance(val, dict):
            for u in val.values():
                if isinstance(u, str) and u.startswith("http"):
                    urls.append(u)
    seen: set[str] = set()
    ordered: list[str] = []
    for prefer_apify in (True, False):
        for u in urls:
            if u in seen:
                continue
            is_apify = "api.apify.com" in u
            if prefer_apify and not is_apify:
                continue
            if not prefer_apify and is_apify:
                continue
            seen.add(u)
            ordered.append(u)
    return ordered


def _to_jpeg(path: Path) -> bytes:
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95, optimize=True, progressive=True)
        return buf.getvalue()


def _append_manifest(row: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _consolidate_pilot() -> int:
    """Copy pilot JPGs into OUT_DIR so skip set is one folder."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in PILOT_DIRS:
        if not d.is_dir():
            continue
        for p in d.glob("*.jpg"):
            dest = OUT_DIR / p.name
            if dest.exists() and dest.stat().st_size > 500:
                continue
            dest.write_bytes(p.read_bytes())
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
        help="Max NEW covers to fetch (default high: drain credit)",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Apify URLs per run")
    args = parser.parse_args()

    if not get_apify_token():
        print("ERROR: APIFY_API_TOKEN missing")
        sys.exit(1)

    try:
        pool = ApifyTokenPool(min_remaining_usd=0.02)
    except RuntimeError as exc:
        print("ERROR:", exc)
        sys.exit(1)
    print(f"token_pool size={len(pool.tokens)} current={pool.fingerprint}")

    copied = _consolidate_pilot()
    have = _existing_ids()
    skip = _load_skip_forever()
    _save_skip_forever(skip)
    print(
        f"consolidated_pilot_copies={copied}  already_have={len(have)}  "
        f"skip_forever={len(skip)}"
    )

    lib = catalog.load_library()
    # Prefer videos that have a real post URL (author + id).
    candidates: list[tuple[str, str]] = []
    for _, row in lib.iterrows():
        vid = str(row["video_id"])
        if vid in have or vid in skip:
            continue
        url = catalog.post_url(vid, str(row.get("author_username") or ""))
        if not url:
            continue
        candidates.append((vid, url))

    todo = candidates[: max(0, args.limit)]
    print(
        f"library={len(lib)}  eligible={len(candidates)}  "
        f"(excluded skip_forever)  this_run={len(todo)}"
    )
    if not todo:
        print("nothing to do")
        return

    ok = 0
    fail = 0
    skipped_quota = False
    t0 = time.perf_counter()

    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i : i + args.batch_size]
        # Re-check skip in case a prior chunk wrote files / permanent fails
        chunk = [
            (vid, url)
            for vid, url in chunk
            if not (OUT_DIR / f"{vid}.jpg").exists() and vid not in skip
        ]
        if not chunk:
            continue
        urls = [u for _, u in chunk]
        print(
            f"\n--- Apify batch {i // args.batch_size + 1}: {len(urls)} urls "
            f"(token={pool.fingerprint}) ---"
        )
        items: list = []
        meta: dict = {}
        while True:
            try:
                items, meta = fetch_videos_by_post_urls(
                    urls,
                    download_videos=False,
                    download_covers=True,
                    api_token=pool.current,
                    max_request_retries=0,
                )
                break
            except ApifyQuotaExceeded as exc:
                print("QUOTA on", pool.fingerprint, "→", exc)
                try:
                    pool.advance()
                    print("retry batch with", pool.fingerprint)
                except ApifyQuotaExceeded:
                    skipped_quota = True
                    break
            except Exception as exc:  # noqa: BLE001
                print("BATCH FAIL:", exc)
                fail += len(chunk)
                for vid, _ in chunk:
                    _append_manifest(
                        {"video_id": vid, "status": "batch_error", "error": str(exc)[:300]}
                    )
                items = None
                break
        if skipped_quota:
            break
        if items is None:
            continue

        print(f"returned {len(items)} items  run={meta.get('id')}")
        by_id = {}
        for item in items:
            vid = video_id_from_apify_item(item)
            if vid:
                by_id[vid] = item

        for vid, _url in chunk:
            dest_jpg = OUT_DIR / f"{vid}.jpg"
            if dest_jpg.exists() and dest_jpg.stat().st_size > 500:
                continue
            item = by_id.get(vid)
            if not item:
                fail += 1
                _append_manifest({"video_id": vid, "status": "missing_item"})
                print("FAIL missing", vid)
                continue
            if item.get("error") or item.get("errorCode"):
                err = str(item.get("error") or item.get("errorCode"))
                fail += 1
                _append_manifest(
                    {"video_id": vid, "status": "apify_error", "error": err}
                )
                if _is_permanent_error(err):
                    _mark_skip(skip, vid, err)
                print("FAIL apify", vid, err)
                continue
            curls = _cover_urls(item)
            if not curls:
                fail += 1
                _append_manifest({"video_id": vid, "status": "no_cover_url"})
                print("FAIL no_cover", vid)
                continue
            bin_path = OUT_DIR / f"{vid}.bin"
            got = None
            last_err = None
            for u in curls:
                try:
                    download_url_to_file(u, bin_path, api_token=pool.current)
                    if bin_path.stat().st_size > 500:
                        got = bin_path
                        break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
            if got is None:
                fail += 1
                _append_manifest(
                    {"video_id": vid, "status": "download_fail", "error": str(last_err)[:200]}
                )
                print("FAIL dl", vid, last_err)
                continue
            try:
                jpeg = _to_jpeg(got)
            except Exception as exc:  # noqa: BLE001
                fail += 1
                _append_manifest(
                    {"video_id": vid, "status": "decode_fail", "error": str(exc)[:200]}
                )
                print("FAIL decode", vid, exc)
                continue
            dest_jpg.write_bytes(jpeg)
            if bin_path.exists():
                bin_path.unlink()
            ok += 1
            _append_manifest(
                {"video_id": vid, "status": "ok", "bytes": len(jpeg), "ts": time.time()}
            )
            print(f"OK  {vid}  {len(jpeg)/1024:.0f}KB")

    elapsed = time.perf_counter() - t0
    have_after = _existing_ids()
    lib_n = len(catalog.load_library())
    summary = {
        "ok_new": ok,
        "fail": fail,
        "quota_stop": skipped_quota,
        "elapsed_sec": round(elapsed, 1),
        "total_local_jpg": len(have_after),
        "skip_forever": len(skip),
        "library": lib_n,
        "still_missing_eligible": max(0, lib_n - len(have_after) - len(skip)),
        "out_dir": str(OUT_DIR),
    }
    (OUT_DIR / "last_run_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    if skipped_quota:
        sys.exit(3)
    if fail and not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
