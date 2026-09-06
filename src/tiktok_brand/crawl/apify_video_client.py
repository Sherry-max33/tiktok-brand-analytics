"""Apify clockworks TikTok clients for hashtag/profile crawls and postURL downloads."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

from tiktok_brand.common.logging import get_logger

log = get_logger("tiktok_brand.crawl.apify_video_client")

DEFAULT_ACTOR_ID = "clockworks/tiktok-scraper"
VIDEO_BY_URL_ACTOR_ID = "clockworks/tiktok-video-scraper"
DEFAULT_RESULTS_PER_PAGE = 200
_VIDEO_ID_RE = re.compile(r"/video/(\d+)")
_QUOTA_HINTS = (
    "monthly usage hard limit exceeded",
    "usage hard limit exceeded",
    "hard limit exceeded",
    "monthly usage limit",
)


class ApifyQuotaExceeded(RuntimeError):
    """Raised when the current Apify token hits its monthly hard limit."""


def _token_fingerprint(token: str) -> str:
    t = token.strip()
    if len(t) <= 10:
        return t[:2] + "…"
    return f"{t[:8]}…{t[-4:]}"


def _is_quota_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if any(h in msg for h in _QUOTA_HINTS):
        return True
    # apify-client sometimes nests message on .message / .type
    for attr in ("message", "type", "name"):
        val = getattr(exc, attr, None)
        if isinstance(val, str) and any(h in val.lower() for h in _QUOTA_HINTS):
            return True
    return False


def _parse_env_file_tokens(path: Path) -> List[str]:
    """Collect every APIFY_API_TOKEN*= value from a .env file (order preserved)."""
    if not path.is_file():
        return []
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key in {"APIFY_API_TOKEN", "APIFY_TOKEN"} or re.fullmatch(
            r"APIFY_API_TOKEN_\d+", key
        ):
            tok = val.strip().strip('"').strip("'")
            if tok:
                out.append(tok)
        elif key == "APIFY_API_TOKENS":
            for part in val.split(","):
                tok = part.strip().strip('"').strip("'")
                if tok:
                    out.append(tok)
    return out


def get_apify_tokens(*, env_path: Optional[Path] = None) -> List[str]:
    """Return deduped Apify tokens in priority order.

    ``.env`` line order is authoritative when present (supports duplicate
    ``APIFY_API_TOKEN=`` lines). Env vars / ``APIFY_API_TOKENS`` append any
    extras not already listed.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path) if env_path else load_dotenv()
    except ImportError:
        pass

    ordered: List[str] = []

    paths: List[Path] = []
    if env_path is not None:
        paths.append(Path(env_path))
    else:
        paths.append(Path(".env"))
        paths.append(Path(__file__).resolve().parents[3] / ".env")
    for candidate in paths:
        try:
            if candidate.is_file():
                ordered.extend(_parse_env_file_tokens(candidate))
                break
        except OSError:
            continue

    multi = os.environ.get("APIFY_API_TOKENS") or ""
    for part in multi.split(","):
        tok = part.strip().strip('"').strip("'")
        if tok:
            ordered.append(tok)

    for key, val in sorted(os.environ.items()):
        if key in {"APIFY_API_TOKEN", "APIFY_TOKEN"} or re.fullmatch(
            r"APIFY_API_TOKEN_\d+", key
        ):
            tok = (val or "").strip().strip('"').strip("'")
            if tok:
                ordered.append(tok)

    seen: set[str] = set()
    out: List[str] = []
    for tok in ordered:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def get_apify_token() -> Optional[str]:
    tokens = get_apify_tokens()
    return tokens[0] if tokens else None


def _filter_usable_tokens(
    tokens: List[str], *, min_remaining_usd: float = 0.05
) -> List[str]:
    """Drop invalid/exhausted tokens so a fresh run starts on a live account."""
    import json
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    usable: List[str] = []
    for tok in tokens:
        fp = _token_fingerprint(tok)
        try:
            req = Request(
                "https://api.apify.com/v2/users/me/limits",
                headers={"Authorization": f"Bearer {tok}"},
            )
            with urlopen(req, timeout=20) as resp:
                data = json.load(resp).get("data") or {}
            used = float((data.get("current") or {}).get("monthlyUsageUsd") or 0)
            cap = float((data.get("limits") or {}).get("maxMonthlyUsageUsd") or 0)
            left = cap - used
            if left < min_remaining_usd:
                log.warning("Skipping exhausted token %s (left=$%.2f)", fp, left)
                continue
            usable.append(tok)
            log.info("Usable token %s (left=$%.2f)", fp, left)
        except HTTPError as exc:
            log.warning("Skipping invalid token %s (HTTP %s)", fp, exc.code)
        except (URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
            log.warning("Skipping token %s (%s)", fp, exc)
    return usable


class ApifyTokenPool:
    """Rotate through tokens when monthly hard limit is hit."""

    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        *,
        skip_exhausted: bool = True,
        min_remaining_usd: float = 0.05,
    ) -> None:
        raw = list(tokens or get_apify_tokens())
        if not raw:
            raise RuntimeError(
                "APIFY_API_TOKEN is not set. Add one or more tokens to .env."
            )
        if skip_exhausted:
            self.tokens = _filter_usable_tokens(
                raw, min_remaining_usd=min_remaining_usd
            )
            if not self.tokens:
                raise RuntimeError(
                    "No Apify tokens with remaining monthly credit "
                    f"(checked {len(raw)}; need > ${min_remaining_usd:.2f})."
                )
        else:
            self.tokens = raw
        self.index = 0
        os.environ["APIFY_API_TOKEN"] = self.tokens[0]

    @property
    def current(self) -> str:
        return self.tokens[self.index]

    @property
    def fingerprint(self) -> str:
        return _token_fingerprint(self.current)

    @property
    def remaining(self) -> int:
        return len(self.tokens) - self.index - 1

    def advance(self) -> str:
        if self.index + 1 >= len(self.tokens):
            raise ApifyQuotaExceeded(
                f"All {len(self.tokens)} Apify token(s) hit monthly hard limit "
                f"(last={self.fingerprint})."
            )
        self.index += 1
        tok = self.tokens[self.index]
        os.environ["APIFY_API_TOKEN"] = tok
        log.warning(
            "Switched Apify token → %s (%d left after this)",
            _token_fingerprint(tok),
            self.remaining,
        )
        return tok


def _base_run_input(scrape_additional_author_meta: bool = True) -> Dict[str, Any]:
    return {
        "shouldDownloadVideos": False,
        "shouldDownloadCovers": False,
        "shouldDownloadSlideshowImages": False,
        "shouldDownloadAvatars": False,
        "shouldDownloadMusicCovers": False,
        "commentsPerPost": 0,
        "topLevelCommentsPerPost": 0,
        "maxRepliesPerComment": 0,
        "scrapeAdditionalAuthorMeta": scrape_additional_author_meta,
        "scrapeRelatedVideos": False,
        "downloadSubtitlesOptions": "NEVER_DOWNLOAD_SUBTITLES",
        "proxyCountryCode": "None",
    }


def _run_actor(
    run_input: Dict[str, Any],
    *,
    actor_id: str = DEFAULT_ACTOR_ID,
    api_token: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run actor; return (dataset items, run metadata)."""
    token = api_token or get_apify_token()
    if not token:
        raise RuntimeError(
            "APIFY_API_TOKEN is not set. Add it to your environment or .env file."
        )

    try:
        from apify_client import ApifyClient
    except ImportError as exc:
        raise RuntimeError(
            "apify-client is not installed. Run: pip install apify-client"
        ) from exc

    client = ApifyClient(token)
    log.info(
        "Starting Apify actor %s (token=%s)",
        actor_id,
        _token_fingerprint(token),
    )
    try:
        run = client.actor(actor_id).call(run_input=run_input)
    except Exception as exc:  # noqa: BLE001 — normalize quota into typed error
        if _is_quota_error(exc):
            raise ApifyQuotaExceeded(
                f"Apify monthly hard limit exceeded (token={_token_fingerprint(token)}): {exc}"
            ) from exc
        raise
    if not run:
        return [], {}

    # apify-client may return a dict or a Run model (snake_case attrs)
    if hasattr(run, "model_dump"):
        run_meta = run.model_dump()
    elif isinstance(run, dict):
        run_meta = run
    else:
        run_meta = {
            "defaultDatasetId": getattr(run, "default_dataset_id", None)
            or getattr(run, "defaultDatasetId", None),
            "default_dataset_id": getattr(run, "default_dataset_id", None),
            "defaultKeyValueStoreId": getattr(run, "default_key_value_store_id", None)
            or getattr(run, "defaultKeyValueStoreId", None),
            "id": getattr(run, "id", None),
        }

    # Some quota failures surface as a finished run with an error status/message.
    status = str(run_meta.get("status") or "").upper()
    status_msg = str(
        run_meta.get("statusMessage")
        or run_meta.get("status_message")
        or ""
    )
    if status in {"FAILED", "ABORTING", "ABORTED"} and _is_quota_error(
        RuntimeError(status_msg)
    ):
        raise ApifyQuotaExceeded(
            f"Apify run {status}: {status_msg} (token={_token_fingerprint(token)})"
        )

    items: List[Dict[str, Any]] = []
    dataset_id = (
        run_meta.get("defaultDatasetId")
        or run_meta.get("default_dataset_id")
    )
    try:
        if dataset_id:
            for item in client.dataset(dataset_id).iterate_items():
                if isinstance(item, dict):
                    items.append(item)
    except Exception as exc:  # noqa: BLE001
        if _is_quota_error(exc):
            raise ApifyQuotaExceeded(
                f"Apify monthly hard limit exceeded (token={_token_fingerprint(token)}): {exc}"
            ) from exc
        raise
    log.info("Apify actor finished: %s items", len(items))
    return items, run_meta


def fetch_hashtag_videos(
    hashtag: str,
    count: int,
    *,
    actor_id: str = DEFAULT_ACTOR_ID,
    api_token: Optional[str] = None,
    scrape_additional_author_meta: bool = True,
) -> List[Dict[str, Any]]:
    tag = hashtag.lstrip("#")
    run_input = {
        **_base_run_input(scrape_additional_author_meta),
        "hashtags": [tag],
        "resultsPerPage": min(count, DEFAULT_RESULTS_PER_PAGE),
    }
    items, _ = _run_actor(run_input, actor_id=actor_id, api_token=api_token)
    return items


def fetch_user_videos(
    username: str,
    count: int,
    *,
    actor_id: str = DEFAULT_ACTOR_ID,
    api_token: Optional[str] = None,
    scrape_additional_author_meta: bool = True,
) -> List[Dict[str, Any]]:
    handle = username.lstrip("@")
    run_input = {
        **_base_run_input(scrape_additional_author_meta),
        "profiles": [handle],
        "resultsPerPage": min(count, DEFAULT_RESULTS_PER_PAGE),
        "profileScrapeSections": ["videos"],
        "profileSorting": "latest",
        "excludePinnedPosts": False,
    }
    items, _ = _run_actor(run_input, actor_id=actor_id, api_token=api_token)
    return items


def fetch_videos_by_post_urls(
    post_urls: List[str],
    *,
    download_videos: bool = True,
    download_covers: bool = False,
    actor_id: str = VIDEO_BY_URL_ACTOR_ID,
    api_token: Optional[str] = None,
    video_kv_store: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Scrape specific TikTok post URLs; optionally download MP4s into Apify KV."""
    urls = [u.strip() for u in post_urls if u and str(u).strip()]
    if not urls:
        return [], {}

    run_input: Dict[str, Any] = {
        "postURLs": urls,
        "shouldDownloadVideos": bool(download_videos),
        "shouldDownloadCovers": bool(download_covers),
        "shouldDownloadSlideshowImages": False,
        "shouldDownloadSubtitles": False,
        "scrapeRelatedVideos": False,
        "downloadSubtitlesOptions": "NEVER_DOWNLOAD_SUBTITLES",
    }
    if video_kv_store:
        run_input["videoKvStoreIdOrName"] = video_kv_store
    return _run_actor(run_input, actor_id=actor_id, api_token=api_token)


def video_id_from_apify_item(item: Dict[str, Any]) -> Optional[str]:
    raw = item.get("id")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    for key in ("webVideoUrl", "submittedVideoUrl", "url"):
        val = item.get(key)
        if not val:
            continue
        m = _VIDEO_ID_RE.search(str(val))
        if m:
            return m.group(1)
    return None


def media_urls_from_apify_item(item: Dict[str, Any]) -> List[str]:
    """Prefer Apify-hosted downloaded media; fall back to TikTok CDN addrs."""
    urls: List[str] = []
    media = item.get("mediaUrls")
    if isinstance(media, list):
        urls.extend(str(u) for u in media if u)
    elif isinstance(media, str) and media.strip():
        urls.append(media.strip())

    video_meta = item.get("videoMeta") if isinstance(item.get("videoMeta"), dict) else {}
    for key in ("downloadAddr", "originalDownloadAddr"):
        val = video_meta.get(key)
        if val:
            urls.append(str(val))

    seen = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def download_url_to_file(
    url: str,
    dest: Path,
    *,
    timeout_sec: int = 180,
    api_token: Optional[str] = None,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fetch_url = url
    # Apify KV record URLs require token unless they already carry ?signature=
    if "api.apify.com/v2/key-value-stores/" in url and "signature=" not in url:
        token = api_token or get_apify_token()
        if token:
            sep = "&" if "?" in url else "?"
            fetch_url = f"{url}{sep}token={token}"
    req = Request(fetch_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout_sec) as resp:
        data = resp.read()
    if not data:
        raise RuntimeError(f"empty download: {url}")
    dest.write_bytes(data)
    return dest


def save_apify_video_item(
    item: Dict[str, Any],
    dest_mp4: Path,
    *,
    timeout_sec: int = 180,
    api_token: Optional[str] = None,
) -> Optional[Path]:
    """Download first usable media URL from an Apify item to ``dest_mp4``."""
    urls = media_urls_from_apify_item(item)
    # Prefer Apify-hosted records over TikTok CDN (often 403 from local IP)
    urls = sorted(urls, key=lambda u: (0 if "api.apify.com" in u else 1, u))
    if not urls:
        return None
    last_err: Optional[Exception] = None
    for url in urls:
        try:
            return download_url_to_file(
                url, dest_mp4, timeout_sec=timeout_sec, api_token=api_token
            )
        except Exception as exc:  # noqa: BLE001 — try next URL
            last_err = exc
            log.debug("media fetch failed %s: %s", url[:120], exc)
    if last_err:
        log.warning("all media URLs failed for %s: %s", dest_mp4, last_err)
    return None
