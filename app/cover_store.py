"""Private cloud cover objects for list-card previews (pipeline pilot).

Priority:
1. Cloudflare R2 when ``R2_ACCOUNT_ID`` + ``R2_ACCESS_KEY_ID`` + ``R2_SECRET_ACCESS_KEY``
   + ``R2_BUCKET`` are set (S3-compatible).
2. Else Supabase Storage when ``SUPABASE_URL`` + ``SUPABASE_SECRET_KEY`` are set
   (bucket ``COVER_STORE_BUCKET``, default ``video-covers-pilot``, private).

Objects are keyed ``covers/{video_id}.jpg`` (pilot may also use ``.webp`` later).
Never commit binaries or credentials. Callers cache bytes locally under ``.cache/``.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

import llm

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CACHE = REPO_ROOT / ".cache" / "cover_store"
APIFY_COVERS = REPO_ROOT / ".cache" / "apify_covers"
DEFAULT_BUCKET = "video-covers-pilot"
OBJECT_PREFIX = "covers/"
PRESIGN_EXPIRES_S = 60 * 60 * 6  # match typical TikTok CDN TTL window


class CoverStoreError(OSError):
    pass


def bucket_name() -> str:
    return (llm.setting("COVER_STORE_BUCKET") or os.getenv("COVER_STORE_BUCKET") or DEFAULT_BUCKET).strip()


def r2_configured() -> bool:
    return bool(
        llm.setting("R2_ACCOUNT_ID")
        and llm.setting("R2_ACCESS_KEY_ID")
        and llm.setting("R2_SECRET_ACCESS_KEY")
        and llm.setting("R2_BUCKET")
    )


def supabase_configured() -> bool:
    return bool(llm.setting("SUPABASE_URL") and llm.setting("SUPABASE_SECRET_KEY"))


def configured() -> bool:
    return r2_configured() or supabase_configured()


def backend() -> str:
    if r2_configured():
        return "r2"
    if supabase_configured():
        return "supabase"
    return "none"


def object_key(video_id: str, *, ext: str = "jpg") -> str:
    ext = ext.lstrip(".")
    return f"{OBJECT_PREFIX}{video_id}.{ext}"


def _local_path(video_id: str, *, ext: str = "jpg") -> Path:
    return LOCAL_CACHE / backend() / f"{video_id}.{ext}"


# ----- R2 (boto3) -----


def _r2_client():
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise CoverStoreError("boto3 is required for R2 (pip install boto3)") from exc
    account = llm.setting("R2_ACCOUNT_ID")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=llm.setting("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=llm.setting("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def _r2_put(key: str, data: bytes, content_type: str) -> None:
    _r2_client().put_object(
        Bucket=llm.setting("R2_BUCKET"),
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def _r2_get(key: str) -> bytes:
    try:
        obj = _r2_client().get_object(Bucket=llm.setting("R2_BUCKET"), Key=key)
    except Exception as exc:  # noqa: BLE001 — normalize missing keys
        raise CoverStoreError(f"R2 get failed for {key}: {exc}") from None
    return obj["Body"].read()


# ----- Supabase Storage -----


def _supabase_headers(*, content_type: str | None = None) -> dict:
    key = llm.setting("SUPABASE_SECRET_KEY")
    headers = {"apikey": key, "Accept": "*/*"}
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _supabase_ensure_bucket() -> None:
    base = llm.setting("SUPABASE_URL").rstrip("/")
    name = bucket_name()
    body = (
        '{"id":"%s","name":"%s","public":false,"file_size_limit":5242880,'
        '"allowed_mime_types":["image/jpeg","image/webp","image/png"]}'
        % (name, name)
    ).encode()
    req = urllib.request.Request(
        f"{base}/storage/v1/bucket",
        data=body,
        headers=_supabase_headers(content_type="application/json"),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
    except urllib.error.HTTPError as error:
        # 409 = already exists
        if error.code in (409, 400):
            raw = error.read().decode("utf-8", errors="ignore")
            if "already exists" in raw.lower() or "duplicate" in raw.lower() or error.code == 409:
                return
            # Some projects return 400 for existing — ignore if list works.
            return
        raise CoverStoreError(f"Supabase create bucket HTTP {error.code}") from None


def _supabase_put(key: str, data: bytes, content_type: str) -> None:
    _supabase_ensure_bucket()
    base = llm.setting("SUPABASE_URL").rstrip("/")
    name = bucket_name()
    # upsert
    url = f"{base}/storage/v1/object/{name}/{key}"
    headers = _supabase_headers(content_type=content_type)
    headers["x-upsert"] = "true"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
    except urllib.error.HTTPError as error:
        # POST may fail if exists without upsert support; try PUT
        if error.code in (400, 409):
            req2 = urllib.request.Request(url, data=data, headers=headers, method="PUT")
            try:
                with urllib.request.urlopen(req2, timeout=60) as resp2:
                    resp2.read()
                return
            except urllib.error.HTTPError as error2:
                raise CoverStoreError(
                    f"Supabase upload HTTP {error2.code}: {error2.read()[:200]!r}"
                ) from None
        raise CoverStoreError(f"Supabase upload HTTP {error.code}: {error.read()[:200]!r}") from None


def _supabase_get(key: str) -> bytes:
    base = llm.setting("SUPABASE_URL").rstrip("/")
    name = bucket_name()
    url = f"{base}/storage/v1/object/{name}/{key}"
    req = urllib.request.Request(url, headers=_supabase_headers(), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as error:
        raise CoverStoreError(f"Supabase get HTTP {error.code} for {key}") from None


# ----- public API -----


def put_cover(video_id: str, data: bytes, *, ext: str = "jpg", content_type: str = "image/jpeg") -> str:
    if not data:
        raise CoverStoreError("empty cover bytes")
    key = object_key(video_id, ext=ext)
    if backend() == "r2":
        _r2_put(key, data, content_type)
    elif backend() == "supabase":
        _supabase_put(key, data, content_type)
    else:
        raise CoverStoreError("No cover store configured (set R2_* or SUPABASE_*)")
    local = _local_path(video_id, ext=ext)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    return key


def _local_candidates(video_id: str, *, ext: str = "jpg") -> list[Path]:
    """Prefer cover_store cache, then Apify download dir (same bytes as R2)."""
    return [
        _local_path(video_id, ext=ext),
        APIFY_COVERS / f"{video_id}.{ext}",
    ]


def get_cover(video_id: str, *, ext: str = "jpg", force_remote: bool = False) -> bytes:
    if not force_remote:
        for path in _local_candidates(video_id, ext=ext):
            if path.is_file() and path.stat().st_size > 0:
                return path.read_bytes()
    key = object_key(video_id, ext=ext)
    if backend() == "r2":
        data = _r2_get(key)
    elif backend() == "supabase":
        data = _supabase_get(key)
    else:
        raise CoverStoreError("No cover store configured")
    local = _local_path(video_id, ext=ext)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    return data


def presigned_cover_url(video_id: str, *, ext: str = "jpg", expires: int = PRESIGN_EXPIRES_S) -> str:
    """Time-limited HTTPS URL for a private R2 object. Empty if not on R2."""
    if backend() != "r2":
        return ""
    try:
        url = _r2_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": llm.setting("R2_BUCKET"), "Key": object_key(video_id, ext=ext)},
            ExpiresIn=max(60, int(expires)),
        )
    except Exception as exc:  # noqa: BLE001
        raise CoverStoreError(f"R2 presign failed for {video_id}: {exc}") from None
    return url or ""


def try_cover_data_uri(video_id: str, *, ext: str = "jpg") -> str:
    """Return data URI or '' if missing / not configured."""
    import base64

    if not configured():
        return ""
    try:
        raw = get_cover(video_id, ext=ext)
    except CoverStoreError:
        return ""
    if not raw:
        return ""
    mime = "image/webp" if ext == "webp" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def try_cover_preview(video_id: str, *, ext: str = "jpg") -> str:
    """Best ``<img src>`` for list cards: local data URI, else R2 presigned URL."""
    if not video_id or not configured():
        return ""
    # Local first — avoids embedding multi-MB HTML and skips network on this machine.
    for path in _local_candidates(video_id, ext=ext):
        if path.is_file() and path.stat().st_size > 0:
            return try_cover_data_uri(video_id, ext=ext)
    if backend() == "r2":
        try:
            return presigned_cover_url(video_id, ext=ext)
        except CoverStoreError:
            return ""
    return try_cover_data_uri(video_id, ext=ext)
