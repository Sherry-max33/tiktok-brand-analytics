"""Supabase (Postgres over its REST API) backend for the live AI cache and daily quota.

Used when SUPABASE_URL and SUPABASE_SECRET_KEY are set (environment or Streamlit Secrets), so
live generations and the daily count survive restarts of a host with an ephemeral disk. The
tables and the atomic quota function are created by scripts/supabase_setup.sql. The secret
key bypasses Row Level Security, so it must stay server-side. Every failure raises
StoreError (an OSError); callers fail closed.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request
from urllib.parse import quote

import llm

TIMEOUT_S = 8


class StoreError(OSError):
    pass


def configured() -> bool:
    return bool(llm.setting("SUPABASE_URL") and llm.setting("SUPABASE_SECRET_KEY"))


def _request(method: str, path: str, body=None, prefer: str | None = None):
    key = llm.setting("SUPABASE_SECRET_KEY")
    headers = {"apikey": key, "Content-Type": "application/json", "Accept": "application/json"}
    # Legacy service_role keys are JWTs and also go in Authorization; the newer sb_secret_
    # keys are rejected there.
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"
    if prefer:
        headers["Prefer"] = prefer
    request = urllib.request.Request(
        llm.setting("SUPABASE_URL").rstrip("/") + "/rest/v1/" + path,
        data=None if body is None else json.dumps(body).encode(),
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raise StoreError(f"Supabase {method} {path.split('?')[0]} returned HTTP {error.code}") from None
    except (urllib.error.URLError, OSError) as error:
        raise StoreError(f"Supabase unreachable: {type(error).__name__}") from None
    try:
        return json.loads(raw) if raw.strip() else None
    except ValueError:
        raise StoreError("Supabase returned a malformed response") from None


def _eq(value: str) -> str:
    return "eq." + quote(value, safe="")


def get(namespace: str, key: str):
    rows = _request("GET", f"ai_cache?namespace={_eq(namespace)}&key={_eq(key)}&select=value")
    if not isinstance(rows, list):
        raise StoreError("Supabase returned an unexpected cache response")
    return rows[0]["value"] if rows else None


def put(namespace: str, key: str, value) -> None:
    _request(
        "POST",
        "ai_cache?on_conflict=namespace,key",
        {"namespace": namespace, "key": key, "value": value},
        prefer="resolution=merge-duplicates,return=minimal",
    )


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def reserve(daily_limit: int) -> bool:
    taken = _request("POST", "rpc/reserve_generation", {"daily_limit": daily_limit})
    if not isinstance(taken, bool):
        raise StoreError("Supabase returned an unexpected quota response")
    return taken


def usage() -> int:
    rows = _request("GET", f"ai_quota?day={_eq(_today())}&select=count")
    if not isinstance(rows, list) or (rows and not isinstance(rows[0].get("count"), int)):
        raise StoreError("Supabase returned an unexpected quota response")
    return rows[0]["count"] if rows else 0
