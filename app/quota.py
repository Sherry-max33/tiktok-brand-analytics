"""Server-side daily (UTC) ceiling for new (uncached) AI generations.

With Supabase configured, the counter is a database row taken by an atomic function
(supabase_store.py), so it survives restarts and holds across server instances. Otherwise a
JSON counter {"date": "YYYY-MM-DD", "count": n} guarded by an exclusive file lock, so
concurrent sessions in one server process tree can't overshoot; its path can be moved with
AI_QUOTA_FILE, and on hosts with ephemeral disks it resets when the container restarts. Any
failure to read, lock or write raises, and callers fail closed (no generation).
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
from pathlib import Path

import ai_cache
import llm
import supabase_store


class QuotaError(RuntimeError):
    pass


def _file() -> Path:
    return Path(llm.setting("AI_QUOTA_FILE") or ai_cache.LOCAL_DIR / "generation_quota.json")


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def reserve(daily_limit: int) -> bool:
    """Take one generation slot for today. False when the ceiling is reached; raises
    QuotaError when the quota state can't be verified."""
    if supabase_store.configured():
        try:
            return supabase_store.reserve(daily_limit)
        except supabase_store.StoreError as error:
            raise QuotaError(str(error)) from error
    path = _file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            raw = handle.read()
            state = json.loads(raw) if raw.strip() else {}
            if state.get("date") != _today():
                state = {"date": _today(), "count": 0}
            if not isinstance(state.get("count"), int):
                raise QuotaError("quota counter is malformed")
            if state["count"] >= daily_limit:
                return False
            state["count"] += 1
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(state))
            handle.flush()
            os.fsync(handle.fileno())
            return True
    except QuotaError:
        raise
    except (OSError, ValueError) as error:
        raise QuotaError(f"quota state unavailable: {error}") from error


def usage() -> dict:
    """Today's count; None when the quota state exists but can't be read."""
    if supabase_store.configured():
        try:
            return {"date": _today(), "count": supabase_store.usage()}
        except supabase_store.StoreError:
            return {"date": _today(), "count": None}
    path = _file()
    try:
        raw = path.read_text() if path.exists() else ""
        state = json.loads(raw) if raw.strip() else {}
    except (OSError, ValueError):
        return {"date": _today(), "count": None}
    if state.get("date") != _today():
        return {"date": _today(), "count": 0}
    return state if isinstance(state.get("count"), int) else {"date": _today(), "count": None}
