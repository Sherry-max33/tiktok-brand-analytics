"""Server-side daily ceiling for new (uncached) AI generations.

A JSON counter {"date": "YYYY-MM-DD", "count": n} guarded by an exclusive file lock, so
concurrent sessions in one server process tree can't overshoot. Any failure to read, lock or
write raises, and callers fail closed (no generation). The file path can be moved with
AI_QUOTA_FILE; on hosts with ephemeral disks the counter resets when the container restarts,
so keep the ceiling conservative.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
from pathlib import Path

import ai_cache
import llm


class QuotaError(RuntimeError):
    pass


def _file() -> Path:
    return Path(llm.setting("AI_QUOTA_FILE") or ai_cache.LOCAL_DIR / "generation_quota.json")


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def reserve(daily_limit: int) -> bool:
    """Take one generation slot for today. False when the ceiling is reached; raises
    QuotaError when the quota state can't be verified."""
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
    path = _file()
    try:
        raw = path.read_text() if path.exists() else ""
        state = json.loads(raw) if raw.strip() else {}
    except (OSError, ValueError):
        return {"date": _today(), "count": None}
    if state.get("date") != _today():
        return {"date": _today(), "count": 0}
    return state if isinstance(state.get("count"), int) else {"date": _today(), "count": None}
