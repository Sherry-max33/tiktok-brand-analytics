from __future__ import annotations
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def now_iso_tz(tz_name: str) -> str:
    tz = ZoneInfo(tz_name)
    return datetime.now(tz=tz).isoformat(timespec="seconds")

def now_ts() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())

def ts_to_iso(ts: int, tz_name: str) -> str:
    tz = ZoneInfo(tz_name)
    return datetime.fromtimestamp(ts, tz=tz).isoformat(timespec="seconds")
