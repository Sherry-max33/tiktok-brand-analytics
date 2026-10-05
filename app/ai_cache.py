"""Persistent cache for LLM outputs (relevance judgments, live generations).

Stored in Supabase when it's configured (supabase_store.py; survives restarts of the deployed
app), otherwise in data/processed/ai/<namespace>_cache.json (gitignored). Keys carry the
prompt or pipeline version, so entries from older versions are simply never looked up again.
Curated showcase outputs don't live here: they are frozen in the committed curated registry
(curated.py). Caching pins what the app shows; it does not make the underlying LLM judgment
deterministic.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import catalog
import supabase_store

LOCAL_DIR = catalog.REPO_ROOT / "data" / "processed" / "ai"

CacheUnavailable = supabase_store.StoreError

_lock = threading.Lock()
_loaded: dict[Path, dict] = {}
# Remote hits only: a stored entry never changes under its versioned key, but a miss may be
# filled by another server instance.
_remote_hits: dict[tuple[str, str], object] = {}


def _path(namespace: str) -> Path:
    return LOCAL_DIR / f"{namespace}_cache.json"


def _load(path: Path) -> dict:
    if path not in _loaded:
        try:
            _loaded[path] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            _loaded[path] = {}
    return _loaded[path]


def get(namespace: str, key: str):
    """The cached value or None; raises CacheUnavailable if the remote store can't be read,
    so a read failure is never mistaken for a miss."""
    if not supabase_store.configured():
        return _load(_path(namespace)).get(key)
    if (namespace, key) not in _remote_hits:
        value = supabase_store.get(namespace, key)
        if value is None:
            return None
        _remote_hits[(namespace, key)] = value
    return _remote_hits[(namespace, key)]


def put(namespace: str, key: str, value) -> None:
    """Persist; raises OSError if the cache can't be written."""
    if supabase_store.configured():
        supabase_store.put(namespace, key, value)
        _remote_hits[(namespace, key)] = value
        return
    with _lock:
        path = _path(namespace)
        data = _load(path)
        data[key] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True))
        tmp.replace(path)
