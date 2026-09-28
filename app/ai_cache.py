"""Persistent local cache for LLM outputs (relevance judgments, live generations).

Stored in data/processed/ai/<namespace>_cache.json (gitignored). Keys carry the prompt or
pipeline version, so entries from older versions are simply never looked up again.
Curated showcase outputs don't live here: they are frozen in the committed curated registry
(curated.py). Caching pins what the app shows; it does not make the underlying LLM judgment
deterministic.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import catalog

LOCAL_DIR = catalog.REPO_ROOT / "data" / "processed" / "ai"

_lock = threading.Lock()
_loaded: dict[Path, dict] = {}


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
    return _load(_path(namespace)).get(key)


def put(namespace: str, key: str, value) -> None:
    """Persist; raises OSError if the cache can't be written."""
    with _lock:
        path = _path(namespace)
        data = _load(path)
        data[key] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True))
        tmp.replace(path)
