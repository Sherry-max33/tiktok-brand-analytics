"""Curated showcase registry: pre-generated, QA-approved AI outputs (committed).

app/frozen_ai/curated.json holds, per showcase video, why it was chosen, every generation
attempt (full pipeline record + metadata), which attempt was selected, and its QA status:

- selected        chosen for the showcase, not generated yet
- pending_review  generated, passed automated QA, awaiting human QA
- auto_failed     generated but failed automated QA (fix the system, then regenerate)
- approved        QA-approved and frozen: served directly, never regenerated live
- rejected        QA-rejected (kept for the record)

Every video in the registry counts as curated whatever its status, so the public app never
triggers live generation for it. Only scripts/curate_ai.py writes this file.
"""

from __future__ import annotations

import json

import catalog

REGISTRY = catalog.APP_DIR / "frozen_ai" / "curated.json"
SCHEMA_VERSION = "curated_registry/v1"
MAX_ATTEMPTS = 3

_cache: dict | None = None


def load(*, fresh: bool = False) -> dict:
    global _cache
    if _cache is None or fresh:
        try:
            _cache = json.loads(REGISTRY.read_text())
        except (OSError, json.JSONDecodeError):
            _cache = {"schema_version": SCHEMA_VERSION, "videos": {}}
    return _cache


def save(registry: dict) -> None:
    global _cache
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=1) + "\n")
    _cache = registry


def entry(video_id: str) -> dict | None:
    return load()["videos"].get(str(video_id))


def is_curated(video_id: str) -> bool:
    return entry(video_id) is not None


def approved_record(video_id: str) -> dict | None:
    item = entry(video_id)
    if not item or item["qa_status"] != "approved" or item.get("selected_attempt") is None:
        return None
    return item["attempts"][item["selected_attempt"] - 1]
