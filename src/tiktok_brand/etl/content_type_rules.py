"""Content type classification from caption using configs/feature_rules.yaml."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .rule_config import get_content_type_rules

# P0–P6 classification priority (match order, not content value ranking)
CLASSIFICATION_PRIORITY: Tuple[str, ...] = (
    "social_viral",
    "official_campaign",
    "story_heritage",
    "tutorial_utility",
    "shopping_haul",
    "vibe_ootd",
    "community_collab",
)


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = phrase.lower().strip()
    if not phrase:
        return False
    if " " in phrase or "-" in phrase or "'" in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _match_strong(text: str, phrases: List[str]) -> bool:
    return any(_contains_phrase(text, p) for p in phrases or [])


def _match_weak_pairs(text: str, pairs: List[List[str]]) -> bool:
    for pair in pairs or []:
        if len(pair) < 2:
            continue
        if all(_contains_phrase(text, p) for p in pair):
            return True
    return False


def _match_bts_heritage(text: str, rule_cfg: Dict[str, Any]) -> bool:
    if _match_strong(text, rule_cfg.get("strong") or []):
        return True
    if _contains_phrase(text, "bts"):
        return _match_strong(text, rule_cfg.get("bts_context") or [])
    for phrase in rule_cfg.get("weak") or []:
        if phrase.lower() == "bts":
            continue
        if _contains_phrase(text, phrase):
            return True
    return _match_weak_pairs(text, rule_cfg.get("require_pairs") or [])


def _match_category(text: str, rule_cfg: Dict[str, Any]) -> bool:
    if _match_strong(text, rule_cfg.get("strong") or []):
        return True
    if _match_weak_pairs(text, rule_cfg.get("require_pairs") or []):
        return True
    weak_hits = sum(1 for p in (rule_cfg.get("weak") or []) if _contains_phrase(text, p))
    return weak_hits >= 2


def infer_content_type(caption: Optional[str]) -> Optional[str]:
    if not caption or not str(caption).strip():
        return None
    text = str(caption).lower()
    rules = get_content_type_rules()

    for category in CLASSIFICATION_PRIORITY:
        cfg = rules.get(category) or {}
        if category == "story_heritage":
            if _match_bts_heritage(text, cfg):
                return category
        elif _match_category(text, cfg):
            return category
    return None
