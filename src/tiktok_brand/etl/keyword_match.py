"""Shared keyword matching for content_type and social_mechanic rules.

Default weak policy (content_type):
  - strong phrase → match
  - require_pairs → match
  - weak: need >= 2 distinct weak phrase hits

Optional ``allow_single_weak_hashtag`` (social_mechanic):
  - also match when exactly 1 weak hit appears as a hashtag (#viral, #trend, …)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


def contains_phrase(text: str, phrase: str) -> bool:
    """Match phrase in text; multi-word phrases also match compacted hashtag forms.

    Example: ``created with adidas`` matches ``#createdwithadidas``.
    """
    phrase = phrase.lower().strip()
    if not phrase:
        return False
    if any(ch in phrase for ch in (" ", "-", "'", "’", "/", ":")):
        if phrase in text:
            return True
        compact = re.sub(r"[^a-z0-9]", "", phrase)
        if not compact:
            return False
        if f"#{compact}" in text:
            return True
        return bool(re.search(rf"\b{re.escape(compact)}\b", text))
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))


def phrase_as_hashtag(text: str, phrase: str) -> bool:
    """True if phrase appears in hashtag form (#trend, #viral, …)."""
    raw = phrase.lower().strip().lstrip("#")
    if not raw:
        return False
    if f"#{raw}" in text:
        return True
    compact = re.sub(r"[^a-z0-9]", "", raw)
    if not compact:
        return False
    return bool(re.search(rf"#\w*{re.escape(compact)}\w*", text))


def match_strong(text: str, phrases: List[str]) -> bool:
    return any(contains_phrase(text, p) for p in phrases or [])


def match_weak_pairs(text: str, pairs: List[List[str]]) -> bool:
    for pair in pairs or []:
        if len(pair) < 2:
            continue
        if all(contains_phrase(text, p) for p in pair):
            return True
    return False


def weak_hit_phrases(text: str, weak: List[str]) -> List[str]:
    return [p for p in (weak or []) if contains_phrase(text, p)]


def weak_hit_count(text: str, weak: List[str]) -> int:
    return len(weak_hit_phrases(text, weak))


def match_weak(
    text: str,
    weak: List[str],
    *,
    allow_single_weak_hashtag: bool = False,
) -> bool:
    hits = weak_hit_phrases(text, weak)
    if len(hits) >= 2:
        return True
    if allow_single_weak_hashtag and len(hits) == 1:
        return phrase_as_hashtag(text, hits[0])
    return False


def match_rule_cfg(
    text: str,
    rule_cfg: Dict[str, Any],
    *,
    allow_single_weak_hashtag: bool = False,
) -> bool:
    """Category match. Default = content_type policy; social_mechanic may enable hashtag gate."""
    if match_strong(text, rule_cfg.get("strong") or []):
        return True
    if match_weak_pairs(text, rule_cfg.get("require_pairs") or []):
        return True
    return match_weak(
        text,
        rule_cfg.get("weak") or [],
        allow_single_weak_hashtag=allow_single_weak_hashtag,
    )
