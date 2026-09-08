"""Content type classification from caption using configs/feature_rules.yaml.

Multi-label: every matching category is returned. ``CLASSIFICATION_ORDER`` only
controls stable output order (not winner-take-all).

Weak matching is shared with ``social_mechanic`` via ``keyword_match.match_rule_cfg``
(strong / require_pairs / >=2 weak hits).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .keyword_match import contains_phrase, match_rule_cfg, match_strong, match_weak_pairs, weak_hit_count
from .rule_config import get_content_type_rules

# Structured collab cue: "adidas x diesel", "@creator x nike" (not bare @handles, not "10 x 10").
_COLLAB_X_RE = re.compile(
    r"(?<![a-z0-9])([a-z0-9@][\w.@&'-]{0,40})\s+x\s+([a-z0-9@][\w.@&'-]{0,40})(?![a-z0-9])",
    re.IGNORECASE,
)

# Stable output order (specific → general). Not mutually exclusive.
CLASSIFICATION_ORDER: Tuple[str, ...] = (
    "official_campaign",
    "collaboration",
    "community_impact",
    "story_heritage",
    "tutorial_utility",
    "product_review",
    "product_promo",
    "product_showcase",
    "vibe_ootd",
)

# Back-compat alias
CLASSIFICATION_PRIORITY = CLASSIFICATION_ORDER


def _match_story_heritage(text: str, rule_cfg: Dict[str, Any]) -> bool:
    """Heritage/craft story. Bare BTS / on-set needs craft/history context."""
    if match_strong(text, rule_cfg.get("strong") or []):
        return True

    bts_triggers = rule_cfg.get("bts_triggers") or [
        "behind the scenes",
        "behind-the-scenes",
        "bts",
        "on set",
    ]
    has_bts = any(contains_phrase(text, p) for p in bts_triggers)
    if has_bts:
        return match_strong(text, rule_cfg.get("bts_context") or [])

    if match_weak_pairs(text, rule_cfg.get("require_pairs") or []):
        return True
    return weak_hit_count(text, rule_cfg.get("weak") or []) >= 2


def _looks_like_collab_x(text: str) -> bool:
    for match in _COLLAB_X_RE.finditer(text):
        left, right = match.group(1), match.group(2)
        if left.isdigit() and right.isdigit():
            continue
        if len(left) >= 2 and len(right) >= 2:
            return True
    return False


def _match_collaboration(text: str, rule_cfg: Dict[str, Any]) -> bool:
    if _looks_like_collab_x(text):
        return True
    return match_rule_cfg(text, rule_cfg)


def infer_content_types(caption: Optional[str]) -> List[str]:
    """Return all matching content_type labels (may be empty)."""
    if not caption or not str(caption).strip():
        return []
    text = str(caption).lower()
    rules = get_content_type_rules()
    hits: List[str] = []
    for category in CLASSIFICATION_ORDER:
        cfg = rules.get(category) or {}
        if category == "story_heritage":
            matched = _match_story_heritage(text, cfg)
        elif category == "collaboration":
            matched = _match_collaboration(text, cfg)
        else:
            matched = match_rule_cfg(text, cfg)
        if matched:
            hits.append(category)
    return hits


def infer_content_type(caption: Optional[str]) -> List[str]:
    """Alias of ``infer_content_types`` (multi-label list)."""
    return infer_content_types(caption)


def content_types_to_flags(
    labels: Optional[Sequence[str]],
) -> Dict[str, bool]:
    """One-hot flags for modeling (all known categories)."""
    present = set(labels or [])
    return {f"is_content_{c}": (c in present) for c in CLASSIFICATION_ORDER}
