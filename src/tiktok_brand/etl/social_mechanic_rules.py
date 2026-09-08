"""TikTok platform propagation mechanics (multi-label) from caption keywords.

Uses ``match_rule_cfg`` with ``allow_single_weak_hashtag=True``:
  strong / require_pairs / >=2 weak hits / or 1 weak as #hashtag.
(content_type keeps the stricter >=2-weak-only policy.)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .keyword_match import match_rule_cfg
from .rule_config import get_social_mechanic_rules

# Stable output order only (not exclusive).
SOCIAL_MECHANIC_ORDER: Tuple[str, ...] = (
    "trend",
    "challenge",
    "pov",
    "duet_stitch",
    "template_remix",
    "audio_driven",
    "bts",
    "grwm_ootd_format",
)


def infer_social_mechanics(caption: Optional[str]) -> List[str]:
    """Return all matching social_mechanic labels (may be empty)."""
    if not caption or not str(caption).strip():
        return []
    text = str(caption).lower()
    rules = get_social_mechanic_rules()
    hits: List[str] = []
    for name in SOCIAL_MECHANIC_ORDER:
        cfg = rules.get(name) or {}
        if cfg and match_rule_cfg(text, cfg, allow_single_weak_hashtag=True):
            hits.append(name)
    return hits


def social_mechanics_to_flags(
    labels: Optional[Sequence[str]],
) -> Dict[str, bool]:
    present = set(labels or [])
    return {f"is_mechanic_{m}": (m in present) for m in SOCIAL_MECHANIC_ORDER}
