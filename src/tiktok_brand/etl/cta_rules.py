"""CTA detection from caption text using regex rules in configs/feature_rules.yaml."""

from __future__ import annotations

import re
from typing import Dict, List, Pattern

from .rule_config import get_cta_pattern_groups


def matches_any(text: str, patterns: List[Pattern[str]]) -> bool:
    if not text or not patterns:
        return False
    return any(p.search(text) for p in patterns)


def detect_cta_flags(caption: str) -> Dict[str, bool]:
    text = (caption or "").lower()
    groups = get_cta_pattern_groups()
    purchase = matches_any(text, groups.get("purchase", []))
    engagement = matches_any(text, groups.get("engagement", []))
    discovery = matches_any(text, groups.get("discovery_traffic", []))
    promo = matches_any(text, groups.get("promo", []))
    return {
        "has_purchase_cta": purchase,
        "has_engagement_cta": engagement,
        "has_discovery_traffic_cta": discovery,
        "has_promo_language": promo,
        "has_cta": purchase or engagement or discovery,
    }
