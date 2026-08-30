"""Load feature engineering rules from configs/feature_rules.yaml."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Pattern, Tuple

import yaml

DEFAULT_RULES_PATH = Path("configs/feature_rules.yaml")


@lru_cache(maxsize=1)
def load_feature_rules(path: str = str(DEFAULT_RULES_PATH)) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _compile_patterns(patterns: List[str]) -> List[Pattern[str]]:
    compiled: List[Pattern[str]] = []
    for p in patterns or []:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error:
            continue
    return compiled


@lru_cache(maxsize=1)
def get_cta_pattern_groups(path: str = str(DEFAULT_RULES_PATH)) -> Dict[str, List[Pattern[str]]]:
    rules = load_feature_rules(path)
    groups = rules.get("cta_patterns") or {}
    return {name: _compile_patterns(patterns) for name, patterns in groups.items()}


@lru_cache(maxsize=1)
def get_content_type_rules(path: str = str(DEFAULT_RULES_PATH)) -> Dict[str, Dict[str, Any]]:
    rules = load_feature_rules(path)
    return rules.get("content_type_rules") or {}


@lru_cache(maxsize=1)
def get_engagement_weights(path: str = str(DEFAULT_RULES_PATH)) -> Tuple[float, float, float]:
    rules = load_feature_rules(path)
    w = rules.get("engagement_weights") or {}
    return float(w.get("like", 0.2)), float(w.get("comment", 0.3)), float(w.get("share", 0.5))
