"""Load feature engineering rules from configs/feature_rules.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

DEFAULT_RULES_PATH = Path("configs/feature_rules.yaml")


@lru_cache(maxsize=1)
def load_feature_rules(path: str = str(DEFAULT_RULES_PATH)) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_content_type_rules(path: str = str(DEFAULT_RULES_PATH)) -> Dict[str, Dict[str, Any]]:
    rules = load_feature_rules(path)
    return rules.get("content_type_rules") or {}


@lru_cache(maxsize=1)
def get_engagement_weights(path: str = str(DEFAULT_RULES_PATH)) -> Tuple[float, float, float, float]:
    rules = load_feature_rules(path)
    w = rules.get("engagement_weights") or {}
    return (
        float(w.get("like", 0.10)),
        float(w.get("comment", 0.25)),
        float(w.get("share", 0.30)),
        float(w.get("collect", 0.35)),
    )
