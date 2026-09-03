"""Load configs/taxonomy.yaml — multi-label product / style features (feature ETL)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

DEFAULT_TAXONOMY_PATH = Path("configs/taxonomy.yaml")

# Fixed display / analysis order (not mutually exclusive within a field)
BRAND_STYLE_ORDER = ("performance", "technical", "lifestyle", "retro")
PRODUCT_CATEGORY_ORDER = ("shoes", "apparel", "accessories", "uncategorized")


@lru_cache(maxsize=1)
def load_taxonomy(path: str = str(DEFAULT_TAXONOMY_PATH)) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ordered_unique(values: Sequence[str], order: Optional[Sequence[str]] = None) -> List[str]:
    seen: set[str] = set()
    collected: List[str] = []
    for v in values:
        if not v or v in seen:
            continue
        seen.add(v)
        collected.append(v)
    if order is None:
        return sorted(collected)
    rank = {name: i for i, name in enumerate(order)}
    return sorted(collected, key=lambda x: (rank.get(x, len(rank)), x))


def infer_brand_styles(
    tags: Optional[List[str]], path: str = str(DEFAULT_TAXONOMY_PATH)
) -> List[str]:
    """All matching brand styles (deduped, fixed order). Empty = unrecognized."""
    style_map = {
        str(k).lower(): str(v) for k, v in (load_taxonomy(path).get("brand_style_map") or {}).items()
    }
    hits: List[str] = []
    for t in tags or []:
        key = str(t).lower()
        if key in style_map:
            hits.append(style_map[key])
    return _ordered_unique(hits, BRAND_STYLE_ORDER)


def infer_product_lines(
    tags: Optional[List[str]], path: str = str(DEFAULT_TAXONOMY_PATH)
) -> List[str]:
    """All matching product lines from normalized hashtags (deduped, sorted)."""
    product_map = {
        str(k).lower(): str(v) for k, v in (load_taxonomy(path).get("product_line_map") or {}).items()
    }
    hits: List[str] = []
    for t in tags or []:
        key = str(t).lower()
        if key in product_map:
            hits.append(product_map[key])
    return _ordered_unique(hits)


def infer_product_categories(
    *,
    product_lines: Optional[List[str]],
    tags: Optional[List[str]],
    caption: Optional[str],
    path: str = str(DEFAULT_TAXONOMY_PATH),
) -> List[str]:
    """
    Cascaded multi-label categories (see configs/taxonomy.yaml):

      1) If product_lines non-empty → map each via line_to_category_map
      2) Else scan hashtags → product_category_map (+ accessories tags)
      3) Else caption / accessories keyword heuristics
      4) Else → ["uncategorized"]
    """
    cfg = load_taxonomy(path)
    line_to_category = {str(k): str(v) for k, v in (cfg.get("line_to_category_map") or {}).items()}
    category_map = {
        str(k).lower(): str(v) for k, v in (cfg.get("product_category_map") or {}).items()
    }
    caption_kw = cfg.get("category_caption_keywords") or {}
    accessories = [str(k).lower() for k in (cfg.get("accessories_keywords") or [])]

    lines = [str(x) for x in (product_lines or []) if x]

    # (1) product_lines present → only line_to_category_map
    if lines:
        hits = [line_to_category[line] for line in lines if line in line_to_category]
        out = _ordered_unique(hits, PRODUCT_CATEGORY_ORDER)
        return out if out else ["uncategorized"]

    # (2) else hashtag → product_category_map
    hits: List[str] = []
    for tag in tags or []:
        t = str(tag).lower()
        if t in category_map:
            hits.append(category_map[t])
        if t in accessories:
            hits.append("accessories")
    if hits:
        return _ordered_unique(hits, PRODUCT_CATEGORY_ORDER)

    # (3) else caption heuristics
    text = str(caption or "").lower()
    for word in caption_kw.get("apparel") or []:
        if word in text:
            hits.append("apparel")
            break
    for word in caption_kw.get("shoes") or []:
        if word in text:
            hits.append("shoes")
            break
    if any(word in text for word in accessories):
        hits.append("accessories")
    if hits:
        return _ordered_unique(hits, PRODUCT_CATEGORY_ORDER)

    # (4) none → uncategorized
    return ["uncategorized"]
