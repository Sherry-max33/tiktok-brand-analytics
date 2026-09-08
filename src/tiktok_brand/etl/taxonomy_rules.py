"""Load configs/taxonomy.yaml — multi-label product / style features (feature ETL)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from .keyword_match import contains_phrase

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


def _evidence_text(caption: Optional[str], tags: Optional[List[str]]) -> str:
    parts: List[str] = []
    if caption:
        parts.append(str(caption).lower())
    for t in tags or []:
        raw = str(t).strip().lstrip("#").lower()
        if raw:
            parts.append(f"#{raw}")
            parts.append(raw)
    return " ".join(parts)


def infer_brand_styles(
    tags: Optional[List[str]] = None,
    path: str = str(DEFAULT_TAXONOMY_PATH),
    *,
    caption: Optional[str] = None,
) -> List[str]:
    """
    Multi-label brand positioning = crawl-seed hashtag prior ∪ caption/hashtag keywords.

    Does **not** map product SKUs (samba/gazelle) to style — only seed_style_map
    + style_keywords evidence. Seed tags come from configs/hashtags.yaml, not @accounts.
    """
    cfg = load_taxonomy(path)
    # Prefer seed_style_map; fall back to short-lived aliases if present.
    seed_map = {
        str(k).lower(): str(v)
        for k, v in (
            cfg.get("seed_style_map")
            or cfg.get("account_style_map")
            or cfg.get("brand_style_map")
            or {}
        ).items()
    }
    style_kw = cfg.get("style_keywords") or {}

    hits: List[str] = []
    for t in tags or []:
        key = str(t).lower().lstrip("#")
        if key in seed_map:
            hits.append(seed_map[key])

    text = _evidence_text(caption, tags)
    if text.strip():
        for style in BRAND_STYLE_ORDER:
            phrases = style_kw.get(style) or []
            if any(contains_phrase(text, str(p)) for p in phrases):
                hits.append(style)

    return _ordered_unique(hits, BRAND_STYLE_ORDER)


def brand_styles_to_flags(labels: Optional[Sequence[str]]) -> Dict[str, bool]:
    """One-hot flags for modeling (all known styles)."""
    present = set(labels or [])
    return {f"brand_style_{s}": (s in present) for s in BRAND_STYLE_ORDER}


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
      2) Else hashtag scan (strong maps + apparel product terms; weak fashion
         tags alone do not assign apparel)
      3) Else caption heuristics (strong apparel terms; weak fashion needs strong)
      4) Else → ["uncategorized"]
    """
    cfg = load_taxonomy(path)
    line_to_category = {str(k): str(v) for k, v in (cfg.get("line_to_category_map") or {}).items()}
    category_map = {
        str(k).lower(): str(v) for k, v in (cfg.get("product_category_map") or {}).items()
    }
    apparel_terms = {
        str(x).lower().lstrip("#") for x in (cfg.get("category_apparel_product_terms") or [])
    }
    weak_fashion = {
        str(x).lower().lstrip("#") for x in (cfg.get("category_weak_fashion_tags") or [])
    }
    caption_kw = cfg.get("category_caption_keywords") or {}
    accessories = [str(k).lower() for k in (cfg.get("accessories_keywords") or [])]

    lines = [str(x) for x in (product_lines or []) if x]

    # (1) product_lines present → only line_to_category_map
    if lines:
        hits = [line_to_category[line] for line in lines if line in line_to_category]
        out = _ordered_unique(hits, PRODUCT_CATEGORY_ORDER)
        return out if out else ["uncategorized"]

    # (2) hashtag scan
    tag_keys = [str(t).lower().lstrip("#") for t in (tags or []) if str(t).strip()]
    hits: List[str] = []
    has_weak_fashion = False
    has_apparel_term = False
    for t in tag_keys:
        if t in weak_fashion:
            has_weak_fashion = True
            continue  # never map weak fashion alone via category_map
        if t in category_map:
            hits.append(category_map[t])
        if t in apparel_terms:
            has_apparel_term = True
            hits.append("apparel")
        if t in accessories:
            hits.append("accessories")

    # weak fashion + explicit apparel product term → apparel (term already added;
    # keep explicit for clarity / future multi-signal rules)
    if has_weak_fashion and has_apparel_term:
        hits.append("apparel")

    if hits:
        return _ordered_unique(hits, PRODUCT_CATEGORY_ORDER)

    # (3) caption heuristics
    text = str(caption or "").lower()
    apparel_strong = [str(w).lower() for w in (caption_kw.get("apparel") or [])]
    apparel_weak = [str(w).lower() for w in (caption_kw.get("apparel_weak") or [])]
    has_apparel_strong = any(word in text for word in apparel_strong)
    has_apparel_weak = any(word in text for word in apparel_weak)
    if has_apparel_strong:
        hits.append("apparel")
    elif has_apparel_weak and has_apparel_term:
        # caption weak fashion + hashtag apparel product term
        hits.append("apparel")
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
