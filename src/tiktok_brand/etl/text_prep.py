"""Caption / hashtag text preparation for rule matching and embeddings."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

_URL_RE = re.compile(r"https?://\S+")
_HASHTAG_TOKEN_RE = re.compile(r"#(\w+)", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")
_BRAND_MENTION_RE = re.compile(r"@(?:nike|adidas)\b", re.IGNORECASE)

# Brand + traffic tokens stripped from clustering text only (not modeling embedding_text).
CLUSTER_STOPWORDS = frozenset(
    {
        "nike",
        "adidas",
        "fyp",
        "foryou",
        "foryoupage",
        "viral",
        "trending",
        "explore",
        "xyzbca",
        "fy",
        "fypp",
        "parati",
        "pourtoi",
        "xuhuong",
        "blowthisup",
    }
)


def _split_camel_case(token: str) -> str:
    if not token:
        return ""
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token)
    parts = parts.replace("_", " ").replace("-", " ")
    return _MULTI_SPACE_RE.sub(" ", parts).strip().lower()


def normalize_hashtag_for_embedding(tag: str) -> str:
    raw = str(tag).strip().lstrip("#")
    if not raw:
        return ""
    split = _split_camel_case(raw)
    return f"{split} {raw.lower()}".strip()


def clean_caption(caption: Optional[str]) -> str:
    if not caption:
        return ""
    text = str(caption)
    text = _URL_RE.sub(" ", text)
    text = _HASHTAG_TOKEN_RE.sub(" ", text)
    text = text.lower()
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def build_embedding_text(caption: Optional[str], hashtags: Optional[List[str]]) -> str:
    caption_part = clean_caption(caption)
    tag_parts: List[str] = []
    for tag in hashtags or []:
        normalized = normalize_hashtag_for_embedding(tag)
        if normalized:
            tag_parts.append(normalized)
    hashtag_part = " ".join(tag_parts)
    if caption_part and hashtag_part:
        return f"{caption_part} [HASHTAGS] {hashtag_part}".strip()
    return caption_part or hashtag_part


def _alnum_key(token: str) -> str:
    """Lowercase alphanumeric-only key for stopword matching (fypシ → fyp)."""
    return re.sub(r"[^a-z0-9]", "", str(token).lower())


def is_cluster_stopword(token: str) -> bool:
    key = _alnum_key(token)
    return bool(key) and key in CLUSTER_STOPWORDS


def _dedupe_preserve_order(tags: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for tag in tags:
        raw = str(tag).strip().lstrip("#")
        if not raw:
            continue
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out


def hashtag_split_for_cluster(tag: str) -> str:
    """Split-only hashtag form (no raw duplicate). Empty if only stopwords remain."""
    raw = str(tag).strip().lstrip("#")
    if not raw:
        return ""
    if is_cluster_stopword(raw):
        return ""
    split = _split_camel_case(raw)
    kept = [w for w in split.split() if w and not is_cluster_stopword(w)]
    return " ".join(kept)


def scrub_caption_for_cluster(caption_clean: Optional[str]) -> str:
    """
    From caption_clean: drop @nike/@adidas; drop standalone brand/traffic tokens;
    keep other @mentions and content words.
    """
    if not caption_clean:
        return ""
    text = _BRAND_MENTION_RE.sub(" ", str(caption_clean))
    pieces: List[str] = []
    for tok in text.split():
        if tok.startswith("@"):
            # brand mentions already removed; keep creators / athletes
            pieces.append(tok.lower())
            continue
        if is_cluster_stopword(tok):
            continue
        pieces.append(tok)
    return _MULTI_SPACE_RE.sub(" ", " ".join(pieces)).strip()


def build_cluster_embedding_text(
    caption: Optional[str],
    hashtags: Optional[Sequence[str]] = None,
    *,
    caption_clean: Optional[str] = None,
) -> str:
    """
    Clustering-only text (does not replace modeling ``embedding_text``).

    - caption_clean base (or clean_caption(caption))
    - remove standalone nike/adidas and @nike/@adidas; keep other mentions
    - append order-deduped hashtags as split-only forms (no raw, no [HASHTAGS])
    - drop brand + traffic stopwords from caption tokens and hashtag splits
    """
    base = caption_clean if caption_clean is not None else clean_caption(caption)
    caption_part = scrub_caption_for_cluster(base)

    tag_parts: List[str] = []
    for tag in _dedupe_preserve_order(hashtags or []):
        split = hashtag_split_for_cluster(tag)
        if split:
            tag_parts.append(split)

    hashtag_part = " ".join(tag_parts)
    if caption_part and hashtag_part:
        return f"{caption_part} {hashtag_part}".strip()
    return caption_part or hashtag_part
