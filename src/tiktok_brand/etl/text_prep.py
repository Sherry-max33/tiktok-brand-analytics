"""Caption / hashtag text preparation for rule matching and embeddings."""

from __future__ import annotations

import re
from typing import List, Optional

_URL_RE = re.compile(r"https?://\S+")
_HASHTAG_TOKEN_RE = re.compile(r"#(\w+)")
_MULTI_SPACE_RE = re.compile(r"\s+")


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
