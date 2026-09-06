"""Caption language detection for sentiment routing (feature ETL)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langdetect import DetectorFactory, LangDetectException, detect_langs

# Deterministic langdetect (short TikTok captions are noisy otherwise)
DetectorFactory.seed = 0

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@[\w.]+")
_HASHTAG_RE = re.compile(r"#\w+", re.UNICODE)
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U00002600-\U000027BF"
    "\U0001FA00-\U0001FAFF"
    "\U0000FE00-\U0000FE0F"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)
_NON_LATIN_RE = re.compile(r"[\u0400-\u04FF\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF\u0600-\u06FF]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")

# Brand / product proper nouns that bias langdetect toward English
_BRAND_PROPER_NOUNS = frozenset(
    {
        "nike",
        "adidas",
        "jordan",
        "jumpman",
        "samba",
        "sambas",
        "gazelle",
        "gazelles",
        "spezial",
        "spezials",
        "ultraboost",
        "airmax",
        "airforce",
        "dunk",
        "yeezy",
        "originals",
        "forum",
        "predator",
        "superstar",
        "stan",
        "smith",
        "boost",
        "techfleece",
        "niketech",
    }
)

_MIN_ALPHA_CHARS = 3
_MIN_DETECT_CHARS = 8


def strip_for_lang_detect(caption: Optional[str]) -> str:
    """Remove URLs, @mentions, hashtags, and brand proper nouns before detection."""
    if not caption:
        return ""
    text = str(caption)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(" ", text)
    tokens = re.findall(r"[A-Za-z0-9']+|[^A-Za-z0-9\s]", text)
    kept: List[str] = []
    for tok in tokens:
        if tok.lower().strip("'") in _BRAND_PROPER_NOUNS:
            continue
        kept.append(tok)
    text = " ".join(kept)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _alpha_char_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _is_emoji_only(original: str, stripped: str) -> bool:
    if not original or not str(original).strip():
        return False
    no_hash = _HASHTAG_RE.sub(" ", str(original))
    no_url = _URL_RE.sub(" ", no_hash)
    no_mention = _MENTION_RE.sub(" ", no_url)
    remainder = _EMOJI_RE.sub(" ", no_mention)
    remainder = re.sub(r"\s+", " ", remainder).strip()
    # After removing emoji/noise, little/no linguistic content left
    if _alpha_char_count(remainder) >= _MIN_ALPHA_CHARS:
        return False
    return bool(_EMOJI_RE.search(str(original))) and _alpha_char_count(stripped) < _MIN_ALPHA_CHARS


def _hashtag_scripts_non_latin(hashtags: Optional[List[str]], caption: Optional[str]) -> bool:
    tags = list(hashtags or [])
    if caption:
        tags.extend(_HASHTAG_RE.findall(str(caption)))
    blob = " ".join(str(t) for t in tags)
    return bool(_NON_LATIN_RE.search(blob))


def _body_has_mixed_scripts(text: str) -> bool:
    has_latin = bool(_LATIN_WORD_RE.search(text))
    has_non_latin = bool(_NON_LATIN_RE.search(text))
    return has_latin and has_non_latin


def detect_caption_language(
    caption: Optional[str],
    hashtags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Detect caption body language for sentiment routing.

    Returns:
      caption_lang: en | es|vi|ru|ko|... | mixed | und
      caption_lang_confidence: float in [0, 1]
      has_mixed_language: True if body lang differs from hashtag script / multi-script body
      is_emoji_only: True when caption is essentially emoji (+ noise)
    """
    original = str(caption or "")
    stripped = strip_for_lang_detect(original)
    emoji_only = _is_emoji_only(original, stripped)
    hashtag_non_latin = _hashtag_scripts_non_latin(hashtags, original)

    und = {
        "caption_lang": "und",
        "caption_lang_confidence": 0.0,
        "has_mixed_language": bool(hashtag_non_latin),
        "is_emoji_only": emoji_only,
    }

    if emoji_only:
        und["has_mixed_language"] = False
        return und

    if _alpha_char_count(stripped) < _MIN_ALPHA_CHARS or len(stripped) < _MIN_DETECT_CHARS:
        # hashtag-only / brand-only / too short
        return und

    if _body_has_mixed_scripts(stripped):
        return {
            "caption_lang": "mixed",
            "caption_lang_confidence": 0.0,
            "has_mixed_language": True,
            "is_emoji_only": False,
        }

    try:
        candidates = detect_langs(stripped)
    except LangDetectException:
        return und

    if not candidates:
        return und

    top = candidates[0]
    lang = str(top.lang).lower()
    conf = float(top.prob)

    has_mixed = False
    if lang == "en" and hashtag_non_latin:
        has_mixed = True
    elif lang != "en" and hashtag_non_latin:
        # non-EN body + other-script tags still mixed signal
        has_mixed = True

    return {
        "caption_lang": lang,
        "caption_lang_confidence": conf,
        "has_mixed_language": has_mixed,
        "is_emoji_only": False,
    }
