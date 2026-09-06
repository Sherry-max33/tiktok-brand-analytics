"""Machine translation helpers for sentiment routing (non-English → English)."""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

TranslateFn = Callable[[str, Optional[str]], str]

# langdetect ISO → deep_translator MyMemory names
_MYMEMORY_LANG = {
    "af": "afrikaans",
    "ar": "arabic",
    "bg": "bulgarian",
    "bn": "bengali",
    "ca": "catalan",
    "cs": "czech",
    "cy": "welsh",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "fa": "persian",
    "fi": "finnish",
    "fr": "french",
    "gu": "gujarati",
    "he": "hebrew",
    "hi": "hindi",
    "hr": "croatian",
    "hu": "hungarian",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "kn": "kannada",
    "ko": "korean",
    "lt": "lithuanian",
    "lv": "latvian",
    "mk": "macedonian",
    "ml": "malayalam",
    "mr": "marathi",
    "ms": "malay",
    "ne": "nepali",
    "nl": "dutch",
    "no": "norwegian bokmål",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sk": "slovak",
    "sl": "slovenian",
    "sq": "albanian",
    "sv": "swedish",
    "sw": "swahili",
    "ta": "tamil",
    "te": "telugu",
    "th": "thai",
    "tl": "filipino",
    "tr": "turkish",
    "uk": "ukrainian",
    "ur": "urdu",
    "vi": "vietnamese",
    "zh-cn": "chinese simplified",
    "zh-tw": "chinese traditional",
    "zh": "chinese simplified",
}

_BAD_MT_RE = re.compile(r"Error \d+|Server Error|unexpected error|try again later", re.I)


def _looks_bad(text: str) -> bool:
    if not text or not str(text).strip():
        return True
    t = str(text).strip()
    if _BAD_MT_RE.search(t):
        return True
    if t.startswith("<!DOCTYPE") or t.startswith("<html"):
        return True
    return False


def _google_translate(text: str, source_lang: Optional[str]) -> str:
    from deep_translator import GoogleTranslator

    src = (source_lang or "auto").lower()
    if src in {"und", "mixed", ""}:
        src = "auto"
    out = GoogleTranslator(source=src, target="en").translate(text)
    if out is None or _looks_bad(str(out)):
        raise RuntimeError(f"Google MT bad result: {out!r}")
    return str(out).strip()


def _mymemory_translate(text: str, source_lang: Optional[str]) -> str:
    from deep_translator import MyMemoryTranslator

    src_iso = (source_lang or "").lower()
    src_name = _MYMEMORY_LANG.get(src_iso)
    if not src_name:
        # MyMemory needs named languages; fall back to spanish-like auto not available
        raise RuntimeError(f"MyMemory has no mapping for lang={source_lang!r}")
    out = MyMemoryTranslator(source=src_name, target="english").translate(text)
    if out is None or _looks_bad(str(out)):
        raise RuntimeError(f"MyMemory MT bad result: {out!r}")
    return str(out).strip()


def _default_translate(text: str, source_lang: Optional[str]) -> str:
    """Prefer Google; fall back to MyMemory (Google free endpoint is often flaky)."""
    errors = []
    try:
        return _google_translate(text, source_lang)
    except Exception as exc:
        errors.append(f"google:{exc}")
        logger.debug("Google MT failed: %s", exc)
    try:
        return _mymemory_translate(text, source_lang)
    except Exception as exc:
        errors.append(f"mymemory:{exc}")
        logger.debug("MyMemory MT failed: %s", exc)
    # Last try: Google auto
    try:
        return _google_translate(text, "auto")
    except Exception as exc:
        errors.append(f"google_auto:{exc}")
    raise RuntimeError("MT failed: " + " | ".join(errors))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.8, min=1, max=8), reraise=True)
def _translate_with_retry(text: str, source_lang: Optional[str], fn: TranslateFn) -> str:
    return fn(text, source_lang)


def translate_to_english(
    text: str,
    source_lang: Optional[str] = None,
    *,
    translate_fn: Optional[TranslateFn] = None,
) -> str:
    """
    Translate text to English.

    ``translate_fn(text, source_lang) -> english`` can be injected for tests / offline runs.
    Default: GoogleTranslator with MyMemory fallback.
    """
    if not text or not str(text).strip():
        raise ValueError("empty text for translation")
    fn = translate_fn or _default_translate
    return _translate_with_retry(str(text).strip(), source_lang, fn)
