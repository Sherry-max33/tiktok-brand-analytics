"""Caption sentiment scoring (VADER) with language routing + optional MT."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .language_rules import detect_caption_language
from .rule_config import load_feature_rules
from .text_prep import clean_caption
from .translation import TranslateFn, translate_to_english

_ANALYZER: Optional[SentimentIntensityAnalyzer] = None

METHOD_VADER_EN = "vader_en"
METHOD_VADER_VIA_MT = "vader_via_mt"
METHOD_NOT_SCORED = "not_scored"
METHOD_NOT_SCORED_NON_EN = "not_scored_non_en"

TRANSLATION_NOT_APPLICABLE = "not_applicable"
TRANSLATION_PENDING = "pending"
TRANSLATION_DONE = "done"
TRANSLATION_FAILED = "failed"
TRANSLATION_SKIPPED = "skipped"


def _analyzer() -> SentimentIntensityAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER


def _sentiment_cfg(rules_path: str = "configs/feature_rules.yaml") -> Dict[str, Any]:
    return load_feature_rules(rules_path).get("sentiment") or {}


def _en_confidence_threshold(rules_path: str = "configs/feature_rules.yaml") -> float:
    return float(_sentiment_cfg(rules_path).get("en_confidence_threshold", 0.80))


def _mt_enabled(rules_path: str = "configs/feature_rules.yaml") -> bool:
    return bool(_sentiment_cfg(rules_path).get("mt_enabled", True))


def _mt_min_chars(rules_path: str = "configs/feature_rules.yaml") -> int:
    return int(_sentiment_cfg(rules_path).get("mt_min_chars", 8))


def vader_compound(text: str) -> float:
    return float(_analyzer().polarity_scores(text)["compound"])


def score_caption_sentiment(
    caption: Optional[str],
    hashtags: Optional[List[str]] = None,
    *,
    rules_path: str = "configs/feature_rules.yaml",
    translate_fn: Optional[TranslateFn] = None,
) -> Dict[str, Any]:
    """
    Language-routed sentiment:

    - High-confidence English → VADER on native text (``vader_en``)
    - Other languages → optional MT to English then VADER (``vader_via_mt``)
    - ``und`` / ``mixed`` / emoji-only / low-conf EN → not scored

    Native EN and MT scores stay distinguishable via ``sentiment_method``.
    """
    original = "" if caption is None else str(caption)
    lang_info = detect_caption_language(original, hashtags)
    lang = lang_info["caption_lang"]
    conf = float(lang_info["caption_lang_confidence"])
    threshold = _en_confidence_threshold(rules_path)
    sentiment_text = clean_caption(original)

    base: Dict[str, Any] = {
        "caption_original": original,
        "caption_lang": lang,
        "caption_lang_confidence": conf,
        "has_mixed_language": bool(lang_info["has_mixed_language"]),
        "is_emoji_only": bool(lang_info["is_emoji_only"]),
        "caption_en": None,
        "translation_status": TRANSLATION_NOT_APPLICABLE,
        "sentiment_score": None,
        "sentiment_method": METHOD_NOT_SCORED,
    }

    if lang == "und" or lang_info["is_emoji_only"] or lang == "mixed":
        return base

    if lang == "en" and conf >= threshold:
        text_for_vader = sentiment_text if sentiment_text.strip() else original
        if not str(text_for_vader).strip():
            return base
        base["caption_en"] = text_for_vader
        base["sentiment_score"] = vader_compound(text_for_vader)
        base["sentiment_method"] = METHOD_VADER_EN
        return base

    if lang == "en" and conf < threshold:
        return base

    # --- Non-English path ---
    if not _mt_enabled(rules_path):
        base["sentiment_method"] = METHOD_NOT_SCORED_NON_EN
        base["translation_status"] = TRANSLATION_PENDING
        return base

    mt_source = sentiment_text if sentiment_text.strip() else original
    if len(mt_source.strip()) < _mt_min_chars(rules_path):
        base["sentiment_method"] = METHOD_NOT_SCORED_NON_EN
        base["translation_status"] = TRANSLATION_SKIPPED
        return base

    try:
        caption_en = translate_to_english(
            mt_source,
            source_lang=lang,
            translate_fn=translate_fn,
        )
    except Exception:
        base["sentiment_method"] = METHOD_NOT_SCORED_NON_EN
        base["translation_status"] = TRANSLATION_FAILED
        return base

    if not caption_en.strip():
        base["sentiment_method"] = METHOD_NOT_SCORED_NON_EN
        base["translation_status"] = TRANSLATION_FAILED
        return base

    base["caption_en"] = caption_en
    base["translation_status"] = TRANSLATION_DONE
    base["sentiment_score"] = vader_compound(caption_en)
    base["sentiment_method"] = METHOD_VADER_VIA_MT
    return base
