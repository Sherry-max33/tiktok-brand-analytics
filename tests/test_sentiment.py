"""Tests for language detection and VADER sentiment routing (+ MT)."""

from tiktok_brand.etl.language_rules import detect_caption_language, strip_for_lang_detect
from tiktok_brand.etl.sentiment_rules import (
    METHOD_NOT_SCORED,
    METHOD_NOT_SCORED_NON_EN,
    METHOD_VADER_EN,
    METHOD_VADER_VIA_MT,
    score_caption_sentiment,
)


def test_strip_removes_brand_and_mentions():
    text = strip_for_lang_detect("Love these @nike kicks from Nike and Adidas!!")
    assert "@nike" not in text.lower()
    assert "nike" not in text.lower()
    assert "adidas" not in text.lower()
    assert "love" in text.lower()
    assert "kicks" in text.lower()


def test_und_hashtag_only():
    info = detect_caption_language("#nike #adidas #fyp", hashtags=["nike", "adidas", "fyp"])
    assert info["caption_lang"] == "und"


def test_und_emoji_only():
    info = detect_caption_language("🔥🔥🔥", hashtags=[])
    assert info["caption_lang"] == "und"
    assert info["is_emoji_only"] is True


def test_english_body_korean_hashtag_mixed_flag():
    info = detect_caption_language(
        "These sneakers look amazing on feet #아디다스 #adidas",
        hashtags=["아디다스", "adidas"],
    )
    assert info["caption_lang"] == "en"
    assert info["has_mixed_language"] is True
    assert info["caption_lang_confidence"] >= 0.80


def test_vader_en_high_confidence():
    out = score_caption_sentiment(
        "I absolutely love these sneakers, best purchase ever!",
        hashtags=["nike"],
    )
    assert out["caption_lang"] == "en"
    assert out["caption_lang_confidence"] >= 0.80
    assert out["sentiment_method"] == METHOD_VADER_EN
    assert out["sentiment_score"] is not None
    assert out["sentiment_score"] > 0
    assert out["caption_en"]
    assert out["translation_status"] == "not_applicable"


def test_non_english_mt_then_vader(tmp_path):
    ru = (
        "adidas — производитель спортивной одежды и обуви, "
        "основанный в 1949 году в Германии. Бренд с тремя полосками."
    )

    def fake_mt(text: str, source_lang=None) -> str:
        assert "производитель" in text or "adidas" in text.lower()
        return "adidas is a great sports brand, I love these shoes!"

    rules = tmp_path / "feature_rules.yaml"
    rules.write_text(
        "sentiment:\n  en_confidence_threshold: 0.80\n  mt_enabled: true\n  mt_min_chars: 8\n",
        encoding="utf-8",
    )
    out = score_caption_sentiment(
        ru, hashtags=["adidas"], translate_fn=fake_mt, rules_path=str(rules)
    )
    assert out["caption_lang"] != "en"
    assert out["translation_status"] == "done"
    assert out["sentiment_method"] == METHOD_VADER_VIA_MT
    assert out["caption_en"] == "adidas is a great sports brand, I love these shoes!"
    assert out["sentiment_score"] is not None
    assert out["sentiment_score"] > 0


def test_non_english_mt_failure(tmp_path):
    ru = (
        "adidas — производитель спортивной одежды и обуви, "
        "основанный в 1949 году в Германии. Бренд с тремя полосками."
    )

    def boom(text: str, source_lang=None) -> str:
        raise RuntimeError("mt down")

    rules = tmp_path / "feature_rules.yaml"
    rules.write_text(
        "sentiment:\n  en_confidence_threshold: 0.80\n  mt_enabled: true\n  mt_min_chars: 8\n",
        encoding="utf-8",
    )
    out = score_caption_sentiment(
        ru, hashtags=["adidas"], translate_fn=boom, rules_path=str(rules)
    )
    assert out["sentiment_score"] is None
    assert out["sentiment_method"] == METHOD_NOT_SCORED_NON_EN
    assert out["translation_status"] == "failed"
    assert out["caption_en"] is None


def test_mt_disabled_leaves_pending(tmp_path, monkeypatch):
    # Point at a temp rules file with mt_enabled false
    rules = tmp_path / "feature_rules.yaml"
    rules.write_text(
        "sentiment:\n  en_confidence_threshold: 0.80\n  mt_enabled: false\n  mt_min_chars: 8\n",
        encoding="utf-8",
    )
    ru = (
        "adidas — производитель спортивной одежды и обуви, "
        "основанный в 1949 году в Германии. Бренд с тремя полосками."
    )
    out = score_caption_sentiment(ru, hashtags=["adidas"], rules_path=str(rules))
    assert out["sentiment_method"] == METHOD_NOT_SCORED_NON_EN
    assert out["translation_status"] == "pending"
    assert out["sentiment_score"] is None


def test_und_not_scored():
    out = score_caption_sentiment("#nike #jordan", hashtags=["nike", "jordan"])
    assert out["caption_lang"] == "und"
    assert out["sentiment_score"] is None
    assert out["sentiment_method"] == METHOD_NOT_SCORED
