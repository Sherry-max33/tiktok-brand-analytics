"""Tests for multilingual text embeddings."""

from tiktok_brand.embeddings.caption_embed import (
    METHOD_NOT_EMBEDDED,
    METHOD_SBERT_MULTILINGUAL,
    compute_text_embeddings,
)


def test_embed_eligible_texts_with_injected_encoder():
    def fake_encode(texts):
        return [[float(i), 1.0] for i, _ in enumerate(texts)]

    rows = compute_text_embeddings(
        ["love these sneakers", "adidas gazelle outfit"],
        encode_batch_fn=fake_encode,
    )
    assert len(rows) == 2
    assert rows[0]["embedding_method"] == METHOD_SBERT_MULTILINGUAL
    assert rows[0]["text_embedding"] == [0.0, 1.0]
    assert rows[1]["text_embedding"] == [1.0, 1.0]
    assert "multilingual" in (rows[0]["embedding_model"] or "")


def test_skip_too_short():
    rows = compute_text_embeddings(
        ["", "ab", "ok enough text"],
        encode_batch_fn=lambda texts: [[1.0, 0.0] for _ in texts],
    )
    assert rows[0]["embedding_method"] == METHOD_NOT_EMBEDDED
    assert rows[0]["text_embedding"] is None
    assert rows[1]["embedding_method"] == METHOD_NOT_EMBEDDED
    assert rows[2]["embedding_method"] == METHOD_SBERT_MULTILINGUAL
    assert rows[2]["text_embedding"] == [1.0, 0.0]


def test_disabled(tmp_path):
    rules = tmp_path / "feature_rules.yaml"
    rules.write_text(
        "text_embedding:\n  enabled: false\n  min_chars: 3\n",
        encoding="utf-8",
    )
    rows = compute_text_embeddings(
        ["plenty of text here"],
        rules_path=str(rules),
        encode_batch_fn=lambda texts: [[1.0] for _ in texts],
    )
    assert rows[0]["embedding_method"] == METHOD_NOT_EMBEDDED
    assert rows[0]["text_embedding"] is None
