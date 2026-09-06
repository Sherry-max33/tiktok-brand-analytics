"""Multilingual caption embeddings (Sentence-BERT).

Uses a multilingual sentence model so non-English captions are embedded
directly — no MT required (unlike VADER sentiment).
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Union

from ..etl.rule_config import load_feature_rules

logger = logging.getLogger(__name__)

METHOD_SBERT_MULTILINGUAL = "sbert_multilingual"
METHOD_NOT_EMBEDDED = "not_embedded"

EncodeBatchFn = Callable[[Sequence[str]], List[List[float]]]

_MODEL = None
_MODEL_NAME: Optional[str] = None


def _embedding_cfg(rules_path: str = "configs/feature_rules.yaml") -> dict:
    return load_feature_rules(rules_path).get("text_embedding") or {}


def _default_model_name(rules_path: str = "configs/feature_rules.yaml") -> str:
    return str(
        _embedding_cfg(rules_path).get(
            "model_name",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    )


def _min_chars(rules_path: str = "configs/feature_rules.yaml") -> int:
    return int(_embedding_cfg(rules_path).get("min_chars", 3))


def _batch_size(rules_path: str = "configs/feature_rules.yaml") -> int:
    return int(_embedding_cfg(rules_path).get("batch_size", 32))


def _embeddings_enabled(rules_path: str = "configs/feature_rules.yaml") -> bool:
    return bool(_embedding_cfg(rules_path).get("enabled", True))


def _load_model(model_name: str):
    global _MODEL, _MODEL_NAME
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformers model: %s", model_name)
    _MODEL = SentenceTransformer(model_name)
    _MODEL_NAME = model_name
    return _MODEL


def _default_encode_batch(texts: Sequence[str], *, model_name: str, batch_size: int) -> List[List[float]]:
    model = _load_model(model_name)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vectors]


def compute_text_embeddings(
    texts: Sequence[Optional[str]],
    *,
    rules_path: str = "configs/feature_rules.yaml",
    encode_batch_fn: Optional[EncodeBatchFn] = None,
) -> List[dict]:
    """
    Embed ``embedding_text`` strings with multilingual Sentence-BERT.

    Returns one dict per input:
      text_embedding: list[float] | None
      embedding_method: sbert_multilingual | not_embedded
      embedding_model: model id or None
    """
    model_name = _default_model_name(rules_path)
    min_chars = _min_chars(rules_path)
    enabled = _embeddings_enabled(rules_path)

    n = len(texts)
    out: List[dict] = [
        {
            "text_embedding": None,
            "embedding_method": METHOD_NOT_EMBEDDED,
            "embedding_model": None,
        }
        for _ in range(n)
    ]

    if not enabled:
        return out

    eligible_idx: List[int] = []
    eligible_texts: List[str] = []
    for i, raw in enumerate(texts):
        text = "" if raw is None else str(raw).strip()
        if len(text) < min_chars:
            continue
        eligible_idx.append(i)
        eligible_texts.append(text)

    if not eligible_texts:
        return out

    try:
        if encode_batch_fn is not None:
            vectors = encode_batch_fn(eligible_texts)
        else:
            vectors = _default_encode_batch(
                eligible_texts,
                model_name=model_name,
                batch_size=_batch_size(rules_path),
            )
    except Exception as exc:
        logger.warning("text embedding failed: %s", exc)
        return out

    if len(vectors) != len(eligible_texts):
        logger.warning(
            "text embedding size mismatch: got %s vectors for %s texts",
            len(vectors),
            len(eligible_texts),
        )
        return out

    for i, vec in zip(eligible_idx, vectors):
        out[i] = {
            "text_embedding": list(vec) if not isinstance(vec, list) else vec,
            "embedding_method": METHOD_SBERT_MULTILINGUAL,
            "embedding_model": model_name,
        }
    return out


def compute_text_embedding(
    text: Optional[str],
    *,
    rules_path: str = "configs/feature_rules.yaml",
    encode_batch_fn: Optional[EncodeBatchFn] = None,
) -> dict:
    """Single-text convenience wrapper around ``compute_text_embeddings``."""
    return compute_text_embeddings([text], rules_path=rules_path, encode_batch_fn=encode_batch_fn)[0]


# Back-compat sentiment shim
def compute_sentiment_score(caption_text: Optional[str]) -> Optional[float]:
    from ..etl.sentiment_rules import score_caption_sentiment

    result = score_caption_sentiment(caption_text)
    score = result.get("sentiment_score")
    return float(score) if score is not None else None
