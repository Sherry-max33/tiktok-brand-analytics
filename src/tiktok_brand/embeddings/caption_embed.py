"""Caption embedding and sentiment (Phase 2)."""

from __future__ import annotations
from typing import Optional


def compute_sentiment_score(caption_text: Optional[str]) -> float:
    """
    Compute text sentiment polarity score from -1 to 1.

    Pipeline interface for NLP model (BERT/VADER). Currently returns placeholder 0.0.
    Replace with actual model inference when Phase 2 is implemented.
    """
    if not caption_text or not str(caption_text).strip():
        return 0.0
    # Placeholder: return 0.0 until BERT/VADER integration
    return 0.0
