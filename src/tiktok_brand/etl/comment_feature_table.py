"""Comment feature table: sentiment + embeddings on comment_text."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..embeddings.caption_embed import compute_text_embeddings
from .sentiment_rules import score_caption_sentiment
from .text_prep import clean_caption


def build_comment_feature_table(
    df: pd.DataFrame,
    *,
    encode_batch_fn=None,
    sentiment_rules_path: str = "configs/feature_rules.yaml",
    disable_mt: bool = True,
) -> pd.DataFrame:
    """
    Derive comment-level analysis features.

    Reuses video sentiment routing and multilingual Sentence-BERT embeddings.

    ``disable_mt=True`` (default): skip Google MT on bulk comments to avoid
    rate limits; non-EN rows stay ``not_scored_non_en`` / ``pending``.
    Set ``disable_mt=False`` to translate then score with ``vader_via_mt``.
    """
    out = df.copy()
    if out.empty:
        return out

    rules_path = sentiment_rules_path
    tmp_rules = None
    if disable_mt:
        import tempfile
        import yaml
        from .rule_config import load_feature_rules

        cfg = load_feature_rules(sentiment_rules_path)
        sent = dict(cfg.get("sentiment") or {})
        sent["mt_enabled"] = False
        cfg = dict(cfg)
        cfg["sentiment"] = sent
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        yaml.safe_dump(cfg, tmp)
        tmp.close()
        tmp_rules = tmp.name
        rules_path = tmp_rules

    try:
        text = out.get("comment_text", pd.Series([""] * len(out), index=out.index)).fillna("")
        out["comment_text_clean"] = text.apply(clean_caption)
        out["comment_length_words"] = (
            out["comment_text_clean"].str.split().str.len().fillna(0).astype(int)
        )
        out["mention_count"] = text.astype(str).str.count(r"@\w+").fillna(0).astype(int)

        for col in ["comment_like_count", "comment_reply_count"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        sentiment_rows = [
            score_caption_sentiment(t, rules_path=rules_path) for t in text.tolist()
        ]
        sent_df = pd.DataFrame(sentiment_rows, index=out.index)
        for col in [
            "caption_original",
            "caption_lang",
            "caption_lang_confidence",
            "has_mixed_language",
            "is_emoji_only",
            "caption_en",
            "translation_status",
            "sentiment_score",
            "sentiment_method",
        ]:
            out[col] = sent_df[col]

        emb_input = [
            (c if isinstance(c, str) and c.strip() else (r if isinstance(r, str) else ""))
            for c, r in zip(out["comment_text_clean"].tolist(), text.tolist())
        ]
        emb_rows = compute_text_embeddings(emb_input, encode_batch_fn=encode_batch_fn)
        emb_df = pd.DataFrame(emb_rows, index=out.index)
        out["text_embedding"] = emb_df["text_embedding"]
        out["embedding_method"] = emb_df["embedding_method"]
        out["embedding_model"] = emb_df["embedding_model"]

        out["crawl_at"] = pd.to_datetime(out.get("crawled_at", pd.NaT), errors="coerce")
        out["crawl_batch_id"] = out.get("crawled_at_ts", pd.NA).astype(str)
    finally:
        if tmp_rules:
            Path(tmp_rules).unlink(missing_ok=True)

    return out


def write_comment_feature_parquet(
    df: pd.DataFrame,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
