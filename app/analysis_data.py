"""Analysis-page data: content profile, audience sentiment, similar high performers, AI.

Everything here is computed from existing pipeline outputs. Nothing is invented to fill the
UI: when a signal is missing the functions return None / an empty list and the page shows
an explicit "not available" state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import catalog

COMMENT_TABLE = catalog.REPO_ROOT / "data" / "processed" / "feature" / "comment_feature_table.parquet"

# Same thresholds as the 04d sentiment notebook (VADER compound).
SENTIMENT_POS_MIN = 0.05
SENTIMENT_NEG_MAX = -0.05

# High performer = Top 10% by WER (the notebooks' Top-10% definition) with real reach.
HIGH_PERFORMER_QUANTILE = 0.90
SIMILAR_COUNT = 3
SIMILARITY_WEIGHTS = {"text": 0.4, "visual": 0.4, "strategy": 0.2}
# Subtracted when both videos have a known product category and they don't overlap
# (e.g. a shoe promo vs an accessories haul); unknown categories are never penalized.
CATEGORY_CONFLICT_PENALTY = 0.1

BRAND_STYLE_LABELS = {
    "lifestyle": "Lifestyle",
    "performance": "Performance",
    "technical": "Technical",
    "retro": "Retro",
}
VISUAL_FORMAT_LABELS = {
    "product_closeup": "Product close-up",
    "on_body_styling": "On-body styling",
    "sports_action": "Sports action",
    "talking_head": "Talking head",
    "campaign_visual": "Campaign visual",
    "archival_retro": "Archival / retro",
    "other": "Other",
}
VISUAL_SETTING_LABELS = {
    "retail_store": "Retail store",
    "outdoor": "Outdoor",
    "studio": "Studio",
    "sports_venue": "Sports venue",
    "gym_training": "Gym / training",
    "event_crowd": "Event / crowd",
    "home_indoor": "Home / indoor",
    "other": "Other",
}
SOCIAL_MECHANIC_LABELS = {
    "trend": "Trend",
    "grwm_ootd_format": "GRWM / OOTD format",
    "template_remix": "Template remix",
    "audio_driven": "Audio-driven",
    "pov": "POV",
    "bts": "Behind the scenes",
    "duet_stitch": "Duet / Stitch",
    "challenge": "Challenge",
}
CTA_LABELS = {
    "has_purchase_cta": "Purchase",
    "has_discovery_traffic_cta": "Discovery / traffic",
    "has_engagement_cta": "Engagement",
    "has_promo_language": "Promo language",
}


def _row(video_id: str) -> pd.Series | None:
    df = catalog.load_library()
    match = df[df["video_id"] == str(video_id)]
    return None if match.empty else match.iloc[0]


def _labels(values: list[str], mapping: dict[str, str]) -> list[str]:
    return [mapping.get(v, v.replace("_", " ").capitalize()) for v in values]


# ---------- Audience sentiment (comment-level only) ----------


@st.cache_data(show_spinner=False)
def _comment_sentiment() -> dict[str, dict]:
    if not COMMENT_TABLE.exists():
        return {}
    comments = pd.read_parquet(COMMENT_TABLE, columns=["video_id", "sentiment_score"])
    comments = comments.dropna(subset=["sentiment_score"])
    out = {}
    for video_id, scores in comments.groupby("video_id")["sentiment_score"]:
        n = len(scores)
        out[str(video_id)] = {
            "comments": n,
            "positive": float((scores >= SENTIMENT_POS_MIN).mean()),
            "negative": float((scores <= SENTIMENT_NEG_MAX).mean()),
        }
    return out


def audience_sentiment(video_id: str) -> dict | None:
    """Share of positive / negative comments. None when no comment data was collected;
    caption sentiment is never used as a stand-in for the audience."""
    return _comment_sentiment().get(str(video_id))


# ---------- Content profile ----------


def content_profile(video_id: str) -> dict[str, list[tuple[str, str]]]:
    """(label, value) rows grouped by source, video frames first; "—" marks a field the
    pipeline left empty for this video. Audience sentiment is shown separately."""
    row = _row(video_id)
    if row is None:
        return {}

    def joined(values: list[str]) -> str:
        return ", ".join(values) if values else "—"

    def frame_label(col: str, labels: dict[str, str]) -> str:
        if row[f"{col}_status"] == "low_confidence":
            likely = row[f"{col}_likely"]
            return f"Likely: {labels[likely]}" if likely in labels else "Not identified"
        return labels.get(row[col] or "", "—")

    cta = [label for col, label in CTA_LABELS.items() if row[col]]
    return {
        "frames": [
            ("Visual format", frame_label("visual_format", VISUAL_FORMAT_LABELS)),
            ("Visual setting", frame_label("visual_setting", VISUAL_SETTING_LABELS)),
        ],
        "caption": [
            ("Content type", joined([catalog.content_type_label(c) for c in row["content_type"]])),
            ("Brand style", joined(_labels(row["brand_styles"], BRAND_STYLE_LABELS))),
            ("Social mechanic", joined(_labels(row["social_mechanic"], SOCIAL_MECHANIC_LABELS))),
            ("CTA signal", joined(cta) if cta else "None detected"),
        ],
    }


# ---------- Similar high performers ----------


def _normalized(vectors: pd.Series, dim: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(vectors), dim), dtype=np.float32)
    present = np.zeros(len(vectors), dtype=bool)
    for i, vec in enumerate(vectors):
        if vec is not None and not (isinstance(vec, float) and np.isnan(vec)) and len(vec) == dim:
            matrix[i] = np.asarray(vec, dtype=np.float32)
            present[i] = True
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms, present


@st.cache_resource(show_spinner=False)
def _embeddings() -> dict | None:
    """SBERT caption and CLIP visual embeddings, row-aligned with load_library()."""
    if not catalog.FEATURE_TABLE.exists():
        return None
    raw = pd.read_parquet(
        catalog.FEATURE_TABLE, columns=["video_id", "text_embedding", "visual_embedding"]
    )
    raw["video_id"] = raw["video_id"].astype(str)
    raw = raw.drop_duplicates("video_id").set_index("video_id")
    ids = catalog.load_library()["video_id"]
    raw = raw.reindex(ids)
    text, has_text = _normalized(raw["text_embedding"], 384)
    visual, has_visual = _normalized(raw["visual_embedding"], 512)
    return {"text": text, "has_text": has_text, "visual": visual, "has_visual": has_visual}


def similarity_active() -> bool:
    return _embeddings() is not None


def _categories(row: pd.Series) -> set[str]:
    """Product categories the app is confident about (see catalog._resolve_products)."""
    return set(row["display_categories"]) - {"uncategorized"}


def _strategy_tokens(row: pd.Series) -> set[str]:
    tokens = {f"ct:{v}" for v in row["content_type"]}
    tokens |= {f"cat:{v}" for v in _categories(row)}
    tokens |= {f"pl:{v}" for v in row["product_lines"]}
    tokens |= {f"bs:{v}" for v in row["brand_styles"]}
    tokens |= {f"sm:{v}" for v in row["social_mechanic"]}
    return tokens


@st.cache_data(show_spinner=False)
def retrieval_candidates(video_id: str, k: int = SIMILAR_COUNT) -> list[dict]:
    """Ranked retrieval: the k high performers (WER Top 10%, 100K+ views) most similar to the
    selected video, from either brand: the question is which strong content is most alike,
    not what else one brand posted. These are candidates only; relevance.py validates them
    before any are shown or used as evidence.

    Performance decides who qualifies; similarity decides who is most comparable:
    SBERT caption cosine + CLIP visual cosine + overlap of caption/hashtag-derived strategy
    labels. Visual labels are left out of the overlap because they come from the same CLIP
    frames as the visual cosine. The overlap is smoothed (|A∩B| / (|A∪B| + 1)) so a single
    shared label doesn't score as a perfect match. Product category counts toward the overlap,
    and a known-but-different category (shoes vs accessories) is penalized. Components the
    selected video lacks are dropped. At most one reference per creator. Scores are internal only.
    """
    emb = _embeddings()
    df = catalog.load_library().reset_index(drop=True)
    idx = df.index[df["video_id"] == str(video_id)]
    if emb is None or idx.empty:
        return []
    i = int(idx[0])

    threshold = df["wer"].quantile(HIGH_PERFORMER_QUANTILE)
    pool = (
        (df["wer"] >= threshold)
        & (df["view_count"] >= catalog.FEATURED_MIN_VIEWS)
        & (df.index != i)
    ).to_numpy()
    if not pool.any():
        return []

    weights = dict(SIMILARITY_WEIGHTS)
    score = np.zeros(len(df), dtype=np.float32)
    used = 0.0
    component_sims = {}
    for part in ("text", "visual"):
        if emb[f"has_{part}"][i]:
            sims = emb[part] @ emb[part][i]
            sims[~emb[f"has_{part}"]] = 0.0
            component_sims[part] = sims
            score += weights[part] * sims
            used += weights[part]

    own = _strategy_tokens(df.loc[i])
    if own:
        candidates = np.flatnonzero(pool)
        jaccard = np.zeros(len(df), dtype=np.float32)
        for j in candidates:
            other = _strategy_tokens(df.loc[j])
            jaccard[j] = len(own & other) / (len(own | other) + 1)
        score += weights["strategy"] * jaccard
        used += weights["strategy"]

    own_categories = _categories(df.loc[i])
    if own_categories:
        for j in np.flatnonzero(pool):
            other = _categories(df.loc[j])
            if other and not own_categories & other:
                score[j] -= CATEGORY_CONFLICT_PENALTY

    if used == 0:
        return []
    score[~pool] = -np.inf

    picks, authors = [], set()
    for j in np.argsort(-score):
        if not np.isfinite(score[j]) or len(picks) >= k:
            break
        author = df.at[j, "author_username"]
        if author in authors:
            continue
        authors.add(author)
        picks.append(j)

    cards = []
    for j in picks:
        row = df.loc[j]
        card = catalog.to_card(row)
        card["wer"] = float(row["wer"]) if pd.notna(row["wer"]) else None
        card["bri"] = float(row["bri"]) if pd.notna(row["bri"]) else None
        pair_sims = {
            part: float(sims[j])
            for part, sims in component_sims.items()
            if emb[f"has_{part}"][j]
        }
        card["shared_labels"] = _shared_labels(df.loc[i], row)
        card["shared"] = _shared_display(card["shared_labels"], pair_sims)
        # Internal evidence for the AI layer; never rendered.
        card["similarity"] = {
            "score": round(float(score[j]), 4),
            "text": round(pair_sims["text"], 4) if "text" in pair_sims else None,
            "visual": round(pair_sims["visual"], 4) if "visual" in pair_sims else None,
            "strategy_overlap": round(float(jaccard[j]), 4) if own else None,
            "category_conflict": bool(
                own_categories
                and _categories(row)
                and not own_categories & _categories(row)
            ),
        }
        cards.append(card)
    return cards


def _shared_labels(own: pd.Series, other: pd.Series) -> list[str]:
    """Labels both videos carry, most specific first."""
    shared = [catalog.product_label(p) for p in own["display_lines"] if p in other["display_lines"]]
    shared += [
        catalog.content_type_label(c) for c in own["content_type"] if c in other["content_type"]
    ]
    shared += _labels(
        [m for m in own["social_mechanic"] if m in other["social_mechanic"]], SOCIAL_MECHANIC_LABELS
    )
    for col, labels in (("visual_format", VISUAL_FORMAT_LABELS), ("visual_setting", VISUAL_SETTING_LABELS)):
        if own[col] and own[col] != "other" and own[col] == other[col]:
            shared.append(labels.get(own[col], own[col]))
    shared += _labels(
        [s for s in own["brand_styles"] if s in other["brand_styles"]], BRAND_STYLE_LABELS
    )
    return list(dict.fromkeys(shared))


def _shared_display(shared: list[str], sims: dict[str, float]) -> list[str]:
    """Card text: up to three shared labels, so the card can say why it was matched without
    exposing scores. With none, names the stronger of the caption / visual similarities."""
    if shared:
        return shared[:3]
    if sims:
        strongest = max(sims, key=sims.get)
        return ["Similar visuals" if strongest == "visual" else "Similar caption"]
    return []
