"""Retrieval relevance validation (Phase 2 of docs/07-ai-layer.md).

Retrieve → Validate → Synthesize. Retrieval returns ranked candidates; each one is judged
for substantive alignment with the selected video on four dimensions. The same judgment
decides which Similar High Performer cards are shown and whether the AI may synthesize a
cross-content pattern; only the minimum counts differ:

- cards: the relevant candidates in retrieval order, up to SIMILAR_COUNT (never padded);
- patterns: allowed only with at least MIN_PATTERN_COMPARABLES relevant comparables.

The judge is an LLM when OPENAI_API_KEY is available, with a deterministic rule-based
fallback. A candidate is relevant only if the judge's verdict and the code-side rule
(two aligned dimensions, one of them subject_product or narrative_intent) both agree, so it
is never relevant on one dimension or on retrieval scores alone.
Rejected candidates and their rationale are kept in the assessment for debugging and
evaluation; they are never rendered.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

import ai_cache
import analysis_data
import catalog
import llm

RETRIEVAL_DEPTH = 10
MIN_PATTERN_COMPARABLES = 2
PROMPT_VERSION = "relevance/v4"

DIMENSIONS = ("subject_product", "narrative_intent", "visual_presentation", "creative_strategy")
ANCHOR_DIMENSIONS = {"subject_product", "narrative_intent"}

SYSTEM_PROMPT = """You validate retrieval results for a TikTok content-analysis tool.
A retrieval system (caption embeddings, video-frame embeddings, rule-based labels) returned
candidate high-performing videos for a selected video. Decide, for each candidate, whether it
is substantively comparable to the selected video.

For each candidate, judge four dimensions independently. For each one, first write a short
evidence note, then decide whether it is aligned.
- subject_product: both center on the same specific subject: the same product line, the
  same product used in the same context (e.g. basketball performance footwear; lifestyle
  sneaker styling), the same sport, athlete or event type, or the same specific topic.
  Merely "both show shoes" or "both mention the brand" is NOT alignment.
- narrative_intent: the same content purpose (e.g. both athlete-signing announcements; both
  product reviews or comparisons; both outfit showcases; both game highlights).
- visual_presentation: a similar visual treatment, judged only from the supplied frame
  labels and visual similarity score (you cannot see the frames). null frame labels are unknown.
- creative_strategy: a similar creative approach (content type, social mechanic,
  CTA/commerce approach, brand style).

Rules:
- Brand is not a criterion. Nike and Adidas videos can be comparable; a different brand is
  never by itself a reason to reject.
- Product lines and labels are produced by hashtag rules and can be wrong. Verify them against
  the caption: e.g. #jordan on a post about traveling to the country Jordan is not Jordan
  Brand; "Jordan Poole" is a person. Lexical overlap, brand-name ambiguity, shared generic
  hashtags, or superficial visual similarity alone are NOT evidence of alignment.
- Use only the supplied evidence. null means unknown: it is neither alignment nor
  misalignment. When a caption is empty or uninformative, be conservative about
  subject_product and narrative_intent.
- Similarity scores are retrieval signals, not proof of relevance.
- comparable: true when at least two dimensions align and one of them is subject_product or
  narrative_intent; otherwise false. Either anchor is enough: a video with the same content
  purpose and creative approach is comparable even if it features a different product, and
  one about the same product is comparable when its presentation or strategy also matches.
- rationale: one short factual sentence (max 25 words) consistent with comparable,
  summarizing why the candidate is or is not a substantive comparable.
Return one judgment per candidate, using the candidate's video_id."""

_DIMENSION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["evidence", "aligned"],
    "properties": {"evidence": {"type": "string"}, "aligned": {"type": "boolean"}},
}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["judgments"],
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["video_id", *DIMENSIONS, "comparable", "rationale"],
                "properties": {
                    "video_id": {"type": "string"},
                    **{d: _DIMENSION_SCHEMA for d in DIMENSIONS},
                    "comparable": {"type": "boolean"},
                    "rationale": {"type": "string"},
                },
            },
        }
    },
}


# ---------- evidence given to the judge ----------


def _video_evidence(row: pd.Series) -> dict:
    import analysis_context as ctx

    return {
        "brand": catalog.brand_label(row["brand"]),
        "account_type": "official_brand_account" if row["is_official_brand"] else "creator",
        "caption": ctx._caption(row),
        "hashtags": list(row["hashtags"])[:15],
        "product": ctx._product(row),
        "content_profile": ctx._content_profile(row),
        "duration_sec": ctx._num(row["video_duration_sec"], 1),
    }


def _row(video_id: str) -> pd.Series:
    df = catalog.load_library()
    return df[df["video_id"] == str(video_id)].iloc[0]


# ---------- judges ----------


def _llm_judge(selected: pd.Series, candidates: list[dict]) -> dict[str, dict]:
    payload = {
        "selected_video": _video_evidence(selected),
        "candidates": [
            {
                "video_id": card["video_id"],
                **_video_evidence(_row(card["video_id"])),
                "retrieval_similarity": {
                    k: card["similarity"][k] for k in ("text", "visual", "strategy_overlap")
                },
                "shared_labels": card["shared_labels"],
            }
            for card in candidates
        ],
    }
    result = llm.complete_json(
        SYSTEM_PROMPT,
        json.dumps(payload, ensure_ascii=False),
        SCHEMA,
        schema_name="retrieval_relevance",
    )
    return {
        j["video_id"]: {
            "aligned_dimensions": [d for d in DIMENSIONS if j[d]["aligned"]],
            "comparable": j["comparable"],
            "evidence": {d: j[d]["evidence"] for d in DIMENSIONS},
            "rationale": j["rationale"],
        }
        for j in result.get("judgments", [])
    }


def _rules_judge(selected: pd.Series, candidates: list[dict]) -> dict[str, dict]:
    """Conservative structured-data fallback when no LLM is available."""
    own_lines = set(selected["display_lines"])
    judgments = {}
    for card in candidates:
        row, sim = _row(card["video_id"]), card["similarity"]
        text, visual = sim["text"] or 0.0, sim["visual"] or 0.0
        same_product = bool(own_lines & set(row["display_lines"]))
        shared_type = bool(set(selected["content_type"]) & set(row["content_type"]))
        shared_strategy = shared_type or bool(
            set(selected["social_mechanic"]) & set(row["social_mechanic"])
            or set(selected["brand_styles"]) & set(row["brand_styles"])
        )
        same_format = selected["visual_format"] not in (None, "other") and (
            selected["visual_format"] == row["visual_format"]
        )
        dims = []
        if same_product and text >= 0.40:
            dims.append("subject_product")
        if text >= 0.55 or (shared_type and text >= 0.40):
            dims.append("narrative_intent")
        if visual >= 0.75 or same_format:
            dims.append("visual_presentation")
        if shared_strategy:
            dims.append("creative_strategy")
        judgments[card["video_id"]] = {
            "aligned_dimensions": dims,
            "rationale": "Rule-based check; aligned on "
            + (", ".join(d.replace("_", " ") for d in dims) or "no dimension")
            + ".",
        }
    return judgments


def _visual_known(selected: pd.Series, other: pd.Series, similarity: dict) -> bool:
    """Visual alignment needs frame evidence on both sides: a visual score or frame labels."""
    return similarity["visual"] is not None or bool(
        selected["visual_format"] and other["visual_format"]
    )


def _passes_rule(dimensions: list[str]) -> bool:
    dims = set(dimensions)
    return len(dims) >= 2 and bool(dims & ANCHOR_DIMENSIONS)


# ---------- cache ----------

CACHE_NAMESPACE = "relevance"


def cache_key(video_id: str, model: str | None = None) -> str:
    ids = [c["video_id"] for c in analysis_data.retrieval_candidates(video_id, RETRIEVAL_DEPTH)]
    digest = hashlib.sha1(",".join(ids).encode()).hexdigest()[:12]
    return f"{PROMPT_VERSION}|{catalog.FRAME_LABEL_POLICY}|{model or llm.model_name()}|{video_id}|{digest}"


# ---------- assessment ----------


def assess(video_id: str, k: int = analysis_data.SIMILAR_COUNT, *, allow_llm: bool = False) -> dict:
    """Validate the ranked candidates and pick the cards.

    Page views call this read-only: a cached (or frozen) LLM judgment if one exists, else the
    rule-based fallback, never a live API call. Only the generation pipeline passes
    allow_llm=True, which judges with the LLM if nothing is cached and raises LLMError on
    failure (generation fails closed instead of silently switching method).

    Returns {"method", "model", "picks": [video_id], "candidates": [...judgment, rank,
    relevant], "retrieval_assessment": {relevant_comparables, excluded_comparables,
    cross_content_synthesis_allowed}}.
    """
    candidates = analysis_data.retrieval_candidates(video_id, RETRIEVAL_DEPTH)
    if not candidates:
        return _assessment([], [], "none", None, k)
    selected = _row(video_id)

    method, model = "llm", llm.model_name()
    key = cache_key(video_id)
    judgments = ai_cache.get(CACHE_NAMESPACE, key)
    if judgments is None and allow_llm:
        judgments = _llm_judge(selected, candidates)
        ai_cache.put(CACHE_NAMESPACE, key, judgments)
    if judgments is None:
        method, model = "rules", None
        judgments = _rules_judge(selected, candidates)

    rows = []
    for rank, card in enumerate(candidates, start=1):
        judgment = judgments.get(card["video_id"]) or {
            "aligned_dimensions": [],
            "rationale": "No judgment returned for this candidate.",
        }
        dims = [d for d in judgment["aligned_dimensions"] if d in DIMENSIONS]
        if not _visual_known(selected, _row(card["video_id"]), card["similarity"]):
            dims = [d for d in dims if d != "visual_presentation"]
        rows.append(
            {
                "rank": rank,
                "video_id": card["video_id"],
                "aligned_dimensions": dims,
                "relevant": judgment.get("comparable", True) and _passes_rule(dims),
                "rationale": judgment["rationale"],
                "evidence": judgment.get("evidence"),
                "similarity": card["similarity"],
            }
        )
    return _assessment(candidates, rows, method, model, k)


def _assessment(candidates: list[dict], rows: list[dict], method: str, model, k: int) -> dict:
    picks, excluded = [], 0
    for row in rows:
        if len(picks) >= k:
            break
        if row["relevant"]:
            picks.append(row["video_id"])
        else:
            excluded += 1
    return {
        "method": method,
        "model": model,
        "picks": picks,
        "candidates": rows,
        "retrieval_assessment": {
            "relevant_comparables": len(picks),
            "excluded_comparables": excluded,
            "cross_content_synthesis_allowed": len(picks) >= MIN_PATTERN_COMPARABLES,
        },
    }


def similar_high_performers(video_id: str, k: int = analysis_data.SIMILAR_COUNT) -> list[dict]:
    """Cards for the relevant comparables only, in retrieval order (may be fewer than k)."""
    picks = assess(video_id, k)["picks"]
    by_id = {c["video_id"]: c for c in analysis_data.retrieval_candidates(video_id, RETRIEVAL_DEPTH)}
    return [by_id[v] for v in picks if v in by_id]
