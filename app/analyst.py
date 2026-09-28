"""AI Content Analyst (Phase 3 of docs/07-ai-layer.md).

Reads only the validated analysis_context (rejected retrieval candidates never reach it) and
returns structured JSON; turning it into prose is the presentation layer's job.

What the evidence does not support is removed from the output schema rather than left to
the prompt:
- cross_content_patterns exists only with >= 2 validated comparables (else null);
- shared_characteristic exists only with exactly 1 validated comparable (else null);
- audience_signal exists only when comment sentiment was collected (else null);
- evidence_refs and comparable_ids are enums built from the fields that are actually present,
  so the model can't cite missing data or an unvalidated video.

Code then checks what a schema can't: patterns cite >= 2 comparables, observations cite only
the selected video, performance is never the sole evidence, and no causal or retrieval
language. Violations get one repair round; items still violating are dropped and logged.
"""

from __future__ import annotations

import copy
import json
import re

import llm

PROMPT_VERSION = "analyst/v12"
SCHEMA_VERSION = "analyst_output/v1"
MAX_INSIGHTS = 3
OBSERVATION_TYPES = ("creative", "visual", "narrative", "strategy", "commerce")

# Observability: the dataset has no OCR, speech or audio data, so these modalities are never
# evidence, present or absent.
CAPTURED_DIMENSIONS = (
    "caption text and hashtags; caption-derived labels (content type, brand style, social "
    "mechanic, CTA signals); product line and category; frame labels (visual format, visual "
    "setting); duration; account type (official brand account or creator); performance "
    "metrics; comment sentiment when collected"
)
UNCAPTURED = re.compile(
    r"\b(on[- ]screen|text overlays?|overlay text|subtitles?|captions? on screen|"
    r"voice[- ]?overs?|narrat(ion|or|ors|ed|es|ing)|spoken|speech|dialogue|talking points|"
    r"music|songs?|soundtracks?|audio|sounds?|lyrics|beats?|sound effects?|trending sound)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are the AI Content Analyst in a TikTok content-analysis tool for Nike and
Adidas videos. You receive a structured evidence object about one selected video and, when
available, comparable high-performing videos. Identify what can be learned from this evidence:
what stands out creatively about the selected video, what it shares with the comparables, and
what may be worth testing.

About the evidence
- The comparables were already validated as substantively relevant by an earlier step. Treat
  them as the comparison set. Do not re-judge their relevance, question their inclusion, rank
  them, or mention retrieval, similarity scores or excluded candidates.
- A comparable's relevance.aligned_dimensions and relevance.rationale describe how it aligns
  with the selected video; shared_labels are labels both videos carry.
- null means unknown (not collected or not computed). An empty list means a rule-based
  classifier ran on the caption/hashtags and found nothing. Missing frame, caption, product or
  audience data is unknown, never negative evidence: do not write findings such as "has no
  visual setting", "no CTA", "lacks product focus" or "no audience data".
- Frame labels (visual_format, visual_setting) are one zero-shot label each from sampled
  frames. Do not describe what the frames show beyond those labels. A frame label that fit
  no specific category is given as null (unknown).
- Observability: the dataset captures only {captured}. It has no on-screen text, voiceover,
  narration, speech, music or audio data. Never describe those, and never infer their
  presence or absence.
- Caption features (content_type, brand_style, social_mechanic, cta_signals) come from the
  caption and hashtags, not from the frames. caption.meaning, when present, is a reading aid
  for the caption. Never mention a caption's language or translation (no "English caption",
  "in French", "translated").
- Catch-all taxonomy values (Other, Unknown, Unclassified …) are given as null or omitted:
  they are unknown, not a category.
- Product lines come from hashtag rules and can be wrong; check them against the caption. A
  bare brand-like hashtag is weak evidence. If the caption uses the word as a person or place
  (e.g. a skit where "Jordan" is someone's name, or a trip to Jordan), do not assert or
  mention the product, and do not build an item on it. Status "unresolved" or "not_identified" means no specific product can
  be asserted.
- Performance: top_percent is the WER percentile in the library (1 = best); BRI is WER relative
  to the brand's median. Counts are a snapshot.

Output rules
- single_video_observations: what stands out about the selected video itself: a creative,
  visual, narrative, strategy or commerce framing with explanatory value (e.g. "culture-led
  rather than product-led"), not a restated label ("this is a basketball video"). Cite only
  the selected video's fields.
- shared_characteristic (present only when exactly one comparable exists): one characteristic
  the selected video shares with that comparable. Call it a shared characteristic; never call
  it a pattern, recurring, consistent or repeated.
- cross_content_patterns (present only when two or more comparables exist): a characteristic
  the selected video shares with at least two comparables. Put exactly the comparables that
  share it in comparable_ids and cite the selected video's fields in evidence_refs. Any label
  you cite must be carried by the selected video and by every cited comparable. Characteristics
  shared by the same comparables belong in one pattern, not several.
- Brand: each comparable has same_brand_as_selected. When a cited comparable is from the
  other brand and that materially strengthens the finding (e.g. an Adidas athlete-signing
  post shares its framing with a Nike one), you may state it, naming the brands. Cite that
  comparable's brand field. Never claim a characteristic generalizes across brands, applies
  regardless of brand, or is universal: one or two comparables can't show that.
- Scope: the evidence covers only this video and its similar high performers. Never
  generalize to a broader population ("common among high-performing Adidas creator content",
  "typical of Nike content", "top-performing videos"); say it "appears in this video and two
  similar high performers" or "recurs within this comparable set".
- In shared characteristics and patterns, describe only characteristics the selected video is
  shown to have. If a field is null for the selected video (e.g. no frame labels), do not
  attribute the comparables' values for that field to it.
- audience_signal (present only when comment sentiment exists): use it only if the sentiment
  is meaningful for interpreting the content, otherwise null. Report what the analyzed
  comments show (shares of positive and negative comments, net sentiment, how many were
  analyzed); do not generalize to the whole audience or demographics, and do not interpret
  what the numbers indicate or suggest about reception.
- Stay at the level of the evidence: do not add details the fields don't contain (e.g. "dynamic
  movement" from a "Sports action" label).
- Every item cites evidence_refs from the allowed paths. Performance fields may support an
  interpretation but cannot be the only evidence; restating metrics is not an insight.
- Correlation, not causation. Valid: "these high-performing comparables share X". Invalid: "X
  caused / drove / led to / explains the performance", "because of X it went viral". Describe
  what the evidence shows, not why the video succeeded. Avoid these words entirely, even as
  descriptive adjectives: cause, drive / driven / driver, led to, result in, due to, because,
  thanks to, boost, fuel, attribute, explain, key to (write "opinion-led" or "focused on
  opinion", not "opinion-driven").
- worth_testing: optionally one short sentence naming what may be worth testing based on this
  item ("... may be worth testing"). Do not say how to test it or predict results. null if
  nothing fits.
- Do not speculate about creator intent, audience demographics or motivation, how viewers
  respond ("may encourage", "appeals to viewers seeking ..."), or brand strategy. Do not name
  creators.
- Prefer omission over weak analysis. At most 3 items in total across all fields; fewer, or
  none, is correct when the evidence is thin. Prefer well-supported cross-content patterns.
  Never make an item out of performance: top percentile, WER, BRI, views and other metrics
  are already on the page. They may support an interpretation, never be the insight or the
  title.
- Voice: write for a brand or content strategist, not about the pipeline. Refer to the
  selected video as "this video" or by what it is ("this tracksuit OOTD"), and to the
  comparables as "similar high performers" (e.g. "two of the three similar high
  performers"), never "the selected video", "comparable 1" or video IDs. Describe labels in
  plain words ("performance-focused captions", "filmed at sports venues") instead of quoting
  field names. Concise, specific, evidence-backed; no added interpretation to make it more
  interesting.
- title: at most 6 words, plain language. finding: 1-2 specific, factual sentences.
- evidence_notes: a brief internal note (max 60 words), written first, on which evidence is
  present and which is unknown."""

SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{captured}", CAPTURED_DIMENSIONS)

REPAIR_PROMPT = """Your previous output broke these rules:
{violations}
Return the complete output again with those problems fixed. Remove an item instead of
stretching the evidence to keep it."""


# ---------- evidence given to the Analyst ----------


CATCH_ALL_VALUES = {"other", "unknown", "unclassified", "uncategorized", "none", "n/a", "misc"}


def _is_catch_all(value) -> bool:
    return isinstance(value, str) and value.strip().lower() in CATCH_ALL_VALUES


def _without_catch_all(value):
    """A catch-all taxonomy value says only that no specific label fit, so it becomes unknown
    (null in a single-label field, dropped from a list)."""
    if isinstance(value, dict):
        return {k: _without_catch_all(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_without_catch_all(v) for v in value if not _is_catch_all(v)]
    return None if _is_catch_all(value) else value


def _caption_without_language(caption: dict | None) -> dict | None:
    """Detected language is unreliable upstream and not analytically important; the English
    rendering is kept only as a reading aid."""
    if not caption:
        return caption
    caption = {k: v for k, v in caption.items() if k != "language"}
    if "english_translation" in caption:
        caption["meaning"] = caption.pop("english_translation")
    return caption


def _clean_video(video: dict) -> dict:
    video = copy.deepcopy(video)
    video["creator"].pop("username", None)
    video["caption"] = _caption_without_language(video.get("caption"))
    if video.get("product"):
        video["product"] = _without_catch_all(video["product"])
    if "content_profile" in video:
        video["content_profile"] = _without_catch_all(video["content_profile"])
    return video


def analyst_input(context: dict) -> dict:
    """The validated context minus what the Analyst must not use: retrieval scores, excluded
    counts, creator usernames, detected caption language, and catch-all taxonomy values
    (given as unknown)."""
    comparables = []
    for item in context["similar_high_performers"]:
        item = _clean_video(item)
        item["shared_labels"] = _without_catch_all(item.pop("similarity")["shared_labels"])
        item["same_brand_as_selected"] = item["brand"] == context["selected_video"]["brand"]
        comparables.append(item)
    selected = _clean_video(context["selected_video"])
    conventions = {
        k: v for k, v in context["conventions"].items() if k not in ("similarity", "relevance")
    }
    if "frame_features" in conventions:
        conventions["frame_features"] = conventions["frame_features"].replace(
            "'Other' = no defined category fit", "null when no defined category fit"
        )
    return {
        "conventions": conventions,
        "evidence_basis": {
            "validated_comparables": len(comparables),
            "cross_content_patterns_allowed": context["retrieval_assessment"][
                "cross_content_synthesis_allowed"
            ],
            "audience_signals_available": context["audience_signals"] is not None,
        },
        "selected_video": selected,
        "performance": context["performance"],
        "content_profile": _without_catch_all(context["content_profile"]),
        "audience_signals": {**context["audience_signals"], "method": "VADER on collected comments"}
        if context["audience_signals"]
        else None,
        "comparables": comparables,
    }


_SKIP_KEYS = {"video_id", "method", "conventions", "evidence_basis"}


def _leaf_paths(value, prefix: str) -> list[str]:
    if isinstance(value, dict):
        paths = []
        for key, child in value.items():
            if key not in _SKIP_KEYS:
                paths += _leaf_paths(child, f"{prefix}.{key}" if prefix else key)
        return paths
    if value is None or value == [] or value == "":
        return []
    return [prefix]


def evidence_refs(payload: dict) -> list[str]:
    """Citable paths: every field that has a value. Comparables are addressed by video ID."""
    refs = _leaf_paths({k: v for k, v in payload.items() if k != "comparables"}, "")
    for item in payload["comparables"]:
        refs += _leaf_paths(item, f"comparables[{item['video_id']}]")
    return refs


# ---------- schema ----------


def _item_schema(refs: list[str], extra: dict) -> dict:
    properties = {
        "title": {"type": "string"},
        "finding": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string", "enum": refs}},
        **extra,
        "worth_testing": {"type": ["string", "null"]},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def output_schema(payload: dict, refs: list[str]) -> dict:
    basis = payload["evidence_basis"]
    ids = [c["video_id"] for c in payload["comparables"]]
    properties: dict = {
        "evidence_notes": {"type": "string"},
        "single_video_observations": {
            "type": "array",
            "items": _item_schema(
                refs, {"type": {"type": "string", "enum": list(OBSERVATION_TYPES)}}
            ),
        },
    }
    comparable_ids = {"comparable_ids": {"type": "array", "items": {"type": "string", "enum": ids}}}
    if len(ids) == 1:
        properties["shared_characteristic"] = {
            "anyOf": [_item_schema(refs, comparable_ids), {"type": "null"}]
        }
    if basis["cross_content_patterns_allowed"]:
        properties["cross_content_patterns"] = {
            "type": "array",
            "items": _item_schema(refs, comparable_ids),
        }
    if basis["audience_signals_available"]:
        properties["audience_signal"] = {"anyOf": [_item_schema(refs, {}), {"type": "null"}]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


# ---------- validation ----------

_CAUSAL = re.compile(
    r"\b(caus(e|es|ed|ing)|drove|drives?|driv(en|ing|ers?)|attribut\w+|(?<!-)led to|leads? to|"
    r"result(s|ed)? in|"
    r"due to|because|responsible for|thanks to|boost(s|ed|ing)?|fuel(s|ed|led)?|"
    r"explains?|key to|secret to|reason (it|for)|made it|why it|contribut\w+ to)\b",
    re.IGNORECASE,
)
_RETRIEVAL = re.compile(
    r"\b(exclu\w+|reject\w+|similarity scores?|retriev\w+|validated|validation)\b", re.IGNORECASE
)
_INTERPRETIVE = re.compile(
    r"\b(indicat\w*|suggest\w*|impl(y|ies|ied)|reflect\w*|well[- ]received|resonat\w*|"
    r"favorabl\w*|appreciat\w*)\b",
    re.IGNORECASE,
)
_RECURRING = re.compile(
    r"\b(recurr\w*|repeat\w*|consistent(ly)?|patterns?|multiple|several)\b", re.IGNORECASE
)
LANGUAGE_CLAIM = re.compile(
    r"\b(translat\w*|english|french|spanish|german|italian|portuguese|dutch|"
    r"norwegian|danish|swedish|korean|japanese|chinese|mandarin|arabic|turkish|polish|"
    r"multilingual|bilingual)\b",
    re.IGNORECASE,
)
_METRIC_TITLE = re.compile(
    r"\b(top \d+|percentile|wer|bri|views?|engagement rate|high[- ]performing|performance metrics?)\b",
    re.IGNORECASE,
)
_SPECULATIVE = re.compile(
    r"\b(encourag\w+|attract\w*|appeal(s|ing)? to|entic\w+|likely to|"
    r"(viewers?|audiences?|fans?|people) (seeking|who|looking|wanting))\b",
    re.IGNORECASE,
)
_GENERALIZED = re.compile(
    r"\b(across (both |all |the two )?brands|regardless of brand|brand[- ]agnostic|universal\w*|"
    r"industry[- ]wide|any brand|all brands|works for (both|any|all))\b",
    re.IGNORECASE,
)
_POPULATION_NOUN = re.compile(
    r"\b((?:high|top|best)[- ]performing (?:\w+ ){0,3}?(?:content|videos?|posts?|creators?|accounts?)"
    r"|(?:nike|adidas|tiktok|creator) (?:creator )?content)\b",
    re.IGNORECASE,
)
_POPULATION_PHRASE = re.compile(
    r"\b((?:common|typical|standard|widespread|prevalent|popular|frequent|a trend|the norm)"
    r"(?: practice| approach)? (?:among|across|in|for|of|on) (?!th(?:ese|is|e (?:similar|comparables?|"
    r"retrieved|selected|three|two|set))\b|its\b|both\b|all three\b|similar\b|comparables?\b)\w+"
    r"|in general|on tiktok|across tiktok|the (?:wider|broader) \w+)\b",
    re.IGNORECASE,
)
_SCOPED = re.compile(
    r"\b(these|this|its|similar|retrieved|comparable|selected|the (two|three)|is|as|an?)\s+$",
    re.IGNORECASE,
)


def population_claim(text: str) -> str | None:
    """A claim about a population broader than this video and its comparables (e.g. "common
    among high-performing Adidas creator content"), which the evidence can't support."""
    text = text or ""
    match = _POPULATION_PHRASE.search(text)
    if match:
        return match.group(0)
    for match in _POPULATION_NOUN.finditer(text):
        if not _SCOPED.search(text[max(0, match.start() - 25):match.start()]):
            return match.group(0)
    return None


_CROSS_BRAND_CLAIM = re.compile(
    r"\b(cross[- ]brand|both brands|different brand|other brand|rival|competitor|"
    r"nike and adidas|adidas and nike)\b",
    re.IGNORECASE,
)


_STRUCTURED_REFS = (
    "content_profile.",
    "selected_video.product.product_lines",
    "selected_video.product.categories",
)


def _text(item: dict) -> str:
    return " ".join(filter(None, (item["title"], item["finding"], item.get("worth_testing"))))


def _resolve(obj, path: str):
    for part in path.split("."):
        obj = obj.get(part) if isinstance(obj, dict) else None
    return obj


def _as_set(value) -> set:
    if value is None:
        return set()
    return set(value) if isinstance(value, list) else {value}


def _unshared_labels(item: dict, payload: dict) -> list[str]:
    """Selected-video label fields cited by a comparison item that some cited comparable
    doesn't share."""
    comparables = {c["video_id"]: c for c in payload["comparables"]}
    unshared = []
    for ref in item["evidence_refs"]:
        if not ref.startswith(_STRUCTURED_REFS):
            continue
        own = _as_set(_resolve(payload, ref))
        comparable_path = ref.removeprefix("selected_video.")
        for video_id in item.get("comparable_ids") or []:
            if video_id in comparables and not own & _as_set(
                _resolve(comparables[video_id], comparable_path)
            ):
                unshared.append(f"{ref} (not shared by {video_id})")
    return unshared


def _item_violations(
    field: str, item: dict, allowed: set[str], ids: set[str], payload: dict
) -> list[str]:
    refs = item["evidence_refs"]
    cited = set(item.get("comparable_ids") or [])
    problems = []
    if not refs:
        problems.append("cites no evidence_refs")
    if set(refs) - allowed:
        problems.append("cites evidence paths that are not in the allowed list")
    if refs and all(r.startswith("performance.") for r in refs):
        problems.append("rests only on performance metrics")
    if cited - ids:
        problems.append("cites a video that is not a validated comparable")
    if _CAUSAL.search(_text(item)):
        problems.append(f"uses causal language ({_CAUSAL.search(_text(item)).group(0)!r})")
    if _RETRIEVAL.search(_text(item)):
        problems.append("discusses retrieval or validation")
    if LANGUAGE_CLAIM.search(_text(item)):
        problems.append(
            f"mentions caption language or translation ({LANGUAGE_CLAIM.search(_text(item)).group(0)!r})"
        )
    if _METRIC_TITLE.search(item["title"]):
        problems.append(
            f"the title restates performance ({_METRIC_TITLE.search(item['title']).group(0)!r}); "
            "metrics may support an insight but are not the insight"
        )
    if UNCAPTURED.search(_text(item)):
        problems.append(
            "describes a modality the dataset doesn't capture "
            f"({UNCAPTURED.search(_text(item)).group(0)!r}); it can't be observed, present or absent"
        )
    if _SPECULATIVE.search(_text(item)):
        problems.append(
            "speculates about audience response or motivation "
            f"({_SPECULATIVE.search(_text(item)).group(0)!r})"
        )
    if _GENERALIZED.search(_text(item)):
        problems.append(
            f"generalizes across brands from limited evidence ({_GENERALIZED.search(_text(item)).group(0)!r})"
        )
    if population_claim(_text(item)):
        problems.append(
            f"generalizes to a broader population ({population_claim(_text(item))!r}); the evidence "
            "covers only this video and its similar high performers, e.g. say it 'recurs within "
            "this comparable set'"
        )
    brands = {c["video_id"]: c["brand"] for c in payload["comparables"]}
    other_brand = any(brands.get(v) != payload["selected_video"]["brand"] for v in cited)
    if _CROSS_BRAND_CLAIM.search(_text(item)) and not other_brand:
        problems.append("claims a cross-brand comparison but no cited comparable is from the other brand")
    selected_refs = [r for r in refs if not r.startswith("comparables[")]
    if field == "single_video_observations" and len(selected_refs) < len(refs):
        problems.append("a single-video observation cites comparable fields")
    if field == "cross_content_patterns":
        if len(cited) < 2:
            problems.append("a cross-content pattern must cite at least 2 comparable_ids")
        if not [r for r in selected_refs if not r.startswith("performance.")]:
            problems.append("a cross-content pattern must cite the selected video's own fields")
    if field == "shared_characteristic":
        if len(cited) != 1:
            problems.append("the shared characteristic must cite exactly 1 comparable_id")
        if _RECURRING.search(_text(item)):
            problems.append(
                f"a single-comparable finding is described as recurring "
                f"({_RECURRING.search(_text(item)).group(0)!r})"
            )
    if field in ("cross_content_patterns", "shared_characteristic"):
        unshared = _unshared_labels(item, payload)
        if unshared:
            problems.append("cites labels the comparables don't share: " + "; ".join(unshared))
    if field == "audience_signal":
        if not any(r.startswith("audience_signals.") for r in refs):
            problems.append("the audience signal must cite audience_signals fields")
        if _INTERPRETIVE.search(_text(item)):
            problems.append(
                "the audience signal interprets beyond the comment numbers "
                f"({_INTERPRETIVE.search(_text(item)).group(0)!r})"
            )
    return problems


def _items(output: dict):
    for field in ("cross_content_patterns", "single_video_observations"):
        for i, item in enumerate(output.get(field) or []):
            yield field, i, item
    for field in ("shared_characteristic", "audience_signal"):
        if output.get(field):
            yield field, None, output[field]


def violations(output: dict, payload: dict, refs: list[str]) -> list[tuple[str, int | None, str]]:
    allowed = set(refs)
    ids = {c["video_id"] for c in payload["comparables"]}
    return [
        (field, i, problem)
        for field, i, item in _items(output)
        for problem in _item_violations(field, item, allowed, ids, payload)
    ]


def _drop(output: dict, found: list[tuple[str, int | None, str]]) -> dict:
    output = copy.deepcopy(output)
    for field in {f for f, _, _ in found}:
        bad = {i for f, i, _ in found if f == field}
        if isinstance(output.get(field), list):
            output[field] = [x for j, x in enumerate(output[field]) if j not in bad]
        else:
            output[field] = None
    return output


def _cap(output: dict) -> tuple[dict, int]:
    """Keep at most MAX_INSIGHTS items: patterns, then the shared characteristic,
    observations, audience signal."""
    output, budget, trimmed = copy.deepcopy(output), MAX_INSIGHTS, 0
    for field in (
        "cross_content_patterns",
        "shared_characteristic",
        "single_video_observations",
        "audience_signal",
    ):
        value = output.get(field)
        if isinstance(value, list):
            trimmed += max(0, len(value) - budget)
            output[field] = value[:budget]
            budget -= len(output[field])
        elif value:
            if budget:
                budget -= 1
            else:
                output[field], trimmed = None, trimmed + 1
    return output, trimmed


# ---------- generation ----------


def _generate(payload: dict, refs: list[str]) -> dict:
    schema = output_schema(payload, refs)
    user = json.dumps({"allowed_evidence_refs": refs, "evidence": payload}, ensure_ascii=False)
    raw = llm.complete_json(SYSTEM_PROMPT, user, schema, schema_name="content_analyst")
    found = violations(raw, payload, refs)
    repaired = False
    if found:
        listing = "\n".join(
            f"- {field}{'' if i is None else f'[{i}]'}: {problem}" for field, i, problem in found
        )
        retry = json.dumps(
            {"allowed_evidence_refs": refs, "evidence": payload, "previous_output": raw},
            ensure_ascii=False,
        )
        raw = llm.complete_json(
            SYSTEM_PROMPT + "\n\n" + REPAIR_PROMPT.format(violations=listing),
            retry,
            schema,
            schema_name="content_analyst",
        )
        repaired = True
        found = violations(raw, payload, refs)
    output, trimmed = _cap(_drop(raw, found))
    return {
        "output": output,
        "validation": {
            "repaired": repaired,
            "dropped": [
                {"field": f, "index": i, "problem": p} for f, i, p in found
            ],
            "trimmed_over_limit": trimmed,
        },
    }


def _result(video_id: str, payload: dict, generated: dict | None, status: str) -> dict:
    basis = payload["evidence_basis"]
    output = (generated or {}).get("output") or {}
    ids = {c["video_id"]: c["brand"] for c in payload["comparables"]}

    def with_cross_brand(item: dict) -> dict:
        own = payload["selected_video"]["brand"]
        return {**item, "cross_brand": any(ids[c] != own for c in item["comparable_ids"])}

    patterns = None
    if basis["cross_content_patterns_allowed"]:
        patterns = [with_cross_brand(p) for p in output.get("cross_content_patterns") or []]
    shared = output.get("shared_characteristic")
    result = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "status": status,
        "evidence_basis": basis,
        "single_video_observations": output.get("single_video_observations") or [],
        "shared_characteristic": with_cross_brand(shared)
        if shared and basis["validated_comparables"] == 1
        else None,
        "cross_content_patterns": patterns,
        "audience_signal": output.get("audience_signal")
        if basis["audience_signals_available"]
        else None,
        "evidence_notes": output.get("evidence_notes"),
        "validation": (generated or {}).get("validation"),
        "model": llm.model_name() if generated else None,
        "prompt_version": PROMPT_VERSION,
    }
    if status == "generated" and not any(
        [
            result["single_video_observations"],
            result["shared_characteristic"],
            result["cross_content_patterns"],
            result["audience_signal"],
        ]
    ):
        result["status"] = "insufficient_evidence"
    return result


def generate(context: dict) -> dict:
    """Run the Analyst over a validated analysis_context. Raises llm.LLMError on failure.

    Returns the structured output with status "generated" (at least one item) or
    "insufficient_evidence" (nothing the evidence supports). Fields the evidence doesn't
    allow are None; [] means allowed but nothing was found. Caching, quotas and serving
    live in ai_service.py.
    """
    payload = analyst_input(context)
    generated = _generate(payload, evidence_refs(payload))
    return _result(context["selected_video"]["video_id"], payload, generated, "generated")
