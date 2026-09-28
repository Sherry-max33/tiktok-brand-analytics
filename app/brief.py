"""Next Content Brief (Phase 5 of docs/07-ai-layer.md).

A separate generation step after the Analyst. It sees the same validated evidence plus the
Analyst's insights (so it can re-check them instead of amplifying an interpretation error)
and returns one testable creative hypothesis as structured JSON.

Grounding follows the Analyst: based_on_insights and evidence_refs are enums of what exists,
and code checks what a schema can't (no causal or predictive claims, controls that aren't
framed as success factors, WER always measured, comment sentiment only when it was
collected). One repair round; a brief that still fails is marked failed_validation and not
shown. A brief is only generated when the Analyst produced at least one insight.
"""

from __future__ import annotations

import json
import re

import analyst
import llm

PROMPT_VERSION = "brief/v13"
SCHEMA_VERSION = "brief_output/v2"
SOURCE_TYPES = ("analyst_insight", "observed_evidence")
METRICS = ("WER", "Share rate", "Save rate", "Comment rate", "Like rate", "Net comment sentiment")

SYSTEM_PROMPT = """You write the Next Content Brief in a TikTok content-analysis tool for Nike
and Adidas videos. Turn the Analyst's insights into ONE executable creative hypothesis worth
testing next. It is an evidence-informed hypothesis, not a performance prediction or a
guaranteed best practice.

Inputs
- evidence: the validated evidence about the selected video and its comparable high
  performers (null = unknown; [] = the caption classifier found nothing).
- insights: the Analyst's findings, each with an id. Re-check each insight against the
  evidence and build only on what the evidence supports.

Fields (each 1-2 concise sentences unless noted)
- concept: one specific creative idea as a short title (max 8 words), not a generic goal like
  "create an engaging Jordan video".
- opening_hook: how the first seconds enter the video, concretely.
- creative_direction: the insight turned into execution (format, setting, framing), consistent
  with the evidence. A detail the selected video isn't shown to have (e.g. a setting only the
  comparables show) must be attributed, e.g. "an outdoor setting, as in the comparables".
- engagement_approach: build on the CTA, social mechanic or commerce signals present in the
  evidence. If none are present, keep it simple and present any engagement element as part of
  the experiment, not as evidence-backed. Never claim one approach performs better.
- why_this_direction: the provenance (which insights and evidence), framed as what the test
  would learn, e.g. "tests whether ...".
- test_plan.variable_to_test: exactly one primary variable. It must be traceable to a
  specific source, recorded in test_plan.variable_source:
  - source_type "analyst_insight": insight_id names the insight whose finding or
    worth_testing mentions the element being varied (also list it in based_on_insights);
  - source_type "observed_evidence": evidence_refs name the specific fields of the selected
    video or of the comparables whose values show the element (e.g. two comparables'
    content_profile.frames.visual_setting = "Studio"). Comparable-derived variables are
    allowed; cite the comparables' own fields.
  - evidence_statement: one plain sentence a reader sees as "why test it", stating only what
    the cited source shows, e.g. "Two of the three similar high performers are filmed in a
    studio, while this video is filmed at home." Count them correctly.
  If no insight or evidence field shows the element, choose a different variable. An element
  that appears nowhere in the evidence (e.g. an on-screen prompt when no CTA is recorded) can't
  be the test variable. A field that is null for the selected video is unknown, so it can't
  be one side of a contrast; look for a contrast in fields both videos have (e.g. their
  caption text).
  Any statement about the comparables must cite their fields: the field that holds the
  value, or, to say a comparable's caption lacks something (e.g. no promo code), its
  caption.text. Never cite unrelated fields to satisfy this, and never write evidence paths
  into the text fields; evidence_refs carry them.
  In short: meaningful evidence-backed differences between this video and its similar high
  performers are candidate test variables; shared observed patterns are candidates to keep
  constant.
- test_plan.variant_a / variant_b: identical except for that variable.
- test_plan.what_to_keep: 2-4 experimental controls held constant across variants so the
  variable can be interpreted, e.g. "Keep the visual setting the same in both variants".
  Controls, not success factors: never justify them by past performance.
- test_plan.measure: WER first, then only the metrics the test needs; at most 3 in total. Use Net comment
  sentiment only if audience signals exist in the evidence.
- based_on_insights: the ids of the insights the brief builds on. evidence_refs: the evidence
  paths it relies on.

Rules
- Observability: the data captures only {captured}. It has no on-screen text, voiceover,
  narration, speech, music or audio data. variable_to_test, variable_source.evidence_statement
  and why_this_direction must stay within the captured dimensions; never state or imply that
  any video has or lacks an uncaptured element. opening_hook, creative_direction,
  engagement_approach and the variants may suggest such elements only as new ideas, in a
  sentence that says so, e.g. "As a new idea not drawn from the data, add on-screen text
  ...". Never write "as seen in" about them.
- Never justify a direction by a video's performance ("strong performance", "performed well",
  "successful"); "similar high performers" only names the comparison set.
- Never mention a caption's language or translation. Catch-all taxonomy values are omitted
  or null: unknown, never a variant.
- Voice: write for a brand or content strategist. In text fields say "this video" and
  "similar high performers" (e.g. "two of the three similar high performers"), never "the
  selected video", "comparable 1" or IDs, and describe labels in plain words rather than
  field names. Concise and specific; no unsupported interpretation. Never generalize beyond
  this video and its similar high performers (e.g. "top-performing Nike content").
- No causal or predictive claims: never "will increase", "drives", "boosts", "guarantees",
  "outperforms", "proven", "best practice", "because it worked". Avoid the words cause, drive,
  driven, boost, fuel, due to, because.
- Do not speculate about audience demographics, creator intent or brand strategy. Do not name
  creators. Mention a product line only if the evidence shows it resolved and the caption
  supports it.
- Frame labels are single zero-shot labels; do not invent visual details, product facts,
  athletes or events that the evidence doesn't contain."""

SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{captured}", analyst.CAPTURED_DIMENSIONS)

REPAIR_PROMPT = """Your previous brief broke these rules:
{violations}
Return the complete brief again with those problems fixed."""

_PREDICTIVE = re.compile(
    r"\b(will (increase|boost|improve|drive|lift|raise|grow|perform|go viral|get|generate)|"
    r"guarantee\w*|outperform\w*|performs? better|proven|best practice\w*|surefire)\b",
    re.IGNORECASE,
)
_SUCCESS_JUSTIFIED = re.compile(
    r"\b(because|since it|drove|driv\w+|work(ed|s)|success\w*|proven|effective|engaging|"
    r"(high|strong|more) engagement|resonat\w*|(high|well|top|best|strong)[- ]perform\w*|perform(s|ed|ing) (well|better|strongly))\b",
    re.IGNORECASE,
)
# "High-performing comparables" names the comparison set; citing a video's performance as the
# reason for a direction implies causation.
_PERFORMANCE_JUSTIFIED = re.compile(
    r"\b((strong|high|good|better|superior|impressive) (performance|results|engagement)|"
    r"perform(s|ed)? (well|better|strongly)|success\w*)\b",
    re.IGNORECASE,
)
# Taxonomy labels are quoted ('Performance' brand style) and are not success claims.
_QUOTED_LABEL = re.compile(r"'[^']+'|\"[^\"]+\"")


def insights_with_ids(analysis: dict) -> list[dict]:
    items = []
    for i, item in enumerate(analysis.get("cross_content_patterns") or [], start=1):
        items.append({"id": f"pattern_{i}", "kind": "cross_content_pattern", **item})
    if analysis.get("shared_characteristic"):
        items.append(
            {"id": "shared_1", "kind": "shared_characteristic", **analysis["shared_characteristic"]}
        )
    for i, item in enumerate(analysis.get("single_video_observations") or [], start=1):
        items.append({"id": f"observation_{i}", "kind": "single_video_observation", **item})
    if analysis.get("audience_signal"):
        items.append({"id": "audience_1", "kind": "audience_signal", **analysis["audience_signal"]})
    return items


def _schema(insight_ids: list[str], refs: list[str], metrics: list[str]) -> dict:
    string = {"type": "string"}
    variable_source = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_type", "insight_id", "evidence_refs", "evidence_statement"],
        "properties": {
            "source_type": {"type": "string", "enum": list(SOURCE_TYPES)},
            "insight_id": {"anyOf": [{"type": "string", "enum": insight_ids}, {"type": "null"}]},
            "evidence_refs": {"type": "array", "items": {"type": "string", "enum": refs}},
            "evidence_statement": string,
        },
    }
    test_plan = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "variable_to_test", "variable_source", "variant_a", "variant_b", "what_to_keep", "measure"
        ],
        "properties": {
            "variable_to_test": string,
            "variable_source": variable_source,
            "variant_a": string,
            "variant_b": string,
            "what_to_keep": {"type": "array", "items": string},
            "measure": {"type": "array", "items": {"type": "string", "enum": metrics}},
        },
    }
    properties = {
        "concept": string,
        "opening_hook": string,
        "creative_direction": string,
        "engagement_approach": string,
        "why_this_direction": string,
        "test_plan": test_plan,
        "based_on_insights": {"type": "array", "items": {"type": "string", "enum": insight_ids}},
        "evidence_refs": {"type": "array", "items": {"type": "string", "enum": refs}},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _texts(brief: dict) -> dict[str, str]:
    plan = brief["test_plan"]
    return {
        "concept": brief["concept"],
        "opening_hook": brief["opening_hook"],
        "creative_direction": brief["creative_direction"],
        "engagement_approach": brief["engagement_approach"],
        "why_this_direction": brief["why_this_direction"],
        "test_plan.variable_to_test": plan["variable_to_test"],
        "test_plan.variable_source.evidence_statement": plan["variable_source"]["evidence_statement"],
        "test_plan.variant_a": plan["variant_a"],
        "test_plan.variant_b": plan["variant_b"],
    }


# Words that name the shape of a test, not the element being varied.
_GENERIC = {
    "versus", "presence", "absence", "video", "videos", "content", "inclusion", "including",
    "include", "includes", "without", "approach", "variant", "whether", "using", "there",
    "their", "these", "those", "which", "other", "different", "specific", "selected",
}
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "both": 2, "all three": 3, "all two": 2}
_PEERS = r"(comparables?|high performers?|peers?|similar videos?)"
_COMPARABLE_COUNT = re.compile(
    rf"\b(\d+|one|two|three|both|all three)\s+(?:of the (?:\w+ )?)?(?:[\w-]+\s+){{0,2}}{_PEERS}\b",
    re.IGNORECASE,
)
_RAW_PATH = re.compile(r"\w+\[[^\]]+\]\.\w+|\b(content_profile|selected_video|audience_signals)\.\w+")


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w[:5] for w in words if len(w) >= 4 and w not in _GENERIC}


def _ref_value(payload: dict, ref: str):
    match = re.match(r"comparables\[([^\]]+)\]\.(.+)", ref)
    if match:
        item = next((c for c in payload["comparables"] if c["video_id"] == match.group(1)), None)
        return analyst._resolve(item or {}, match.group(2))
    return analyst._resolve(payload, ref)


def _source_violations(plan: dict, brief: dict, insights: dict, payload: dict) -> list[str]:
    """The test variable must trace to a specific insight or observed evidence field."""
    source = plan["variable_source"]
    variable = _stems(plan["variable_to_test"])
    problems = []
    if source["source_type"] == "analyst_insight":
        insight = insights.get(source["insight_id"])
        if insight is None:
            return ["variable_source names no insight_id"]
        if source["insight_id"] not in brief["based_on_insights"]:
            problems.append("the insight behind the test variable is missing from based_on_insights")
        source_text = " ".join(
            filter(None, (insight.get("title"), insight.get("finding"), insight.get("worth_testing")))
        )
    else:
        if not source["evidence_refs"]:
            return ["variable_source cites no evidence_refs"]
        values = [_ref_value(payload, r) for r in source["evidence_refs"]]
        for ref in source["evidence_refs"]:
            if ref.endswith(("caption.text", "hashtags")):
                # Caption-derived labels are a reading of the cited caption.
                prefix = ref.split("caption.text")[0].split("hashtags")[0]
                owner = "" if prefix in ("", "selected_video.") else prefix
                values.append(_ref_value(payload, f"{owner}content_profile.caption_and_hashtags"))
        source_text = " ".join(json.dumps(v, ensure_ascii=False) for v in values)
    if not variable & _stems(source_text):
        problems.append(
            f"the test variable ({plan['variable_to_test']!r}) doesn't appear in its cited source; "
            "choose a variable the source actually shows"
        )
    cited_comparables = {
        m.group(1) for r in source["evidence_refs"] if (m := re.match(r"comparables\[([^\]]+)\]", r))
    }
    if source["source_type"] == "analyst_insight":
        cited_comparables |= set(insights[source["insight_id"]].get("comparable_ids") or [])
    if re.search(rf"\b{_PEERS}\b", source["evidence_statement"], re.IGNORECASE) and not cited_comparables:
        problems.append(
            "evidence_statement describes the comparables, but variable_source.evidence_refs cites "
            "none of their fields: add the comparables' field for the same element (or their "
            "caption.text) to variable_source.evidence_refs; do not write paths in the text"
        )
    own_fields = {r.removeprefix("selected_video.") for r in source["evidence_refs"] if not r.startswith("comparables[")}
    for ref in source["evidence_refs"]:
        match = re.match(r"comparables\[[^\]]+\]\.(.+)", ref)
        field = match.group(1) if match else None
        related = own_fields | {"caption.text", "hashtags"}
        if field and field.startswith("product.") and any(f.startswith("product.") for f in own_fields):
            related.add(field)
        if field and field not in related:
            problems.append(
                f"variable_source cites {ref}, a comparable field unrelated to the selected "
                "video's cited fields; cite the same field for both, or the comparable's caption.text"
            )
    variants = " ".join((plan["variable_to_test"], plan["variant_a"], plan["variant_b"]))
    if re.search(r"\bOther\b", variants):
        problems.append("the test uses the catch-all 'Other' label, which isn't an actionable variant")
    for match in _COMPARABLE_COUNT.finditer(source["evidence_statement"]):
        word = match.group(1).lower()
        count = int(word) if word.isdigit() else _NUMBER_WORDS[word]
        if count > len(cited_comparables):
            problems.append(
                f"evidence_statement mentions {count} comparables but variable_source cites fields "
                f"of {len(cited_comparables)}; cite the same field for every comparable it counts"
            )
    return problems


_EVIDENCE_FIELDS = (
    "why_this_direction",
    "test_plan.variable_to_test",
    "test_plan.variable_source.evidence_statement",
)
_NEW_IDEA = re.compile(r"\b(new|idea|optional|untested)\b", re.IGNORECASE)
_ATTRIBUTED = re.compile(
    r"\b(as (seen|observed|shown|used|present)|like the (selected|comparable)|as in the)\b",
    re.IGNORECASE,
)


def _observability_violations(brief: dict) -> list[str]:
    """Uncaptured modalities are never evidence; in execution fields they must be framed as
    new ideas."""
    problems = []
    for field, text in _texts(brief).items():
        if field in _EVIDENCE_FIELDS:
            match = analyst.UNCAPTURED.search(text or "")
            if match:
                problems.append(
                    f"{field} relies on {match.group(0)!r}, which the dataset doesn't capture; "
                    "evidence-based fields must stay within the captured dimensions"
                )
            continue
        for sentence in re.split(r"(?<=[.!?;])\s+", text or ""):
            match = analyst.UNCAPTURED.search(sentence)
            if match and (not _NEW_IDEA.search(sentence) or _ATTRIBUTED.search(sentence)):
                problems.append(
                    f"{field} mentions {match.group(0)!r} (not captured by the dataset) without "
                    "framing it as a new idea, or attributes it to the evidence"
                )
    return problems


def violations(brief: dict, audience_available: bool, insights: dict, payload: dict) -> list[str]:
    problems = _source_violations(brief["test_plan"], brief, insights, payload)
    problems += _observability_violations(brief)
    for field, text in _texts(brief).items():
        match = analyst.LANGUAGE_CLAIM.search(text or "")
        if match:
            problems.append(f"{field} mentions caption language or translation ({match.group(0)!r})")
        claim = analyst.population_claim(text)
        if claim:
            problems.append(
                f"{field} generalizes beyond this video and its similar high performers ({claim!r})"
            )
    if not brief["based_on_insights"]:
        problems.append("based_on_insights is empty")
    if not brief["evidence_refs"]:
        problems.append("evidence_refs is empty")
    for field, text in _texts(brief).items():
        for pattern, label in (
            (analyst._CAUSAL, "causal"),
            (_PREDICTIVE, "predictive"),
            (analyst._RETRIEVAL, "retrieval"),
        ):
            match = pattern.search(text or "")
            if match:
                problems.append(f"{field} uses {label} language ({match.group(0)!r})")
        match = _PERFORMANCE_JUSTIFIED.search(text or "")
        if match:
            problems.append(f"{field} cites performance as a justification ({match.group(0)!r})")
        match = _RAW_PATH.search(text or "")
        if match:
            problems.append(f"{field} pastes an evidence path into the text ({match.group(0)!r})")
    plan = brief["test_plan"]
    if not 2 <= len(plan["what_to_keep"]) <= 4:
        problems.append("what_to_keep must list 2-4 controls")
    for control in plan["what_to_keep"]:
        unquoted = _QUOTED_LABEL.sub("", control)
        match = _SUCCESS_JUSTIFIED.search(unquoted) or analyst._CAUSAL.search(unquoted)
        if match:
            problems.append(
                f"what_to_keep states a success factor, not a control ({match.group(0)!r})"
            )
    if not plan["measure"] or plan["measure"][0] != "WER":
        problems.append("measure must start with WER")
    if len(plan["measure"]) > 3:
        problems.append("measure lists more than 3 metrics")
    if "Net comment sentiment" in plan["measure"] and not audience_available:
        problems.append("Net comment sentiment is measured but no audience signals exist")
    if plan["variant_a"].strip().lower() == plan["variant_b"].strip().lower():
        problems.append("variant_a and variant_b are identical")
    return problems


def generate(context: dict, analysis: dict) -> dict:
    """Brief for a validated context + Analyst result. Raises llm.LLMError on API failure.

    status: "generated", "not_generated" (the Analyst found nothing to build on) or
    "failed_validation" (kept for debugging, never shown).
    """
    base = {"schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION}
    insights = insights_with_ids(analysis)
    if analysis.get("status") != "generated" or not insights:
        return {**base, "status": "not_generated", "brief": None, "validation": None}

    payload = analyst.analyst_input(context)
    refs = analyst.evidence_refs(payload)
    audience = payload["evidence_basis"]["audience_signals_available"]
    metrics = [m for m in METRICS if audience or m != "Net comment sentiment"]
    schema = _schema([i["id"] for i in insights], refs, metrics)
    user = {"evidence": payload, "insights": insights}

    by_id = {i["id"]: i for i in insights}
    brief = llm.complete_json(
        SYSTEM_PROMPT, json.dumps(user, ensure_ascii=False), schema, schema_name="content_brief"
    )
    found, repaired = violations(brief, audience, by_id, payload), False
    initial = found
    if found:
        repaired = True
        brief = llm.complete_json(
            SYSTEM_PROMPT + "\n\n" + REPAIR_PROMPT.format(violations="\n".join(f"- {p}" for p in found)),
            json.dumps({**user, "previous_brief": brief}, ensure_ascii=False),
            schema,
            schema_name="content_brief",
        )
        found = violations(brief, audience, by_id, payload)
    return {
        **base,
        "status": "failed_validation" if found else "generated",
        "brief": brief,
        "validation": {"repaired": repaired, "initial_violations": initial, "violations": found},
    }
