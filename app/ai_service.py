"""Serving and generation for the AI layer (Analyst + Brief).

Pipeline (the same code for curated and live generations):
    structured evidence -> retrieval validation -> Analyst -> Brief -> automated QA

Serving order for a video:
1. Curated showcase video: its QA-approved frozen record is served directly. Curated videos
   never trigger live generation, even before approval.
2. Persistent cache of live generations, served without using any quota. A live generation
   has two stages, cached separately: the analysis (relevance + Analyst), keyed by the
   analysis version, and the Brief, keyed by the full pipeline version.
3. Otherwise the analysis page starts a live analysis when it opens; the Brief is generated
   only when the user asks for it. Both stages pass the same admission: the session limit
   and the server-side daily ceiling count videos, so one video's analysis and Brief take a
   single slot within a session. Anything that can't be verified (quota file, key, live
   switch) fails closed, and a failed stage is not retried in the same session.

Settings (environment or Streamlit Secrets): OPENAI_API_KEY, OPENAI_MODEL,
AI_LIVE_GENERATION ("off" disables live generation), AI_SESSION_LIMIT (default 2),
AI_DAILY_LIMIT (default 10), AI_QUOTA_FILE.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import MutableMapping

import ai_cache
import analysis_context
import analysis_data
import analyst
import brief
import curated
import llm
import quota
import relevance

# Bump when evidence, retrieval or generation logic changes in a way prompt versions don't
# capture. The full version also embeds each prompt version, so any prompt bump invalidates
# cached generations automatically.
PIPELINE_REVISION = "ai-pipeline/2"
ANALYSIS_NAMESPACE = "live_analysis"
BRIEF_NAMESPACE = "live_brief"
DEFAULT_SESSION_LIMIT = 2
DEFAULT_DAILY_LIMIT = 10
SESSION_KEY = "ai_generation_log"

REFUSALS = {
    "disabled": "AI analysis isn't available for this video.",
    "session_limit": "This session has reached its AI generation limit.",
    "daily_limit": "Today's AI generation limit has been reached. Please check back tomorrow.",
    "quota_unavailable": "AI generation limits can't be verified right now.",
    "already_failed": "AI generation failed for this video. The rest of the analysis is unaffected.",
}


def analysis_version() -> str:
    return "+".join([PIPELINE_REVISION, relevance.PROMPT_VERSION, analyst.PROMPT_VERSION])


def pipeline_version() -> str:
    return f"{analysis_version()}+{brief.PROMPT_VERSION}"


def _analysis_key(video_id: str) -> str:
    return f"{video_id}|{analysis_version()}|{llm.model_name()}"


def _brief_key(video_id: str) -> str:
    return f"{video_id}|{pipeline_version()}|{llm.model_name()}"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---------- pipeline ----------


def auto_qa(record: dict) -> dict:
    """Mechanical checks. Showcase videos must pass all of them; live generations show only
    what passed the generators' own validation. A Brief that fails validation is withheld
    (a warning, not a failure): omission is preferred over an unsupported Brief."""
    issues, warnings = [], []
    analysis, generated_brief = record["analyst"], record["brief"]
    basis = analysis["evidence_basis"]
    n = basis["validated_comparables"]
    if analysis["status"] != "generated":
        issues.append(f"analyst status is {analysis['status']}")
    if n < 2 and analysis["cross_content_patterns"] is not None:
        issues.append("cross_content_patterns present with fewer than 2 comparables")
    if n != 1 and analysis["shared_characteristic"] is not None:
        issues.append("shared_characteristic present without exactly 1 comparable")
    if not basis["audience_signals_available"] and analysis["audience_signal"] is not None:
        issues.append("audience_signal present without audience data")
    for pattern in analysis["cross_content_patterns"] or []:
        if len(set(pattern["comparable_ids"])) < 2:
            issues.append("a pattern cites fewer than 2 comparables")
    items = (
        len(analysis["single_video_observations"])
        + len(analysis["cross_content_patterns"] or [])
        + bool(analysis["shared_characteristic"])
        + bool(analysis["audience_signal"])
    )
    if items > analyst.MAX_INSIGHTS:
        issues.append(f"{items} insights exceed the limit of {analyst.MAX_INSIGHTS}")
    if not items:
        issues.append("no insights to show")
    validation = analysis.get("validation") or {}
    if validation.get("dropped"):
        warnings.append(f"analyst dropped {len(validation['dropped'])} item(s) after repair")
    if generated_brief["status"] != "generated":
        warnings.append(f"brief withheld (status {generated_brief['status']}); insights only")
    elif (generated_brief.get("validation") or {}).get("repaired"):
        warnings.append("brief needed a repair round")
    return {"passed": not issues, "issues": issues, "warnings": warnings}


def servable_brief(record: dict) -> dict | None:
    """The Brief to display, or None when it was withheld: a Brief that failed validation or
    that human QA withheld is never shown, and the Analyst insights are served on their own."""
    generated_brief = record["brief"]
    if generated_brief is None:
        return None
    if generated_brief["status"] != "generated" or (record.get("qa") or {}).get("brief_withheld"):
        return None
    return generated_brief["brief"]


def brief_requestable(record: dict) -> bool:
    """A live analysis whose Brief hasn't been generated yet and could be."""
    return record["brief"] is None and record["analyst"]["status"] == "generated"


def run_pipeline(video_id: str, *, attempt: int = 1) -> dict:
    """One full generation. Raises llm.LLMError on any API failure (never partial)."""
    assessment = relevance.assess(video_id, allow_llm=True)
    context = analysis_context.build_analysis_context(video_id)
    analysis = analyst.generate(context)
    generated_brief = brief.generate(context, analysis)
    record = {
        "video_id": str(video_id),
        "model": llm.model_name(),
        "pipeline_version": pipeline_version(),
        "prompt_versions": {
            "relevance": relevance.PROMPT_VERSION,
            "analyst": analyst.PROMPT_VERSION,
            "brief": brief.PROMPT_VERSION,
        },
        "generation_attempt": attempt,
        "generated_at": _now(),
        "relevance": {
            "method": assessment["method"],
            "picks": assessment["picks"],
            "retrieval_assessment": assessment["retrieval_assessment"],
            "candidates": assessment["candidates"],
        },
        "analyst": analysis,
        "brief": generated_brief,
    }
    record["auto_qa"] = auto_qa(record)
    return record


# ---------- serving ----------


def lookup(video_id: str) -> dict:
    """Read-only; never calls the API. Returns {"source", "record", "status"}:
    source "curated" | "cache" | None; status "ready", "curated_pending" (a showcase video
    whose output isn't approved yet) or "not_generated". A cached record's "brief" is None
    until its Brief has been generated."""
    video_id = str(video_id)
    if curated.is_curated(video_id):
        record = curated.approved_record(video_id)
        if record:
            return {"source": "curated", "record": record, "status": "ready"}
        return {"source": None, "record": None, "status": "curated_pending"}
    analysis = ai_cache.get(ANALYSIS_NAMESPACE, _analysis_key(video_id))
    if analysis:
        record = {**analysis, "brief": ai_cache.get(BRIEF_NAMESPACE, _brief_key(video_id))}
        return {"source": "cache", "record": record, "status": "ready"}
    return {"source": None, "record": None, "status": "not_generated"}


def similar_cards(video_id: str) -> list[dict]:
    """Similar High Performer cards consistent with the AI evidence: a curated record's frozen
    picks, else the read-only relevance assessment (cached LLM judgment or rules)."""
    record = curated.approved_record(video_id)
    picks = record["relevance"]["picks"] if record else relevance.assess(video_id)["picks"]
    by_id = {
        c["video_id"]: c
        for c in analysis_data.retrieval_candidates(video_id, relevance.RETRIEVAL_DEPTH)
    }
    return [by_id[v] for v in picks if v in by_id]


def _int_setting(name: str, default: int) -> int:
    raw = llm.setting(name)
    return int(raw) if raw else default


def live_generation_enabled() -> bool:
    return llm.available() and llm.setting("AI_LIVE_GENERATION").lower() not in ("off", "0", "false")


def _log(session: MutableMapping) -> dict:
    return session.setdefault(SESSION_KEY, {"videos": [], "failed": []})


def admission(video_id: str, stage: str, session: MutableMapping) -> str | None:
    """Read-only check (reserves nothing) of whether a live `stage` ("analysis" | "brief")
    may run for this video: None, or a refusal status from REFUSALS."""
    log = _log(session)
    if f"{video_id}:{stage}" in log["failed"]:
        return "already_failed"
    if not live_generation_enabled():
        return "disabled"
    if video_id in log["videos"]:
        return None
    try:
        session_limit = _int_setting("AI_SESSION_LIMIT", DEFAULT_SESSION_LIMIT)
        daily_limit = _int_setting("AI_DAILY_LIMIT", DEFAULT_DAILY_LIMIT)
    except ValueError:
        return "quota_unavailable"
    if len(log["videos"]) >= session_limit:
        return "session_limit"
    used = quota.usage()["count"]
    if used is None:
        return "quota_unavailable"
    if used >= daily_limit:
        return "daily_limit"
    return None


def _admit(video_id: str, stage: str, session: MutableMapping) -> str | None:
    """admission(), then take the video's daily slot if this session hasn't yet."""
    refusal = admission(video_id, stage, session)
    log = _log(session)
    if refusal or video_id in log["videos"]:
        return refusal
    try:
        if not quota.reserve(_int_setting("AI_DAILY_LIMIT", DEFAULT_DAILY_LIMIT)):
            return "daily_limit"
    except quota.QuotaError:
        return "quota_unavailable"
    log["videos"].append(video_id)
    return None


def generate_analysis(video_id: str, session: MutableMapping) -> str | None:
    """Live relevance + Analyst for a non-curated video, cached on success. Returns None
    (ready, or nothing to do) or a refusal status; "failed" when the API call failed."""
    video_id = str(video_id)
    if lookup(video_id)["status"] != "not_generated":
        return None
    refusal = _admit(video_id, "analysis", session)
    if refusal:
        return refusal
    try:
        assessment = relevance.assess(video_id, allow_llm=True)
        analysis = analyst.generate(analysis_context.build_analysis_context(video_id))
        ai_cache.put(
            ANALYSIS_NAMESPACE,
            _analysis_key(video_id),
            {
                "video_id": video_id,
                "model": llm.model_name(),
                "pipeline_version": analysis_version(),
                "prompt_versions": {
                    "relevance": relevance.PROMPT_VERSION,
                    "analyst": analyst.PROMPT_VERSION,
                },
                "generated_at": _now(),
                "relevance": {
                    "method": assessment["method"],
                    "picks": assessment["picks"],
                    "retrieval_assessment": assessment["retrieval_assessment"],
                    "candidates": assessment["candidates"],
                },
                "analyst": analysis,
            },
        )
    except (llm.LLMError, OSError):
        _log(session)["failed"].append(f"{video_id}:analysis")
        return "failed"
    return None


def generate_brief(video_id: str, session: MutableMapping) -> str | None:
    """User-requested Brief for a cached live analysis, cached on success (including a
    withheld Brief, so it isn't regenerated). Same return convention as generate_analysis."""
    video_id = str(video_id)
    found = lookup(video_id)
    if found["source"] != "cache" or not brief_requestable(found["record"]):
        return None
    refusal = _admit(video_id, "brief", session)
    if refusal:
        return refusal
    try:
        # The relevance judgment is cached by the analysis stage, so this rebuilds the
        # exact context the Analyst saw without another call.
        relevance.assess(video_id, allow_llm=True)
        context = analysis_context.build_analysis_context(video_id)
        generated = brief.generate(context, found["record"]["analyst"])
        ai_cache.put(
            BRIEF_NAMESPACE,
            _brief_key(video_id),
            {**generated, "pipeline_version": pipeline_version(), "generated_at": _now()},
        )
    except (llm.LLMError, OSError):
        _log(session)["failed"].append(f"{video_id}:brief")
        return "failed"
    return None
