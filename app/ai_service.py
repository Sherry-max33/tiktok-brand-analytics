"""Serving and generation for the AI layer (Analyst + Brief).

Pipeline (the same code for curated and live generations):
    structured evidence -> retrieval validation -> Analyst -> Brief -> automated QA

Serving order for a video:
1. Curated showcase video: its QA-approved frozen record is served directly. Curated videos
   never trigger live generation, even before approval.
2. Persistent cache, keyed by (video_id, AI pipeline version, model): served without using
   any quota.
3. Otherwise nothing is generated on page view. A user can request an on-demand generation,
   which runs only after the session limit and the server-side daily ceiling both pass.
   Anything that can't be verified (quota file, key, live switch) fails closed, and a failed
   API call is not retried.

Settings (environment or Streamlit Secrets): OPENAI_API_KEY, OPENAI_MODEL,
AI_LIVE_GENERATION ("off" disables live generation), AI_SESSION_LIMIT (default 2),
AI_DAILY_LIMIT (default 25), AI_QUOTA_FILE.
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
CACHE_NAMESPACE = "generations"
DEFAULT_SESSION_LIMIT = 2
DEFAULT_DAILY_LIMIT = 25
SESSION_KEY = "ai_generation_log"


def pipeline_version() -> str:
    return "+".join(
        [PIPELINE_REVISION, relevance.PROMPT_VERSION, analyst.PROMPT_VERSION, brief.PROMPT_VERSION]
    )


def cache_key(video_id: str) -> str:
    return f"{video_id}|{pipeline_version()}|{llm.model_name()}"


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
    if generated_brief["status"] != "generated" or (record.get("qa") or {}).get("brief_withheld"):
        return None
    return generated_brief["brief"]


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
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
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
    whose output isn't approved yet) or "not_generated"."""
    video_id = str(video_id)
    if curated.is_curated(video_id):
        record = curated.approved_record(video_id)
        if record:
            return {"source": "curated", "record": record, "status": "ready"}
        return {"source": None, "record": None, "status": "curated_pending"}
    record = ai_cache.get(CACHE_NAMESPACE, cache_key(video_id))
    if record:
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


def generate_on_demand(video_id: str, session: MutableMapping) -> dict:
    """User-requested generation for a non-curated video. Returns {"status", "record",
    "message"}; status "ready" (served from curated/cache or newly generated), or one of
    "curated_pending", "disabled", "session_limit", "daily_limit", "quota_unavailable",
    "already_failed", "failed"."""
    video_id = str(video_id)
    found = lookup(video_id)
    if found["status"] != "not_generated":
        return {**found, "message": None}

    log = session.setdefault(SESSION_KEY, {"count": 0, "failed": []})
    if video_id in log["failed"]:
        return _refusal("already_failed", "Generation failed for this video; it won't be retried.")
    if not live_generation_enabled():
        return _refusal("disabled", "Live AI generation is not available right now.")
    try:
        session_limit = _int_setting("AI_SESSION_LIMIT", DEFAULT_SESSION_LIMIT)
        daily_limit = _int_setting("AI_DAILY_LIMIT", DEFAULT_DAILY_LIMIT)
    except ValueError:
        return _refusal("quota_unavailable", "Generation limits are misconfigured.")
    if log["count"] >= session_limit:
        return _refusal("session_limit", "This session has reached its AI generation limit.")
    try:
        if not quota.reserve(daily_limit):
            return _refusal("daily_limit", "Today's AI generation limit has been reached.")
    except quota.QuotaError:
        return _refusal("quota_unavailable", "Generation limits can't be verified right now.")

    log["count"] += 1
    try:
        record = run_pipeline(video_id)
        ai_cache.put(CACHE_NAMESPACE, cache_key(video_id), record)
    except (llm.LLMError, OSError):
        log["failed"].append(video_id)
        return _refusal("failed", "AI generation failed. The rest of the analysis is unaffected.")
    return {"source": "live", "record": record, "status": "ready", "message": None}


def _refusal(status: str, message: str) -> dict:
    return {"source": None, "record": None, "status": status, "message": message}
