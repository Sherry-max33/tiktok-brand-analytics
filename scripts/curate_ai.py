"""Curate the AI showcase set: propose → add → generate → review → approve (freeze).

Curated outputs come only from the real pipeline (evidence → retrieval validation → Analyst →
Brief → automated QA); nothing here authors or edits AI text. Approved records are frozen in
app/frozen_ai/curated.json and served without any API call.

    python scripts/curate_ai.py propose              # candidate set + coverage report (no API calls)
    python scripts/curate_ai.py add ID [ID ...] --reason "..."
    python scripts/curate_ai.py add --from-proposal  # register the proposed set with its reasons
    python scripts/curate_ai.py generate ID [ID ...] # new attempt per video (needs OPENAI_API_KEY)
    python scripts/curate_ai.py show ID [--attempt N]
    python scripts/curate_ai.py select ID N          # choose the stronger fully grounded attempt
    python scripts/curate_ai.py qa ID approve|reject --note "..."
    python scripts/curate_ai.py status

Regeneration policy: if an output fails because of a systematic prompt, retrieval, grounding or
data problem, fix the system (which bumps the pipeline version) and regenerate. For normal
model variance, at most curated.MAX_ATTEMPTS attempts per pipeline version; every attempt is
kept in the registry.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
logging.getLogger("streamlit").setLevel(logging.ERROR)

import ai_service  # noqa: E402
import analysis_data  # noqa: E402
import catalog  # noqa: E402
import curated  # noqa: E402
import llm  # noqa: E402
import relevance  # noqa: E402
from components.hero import SUGGESTIONS  # noqa: E402

PROPOSAL_FILE = catalog.REPO_ROOT / "data" / "processed" / "ai" / "curation_proposal.json"
TARGET_MAX = 25
PER_KEYWORD = 2
KEY_CONTENT_TYPES = ("vibe_ootd", "product_showcase", "collaboration")
KEY_PRODUCT_LINES = ("samba", "spezial", "gazelle", "tech_fleece", "jordan", "air_max", "af1", "f50")
MIN_PER_CONTENT_TYPE = 2
MIN_WITH_SENTIMENT = 4
MIN_CROSS_BRAND = 3
PACE_S = 45  # one full pipeline run is close to the account's 30K tokens-per-minute limit


# ---------- screening (no API calls) ----------


def _profile(video_id: str) -> dict:
    df = catalog.load_library()
    row = df[df["video_id"] == video_id].iloc[0]
    assessment = relevance.assess(video_id)
    brands = {df.loc[df["video_id"] == v, "brand"].iloc[0] for v in assessment["picks"]}
    return {
        "video_id": video_id,
        "brand": row["brand"],
        "content_types": list(row["content_type"]),
        "product_lines": list(row["display_lines"]),
        "has_caption": bool(catalog.display_caption(row["caption_raw"])),
        "product_unresolved": bool(row["product_unresolved"]),
        "has_frames": bool(row["visual_format"]),
        "has_sentiment": analysis_data.audience_sentiment(video_id) is not None,
        "comparables": len(assessment["picks"]),
        # The rules screen accepts cross-brand candidates the LLM judgment usually rejects.
        "cross_brand": assessment["method"] == "llm" and bool(brands - {row["brand"]}),
        "relevance_method": assessment["method"],
        "top_pct": int(row["top_pct"]) if row["top_pct"] == row["top_pct"] else None,
        "caption": (catalog.display_caption(row["caption_raw"]) or "")[:70],
    }


def _problems(p: dict) -> list[str]:
    problems = []
    if not p["has_caption"]:
        problems.append("no caption")
    if p["product_unresolved"]:
        problems.append("product line unresolved (taxonomy)")
    if not p["has_frames"]:
        problems.append("no frame labels")
    if p["comparables"] == 0:
        problems.append("no validated comparables")
    return problems


def _keyword_results(dim: str, value: str) -> list[str]:
    kwargs = {"product": (value,)} if dim == "product" else {"content": (value,)}
    cards, _ = catalog.search_videos(**kwargs, limit=15)
    return [c["video_id"] for c in cards]


def _coverage(profiles: list[dict]) -> dict:
    return {
        "brands": {b: sum(p["brand"] == b for p in profiles) for b in ("nike", "adidas")},
        "content_types": {
            t: sum(t in p["content_types"] for p in profiles) for t in KEY_CONTENT_TYPES
        },
        "product_lines": {
            line: sum(line in p["product_lines"] for p in profiles) for line in KEY_PRODUCT_LINES
        },
        "with_sentiment": sum(p["has_sentiment"] for p in profiles),
        "cross_brand_retrieval": sum(p["cross_brand"] for p in profiles),
    }


def _gaps(cov: dict, n: int) -> list[tuple[str, object]]:
    gaps = []
    for t, count in cov["content_types"].items():
        if count < MIN_PER_CONTENT_TYPE:
            gaps.append(("content_type", t))
    for line, count in cov["product_lines"].items():
        if count == 0:
            gaps.append(("product_line", line))
    if cov["with_sentiment"] < MIN_WITH_SENTIMENT:
        gaps.append(("sentiment", None))
    if cov["cross_brand_retrieval"] < MIN_CROSS_BRAND:
        gaps.append(("cross_brand", None))
    low = min(cov["brands"], key=cov["brands"].get)
    if n and cov["brands"][low] / n < 0.4:
        gaps.append(("brand", low))
    return gaps


def _fits(p: dict, gap: tuple[str, object]) -> bool:
    kind, value = gap
    return {
        "content_type": lambda: value in p["content_types"],
        "product_line": lambda: value in p["product_lines"],
        "sentiment": lambda: p["has_sentiment"],
        "cross_brand": lambda: p["cross_brand"],
        "brand": lambda: p["brand"] == value,
    }[kind]()


def propose() -> None:
    chosen: dict[str, dict] = {}
    skipped: list[dict] = []
    df = catalog.load_library()
    authors = dict(zip(df["video_id"], df["author_username"]))

    def creator(video_id: str) -> str:
        return authors.get(video_id) or video_id

    def creators() -> set[str]:
        return {creator(v) for v in chosen}

    def take(video_id: str, reason: str) -> None:
        if video_id in chosen:
            chosen[video_id]["reasons"].append(reason)
        else:
            chosen[video_id] = {**_profile(video_id), "reasons": [reason]}

    for card in catalog.featured_videos():
        take(card["video_id"], "homepage featured")

    for dim, value in SUGGESTIONS:
        taken = 0
        for video_id in _keyword_results(dim, value):
            if taken == PER_KEYWORD:
                break
            if video_id in chosen:
                chosen[video_id]["reasons"].append(f"try: {value}")
                taken += 1
                continue
            profile = _profile(video_id)
            problems = _problems(profile)
            if creator(video_id) in creators():
                problems.append("creator already in the set")
            if problems:
                skipped.append({"video_id": video_id, "source": f"try: {value}", "problems": problems})
                continue
            take(video_id, f"try: {value}")
            taken += 1

    pool = df[(df["view_count"] >= catalog.FEATURED_MIN_VIEWS) & df["wer"].notna()]
    pool = pool.sort_values("wer", ascending=False)["video_id"].head(400).tolist()
    profiles_cache: dict[str, dict] = {}
    unfillable: set = set()
    while len(chosen) < TARGET_MAX:
        gaps = [g for g in _gaps(_coverage(list(chosen.values())), len(chosen)) if g not in unfillable]
        if not gaps:
            break
        gap = gaps[0]
        for video_id in pool:
            if video_id in chosen or creator(video_id) in creators():
                continue
            profile = profiles_cache.setdefault(video_id, _profile(video_id))
            if not _problems(profile) and _fits(profile, gap):
                take(video_id, f"coverage: {gap[0]}{'' if gap[1] is None else ' ' + str(gap[1])}")
                break
        else:
            unfillable.add(gap)

    profiles = list(chosen.values())
    proposal = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "videos": profiles,
        "coverage": _coverage(profiles),
        "remaining_gaps": _gaps(_coverage(profiles), len(profiles)),
        "skipped": skipped,
    }
    PROPOSAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROPOSAL_FILE.write_text(json.dumps(proposal, indent=1, ensure_ascii=False))

    print(f"{len(profiles)} proposed ({PROPOSAL_FILE.relative_to(catalog.REPO_ROOT)})")
    for p in profiles:
        flags = ",".join(
            f for f, on in (
                ("sentiment", p["has_sentiment"]),
                ("cross-brand", p["cross_brand"]),
                ("no-frames", not p["has_frames"]),
                ("unresolved", p["product_unresolved"]),
            ) if on
        )
        print(
            f"  {p['video_id']} {p['brand']:<6} top{p['top_pct']:>3}% comps={p['comparables']}"
            f"({p['relevance_method']}) {'/'.join(p['content_types']) or '-':<28} "
            f"{'/'.join(p['product_lines']) or '-':<22} [{flags}] {'; '.join(p['reasons'])}"
        )
        print(f"      {p['caption']!r}")
    print("coverage:", json.dumps(proposal["coverage"]))
    print("remaining gaps:", proposal["remaining_gaps"] or "none")
    print("skipped keyword results:")
    for s in skipped:
        print(f"  {s['video_id']} ({s['source']}): {', '.join(s['problems'])}")


# ---------- registry workflow ----------


def add(video_ids: list[str], reason: str | None, from_proposal: bool) -> None:
    registry = curated.load(fresh=True)
    reasons = {video_id: reason for video_id in video_ids}
    if from_proposal:
        for video in json.loads(PROPOSAL_FILE.read_text())["videos"]:
            reasons[video["video_id"]] = "; ".join(video["reasons"])
    for video_id, why in reasons.items():
        if not why:
            sys.exit(f"{video_id}: --reason is required")
        registry["videos"].setdefault(
            video_id,
            {"reason": why, "qa_status": "selected", "qa_notes": None, "selected_attempt": None, "attempts": []},
        )
    curated.save(registry)
    print(f"registry: {len(registry['videos'])} videos")


def generate(video_ids: list[str]) -> None:
    llm.DEFAULT_RETRIES = 2
    registry = curated.load(fresh=True)
    version = ai_service.pipeline_version()
    for i, video_id in enumerate(video_ids):
        item = registry["videos"].get(video_id)
        if item is None:
            print(f"{video_id}: not in the registry (add it first)")
            continue
        if i:
            time.sleep(PACE_S)
        if item["qa_status"] == "approved":
            print(f"{video_id}: already approved; reject it first to regenerate")
            continue
        same_version = [a for a in item["attempts"] if a["pipeline_version"] == version]
        if len(same_version) >= curated.MAX_ATTEMPTS:
            print(f"{video_id}: {curated.MAX_ATTEMPTS} attempts already at {version}; fix the system instead")
            continue
        try:
            record = ai_service.run_pipeline(video_id, attempt=len(item["attempts"]) + 1)
        except llm.LLMError as error:
            print(f"{video_id}: generation failed ({error}); not retried")
            selected = item["selected_attempt"]
            if selected and item["attempts"][selected - 1]["pipeline_version"] != version:
                item["selected_attempt"], item["qa_status"] = None, "auto_failed"
                curated.save(registry)
            continue
        record["qa"] = {"status": "pending_review" if record["auto_qa"]["passed"] else "auto_failed"}
        item["attempts"].append(record)
        if record["auto_qa"]["passed"] and (
            item["selected_attempt"] is None
            or item["attempts"][item["selected_attempt"] - 1]["pipeline_version"] != version
        ):
            item["selected_attempt"] = record["generation_attempt"]
        if (
            item["selected_attempt"]
            and item["attempts"][item["selected_attempt"] - 1]["pipeline_version"] != version
        ):
            item["selected_attempt"] = None
        if item["selected_attempt"] and item["attempts"][item["selected_attempt"] - 1]["auto_qa"]["passed"]:
            item["qa_status"] = "pending_review"
        else:
            item["qa_status"] = "auto_failed"
        curated.save(registry)
        qa = record["auto_qa"]
        print(
            f"{video_id}: attempt {record['generation_attempt']} "
            f"{'passed' if qa['passed'] else 'FAILED'} auto QA {qa['issues'] or ''} {qa['warnings'] or ''}"
        )
    print(f"usage: {llm.USAGE}")


def show(video_id: str, attempt: int | None) -> None:
    item = curated.load(fresh=True)["videos"][video_id]
    n = attempt or item["selected_attempt"] or len(item["attempts"])
    record = item["attempts"][n - 1]
    a, b = record["analyst"], record["brief"]
    print(f"{video_id} attempt {n}/{len(item['attempts'])} status={item['qa_status']} "
          f"{record['pipeline_version']} {record['model']} {record['generated_at']}")
    print(f"auto_qa: {record['auto_qa']}")
    print(f"relevance: {record['relevance']['retrieval_assessment']} picks={record['relevance']['picks']}")
    print(f"evidence notes: {a['evidence_notes']}")

    def item_line(label, x):
        label += " cross-brand" if x.get("cross_brand") and "cross-brand" not in label else ""
        print(f"  [{label}] {x['title']}\n      {x['finding']}")
        print(f"      refs={x['evidence_refs']} comps={x.get('comparable_ids')}")
        if x.get("worth_testing"):
            print(f"      worth testing: {x['worth_testing']}")

    for x in a["cross_content_patterns"] or []:
        item_line("PATTERN" + (" cross-brand" if x.get("cross_brand") else ""), x)
    if a["shared_characteristic"]:
        item_line("SHARED", a["shared_characteristic"])
    for x in a["single_video_observations"]:
        item_line(f"OBSERVATION/{x['type']}", x)
    if a["audience_signal"]:
        item_line("AUDIENCE", a["audience_signal"])
    withheld = "" if ai_service.servable_brief(record) else " (WITHHELD: never shown)"
    print(f"brief: {b['status']}{withheld} {b.get('validation')}")
    if b["brief"]:
        brief = b["brief"]
        for field in ("concept", "opening_hook", "creative_direction", "engagement_approach", "why_this_direction"):
            print(f"  {field}: {brief[field]}")
        plan = brief["test_plan"]
        print(f"  test variable: {plan['variable_to_test']}")
        source = plan.get("variable_source")
        if source:
            print(f"    source: {source['source_type']} {source['insight_id'] or ''} {source['evidence_refs']}")
            print(f"    why test it: {source['evidence_statement']}")
        print(f"    A: {plan['variant_a']}\n    B: {plan['variant_b']}")
        print(f"    keep: {plan['what_to_keep']}\n    measure: {plan['measure']}")
        print(f"  based on: {brief['based_on_insights']} refs={brief['evidence_refs']}")


def select(video_id: str, attempt: int) -> None:
    registry = curated.load(fresh=True)
    item = registry["videos"][video_id]
    record = item["attempts"][attempt - 1]
    if not record["auto_qa"]["passed"]:
        sys.exit("that attempt failed automated QA and can't be selected")
    item["selected_attempt"] = attempt
    item["qa_status"] = "pending_review"
    curated.save(registry)


def qa(video_id: str, decision: str, note: str, withhold_brief: bool = False) -> None:
    registry = curated.load(fresh=True)
    item = registry["videos"][video_id]
    if decision == "approve":
        if not item["selected_attempt"]:
            sys.exit("no selected attempt")
        if not item["attempts"][item["selected_attempt"] - 1]["auto_qa"]["passed"]:
            sys.exit("the selected attempt failed automated QA")
        if item["attempts"][item["selected_attempt"] - 1]["pipeline_version"] != ai_service.pipeline_version():
            sys.exit("the selected attempt is from an older pipeline version; regenerate first")
    item["qa_status"] = "approved" if decision == "approve" else "rejected"
    item["qa_notes"] = note
    item["qa_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if item["selected_attempt"]:
        item["attempts"][item["selected_attempt"] - 1]["qa"] = {
            "status": item["qa_status"], "note": note, "at": item["qa_at"],
            "brief_withheld": withhold_brief,
        }
    curated.save(registry)
    print(f"{video_id}: {item['qa_status']}")


def status() -> None:
    videos = curated.load(fresh=True)["videos"]
    print(f"{len(videos)} curated videos; pipeline {ai_service.pipeline_version()}")
    for video_id, item in videos.items():
        print(f"  {video_id} {item['qa_status']:<15} attempts={len(item['attempts'])} "
              f"selected={item['selected_attempt']} {item['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("propose")
    p = sub.add_parser("add")
    p.add_argument("ids", nargs="*")
    p.add_argument("--reason")
    p.add_argument("--from-proposal", action="store_true")
    p = sub.add_parser("generate")
    p.add_argument("ids", nargs="+")
    p = sub.add_parser("show")
    p.add_argument("id")
    p.add_argument("--attempt", type=int)
    p = sub.add_parser("select")
    p.add_argument("id")
    p.add_argument("attempt", type=int)
    p = sub.add_parser("qa")
    p.add_argument("id")
    p.add_argument("decision", choices=["approve", "reject"])
    p.add_argument("--note", required=True)
    p.add_argument("--withhold-brief", action="store_true", help="serve the Analyst insights only")
    sub.add_parser("status")
    args = parser.parse_args()
    {
        "propose": propose,
        "add": lambda: add(args.ids, args.reason, args.from_proposal),
        "generate": lambda: generate(args.ids),
        "show": lambda: show(args.id, args.attempt),
        "select": lambda: select(args.id, args.attempt),
        "qa": lambda: qa(args.id, args.decision, args.note, args.withhold_brief),
        "status": status,
    }[args.command]()


if __name__ == "__main__":
    main()
