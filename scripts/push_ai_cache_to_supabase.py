"""Copy live AI generations from the local cache (data/processed/ai/) to Supabase, so the
deployed app serves them instead of generating them again. Never calls an LLM.

Uploads only what serving needs for each live-generated video: its analysis, its Brief (if
generated) and the relevance judgment under the current cache key. Entries already in
Supabase are left as they are. Needs SUPABASE_URL and SUPABASE_SECRET_KEY (environment or
.env). Dry run unless --apply is given.

    python scripts/push_ai_cache_to_supabase.py [--apply]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import ai_cache  # noqa: E402
import ai_service  # noqa: E402
import relevance  # noqa: E402
import supabase_store  # noqa: E402


def _local(namespace: str) -> dict:
    path = ai_cache.LOCAL_DIR / f"{namespace}_cache.json"
    return json.loads(path.read_text()) if path.exists() else {}


def entries() -> list[tuple[str, str, object]]:
    analyses = _local(ai_service.ANALYSIS_NAMESPACE)
    briefs = _local(ai_service.BRIEF_NAMESPACE)
    judgments = _local(relevance.CACHE_NAMESPACE)
    out = []
    for key, analysis in analyses.items():
        if key != ai_service._analysis_key(analysis["video_id"]):
            continue  # an older pipeline version the app no longer looks up
        video_id = analysis["video_id"]
        out.append((ai_service.ANALYSIS_NAMESPACE, key, analysis))
        brief_key = ai_service._brief_key(video_id)
        if brief_key in briefs:
            out.append((ai_service.BRIEF_NAMESPACE, brief_key, briefs[brief_key]))
        rel_key = relevance.cache_key(video_id)
        if rel_key in judgments:
            out.append((relevance.CACHE_NAMESPACE, rel_key, judgments[rel_key]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="upload (default: dry run)")
    args = parser.parse_args()
    if not supabase_store.configured():
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SECRET_KEY first.")

    for namespace, key, value in entries():
        if supabase_store.get(namespace, key) is not None:
            print(f"exists    {namespace}  {key}")
        elif args.apply:
            supabase_store.put(namespace, key, value)
            print(f"uploaded  {namespace}  {key}")
        else:
            print(f"would add {namespace}  {key}")
    if not args.apply:
        print("Dry run; rerun with --apply to upload.")


if __name__ == "__main__":
    main()
