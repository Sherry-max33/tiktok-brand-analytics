"""Suggest content_type / social_mechanic rule phrases via OpenAI (suggest-only).

Does NOT write configs/feature_rules.yaml. Review audit outputs, then edit YAML yourself.

Pool: content_type == [] (rule misses). Input per row: caption_en (fallback caption_clean)
+ hashtags. One API call returns both label axes.

Usage (repo root):
  # 1) Preview stratified sample only (no API spend)
  PYTHONPATH=src python -m scripts.llm_suggest_rules --sample-only

  # 2) Call gpt-4o-mini (needs OPENAI_API_KEY in .env)
  PYTHONPATH=src python -m scripts.llm_suggest_rules --n-samples 400

Outputs under data/processed/feature/audits/:
  llm_rule_sample.csv
  llm_rule_suggestions.jsonl
  llm_rule_phrase_candidates.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv

from tiktok_brand.etl.content_type_rules import CLASSIFICATION_ORDER
from tiktok_brand.etl.social_mechanic_rules import SOCIAL_MECHANIC_ORDER

# gpt-4o-mini list prices (USD / 1M tokens) — used only for soft budget guardrails
_INPUT_USD_PER_1M = 0.15
_OUTPUT_USD_PER_1M = 0.60

CONTENT_TYPE_LABELS = list(CLASSIFICATION_ORDER)
SOCIAL_MECHANIC_LABELS = list(SOCIAL_MECHANIC_ORDER)


def _as_list(val: Any) -> List[Any]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return val
    try:
        import numpy as np

        if isinstance(val, np.ndarray):
            return val.tolist()
    except ImportError:
        pass
    s = str(val).strip()
    if not s or s in {"[]", "None", "nan"}:
        return []
    return [val]


def _is_empty_labels(val: Any) -> bool:
    return len(_as_list(val)) == 0


def _hashtag_strings(val: Any) -> List[str]:
    out: List[str] = []
    for t in _as_list(val):
        raw = str(t).strip()
        if not raw:
            continue
        if not raw.startswith("#"):
            raw = f"#{raw}"
        out.append(raw)
    return out


def _caption_for_llm(row: pd.Series) -> str:
    en = str(row.get("caption_en") or "").strip()
    if en:
        return en
    clean = str(row.get("caption_clean") or "").strip()
    if clean:
        return clean
    return str(row.get("caption_raw") or row.get("caption_text") or "").strip()


def _stratified_sample(
    pool: pd.DataFrame,
    n_samples: int,
    *,
    brand_col: str,
    cluster_col: str,
    seed: int,
) -> pd.DataFrame:
    """Proportional sample by brand × cluster; leftover filled randomly."""
    if len(pool) <= n_samples:
        return pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    work = pool.copy()
    if cluster_col not in work.columns:
        work["_stratum"] = work[brand_col].astype(str)
    else:
        cl = work[cluster_col].fillna(-1).astype(str)
        work["_stratum"] = work[brand_col].astype(str) + "|" + cl

    sizes = work.groupby("_stratum").size()
    target = (sizes / sizes.sum() * n_samples).round().astype(int).clip(lower=0)
    # Fix rounding so sum == n_samples
    while int(target.sum()) > n_samples:
        target.loc[target.idxmax()] -= 1
    while int(target.sum()) < n_samples:
        # prefer largest remaining strata capacity
        remain = sizes - target
        remain = remain[remain > 0]
        if remain.empty:
            break
        target.loc[remain.idxmax()] += 1

    parts: List[pd.DataFrame] = []
    for stratum, k in target.items():
        if k <= 0:
            continue
        g = work[work["_stratum"] == stratum]
        parts.append(g.sample(n=min(int(k), len(g)), random_state=seed))

    out = pd.concat(parts, ignore_index=False) if parts else work.head(0)
    if len(out) < n_samples:
        need = n_samples - len(out)
        leftover = work.loc[~work.index.isin(out.index)]
        if len(leftover) > 0:
            out = pd.concat(
                [out, leftover.sample(n=min(need, len(leftover)), random_state=seed + 1)],
                ignore_index=False,
            )
    return out.drop(columns=["_stratum"], errors="ignore").reset_index(drop=True)


def _build_messages(brand: str, caption: str, hashtags: Sequence[str]) -> List[Dict[str, str]]:
    system = (
        "You suggest keyword/phrase rules for TikTok brand-video labeling. "
        "Only use labels from the allowed lists. Prefer short evidence phrases that "
        "literally appear in the caption or hashtags (good candidates for YAML strong/weak lists). "
        "If nothing fits confidently, return empty lists. "
        "Respond with a single JSON object only."
    )
    user = {
        "brand": brand,
        "caption_en": caption,
        "hashtags": list(hashtags),
        "allowed_content_type": CONTENT_TYPE_LABELS,
        "allowed_social_mechanic": SOCIAL_MECHANIC_LABELS,
        "output_schema": {
            "content_type": ["label", "..."],
            "social_mechanic": ["label", "..."],
            "content_type_evidence": [{"label": "...", "phrases": ["..."]}],
            "social_mechanic_evidence": [{"label": "...", "phrases": ["..."]}],
            "notes": "optional short note",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000.0 * _INPUT_USD_PER_1M
        + completion_tokens / 1_000_000.0 * _OUTPUT_USD_PER_1M
    )


def _openai_chat_json(
    messages: List[Dict[str, str]],
    *,
    api_key: str,
    model: str,
    timeout_sec: float = 60.0,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    body = {
        "model": model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP {e.code}: {detail[:500]}") from e

    usage = payload.get("usage") or {}
    tokens = {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("Model returned non-object JSON")
    return parsed, tokens


def _filter_labels(labels: Any, allowed: Sequence[str]) -> List[str]:
    allow = set(allowed)
    out: List[str] = []
    for x in _as_list(labels):
        s = str(x).strip()
        if s in allow and s not in out:
            out.append(s)
    return out


def _collect_phrases(
    evidence: Any,
    allowed: Sequence[str],
    bucket: Dict[str, Counter],
) -> None:
    allow = set(allowed)
    for item in _as_list(evidence):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if label not in allow:
            continue
        for p in _as_list(item.get("phrases")):
            phrase = str(p).strip().lower()
            if len(phrase) < 2:
                continue
            bucket[label][phrase] += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM suggest-only for content_type / social_mechanic rule phrases"
    )
    parser.add_argument("--n-samples", type=int, default=400, help="Stratified sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=2.0,
        help="Stop before starting another call if estimated spend would exceed this",
    )
    parser.add_argument("--sleep", type=float, default=0.05, help="Pause between API calls")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Write stratified sample CSV only (no OpenAI calls)",
    )
    parser.add_argument(
        "--feature-table",
        default=None,
        help="Override path to feature_table.parquet",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override audits output directory",
    )
    args = parser.parse_args()

    load_dotenv()
    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    feature_dir = Path(project_cfg["output"].get("feature_dir", "data/processed/feature"))
    video_path = Path(args.feature_table) if args.feature_table else feature_dir / "feature_table.parquet"
    out_dir = Path(args.out_dir) if args.out_dir else feature_dir / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(video_path)
    if "content_type" not in df.columns:
        raise SystemExit("feature table missing content_type")
    if "brand" not in df.columns:
        raise SystemExit("feature table missing brand")

    pool = df[df["content_type"].map(_is_empty_labels)].copy()
    pool["_caption_llm"] = pool.apply(_caption_for_llm, axis=1)
    pool = pool[pool["_caption_llm"].str.len() >= 8].copy()
    print(
        f"Pool content_type==[] with usable caption: {len(pool)} / {len(df)} "
        f"(from table {video_path})"
    )

    cluster_col = (
        "content_cluster_id_clean_k12"
        if "content_cluster_id_clean_k12" in pool.columns
        else "content_cluster_id_clean"
        if "content_cluster_id_clean" in pool.columns
        else ""
    )
    sample = _stratified_sample(
        pool,
        int(args.n_samples),
        brand_col="brand",
        cluster_col=cluster_col,
        seed=int(args.seed),
    )
    print(
        f"Sampled {len(sample)} rows "
        f"(strata=brand×{cluster_col or 'none'}, seed={args.seed})"
    )

    tag_col = "hashtags" if "hashtags" in sample.columns else "normalized_hashtags"
    sample_rows = []
    for _, row in sample.iterrows():
        tags = _hashtag_strings(row.get(tag_col))
        sample_rows.append(
            {
                "video_id": row.get("video_id"),
                "brand": row.get("brand"),
                "cluster_id": row.get(cluster_col) if cluster_col else None,
                "caption_llm": row.get("_caption_llm"),
                "hashtags": " ".join(tags),
                "social_mechanic_existing": json.dumps(
                    _as_list(row.get("social_mechanic")), ensure_ascii=False
                ),
            }
        )
    sample_df = pd.DataFrame(sample_rows)
    sample_path = out_dir / "llm_rule_sample.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"Wrote sample → {sample_path}")

    if args.sample_only:
        print("sample-only: skipping OpenAI. Review sample, then re-run without --sample-only.")
        return

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY missing. Put it in .env, or use --sample-only first.")

    jsonl_path = out_dir / "llm_rule_suggestions.jsonl"
    phrase_ct: Dict[str, Counter] = defaultdict(Counter)
    phrase_sm: Dict[str, Counter] = defaultdict(Counter)
    prompt_tok = completion_tok = 0
    ok = fail = 0
    est_cost = 0.0

    with jsonl_path.open("w", encoding="utf-8") as fout:
        for i, row in sample_df.iterrows():
            if est_cost >= float(args.max_cost_usd):
                print(
                    f"Stopped at row {i}: estimated cost ${est_cost:.4f} "
                    f">= max ${args.max_cost_usd:.2f}"
                )
                break

            brand = str(row.get("brand") or "")
            caption = str(row.get("caption_llm") or "")
            tags = [t for t in str(row.get("hashtags") or "").split() if t]
            messages = _build_messages(brand, caption, tags)

            try:
                parsed, tokens = _openai_chat_json(
                    messages, api_key=api_key, model=str(args.model)
                )
            except Exception as exc:  # noqa: BLE001 — keep batch going
                fail += 1
                rec = {
                    "video_id": row.get("video_id"),
                    "brand": brand,
                    "error": str(exc),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  fail {i + 1}/{len(sample_df)}: {exc}")
                continue

            prompt_tok += tokens["prompt_tokens"]
            completion_tok += tokens["completion_tokens"]
            est_cost = _estimate_cost_usd(prompt_tok, completion_tok)

            ct = _filter_labels(parsed.get("content_type"), CONTENT_TYPE_LABELS)
            sm = _filter_labels(parsed.get("social_mechanic"), SOCIAL_MECHANIC_LABELS)
            _collect_phrases(parsed.get("content_type_evidence"), CONTENT_TYPE_LABELS, phrase_ct)
            _collect_phrases(
                parsed.get("social_mechanic_evidence"), SOCIAL_MECHANIC_LABELS, phrase_sm
            )

            rec = {
                "video_id": row.get("video_id"),
                "brand": brand,
                "cluster_id": row.get("cluster_id"),
                "caption_llm": caption,
                "hashtags": tags,
                "content_type_suggested": ct,
                "social_mechanic_suggested": sm,
                "content_type_evidence": parsed.get("content_type_evidence"),
                "social_mechanic_evidence": parsed.get("social_mechanic_evidence"),
                "notes": parsed.get("notes"),
                "usage": tokens,
                "est_cost_usd_cum": round(est_cost, 6),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1
            if (ok + fail) % 25 == 0 or (ok + fail) == len(sample_df):
                print(
                    f"  … {ok + fail}/{len(sample_df)} ok={ok} fail={fail} "
                    f"est_cost=${est_cost:.4f}",
                    flush=True,
                )
            if args.sleep > 0:
                time.sleep(float(args.sleep))

    phrase_rows: List[Dict[str, Any]] = []
    for axis, bucket in (("content_type", phrase_ct), ("social_mechanic", phrase_sm)):
        for label, ctr in bucket.items():
            for phrase, count in ctr.most_common():
                phrase_rows.append(
                    {
                        "axis": axis,
                        "label": label,
                        "phrase": phrase,
                        "suggest_count": count,
                    }
                )
    phrase_df = pd.DataFrame(phrase_rows)
    if not phrase_df.empty:
        phrase_df = phrase_df.sort_values(
            ["axis", "label", "suggest_count"], ascending=[True, True, False]
        )
    phrase_path = out_dir / "llm_rule_phrase_candidates.csv"
    phrase_df.to_csv(phrase_path, index=False)

    print(f"Wrote suggestions → {jsonl_path}")
    print(f"Wrote phrase candidates → {phrase_path}")
    print(
        f"Done: ok={ok} fail={fail} tokens_in={prompt_tok} tokens_out={completion_tok} "
        f"est_cost=${est_cost:.4f} (list-price estimate; not a bill)"
    )
    print("Next: review phrase CSV → hand-edit feature_rules.yaml → refresh_content_labels.")


if __name__ == "__main__":
    main()
