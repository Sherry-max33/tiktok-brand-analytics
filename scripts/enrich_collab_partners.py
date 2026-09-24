"""Enrich collaboration partner fields via OpenAI (strict json_schema).

Writes:
  collab_partner_type  ∈ athlete|artist|creator|brand|other|none
  collab_partner_name  string|null
  collab_role          ∈ product_collab|campaign|gifted|feature|unclear
  collab_enrich_pool   collaboration|mention_expand
  collab_enrich_method llm

Candidate pools (default: both):
  1) is_collaboration  (content_type contains collaboration)
  2) mention_count >= 2 and not collaboration

Usage (repo root):
  PYTHONPATH=src python -m scripts.enrich_collab_partners --dry-run
  PYTHONPATH=src python -m scripts.enrich_collab_partners
  PYTHONPATH=src python -m scripts.enrich_collab_partners --pool collaboration
  PYTHONPATH=src python -m scripts.enrich_collab_partners --limit 5   # smoke

Requires OPENAI_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover

    def load_dotenv(*_a, **_k) -> bool:
        """Minimal .env loader when python-dotenv is not installed."""
        env_path = Path(".env")
        if not env_path.exists():
            return False
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, _, v = s.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
        return True


def write_partitioned_parquet(
    df: pd.DataFrame,
    base_path: Path,
    partition_cols: Optional[List[str]] = None,
) -> Path:
    """Local copy to avoid importing heavy feature_table deps at module import."""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    partition_cols = partition_cols or ["brand"]
    out = df.copy()
    for c in partition_cols:
        if c not in out.columns:
            out[c] = "__unknown__"
        else:
            out[c] = out[c].fillna("__unknown__").astype(str)
    out.to_parquet(base_path, partition_cols=partition_cols, index=False)
    return base_path

PARTNER_TYPES = ("athlete", "artist", "creator", "brand", "other", "none")
COLLAB_ROLES = ("product_collab", "campaign", "gifted", "feature", "unclear")

OUT_COLS = (
    "collab_partner_type",
    "collab_partner_name",
    "collab_role",
    "collab_enrich_pool",
    "collab_enrich_method",
)

# gpt-4o-mini list prices (USD / 1M tokens) — soft budget only
_INPUT_USD_PER_1M = 0.15
_OUTPUT_USD_PER_1M = 0.60

JSON_SCHEMA: Dict[str, Any] = {
    "name": "collab_partner_enrichment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "partner_type": {"type": "string", "enum": list(PARTNER_TYPES)},
            "partner_name": {"type": ["string", "null"]},
            "collab_role": {"type": "string", "enum": list(COLLAB_ROLES)},
        },
        "required": ["partner_type", "partner_name", "collab_role"],
        "additionalProperties": False,
    },
}


def _as_list(val: Any) -> List[Any]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return val
    try:
        if isinstance(val, np.ndarray):
            return val.tolist()
    except Exception:
        pass
    s = str(val).strip()
    if not s or s in {"[]", "None", "nan"}:
        return []
    return [val]


def _has_collaboration(val: Any) -> bool:
    return "collaboration" in [str(x) for x in _as_list(val)]


def _caption_for_llm(row: pd.Series) -> str:
    for col in ("caption_en", "caption_clean", "caption_raw", "caption_text"):
        text = str(row.get(col) or "").strip()
        if text:
            return text
    return ""


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


def _estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000.0 * _INPUT_USD_PER_1M
        + completion_tokens / 1_000_000.0 * _OUTPUT_USD_PER_1M
    )


def build_messages(
    *,
    brand: str,
    caption: str,
    hashtags: Sequence[str],
    author_username: str,
    mention_count: int,
) -> List[Dict[str, str]]:
    """Flat prompt: enums in text, no fake schema example with placeholders."""
    system = (
        "You label TikTok brand-video collaboration partners from caption text only. "
        "Identify the main non-self collaboration partner if any. "
        "partner_type must be one of: athlete, artist, creator, brand, other, none. "
        "collab_role must be one of: product_collab, campaign, gifted, feature, unclear. "
        "If there is no clear external partner, use partner_type=none and partner_name=null. "
        "Do not treat the focal brand (Nike/Adidas) or the posting account as the partner "
        "unless a different brand is clearly co-named. "
        "partner_name should be a short proper name when known, else null. "
        "Respond only with the structured JSON object."
    )
    user = {
        "brand": brand,
        "author_username": author_username,
        "mention_count": int(mention_count),
        "caption_en": caption,
        "hashtags": list(hashtags),
        "instructions": {
            "partner_type": list(PARTNER_TYPES),
            "collab_role": list(COLLAB_ROLES),
            "partner_name": "string or null",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def normalize_result(
    parsed: Dict[str, Any],
    *,
    brand: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate / coerce model output into stable columns."""
    ptype = str(parsed.get("partner_type") or "").strip().lower()
    if ptype not in PARTNER_TYPES:
        ptype = "other"

    role = str(parsed.get("collab_role") or "").strip().lower()
    if role not in COLLAB_ROLES:
        role = "unclear"

    name_raw = parsed.get("partner_name")
    if name_raw is None or (isinstance(name_raw, float) and pd.isna(name_raw)):
        name: Optional[str] = None
    else:
        name = str(name_raw).strip() or None
        if name and name.lower() in {"null", "none", "n/a", "na"}:
            name = None

    if ptype == "none":
        name = None

    # Focal brand is not an external partner
    brand_l = str(brand or "").strip().lower()
    aliases = {
        "nike": {"nike", "nikesportswear", "jumpman23", "jordan"},
        "adidas": {"adidas", "adidasoriginals", "adidasfootball", "adidastennis"},
    }
    if name and brand_l in aliases and name.lower() in aliases[brand_l]:
        ptype = "none"
        name = None

    return {
        "collab_partner_type": ptype,
        "collab_partner_name": name,
        "collab_role": role,
    }


def openai_chat_json_schema(
    messages: List[Dict[str, str]],
    *,
    api_key: str,
    model: str,
    timeout_sec: float = 60.0,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    body = {
        "model": model,
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": JSON_SCHEMA,
        },
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
        raise RuntimeError(f"OpenAI HTTP {e.code}: {detail[:800]}") from e

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


def select_candidates(
    df: pd.DataFrame,
    *,
    pools: Sequence[str],
) -> pd.DataFrame:
    """Return candidate rows with ``_enrich_pool`` attached."""
    if "content_type" not in df.columns:
        raise KeyError("feature table missing content_type")
    if "mention_count" not in df.columns:
        raise KeyError("feature table missing mention_count")

    is_collab = df["content_type"].map(_has_collaboration)
    mention_n = pd.to_numeric(df["mention_count"], errors="coerce").fillna(0).astype(int)

    frames: List[pd.DataFrame] = []
    if "collaboration" in pools:
        sub = df.loc[is_collab].copy()
        sub["_enrich_pool"] = "collaboration"
        frames.append(sub)
    if "mention_expand" in pools:
        sub = df.loc[(~is_collab) & (mention_n >= 2)].copy()
        sub["_enrich_pool"] = "mention_expand"
        frames.append(sub)

    if not frames:
        return df.head(0).copy()

    out = pd.concat(frames, ignore_index=False)
    # Prefer collaboration label if a row somehow appears in both (shouldn't)
    out = out[~out.index.duplicated(keep="first")]
    out["_caption_llm"] = out.apply(_caption_for_llm, axis=1)
    out = out[out["_caption_llm"].str.len() >= 3].copy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM enrich collab partner fields")
    parser.add_argument(
        "--pool",
        action="append",
        choices=["collaboration", "mention_expand"],
        default=None,
        help="Repeatable. Default: both pools.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows to call (smoke)")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--max-cost-usd", type=float, default=3.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print candidate counts / sample; no API, no writeback",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-label rows that already have collab_partner_type",
    )
    parser.add_argument("--feature-table", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--no-writeback",
        action="store_true",
        help="Write audit JSONL only; do not update feature_table.parquet",
    )
    args = parser.parse_args()

    load_dotenv()
    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    feature_dir = Path(project_cfg["output"].get("feature_dir", "data/processed/feature"))
    video_path = Path(args.feature_table) if args.feature_table else feature_dir / "feature_table.parquet"
    out_dir = Path(args.out_dir) if args.out_dir else feature_dir / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)

    pools = args.pool or ["collaboration", "mention_expand"]
    df = pd.read_parquet(video_path)
    print(f"Loaded {len(df):,} rows from {video_path}")

    cand = select_candidates(df, pools=pools)
    print(
        "Candidates by pool:\n"
        + cand.groupby("_enrich_pool").size().rename("n").to_string()
        if len(cand)
        else "Candidates by pool: (none)"
    )
    print(f"Total candidates: {len(cand)}")

    # Skip already enriched unless --force
    if "collab_partner_type" in df.columns and not args.force:
        already = df["collab_partner_type"].notna() & (
            df["collab_partner_type"].astype(str).str.strip() != ""
        )
        before = len(cand)
        cand = cand.loc[~cand.index.isin(df.index[already])].copy()
        print(f"Skip already enriched: {before - len(cand)} → remaining {len(cand)}")

    if args.limit is not None:
        cand = cand.head(int(args.limit)).copy()
        print(f"Limited to {len(cand)} rows")

    sample_path = out_dir / "collab_partner_candidates.csv"
    tag_col = "hashtags" if "hashtags" in cand.columns else "normalized_hashtags"
    sample_rows = []
    for idx, row in cand.iterrows():
        sample_rows.append(
            {
                "row_index": int(idx) if isinstance(idx, (int, np.integer)) else idx,
                "video_id": row.get("video_id"),
                "brand": row.get("brand"),
                "author_username": row.get("author_username"),
                "mention_count": int(pd.to_numeric(row.get("mention_count"), errors="coerce") or 0),
                "enrich_pool": row.get("_enrich_pool"),
                "caption_llm": row.get("_caption_llm"),
                "hashtags": " ".join(_hashtag_strings(row.get(tag_col))),
            }
        )
    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(sample_path, index=False)
    print(f"Wrote candidates → {sample_path}")

    if args.dry_run:
        print("dry-run: no API calls. Re-run without --dry-run to enrich.")
        return

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY missing. Put it in .env.")

    jsonl_path = out_dir / "collab_partner_enrichment.jsonl"
    prompt_tok = completion_tok = 0
    ok = fail = 0
    est_cost = 0.0
    results: Dict[Any, Dict[str, Any]] = {}

    with jsonl_path.open("w", encoding="utf-8") as fout:
        for i, (_, row) in enumerate(sample_df.iterrows(), start=1):
            if est_cost >= float(args.max_cost_usd):
                print(
                    f"Stopped at {i}: estimated cost ${est_cost:.4f} "
                    f">= max ${args.max_cost_usd:.2f}"
                )
                break

            brand = str(row.get("brand") or "")
            caption = str(row.get("caption_llm") or "")
            tags = [t for t in str(row.get("hashtags") or "").split() if t]
            messages = build_messages(
                brand=brand,
                caption=caption,
                hashtags=tags,
                author_username=str(row.get("author_username") or ""),
                mention_count=int(row.get("mention_count") or 0),
            )

            try:
                parsed, tokens = openai_chat_json_schema(
                    messages, api_key=api_key, model=str(args.model)
                )
                norm = normalize_result(parsed, brand=brand)
            except Exception as exc:  # noqa: BLE001
                fail += 1
                rec = {
                    "video_id": row.get("video_id"),
                    "brand": brand,
                    "enrich_pool": row.get("enrich_pool"),
                    "error": str(exc),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  fail {i}/{len(sample_df)}: {exc}")
                if args.sleep > 0:
                    time.sleep(args.sleep)
                continue

            prompt_tok += tokens["prompt_tokens"]
            completion_tok += tokens["completion_tokens"]
            est_cost = _estimate_cost_usd(prompt_tok, completion_tok)
            ok += 1

            rec = {
                "video_id": row.get("video_id"),
                "brand": brand,
                "enrich_pool": row.get("enrich_pool"),
                "caption_llm": caption[:240],
                **norm,
                "raw": parsed,
                "usage": tokens,
                "est_cost_usd_cum": round(est_cost, 5),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results[row.get("row_index")] = {
                **norm,
                "collab_enrich_pool": row.get("enrich_pool"),
                "collab_enrich_method": "llm",
            }

            if i % 25 == 0 or i == len(sample_df):
                print(
                    f"  … {i}/{len(sample_df)} ok={ok} fail={fail} "
                    f"est_cost=${est_cost:.4f}",
                    flush=True,
                )
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Wrote audit → {jsonl_path}")
    print(
        f"Done calls: ok={ok} fail={fail} tokens_in={prompt_tok} "
        f"tokens_out={completion_tok} est_cost=${est_cost:.4f}"
    )

    if args.no_writeback:
        print("--no-writeback: feature table unchanged")
        return

    if not results:
        print("No successful results to write back")
        return

    out = df.copy()
    for col in OUT_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    written = 0
    for idx, vals in results.items():
        if idx not in out.index:
            continue
        for col, val in vals.items():
            out.at[idx, col] = val
        written += 1

    if written == 0:
        print("Warning: no rows matched by index; feature table unchanged")
        return

    out.to_parquet(video_path, index=False)
    write_partitioned_parquet(out, feature_dir / "feature_table", partition_cols=["brand"])
    labeled = out["collab_partner_type"].notna().sum()
    print(
        f"Writeback complete → {video_path} "
        f"(updated={written}, rows with partner_type={int(labeled):,})"
    )
    print(out["collab_partner_type"].value_counts(dropna=False).head(12).to_string())


if __name__ == "__main__":
    main()
