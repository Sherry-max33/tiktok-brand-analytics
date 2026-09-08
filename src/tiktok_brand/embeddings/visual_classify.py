"""CLIP zero-shot visual_format / visual_setting classification.

Replaces appearance_type (person/product detector ratios).

Aggregation (per axis):
  for each class: mean over 3 prompt similarities within a frame
  then mean those frame-class scores over valid frames
  top-1 / top-2 / margin from video-level class scores

Embeddings must be L2-normalized so cosine == dot product
(``normalize_embeddings=True`` in CLIP encode).
No valid frames → labels ``unknown``; prompts are not used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..etl.rule_config import load_feature_rules

STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_UNKNOWN = "unknown"
LABEL_UNKNOWN = "unknown"

FORMAT_ORDER: Tuple[str, ...] = (
    "product_closeup",
    "on_body_styling",
    "sports_action",
    "talking_head",
    "campaign_visual",
    "archival_retro",
    "other",
)
SETTING_ORDER: Tuple[str, ...] = (
    "studio",
    "outdoor",
    "gym_training",
    "sports_venue",
    "event_crowd",
    "home_indoor",
    "retail_store",
    "other",
)

_PROMPT_CACHE: Dict[str, Dict[str, List[List[float]]]] = {}


def _zero_shot_cfg(rules_path: str) -> dict:
    return load_feature_rules(rules_path).get("visual_zero_shot") or {}


def zero_shot_enabled(rules_path: str = "configs/feature_rules.yaml") -> bool:
    return bool(_zero_shot_cfg(rules_path).get("enabled", True))


def format_labels(rules_path: str = "configs/feature_rules.yaml") -> List[str]:
    cfg = _zero_shot_cfg(rules_path).get("visual_format") or {}
    return [k for k in FORMAT_ORDER if k in cfg] or list(FORMAT_ORDER)


def setting_labels(rules_path: str = "configs/feature_rules.yaml") -> List[str]:
    cfg = _zero_shot_cfg(rules_path).get("visual_setting") or {}
    return [k for k in SETTING_ORDER if k in cfg] or list(SETTING_ORDER)


def class_score_column(axis: str, label: str) -> str:
    prefix = "vf" if axis == "visual_format" else "vs"
    return f"{prefix}_{label}_score"


def empty_classification(
    *,
    expected_frames: int = 4,
    valid_frames: int = 0,
    rules_path: str = "configs/feature_rules.yaml",
) -> Dict[str, Any]:
    """Unknown / empty payload (no prompt scoring)."""
    if valid_frames <= 0:
        status = STATUS_UNKNOWN
    elif valid_frames < int(expected_frames):
        status = STATUS_PARTIAL
    else:
        status = STATUS_OK

    out: Dict[str, Any] = {
        "visual_format": LABEL_UNKNOWN if valid_frames <= 0 else None,
        "visual_setting": LABEL_UNKNOWN if valid_frames <= 0 else None,
        "visual_format_score": None,
        "visual_setting_score": None,
        "visual_format_margin": None,
        "visual_setting_margin": None,
        "visual_format_second": None,
        "visual_setting_second": None,
        "visual_format_second_score": None,
        "visual_setting_second_score": None,
        "visual_valid_frames": int(valid_frames),
        "visual_classification_status": status,
    }
    for label in format_labels(rules_path):
        out[class_score_column("visual_format", label)] = None
    for label in setting_labels(rules_path):
        out[class_score_column("visual_setting", label)] = None
    if valid_frames <= 0:
        out["visual_format"] = LABEL_UNKNOWN
        out["visual_setting"] = LABEL_UNKNOWN
    return out


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _rank_scores(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def _axis_result(scores: Dict[str, float], *, prefix: str) -> Dict[str, Any]:
    ranked = _rank_scores(scores)
    if not ranked:
        return {
            prefix: LABEL_UNKNOWN,
            f"{prefix}_score": None,
            f"{prefix}_margin": None,
            f"{prefix}_second": None,
            f"{prefix}_second_score": None,
        }
    top_label, top_score = ranked[0]
    if len(ranked) > 1:
        second_label, second_score = ranked[1]
        margin = float(top_score - second_score)
    else:
        second_label, second_score, margin = None, None, None
    out = {
        prefix: top_label,
        f"{prefix}_score": float(top_score),
        f"{prefix}_margin": margin,
        f"{prefix}_second": second_label,
        f"{prefix}_second_score": float(second_score) if second_score is not None else None,
    }
    return out


def _load_prompt_vectors(
    *,
    rules_path: str,
    model_name: str,
    batch_size: int,
    encode_texts_fn,
) -> Dict[str, Dict[str, List[List[float]]]]:
    """axis → label → list[prompt_vector]."""
    cache_key = f"{rules_path}|{model_name}"
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]

    cfg = _zero_shot_cfg(rules_path)
    axes = {
        "visual_format": cfg.get("visual_format") or {},
        "visual_setting": cfg.get("visual_setting") or {},
    }
    flat: List[str] = []
    owners: List[Tuple[str, str]] = []  # (axis, label)
    for axis, mapping in axes.items():
        order = FORMAT_ORDER if axis == "visual_format" else SETTING_ORDER
        for label in order:
            prompts = mapping.get(label) or []
            for p in prompts:
                text = str(p).strip()
                if not text:
                    continue
                flat.append(text)
                owners.append((axis, label))

    encoded = encode_texts_fn(flat, model_name=model_name, batch_size=batch_size) if flat else []
    out: Dict[str, Dict[str, List[List[float]]]] = {
        "visual_format": {k: [] for k in format_labels(rules_path)},
        "visual_setting": {k: [] for k in setting_labels(rules_path)},
    }
    for (axis, label), vec in zip(owners, encoded):
        out.setdefault(axis, {}).setdefault(label, []).append(
            list(vec) if not isinstance(vec, list) else vec
        )
    _PROMPT_CACHE[cache_key] = out
    return out


def classify_frame_vectors(
    frame_vectors: Sequence[Sequence[float]],
    *,
    rules_path: str = "configs/feature_rules.yaml",
    expected_frames: int = 4,
    model_name: str = "sentence-transformers/clip-ViT-B-32",
    batch_size: int = 16,
    encode_texts_fn=None,
) -> Dict[str, Any]:
    """
    Classify one video from per-frame L2-normalized CLIP image vectors.

    ``frame_vectors`` should only include successfully encoded frames.
    """
    valid = [list(v) for v in frame_vectors if v]
    n_valid = len(valid)
    base = empty_classification(
        expected_frames=expected_frames, valid_frames=n_valid, rules_path=rules_path
    )
    if n_valid <= 0:
        return base
    if not zero_shot_enabled(rules_path):
        base["visual_format"] = LABEL_UNKNOWN
        base["visual_setting"] = LABEL_UNKNOWN
        base["visual_classification_status"] = STATUS_UNKNOWN
        return base
    if encode_texts_fn is None:
        raise ValueError("encode_texts_fn is required for zero-shot classification")

    prompt_vecs = _load_prompt_vectors(
        rules_path=rules_path,
        model_name=model_name,
        batch_size=batch_size,
        encode_texts_fn=encode_texts_fn,
    )

    def _video_class_scores(axis: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for label, pvecs in (prompt_vecs.get(axis) or {}).items():
            if not pvecs:
                continue
            frame_scores: List[float] = []
            for fv in valid:
                sims = [_dot(fv, pv) for pv in pvecs]
                frame_scores.append(sum(sims) / len(sims))
            scores[label] = sum(frame_scores) / len(frame_scores)
        return scores

    fmt_scores = _video_class_scores("visual_format")
    set_scores = _video_class_scores("visual_setting")

    out = dict(base)
    out.update(_axis_result(fmt_scores, prefix="visual_format"))
    out.update(_axis_result(set_scores, prefix="visual_setting"))
    for label, score in fmt_scores.items():
        out[class_score_column("visual_format", label)] = float(score)
    for label, score in set_scores.items():
        out[class_score_column("visual_setting", label)] = float(score)
    return out


def classification_column_names(rules_path: str = "configs/feature_rules.yaml") -> List[str]:
    """Flat list of zero-shot output columns (labels, margins, per-class scores)."""
    cols = [
        "visual_format",
        "visual_setting",
        "visual_format_score",
        "visual_setting_score",
        "visual_format_margin",
        "visual_setting_margin",
        "visual_format_second",
        "visual_setting_second",
        "visual_format_second_score",
        "visual_setting_second_score",
        "visual_valid_frames",
        "visual_classification_status",
    ]
    for label in format_labels(rules_path):
        cols.append(class_score_column("visual_format", label))
    for label in setting_labels(rules_path):
        cols.append(class_score_column("visual_setting", label))
    return cols


def clear_prompt_cache() -> None:
    _PROMPT_CACHE.clear()
