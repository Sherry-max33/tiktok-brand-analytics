"""Visual embeddings (CLIP) + zero-shot visual_format / visual_setting.

Frame extraction / cover download happens once via ``frames.resolve_video_frames``.
CLIP image encode, caption alignment, and zero-shot axes all reuse those paths.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from ..etl.rule_config import load_feature_rules
from .frames import DEFAULT_FRAME_FRACTIONS, resolve_frames_batch
from .visual_classify import classify_frame_vectors, empty_classification

logger = logging.getLogger(__name__)

METHOD_CLIP = "clip_visual"
METHOD_NOT_EMBEDDED = "not_embedded"

# Deprecated appearance_type constants (kept for old imports).
APPEARANCE_UNKNOWN = "unknown"
APPEARANCE_PERSON_PRESENT = "person_present"
APPEARANCE_PRODUCT_ONLY = "product_only"
APPEARANCE_MIXED = "mixed"
APPEARANCE_OTHER = "other"

EncodeImagesFn = Callable[[Sequence[object]], List[List[float]]]

_MODEL = None
_MODEL_NAME: Optional[str] = None


def _visual_cfg(rules_path: str = "configs/feature_rules.yaml") -> dict:
    return load_feature_rules(rules_path).get("visual_embedding") or {}


def _enabled(rules_path: str = "configs/feature_rules.yaml") -> bool:
    return bool(_visual_cfg(rules_path).get("enabled", True))


def _model_name(rules_path: str = "configs/feature_rules.yaml") -> str:
    return str(
        _visual_cfg(rules_path).get(
            "model_name",
            "sentence-transformers/clip-ViT-B-32",
        )
    )


def _batch_size(rules_path: str = "configs/feature_rules.yaml") -> int:
    return int(_visual_cfg(rules_path).get("batch_size", 16))


def _max_frames(rules_path: str = "configs/feature_rules.yaml") -> int:
    return int(_visual_cfg(rules_path).get("max_frames", 4))


def _frame_fractions(rules_path: str = "configs/feature_rules.yaml") -> list[float]:
    raw = _visual_cfg(rules_path).get("frame_fractions")
    if isinstance(raw, list) and raw:
        return [float(x) for x in raw]
    n = _max_frames(rules_path)
    if n == 4:
        return list(DEFAULT_FRAME_FRACTIONS)
    return [round((i + 1) / (n + 1), 4) for i in range(n)]


def _frames_root(rules_path: str = "configs/feature_rules.yaml") -> str:
    return str(_visual_cfg(rules_path).get("frames_dir", "data/processed/frames"))


def _download_max_height(rules_path: str = "configs/feature_rules.yaml") -> int:
    return int(_visual_cfg(rules_path).get("download_max_height", 480))


def _download_video_enabled(rules_path: str = "configs/feature_rules.yaml") -> bool:
    import os

    env = os.environ.get("VISUAL_DOWNLOAD", "").strip()
    if env == "1":
        return True
    if env == "0":
        return False
    return bool(_visual_cfg(rules_path).get("download_video", False))


def _cookies_from_browser(rules_path: str = "configs/feature_rules.yaml") -> str | None:
    import os

    env = os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "").strip()
    if env:
        return env
    raw = _visual_cfg(rules_path).get("cookies_from_browser")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _load_clip(model_name: str):
    global _MODEL, _MODEL_NAME
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL
    from sentence_transformers import SentenceTransformer

    logger.info("Loading CLIP visual model: %s", model_name)
    _MODEL = SentenceTransformer(model_name)
    _MODEL_NAME = model_name
    return _MODEL


def _open_image(source: Union[str, Path]):
    from PIL import Image

    src = str(source)
    if src.startswith("http://") or src.startswith("https://"):
        import urllib.request

        with urllib.request.urlopen(src, timeout=20) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    path = Path(src)
    if not path.is_file():
        raise FileNotFoundError(src)
    return Image.open(path).convert("RGB")


def _default_encode_images(
    images: Sequence[object],
    *,
    model_name: str,
    batch_size: int,
) -> List[List[float]]:
    model = _load_clip(model_name)
    vectors = model.encode(
        list(images),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vectors]


def _mean_vector(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += float(x)
    n = float(len(vectors))
    mean = [x / n for x in acc]
    norm = sum(x * x for x in mean) ** 0.5
    if norm <= 0:
        return mean
    return [x / norm for x in mean]


def infer_appearance_type(
    *,
    has_image: bool,
    person_ratio: Optional[float] = None,
    product_ratio: Optional[float] = None,
    rules_path: str = "configs/feature_rules.yaml",
) -> str:
    """Deprecated stub. Prefer visual_format / visual_setting zero-shot."""
    if not has_image:
        return APPEARANCE_UNKNOWN
    if person_ratio is None or product_ratio is None:
        return APPEARANCE_OTHER
    cfg = load_feature_rules(rules_path).get("appearance_type") or {}
    person_thr = float(cfg.get("person_ratio", 0.6))
    product_thr = float(cfg.get("product_only_ratio", 0.6))
    mixed_min = float(cfg.get("mixed_min_ratio", 0.2))
    person_hi = person_ratio >= person_thr
    product_hi = product_ratio >= product_thr
    if person_hi and product_hi:
        return APPEARANCE_MIXED
    if person_hi:
        return APPEARANCE_PERSON_PRESENT
    if product_hi:
        return APPEARANCE_PRODUCT_ONLY
    if person_ratio >= mixed_min and product_ratio >= mixed_min:
        return APPEARANCE_MIXED
    return APPEARANCE_OTHER


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("nan")
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _is_missing(x: object) -> bool:
    if x is None:
        return True
    try:
        import pandas as pd

        return bool(pd.isna(x))
    except Exception:
        return False


def _default_encode_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    batch_size: int,
) -> List[List[float]]:
    model = _load_clip(model_name)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vectors]


def _alignment_stats(
    text_vec: List[float], frame_vecs: List[List[float]]
) -> dict:
    sims = [_cosine(text_vec, fv) for fv in frame_vecs]
    sims = [s for s in sims if s == s]
    if not sims:
        return {
            "clip_alignment_mean": None,
            "clip_alignment_max": None,
            "clip_alignment_min": None,
            "clip_alignment_std": None,
            "clip_text_embedding": text_vec,
        }
    mean = sum(sims) / len(sims)
    var = sum((s - mean) ** 2 for s in sims) / len(sims)
    return {
        "clip_alignment_mean": float(mean),
        "clip_alignment_max": float(max(sims)),
        "clip_alignment_min": float(min(sims)),
        "clip_alignment_std": float(var**0.5),
        "clip_text_embedding": text_vec,
    }


def _blank_visual_row(expected_frames: int, rules_path: str) -> dict:
    empty_align = {
        "clip_alignment_mean": None,
        "clip_alignment_max": None,
        "clip_alignment_min": None,
        "clip_alignment_std": None,
        "clip_text_embedding": None,
    }
    return {
        "visual_embedding": None,
        "visual_embedding_method": METHOD_NOT_EMBEDDED,
        "visual_embedding_model": None,
        "frame_paths": [],
        **empty_align,
        **empty_classification(
            expected_frames=expected_frames, valid_frames=0, rules_path=rules_path
        ),
    }


def compute_visual_features(
    video_ids: Sequence[Optional[str]],
    *,
    video_urls: Optional[Sequence[Optional[str]]] = None,
    page_urls: Optional[Sequence[Optional[str]]] = None,
    author_usernames: Optional[Sequence[Optional[str]]] = None,
    cover_urls: Optional[Sequence[Optional[str]]] = None,
    cover_paths: Optional[Sequence[Optional[str]]] = None,
    video_paths: Optional[Sequence[Optional[str]]] = None,
    frame_paths_per_video: Optional[Sequence[Optional[Sequence[str]]]] = None,
    captions_en: Optional[Sequence[Optional[str]]] = None,
    person_ratios: Optional[Sequence[Optional[float]]] = None,
    product_ratios: Optional[Sequence[Optional[float]]] = None,
    rules_path: str = "configs/feature_rules.yaml",
    encode_images_fn: Optional[EncodeImagesFn] = None,
    encode_texts_fn: Optional[Callable] = None,
) -> List[dict]:
    """
    Frames → CLIP visual_embedding (+ optional caption alignment)
    + zero-shot visual_format / visual_setting scores.

    ``person_ratios`` / ``product_ratios`` are ignored (legacy appearance_type removed).
    """
    _ = person_ratios, product_ratios  # legacy unused
    model_name = _model_name(rules_path)
    fractions = _frame_fractions(rules_path)
    expected_frames = len(fractions)
    n = len(video_ids)
    out: List[dict] = [_blank_visual_row(expected_frames, rules_path) for _ in range(n)]

    if not _enabled(rules_path):
        return out

    if frame_paths_per_video is None:
        frame_paths_per_video = resolve_frames_batch(
            video_ids,
            video_urls=video_urls,
            page_urls=page_urls,
            author_usernames=author_usernames,
            cover_urls=cover_urls,
            cover_paths=cover_paths,
            video_paths=video_paths,
            frames_root=_frames_root(rules_path),
            fractions=fractions,
            download_video=_download_video_enabled(rules_path),
            max_height=_download_max_height(rules_path),
            cookies_from_browser=_cookies_from_browser(rules_path),
        )

    flat_images: List[object] = []
    owners: List[int] = []
    local_paths: List[List[str]] = [list(x or []) for x in frame_paths_per_video]

    for i, paths in enumerate(local_paths):
        out[i]["frame_paths"] = paths
        for p in paths:
            try:
                flat_images.append(_open_image(p))
                owners.append(i)
            except Exception as exc:
                logger.debug("skip frame %s: %s", p, exc)

    vectors_by_row: dict[int, List[List[float]]] = {i: [] for i in range(n)}
    if flat_images:
        try:
            if encode_images_fn is not None:
                vectors = encode_images_fn(flat_images)
            else:
                vectors = _default_encode_images(
                    flat_images,
                    model_name=model_name,
                    batch_size=_batch_size(rules_path),
                )
            for row_i, vec in zip(owners, vectors):
                vectors_by_row[row_i].append(list(vec) if not isinstance(vec, list) else vec)
        except Exception as exc:
            logger.warning("visual embedding failed: %s", exc)

    text_encode = encode_texts_fn or _default_encode_texts

    text_vecs: List[Optional[List[float]]] = [None] * n
    if captions_en is not None:
        text_jobs: List[tuple[int, str]] = []
        for i, cap in enumerate(captions_en):
            if cap is None or _is_missing(cap):
                continue
            text = str(cap).strip()
            if len(text) < 3:
                continue
            text_jobs.append((i, text))
        if text_jobs:
            try:
                encoded = text_encode(
                    [t for _, t in text_jobs],
                    model_name=model_name,
                    batch_size=_batch_size(rules_path),
                )
                for (i, _), vec in zip(text_jobs, encoded):
                    text_vecs[i] = list(vec) if not isinstance(vec, list) else vec
            except Exception as exc:
                logger.warning("CLIP text embedding failed: %s", exc)

    empty_align = {
        "clip_alignment_mean": None,
        "clip_alignment_max": None,
        "clip_alignment_min": None,
        "clip_alignment_std": None,
        "clip_text_embedding": None,
    }

    for i in range(n):
        paths = local_paths[i]
        vecs = vectors_by_row[i]
        align = dict(empty_align)
        if text_vecs[i] is not None and vecs:
            align = _alignment_stats(text_vecs[i], vecs)
        elif text_vecs[i] is not None:
            align["clip_text_embedding"] = text_vecs[i]

        try:
            classified = classify_frame_vectors(
                vecs,
                rules_path=rules_path,
                expected_frames=expected_frames,
                model_name=model_name,
                batch_size=_batch_size(rules_path),
                encode_texts_fn=text_encode,
            )
        except Exception as exc:
            logger.warning("visual zero-shot failed for row %s: %s", i, exc)
            classified = empty_classification(
                expected_frames=expected_frames,
                valid_frames=len(vecs),
                rules_path=rules_path,
            )

        if not vecs:
            out[i].update({**align, **classified})
            continue

        out[i] = {
            "visual_embedding": _mean_vector(vecs),
            "visual_embedding_method": METHOD_CLIP,
            "visual_embedding_model": model_name,
            "frame_paths": paths,
            **align,
            **classified,
        }
    return out


def compute_visual_embeddings(
    cover_sources: Sequence[Optional[str]],
    *,
    rules_path: str = "configs/feature_rules.yaml",
    encode_images_fn: Optional[EncodeImagesFn] = None,
) -> List[dict]:
    """Back-compat: treat each source as a direct local video/cover path."""
    video_ids = [f"direct_{i}" if s else None for i, s in enumerate(cover_sources)]
    video_paths = []
    cover_paths = []
    cover_urls = []
    for s in cover_sources:
        if not s:
            video_paths.append(None)
            cover_paths.append(None)
            cover_urls.append(None)
        elif str(s).startswith(("http://", "https://")):
            video_paths.append(None)
            cover_paths.append(None)
            cover_urls.append(str(s))
        else:
            video_paths.append(str(s))
            cover_paths.append(None)
            cover_urls.append(None)
    return compute_visual_features(
        video_ids,
        video_paths=video_paths,
        cover_paths=cover_paths,
        cover_urls=cover_urls,
        rules_path=rules_path,
        encode_images_fn=encode_images_fn,
    )


def compute_visual_embedding_id(
    video_id: str,
    video_path_or_url: Optional[str] = None,
) -> Union[str, List[float], None]:
    """Back-compat single-video helper."""
    is_url = bool(video_path_or_url and str(video_path_or_url).startswith("http"))
    rows = compute_visual_features(
        [video_id],
        video_urls=[video_path_or_url if is_url else None],
        video_paths=[None if is_url else video_path_or_url],
    )
    return rows[0]["visual_embedding"]
