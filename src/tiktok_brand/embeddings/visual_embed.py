"""Visual embedding (Phase 2)."""

from __future__ import annotations
from typing import Optional, Union, List


def compute_visual_embedding_id(
    video_id: str,
    video_path_or_url: Optional[str] = None,
) -> Union[str, List[float]]:
    """
    Placeholder for CLIP/CNN visual embedding.

    Returns a feature_id (path or ID) for now; later will store full embedding.
    Parquet stores feature_id to avoid large vector columns.
    """
    # Placeholder: return empty string (no embedding yet)
    return ""
