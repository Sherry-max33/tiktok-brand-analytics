from __future__ import annotations
from typing import List, Dict

def normalize_hashtags(tags: List[str], mapping: Dict[str, str]) -> List[str]:
    """Normalize hashtags using mapping (e.g. plurals -> singular)."""
    out = []
    for t in tags or []:
        t_l = t.lower()
        out.append(mapping.get(t_l, t_l))
    return out
