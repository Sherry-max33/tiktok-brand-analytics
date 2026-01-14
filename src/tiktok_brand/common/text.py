import re
from typing import List

_HASHTAG_RE = re.compile(r"#(\w+)")

def extract_hashtags(caption: str) -> List[str]:
    """Extract hashtags from caption. Returns lowercased tags without '#'."""
    if not caption:
        return []
    return [m.group(1).lower() for m in _HASHTAG_RE.finditer(caption)]
