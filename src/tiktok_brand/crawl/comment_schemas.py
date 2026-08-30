from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class CommentRecord:
    platform: str
    video_id: str
    comment_id: str
    comment_text: str

    comment_like_count: Optional[int]
    comment_reply_count: Optional[int]
    comment_create_time: Optional[str]

    # commenter identity (NOT the video post author)
    comment_author_id: Optional[str]
    comment_author_username: Optional[str]
    comment_author_nickname: Optional[str]
    region: Optional[str]

    crawled_at: str
    crawled_at_ts: int
    raw_payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
