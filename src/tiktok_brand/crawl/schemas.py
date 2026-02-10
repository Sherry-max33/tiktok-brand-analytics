from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

@dataclass
class VideoRecord:
    platform: str
    source_type: str         # 'user' | 'hashtag'
    source_query: str        # username or seed hashtag
    seed_hashtag: Optional[str]

    video_id: Optional[str]
    create_time_ts: Optional[int]
    caption_raw: Optional[str]
    hashtags: List[str]

    author_id: Optional[str]
    author_username: Optional[str]
    author_verified: Optional[bool]

    view_count: Optional[int]
    like_count: Optional[int]
    comment_count: Optional[int]
    share_count: Optional[int]
    save_count: Optional[int]

    crawled_at: str
    crawled_at_ts: int

    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
