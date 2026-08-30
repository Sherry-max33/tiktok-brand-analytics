from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VideoRecord:
    platform: str

    # data lineage
    source_type: str  # 'user' | 'hashtag'
    source_query: str
    brand: Optional[str]

    # video identity
    video_id: Optional[str]
    create_time_ts: Optional[int]

    # content
    caption_raw: Optional[str]
    hashtags: List[str]

    # video creator (post author)
    author_id: Optional[str]
    author_username: Optional[str]
    author_verified: Optional[bool]
    author_follower_count: Optional[int]
    author_signature: Optional[str]

    # video / music
    video_duration_sec: Optional[float]
    music_id: Optional[str]
    has_music: bool

    # engagement stats
    view_count: Optional[int]
    like_count: Optional[int]
    comment_count: Optional[int]
    share_count: Optional[int]
    collect_count: Optional[int]

    # crawl metadata
    crawled_at: str
    crawled_at_ts: int
    raw_payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
