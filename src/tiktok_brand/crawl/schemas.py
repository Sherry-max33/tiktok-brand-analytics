from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

@dataclass
class VideoRecord:
    platform: str

    # data lineage
    source_type: str         # 'user' | 'hashtag'
    source_query: str        # username (user crawl) or hashtag (hashtag crawl/seed hashtag)
    brand: Optional[str]     # nike | adidas | None

    # video identity
    video_id: Optional[str]
    create_time_ts: Optional[int]

    # content
    caption_raw: Optional[str]
    hashtags: List[str]     #video caption hashtags

    # creator
    author_id: Optional[str]
    author_username: Optional[str]
    author_verified: Optional[bool]
    author_follower_count: Optional[int]  # authorStats.followerCount
    author_signature: Optional[str]      # KOL 主页描述/签名 (author.signature)，用于 creator_type 分类

    # video / music (raw API)
    video_duration_sec: Optional[float]  # video.duration (ms→s if needed)
    music_id: Optional[str]              # music.id or music.idStr
    has_music: bool                      # True if music non-empty and has valid id

    # # engagement stats
    view_count: Optional[int]
    like_count: Optional[int]
    comment_count: Optional[int]
    share_count: Optional[int]
    save_count: Optional[int]

    # crawl metadata
    crawled_at: str
    crawled_at_ts: int

    # raw payload
    raw_payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
