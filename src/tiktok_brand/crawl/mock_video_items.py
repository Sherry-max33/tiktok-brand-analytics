"""Apify-shaped mock video items for local smoke tests."""

from __future__ import annotations

import time
from typing import Any, Dict, List


def mock_hashtag_video_items(hashtag: str, count: int) -> List[Dict[str, Any]]:
    tag = hashtag.lstrip("#")
    items: List[Dict[str, Any]] = []
    for i in range(min(count, 5)):
        items.append(
            {
                "id": f"mock_video_{tag}_{i}",
                "text": f"Mock video about {tag} #{tag} #fashion",
                "createTime": int(time.time()) - (i * 3600),
                "authorMeta": {
                    "id": f"mock_author_{i}",
                    "name": f"user_{i}",
                    "verified": i % 3 == 0,
                    "signature": "Fitness & sports content creator"
                    if i % 3 == 0
                    else "Lifestyle | fashion",
                    "fans": 10000 + (i * 1000),
                },
                "videoMeta": {"duration": 15 + i},
                "musicMeta": {"musicId": f"mock_music_{i}"} if i % 2 == 0 else {},
                "playCount": 1000 + (i * 500),
                "diggCount": 50 + (i * 20),
                "commentCount": 10 + (i * 5),
                "shareCount": 5 + (i * 2),
                "collectCount": 20 + (i * 8) if i % 2 == 0 else 0,
                "hashtags": [{"name": tag}, {"name": "fashion"}, {"name": "style"}],
            }
        )
    return items


def mock_user_video_items(username: str, count: int) -> List[Dict[str, Any]]:
    handle = username.lstrip("@")
    items: List[Dict[str, Any]] = []
    for i in range(min(count, 5)):
        items.append(
            {
                "id": f"mock_user_{handle}_{i}",
                "text": f"Official post #{handle} #fashion",
                "createTime": int(time.time()) - (i * 3600),
                "authorMeta": {
                    "id": f"mock_author_{handle}",
                    "name": handle,
                    "verified": True,
                    "signature": "Official brand account",
                    "fans": 50000 + (i * 1000),
                },
                "videoMeta": {"duration": 12 + i},
                "musicMeta": {"musicId": f"music_{handle}_{i}"},
                "playCount": 2000 + (i * 500),
                "diggCount": 100 + (i * 20),
                "commentCount": 20 + (i * 5),
                "shareCount": 10 + (i * 2),
                "collectCount": 30 + (i * 10) if i % 2 == 0 else 0,
                "hashtags": [{"name": handle}, {"name": "fashion"}],
            }
        )
    return items
