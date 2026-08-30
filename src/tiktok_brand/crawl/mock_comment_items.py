"""Apify-shaped mock comment items for local smoke tests."""

from __future__ import annotations

from typing import Any, Dict, List


def mock_comment_items(video_id: str, count: int) -> List[Dict[str, Any]]:
    vid = str(video_id).strip()
    items: List[Dict[str, Any]] = []
    for i in range(min(count, 5)):
        items.append(
            {
                "cid": f"mock_comment_{vid}_{i}",
                "text": f"Mock comment {i} on video {vid}",
                "diggCount": 10 + (i * 3),
                "replyCommentTotal": i,
                "createTimeISO": "2025-07-16T03:24:56.000Z",
                "uid": f"mock_commenter_{i}",
                "uniqueId": f"commenter_{i}",
                "nickName": f"Commenter {i}",
                "videoWebUrl": f"https://www.tiktok.com/video/{vid}",
                "region": "US",
            }
        )
    return items
