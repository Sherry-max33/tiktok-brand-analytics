"""Re-exports for crawl record schemas."""

from tiktok_brand.crawl.comment_schemas import CommentRecord
from tiktok_brand.crawl.video_schemas import VideoRecord

__all__ = ["VideoRecord", "CommentRecord"]
