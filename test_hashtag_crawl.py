#!/usr/bin/env python3
"""
Quick test script for hashtag crawler.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from tiktok_brand.crawl.hashtag_crawler import crawl_hashtag
from tiktok_brand.common.io import write_jsonl
from tiktok_brand.common.time import now_ts

def test_single_hashtag():
    """Test crawling a single hashtag."""
    print("Testing hashtag crawler with 'nike'...")

    try:
        # Test with a small count
        records = crawl_hashtag(
            seed_hashtag="nike",
            count=3,  # Very small for testing
            tz_name="America/New_York",
            brand="nike"
        )

        print(f"✓ Crawled {len(records)} records")

        if records:
            # Show first record structure
            record = records[0]
            print("✓ Sample record keys:", list(record.keys()))
            print("✓ Platform:", record.get('platform'))
            print("✓ Source type:", record.get('source_type'))
            print("✓ Brand:", record.get('brand'))
            print("✓ Seed hashtag:", record.get('seed_hashtag'))
            print("✓ Has crawled_at:", 'crawled_at' in record)
            print("✓ Has crawled_at_ts:", 'crawled_at_ts' in record)

            # Save to test file
            test_file = Path("data/raw") / f"test_hashtag_nike_{now_ts()}.jsonl"
            write_jsonl(test_file, records)
            print(f"✓ Saved to {test_file}")

            # Check file exists and has content
            if test_file.exists():
                with open(test_file, 'r') as f:
                    lines = f.readlines()
                print(f"✓ File has {len(lines)} lines")

                # Try to parse first line as JSON
                import json
                first_record = json.loads(lines[0])
                print("✓ First line is valid JSON")
                print("✓ Contains video_id:", 'video_id' in first_record)
            else:
                print("✗ Test file was not created")

        else:
            print("✗ No records returned")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_hashtag()