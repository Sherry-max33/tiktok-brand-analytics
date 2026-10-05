"""Build the small aggregate tables the executive notebook (notebooks/06) reads.

Each table comes from the same analysis function the evidence notebooks use, so every number in
06 can be traced to 02 / 03 / 04b / 04c / 04d. Output is grouped aggregates only (medians,
shares, counts per group): no video IDs, account IDs, handles, captions or comment text.

    python scripts/build_executive_data.py

Re-run after rebuilding the feature tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tiktok_brand.analysis.commerce import add_commerce_intensity, commerce_engagement_trend  # noqa: E402
from tiktok_brand.analysis.data import load_feature_table, project_paths  # noqa: E402
from tiktok_brand.analysis.influencer import add_creator_flags, scale_x_dimension_wide  # noqa: E402
from tiktok_brand.analysis.sentiment_topics import (  # noqa: E402
    add_polarity_bin,
    brand_perception_summary,
    content_type_sentiment,
    load_topic_taxonomy,
    topic_id_to_label,
    topic_sentiment_by_brand,
    topic_sentiment_gap,
)
from tiktok_brand.analysis.within_brand import relative_performance_by_category  # noqa: E402

OUT = ROOT / "notebooks" / "executive_data"
BRI = "brand_relative_engagement_index"

# 03 §11 (Top-10% screening, author-grouped 5-fold CV, n = 4,418). Not recomputed here: it needs
# the full CV run; Lift@10 is printed by 03's screening diagnostics cell and all three match
# 03's saved Lift@K chart.
SCREENING_LIFT = {"lift_at_5": 3.10, "lift_at_10": 2.39, "lift_at_20": 2.06, "n_videos": 4418, "n_creators": 2614}


def _has_label(value) -> bool:
    return value is not None and len(value) > 0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = project_paths(ROOT / "configs" / "project.yaml")
    df = load_feature_table(paths["feature_table"])

    # 02 §4.2.1 / §4.2.3: content type vs each brand's own median WER.
    content = relative_performance_by_category(df, "content_type", min_count=5)
    content.to_csv(OUT / "content_type_relative.csv", index=False)

    # 04b §4: commerce intensity bins vs median BRI / top-10% rate.
    commerce = commerce_engagement_trend(add_commerce_intensity(df.copy()))
    commerce.to_csv(OUT / "commerce_trend.csv", index=False)

    # 04c §3.1: UGC creator tier x content type, cells with n >= 15.
    creators = add_creator_flags(df.copy())
    wide = scale_x_dimension_wide(creators, "content_type", min_count=15)
    wide.to_csv(OUT / "ugc_tier_content_bri.csv")
    wide.attrs["n_counts"].to_csv(OUT / "ugc_tier_content_n.csv")

    # 04c §1: official vs UGC channel snapshot.
    channel = (
        creators.groupby("creator_channel", observed=True)
        .agg(n=(BRI, "size"), median_bri=(BRI, "median"), median_views=("view_count", "median"))
        .reset_index()
    )
    channel["share"] = channel["n"] / channel["n"].sum()
    channel.to_csv(OUT / "channel_summary.csv", index=False)

    # 04d §1, §2.1, §4: comment sentiment on the high-engagement subsample.
    comments = add_polarity_bin(
        pd.read_parquet(paths["feature_table"].parent / "comment_feature_table.parquet")
    )
    labels = topic_id_to_label(load_topic_taxonomy())
    topics = topic_sentiment_by_brand(comments, min_count=25, layers=["substantive"], taxonomy=labels)
    topic_sentiment_gap(topics, metric="net").to_csv(OUT / "topic_sentiment_net.csv", index=False)
    content_type_sentiment(comments, df, min_count=30).to_csv(OUT / "content_type_sentiment.csv", index=False)
    brand_perception_summary(comments).to_csv(OUT / "brand_sentiment.csv", index=False)

    labeled = df["content_type"].map(_has_label)
    summary = {
        "videos": int(len(df)),
        "videos_by_brand": df["brand"].value_counts().to_dict(),
        "valid_engagement": int(df["weighted_engagement_rate"].notna().sum()),
        "content_labeled": int(labeled.sum()),
        "content_labeled_by_brand": df.loc[labeled, "brand"].value_counts().to_dict(),
        "collaboration_videos": int(creators["is_collaboration"].sum()),
        "comment_videos": int(comments["video_id"].nunique()),
        "comments": int(len(comments)),
        "comments_scored": int(comments["sentiment_score"].notna().sum()),
        "top10_wer_threshold": float(commerce.attrs["wer_threshold"]),
        "screening": SCREENING_LIFT,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
