"""Theme 3 creator & collaboration helpers."""

from __future__ import annotations

import pandas as pd

from tiktok_brand.analysis.influencer import (
    add_creator_flags,
    brand_channel_summary,
    collab_vs_rest,
    scale_x_dimension,
    ugc_tier_mix,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "brand": ["nike", "nike", "adidas", "adidas", "nike", "adidas"],
            "is_official_brand": [True, False, False, False, False, True],
            "creator_tier": ["macro", "nano", "micro", "mid", "nano", "mega"],
            "author_id": ["o1", "u1", "u2", "u3", "u4", "o2"],
            "author_follower_count": [8_000_000, 2_000, 50_000, 400_000, 8_000, 10_000_000],
            "content_type": [
                ["official_campaign"],
                ["collaboration", "product_review"],
                ["vibe_ootd"],
                ["collaboration"],
                ["product_showcase"],
                ["official_campaign"],
            ],
            "brand_relative_engagement_index": [0.8, 1.4, 1.1, 1.3, 0.9, 0.7],
            "weighted_engagement_rate": [0.005, 0.012, 0.009, 0.011, 0.007, 0.004],
            "view_count": [100_000, 50_000, 80_000, 60_000, 40_000, 200_000],
            "sentiment_score": [0.2, 0.5, 0.1, 0.4, 0.0, 0.3],
        }
    )


def test_add_creator_flags_and_channel():
    out = add_creator_flags(_sample_df())
    assert out["is_ugc"].tolist() == [False, True, True, True, True, False]
    assert out["creator_channel"].astype(str).tolist() == [
        "official",
        "ugc",
        "ugc",
        "ugc",
        "ugc",
        "official",
    ]
    assert out["is_collaboration"].tolist() == [False, True, False, True, False, False]


def test_brand_channel_and_tier_mix():
    df = add_creator_flags(_sample_df())
    summary = brand_channel_summary(df)
    nike = summary.loc[summary["brand"] == "nike"].iloc[0]
    assert int(nike["n"]) == 3
    assert int(nike["official_n"]) == 1
    assert int(nike["ugc_n"]) == 2
    assert int(nike["collab_n"]) == 1

    mix = ugc_tier_mix(df)
    all_nano = mix.loc[(mix["scope"] == "all") & (mix["creator_tier"].astype(str) == "nano")]
    assert int(all_nano["authors"].iloc[0]) == 2
    assert int(all_nano["n"].iloc[0]) == 2
    assert "author_share" in mix.columns


def test_scale_x_dimension_and_collab():
    df = add_creator_flags(_sample_df())
    tab = scale_x_dimension(df, "content_type", min_count=1)
    assert set(tab.columns) >= {"scale", "level", "n", "median_bri"}
    # collaboration cells from UGC only
    collab = tab.loc[tab["level"] == "collaboration"]
    assert int(collab["n"].sum()) == 2

    cmp_ = collab_vs_rest(df)
    assert set(cmp_["group"]) == {"collaboration", "non_collaboration"}
    assert int(cmp_.loc[cmp_["group"] == "collaboration", "n"].iloc[0]) == 2
