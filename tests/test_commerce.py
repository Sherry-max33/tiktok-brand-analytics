"""Theme 2 commerce intensity helpers."""

from __future__ import annotations

import pandas as pd

from tiktok_brand.analysis.commerce import (
    add_commerce_intensity,
    commerce_by_dimension,
    commerce_engagement_trend,
    score_to_intensity,
)
from tiktok_brand.etl.cta_rules import detect_cta_flags


def test_detect_giveaway_flag():
    flags = detect_cta_flags("Enter our giveaway and tag a friend")
    assert flags["has_giveaway"] is True
    assert flags["has_promo_language"] is False


def test_score_bins():
    s = pd.Series([0, 1, 2, 3, 4])
    bins = score_to_intensity(s).astype(str).tolist()
    assert bins == ["None", "Low", "Medium", "High", "High"]


def test_add_commerce_intensity_and_trend():
    df = pd.DataFrame(
        {
            "caption_raw": [
                "shop now use code SAVE",
                "link in bio",
                "nice shoes",
                "giveaway win a pair + on sale",
            ],
            "has_purchase_cta": [True, False, False, False],
            "has_discovery_traffic_cta": [False, True, False, False],
            "has_promo_language": [True, False, False, True],
            "brand_relative_engagement_index": [1.2, 1.0, 0.8, 0.6],
            "weighted_engagement_rate": [0.02, 0.015, 0.01, 0.005],
            "content_type": [
                ["product_promo"],
                ["lifestyle"],
                ["vibe_ootd"],
                ["product_promo"],
            ],
        }
    )
    out = add_commerce_intensity(df)
    assert "has_giveaway" in out.columns
    assert out["commerce_intensity_score"].tolist() == [2, 1, 0, 2]
    assert out["commerce_intensity"].astype(str).tolist() == [
        "Medium",
        "Low",
        "None",
        "Medium",
    ]

    trend = commerce_engagement_trend(out)
    assert list(trend["commerce_intensity"].astype(str)) == [
        "None",
        "Low",
        "Medium",
        "High",
    ]
    assert trend.loc[trend["commerce_intensity"].astype(str) == "None", "n"].iloc[0] == 1

    tab = commerce_by_dimension(out, "content_type", min_count=1)
    assert "product_promo" in set(tab["level"])
    assert tab.loc[tab["level"] == "product_promo", "mean_commerce_intensity"].iloc[0] == 2.0
