"""Unit tests for collab partner enrich helpers (no API)."""

from __future__ import annotations

import pandas as pd

from scripts.enrich_collab_partners import (
    normalize_result,
    select_candidates,
)


def test_normalize_forces_none_name_null():
    out = normalize_result(
        {"partner_type": "none", "partner_name": "Someone", "collab_role": "feature"}
    )
    assert out["collab_partner_type"] == "none"
    assert out["collab_partner_name"] is None
    assert out["collab_role"] == "feature"


def test_normalize_invalid_enums():
    out = normalize_result(
        {"partner_type": "influencer", "partner_name": "n/a", "collab_role": "sponsored"}
    )
    assert out["collab_partner_type"] == "other"
    assert out["collab_partner_name"] is None
    assert out["collab_role"] == "unclear"


def test_normalize_focal_brand_not_partner():
    out = normalize_result(
        {"partner_type": "brand", "partner_name": "Nike", "collab_role": "product_collab"},
        brand="nike",
    )
    assert out["collab_partner_type"] == "none"
    assert out["collab_partner_name"] is None


def test_select_candidates_two_pools():
    df = pd.DataFrame(
        {
            "content_type": [
                ["collaboration"],
                ["vibe_ootd"],
                ["product_review"],
                ["collaboration", "product_promo"],
            ],
            "mention_count": [0, 2, 1, 3],
            "caption_en": ["a x b", "hi @x @y", "short", "capsule with artist"],
        }
    )
    both = select_candidates(df, pools=["collaboration", "mention_expand"])
    assert set(both["_enrich_pool"]) == {"collaboration", "mention_expand"}
    assert len(both) == 3  # two collab + one mention_expand

    only_c = select_candidates(df, pools=["collaboration"])
    assert len(only_c) == 2
    assert (only_c["_enrich_pool"] == "collaboration").all()
