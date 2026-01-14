import pandas as pd
from tiktok_brand.etl.feature_table import add_derived_metrics

def test_add_derived_metrics():
    df = pd.DataFrame([{"view_count": 100, "like_count": 10, "comment_count": 2, "share_count": 1}])
    out = add_derived_metrics(df)
    assert out.loc[0, "engagement"] == 13
    assert abs(out.loc[0, "engagement_rate"] - 0.13) < 1e-9
