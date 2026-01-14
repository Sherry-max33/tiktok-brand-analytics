from __future__ import annotations
import pandas as pd

def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["view_count","like_count","comment_count","share_count","save_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["engagement"] = df[["like_count","comment_count","share_count"]].sum(axis=1, min_count=1)
    df["engagement_rate"] = df["engagement"] / df["view_count"]
    df.loc[df["view_count"].fillna(0) <= 0, "engagement_rate"] = pd.NA
    return df
