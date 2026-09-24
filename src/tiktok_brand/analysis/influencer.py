"""Theme 3: Creator & Collaboration Strategy.

Core question
-------------
Who posts brand-related content, with what content form, for which brand,
and how audiences respond (within-brand BRI).

This chapter does **not** claim that UGC authors are paid influencers, or that
``content_type=collaboration`` implies a verified commercial partnership.

Creator scale
-------------
Primary axis: existing industry-style ``creator_tier`` from follower cutoffs
(nano / micro / mid / macro / mega). These are practical bins for a top-search
sample, not universally accepted industry standards.

Scale × performance tables run on **UGC only** (``is_official_brand == False``),
because official brand accounts sit in macro/mega and would otherwise dominate
those bins. Official vs UGC is analyzed separately as channel mix.

Optional author-level follower quantiles are a robustness check only
(relative scale within the observed UGC sample), not the definition of creator type.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .data import BRI_COL, CLUSTER_COL, LIST_LABEL_COLS, WER_COL, explode_list_col

TIER_ORDER = ("nano", "micro", "mid", "macro", "mega")
TIER_CAT = pd.CategoricalDtype(categories=list(TIER_ORDER), ordered=True)

CHANNEL_ORDER = ("official", "ugc")
CHANNEL_CAT = pd.CategoricalDtype(categories=list(CHANNEL_ORDER), ordered=True)

# Primary Who × What dimensions (UGC)
CREATOR_CONTENT_DIMS = (
    "content_type",
    "brand_styles",
    "content_cluster",
)

# Optional LLM enrichment columns (if present on the feature table)
PARTNER_COLS = (
    "collab_partner_type",
    "collab_partner_name",
    "collab_role",
)


def _has_label(series: pd.Series, label: str) -> pd.Series:
    def _check(val) -> bool:
        if isinstance(val, (list, tuple, np.ndarray)):
            return label in list(val)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return False
        return str(val) == label

    return series.map(_check)


def add_creator_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Attach channel, collaboration, and ordered creator_tier columns."""
    out = df.copy()
    if "is_official_brand" not in out.columns:
        raise KeyError("is_official_brand required")
    official = out["is_official_brand"].fillna(False).astype(bool)
    out["is_official_brand"] = official
    out["is_ugc"] = ~official
    out["creator_channel"] = np.where(official, "official", "ugc")
    out["creator_channel"] = out["creator_channel"].astype(CHANNEL_CAT)

    if "content_type" not in out.columns:
        out["is_collaboration"] = False
    else:
        out["is_collaboration"] = _has_label(out["content_type"], "collaboration")

    if "creator_tier" in out.columns:
        out["creator_tier"] = (
            out["creator_tier"]
            .astype(str)
            .replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan})
        )
        out["creator_tier"] = pd.Categorical(
            out["creator_tier"], categories=list(TIER_ORDER), ordered=True
        )
    return out


def ugc_frame(df: pd.DataFrame) -> pd.DataFrame:
    """UGC subset for creator-scale analysis."""
    if "is_ugc" not in df.columns:
        df = add_creator_flags(df)
    return df.loc[df["is_ugc"]].copy()


def add_ugc_follower_quartile(
    df: pd.DataFrame,
    *,
    author_col: str = "author_id",
    follower_col: str = "author_follower_count",
    out_col: str = "ugc_follower_quartile",
) -> pd.DataFrame:
    """Optional robustness: author-level follower quartiles within UGC.

    Labels Q1-Q4 on unique authors, then mapped back to videos.
    Does not redefine creator type; answers relative scale in this sample only.
    """
    out = df.copy()
    if "is_ugc" not in out.columns:
        out = add_creator_flags(out)

    ugc = out.loc[out["is_ugc"], [author_col, follower_col]].drop_duplicates(author_col)
    followers = pd.to_numeric(ugc[follower_col], errors="coerce")
    # qcut on authors; duplicates='drop' if ties collapse bins
    try:
        q = pd.qcut(followers, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    except ValueError:
        out[out_col] = pd.NA
        return out

    author_q = pd.Series(q.to_numpy(), index=ugc[author_col].to_numpy())
    out[out_col] = out[author_col].map(author_q)
    out.loc[~out["is_ugc"], out_col] = pd.NA
    return out


def _prepare_dim(df: pd.DataFrame, label: str) -> pd.DataFrame:
    work = df
    if label == "content_cluster" and label not in work.columns and CLUSTER_COL in work.columns:
        work = work.copy()
        work["content_cluster"] = work[CLUSTER_COL]
    if label in LIST_LABEL_COLS or label in {
        "content_type",
        "brand_styles",
        "product_lines",
        "product_categories",
    }:
        if label in work.columns:
            work = explode_list_col(work, label)
    return work


def brand_channel_summary(
    df: pd.DataFrame,
    *,
    brand_col: str = "brand",
    brands: Sequence[str] = ("nike", "adidas"),
) -> pd.DataFrame:
    """Brand × Official/UGC mix + collaboration rate."""
    if "creator_channel" not in df.columns:
        raise KeyError("Run add_creator_flags() first")

    rows = []
    for b in brands:
        sub = df.loc[df[brand_col] == b]
        if sub.empty:
            continue
        n = len(sub)
        official_n = int(sub["is_official_brand"].sum())
        ugc_n = n - official_n
        collab_n = int(sub["is_collaboration"].sum())
        rows.append(
            {
                "brand": b,
                "n": n,
                "official_n": official_n,
                "ugc_n": ugc_n,
                "official_share": official_n / n if n else np.nan,
                "ugc_share": ugc_n / n if n else np.nan,
                "collab_n": collab_n,
                "collab_rate": collab_n / n if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ugc_tier_mix(
    df: pd.DataFrame,
    *,
    brand_col: Optional[str] = "brand",
    brands: Sequence[str] = ("nike", "adidas"),
    author_col: str = "author_id",
) -> pd.DataFrame:
    """UGC creator_tier mix: unique authors + video n (overall and by brand).

    Primary share is **author_share** (avoids overweighting prolific posters).
    ``n`` = video count; ``share`` aliases ``author_share`` for plot helpers.
    """
    ugc = ugc_frame(df)
    if "creator_tier" not in ugc.columns:
        raise KeyError("creator_tier required")
    if author_col not in ugc.columns:
        raise KeyError(f"{author_col} required for author-level tier mix")

    frames = []
    scopes = [("all", ugc)]
    if brand_col and brand_col in ugc.columns:
        for b in brands:
            scopes.append((b, ugc.loc[ugc[brand_col] == b]))

    for scope, sub in scopes:
        if sub.empty:
            continue
        total_videos = int(len(sub))
        # one tier per author (first video's tier; follower count is author-level)
        authors = sub.drop_duplicates(author_col)
        total_authors = int(len(authors))
        video_counts = sub["creator_tier"].value_counts(dropna=False)
        author_counts = authors["creator_tier"].value_counts(dropna=False)
        for tier in TIER_ORDER:
            n_vid = int(video_counts.get(tier, 0))
            n_auth = int(author_counts.get(tier, 0))
            author_share = n_auth / total_authors if total_authors else np.nan
            frames.append(
                {
                    "scope": scope,
                    "creator_tier": tier,
                    "authors": n_auth,
                    "n": n_vid,
                    "author_share": author_share,
                    "video_share": n_vid / total_videos if total_videos else np.nan,
                    "share": author_share,  # plot default = author mix
                    "total_authors": total_authors,
                    "total_n": total_videos,
                }
            )
    out = pd.DataFrame(frames)
    out["creator_tier"] = out["creator_tier"].astype(TIER_CAT)
    return out


def scale_x_dimension(
    df: pd.DataFrame,
    label: str,
    *,
    scale_col: str = "creator_tier",
    bri_col: str = BRI_COL,
    wer_col: str = WER_COL,
    min_count: int = 15,
    ugc_only: bool = True,
) -> pd.DataFrame:
    """Creator scale × annotation dimension → n + median BRI (+ median WER).

    Default: UGC only with industry-style ``creator_tier``.
    """
    work = ugc_frame(df) if ugc_only else df.copy()
    if scale_col not in work.columns:
        raise KeyError(f"Missing scale column: {scale_col}")
    work = _prepare_dim(work, label)
    if label not in work.columns:
        raise KeyError(f"Column not found: {label}")

    g = work.groupby([scale_col, label], dropna=False, observed=False)
    out = g.size().rename("n").to_frame()
    out["median_bri"] = g[bri_col].median()
    if wer_col in work.columns:
        out["median_wer"] = g[wer_col].median()
    out = out.reset_index().rename(columns={label: "level", scale_col: "scale"})
    out = out[out["n"] >= min_count].copy()
    if scale_col == "creator_tier":
        out["scale"] = out["scale"].astype(TIER_CAT)
        out = out.sort_values(["scale", "median_bri"], ascending=[True, False])
    else:
        out = out.sort_values(["scale", "median_bri"], ascending=[True, False])
    out.attrs["label"] = label
    out.attrs["scale_col"] = scale_col
    out.attrs["ugc_only"] = ugc_only
    out.attrs["source_n"] = len(work)
    return out.reset_index(drop=True)


def scale_x_dimension_wide(
    df: pd.DataFrame,
    label: str,
    *,
    scale_col: str = "creator_tier",
    value_col: str = "median_bri",
    min_count: int = 15,
    ugc_only: bool = True,
) -> pd.DataFrame:
    """Wide pivot: rows = scale, columns = dimension levels, values = median BRI."""
    long = scale_x_dimension(
        df,
        label,
        scale_col=scale_col,
        min_count=min_count,
        ugc_only=ugc_only,
    )
    if long.empty:
        return long
    wide = long.pivot(index="scale", columns="level", values=value_col)
    n_wide = long.pivot(index="scale", columns="level", values="n")
    wide.attrs["n_counts"] = n_wide
    wide.attrs["label"] = label
    return wide


def collab_brand_channel(
    df: pd.DataFrame,
    *,
    brand_col: str = "brand",
    brands: Sequence[str] = ("nike", "adidas"),
) -> pd.DataFrame:
    """Collaboration counts: brand × official/UGC."""
    if "is_collaboration" not in df.columns:
        raise KeyError("Run add_creator_flags() first")
    sub = df.loc[df["is_collaboration"]]
    rows = []
    for b in brands:
        bsub = sub.loc[sub[brand_col] == b]
        for channel, mask in (
            ("official", bsub["is_official_brand"]),
            ("ugc", ~bsub["is_official_brand"]),
        ):
            part = bsub.loc[mask]
            rows.append(
                {
                    "brand": b,
                    "channel": channel,
                    "n": int(len(part)),
                    "median_bri": float(pd.to_numeric(part[BRI_COL], errors="coerce").median())
                    if len(part)
                    else np.nan,
                    "median_wer": float(pd.to_numeric(part[WER_COL], errors="coerce").median())
                    if len(part)
                    else np.nan,
                    "median_sentiment": float(
                        pd.to_numeric(part["sentiment_score"], errors="coerce").median()
                    )
                    if "sentiment_score" in part.columns and len(part)
                    else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out["channel"] = out["channel"].astype(CHANNEL_CAT)
    return out


def collab_vs_rest(
    df: pd.DataFrame,
    *,
    bri_col: str = BRI_COL,
    wer_col: str = WER_COL,
    sentiment_col: str = "sentiment_score",
) -> pd.DataFrame:
    """Collaboration vs non-collaboration engagement / sentiment summary."""
    if "is_collaboration" not in df.columns:
        raise KeyError("Run add_creator_flags() first")

    rows = []
    for label, mask in (
        ("collaboration", df["is_collaboration"]),
        ("non_collaboration", ~df["is_collaboration"]),
    ):
        sub = df.loc[mask]
        row = {
            "group": label,
            "n": int(len(sub)),
            "median_bri": float(pd.to_numeric(sub[bri_col], errors="coerce").median()),
            "median_wer": float(pd.to_numeric(sub[wer_col], errors="coerce").median()),
            "median_views": float(pd.to_numeric(sub["view_count"], errors="coerce").median())
            if "view_count" in sub.columns
            else np.nan,
        }
        if sentiment_col in sub.columns:
            sent = pd.to_numeric(sub[sentiment_col], errors="coerce")
            row["median_sentiment"] = float(sent.median())
            row["sentiment_coverage"] = float(sent.notna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def collab_content_mix(
    df: pd.DataFrame,
    *,
    min_count: int = 5,
    bri_col: str = BRI_COL,
) -> pd.DataFrame:
    """Among collaboration videos, co-occurring content_type labels + median BRI."""
    if "is_collaboration" not in df.columns:
        raise KeyError("Run add_creator_flags() first")
    sub = df.loc[df["is_collaboration"]]
    if sub.empty:
        return pd.DataFrame(columns=["level", "n", "median_bri", "share"])

    work = explode_list_col(sub, "content_type")
    # Drop self-label: every row is already collaboration, so share would be 100%
    work = work.loc[work["content_type"].astype(str) != "collaboration"]
    if work.empty:
        return pd.DataFrame(columns=["level", "n", "median_bri", "share"])
    g = work.groupby("content_type", dropna=False)
    out = g.size().rename("n").to_frame()
    out["median_bri"] = g[bri_col].median()
    out["share"] = out["n"] / len(sub)
    out = out[out["n"] >= min_count].sort_values("n", ascending=False)
    return out.reset_index().rename(columns={"content_type": "level"})


def partner_type_mix(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """If LLM partner columns exist, summarize partner_type on collab videos."""
    if "is_collaboration" not in df.columns:
        df = add_creator_flags(df)
    missing = [c for c in PARTNER_COLS if c not in df.columns]
    if missing:
        return None
    sub = df.loc[df["is_collaboration"]]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby("collab_partner_type", dropna=False)
    out = g.size().rename("n").to_frame()
    out["median_bri"] = g[BRI_COL].median()
    out["share"] = out["n"] / len(sub)
    return out.reset_index().sort_values("n", ascending=False)
