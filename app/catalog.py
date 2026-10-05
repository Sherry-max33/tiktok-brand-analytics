"""Data layer for the app: loading, featured selection, and search.

Nothing in here renders UI. Components receive plain dicts from these functions.

TikTok media is never downloaded or re-hosted. Each record keeps the original post URL
and video ID, so the UI can link out or use TikTok's official embed.
"""

from __future__ import annotations

import base64
import math
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
# Slim, committed exports of the research tables (scripts/export_app_data.py).
APP_DATA_DIR = APP_DIR / "data"
FEATURE_TABLE = APP_DATA_DIR / "videos.parquet"
COMMENT_SENTIMENT = APP_DATA_DIR / "comment_sentiment.parquet"
TAXONOMY = REPO_ROOT / "configs" / "taxonomy.yaml"
COVERS_DIR = APP_DIR / "assets" / "covers"
COVER_EXTS = (".webp", ".jpg", ".png")

# Official TikTok Embed Player; caption and music are hidden because the page shows them.
TIKTOK_PLAYER_URL = "https://www.tiktok.com/player/v1/{post_id}?music_info=0&description=0&rel=0"
_POST_ID = re.compile(r"/(?:video|photo)/(\d+)")
TIKTOK_POST_URL = "https://www.tiktok.com/@{author}/video/{video_id}"

# Rate-based index is noisy on tiny audiences, so featured picks need real reach.
FEATURED_MIN_VIEWS = 100_000
FEATURED_COUNT = 6
# Hand-picked demo cases: real high performers (WER Top 10%) that also have scraped
# comments, so Audience Sentiment is visible from the homepage. They must still pass the
# Featured filters below, except that content type may be missing (its card row stays
# empty); an ID that doesn't qualify is silently skipped.
FEATURED_DEMO_IDS = ("7336649392444689707", "7549618788073737485")
# Fixed card positions (0-based) on the homepage; 3 is the first card of the second row in
# the 3-column desktop grid. Other cards keep the automatic order around them. A slotted
# video may lack a product line (its card then shows the brand only).
FEATURED_SLOTS = {
    "7465873104259009835": 0,
    "7597641094410669367": 2,
    "7336649392444689707": 3,
    "7610110515477630222": 4,
    "7362654592309464363": 5,
}
# Never shown on the homepage (still searchable).
FEATURED_EXCLUDED = ("7598744349328854302",)
SEARCH_LIMIT = 20

# Hand-reviewed quality of the curated AI outputs (lower ranks first). Search results are
# ordered by this tier before performance; unrated videos sit between "ok" and "not ok".
_GOOD, _ABOVE_OK, _OK, _BELOW_OK, _UNRATED, _NOT_OK = range(6)
SHOWCASE_TIER = {
    "7465873104259009835": _GOOD,
    "7610110515477630222": _GOOD,
    "7597198442552773902": _GOOD,
    "7306593126414699822": _GOOD,
    "7621515669506411797": _GOOD,
    "7623157311124557086": _GOOD,
    "7564479769564155158": _GOOD,
    "7621276091654638862": _ABOVE_OK,
    "7336649392444689707": _OK,
    "7549618788073737485": _OK,
    "7597641094410669367": _OK,
    "7362654592309464363": _OK,
    "7353406737233300779": _OK,
    "7591946281082424598": _OK,
    "7614154167984147734": _OK,
    "7584803419152567570": _OK,
    "7589786563266104598": _OK,
    "7611337962038562078": _OK,
    "7612887996542536974": _BELOW_OK,
    "7598744349328854302": _NOT_OK,
    "7594922207696932128": _NOT_OK,
    "7467118868713049377": _NOT_OK,
    "7623955551726161183": _NOT_OK,
    "7588663576416570655": _NOT_OK,
    "7507053168048540971": _NOT_OK,
}

COLUMNS = [
    "video_id",
    "brand",
    "author_username",
    "caption_raw",
    "caption_en",
    "hashtags",
    "product_lines",
    "content_type",
    "view_count",
    "like_count",
    "comment_count",
    "share_count",
    "collect_count",
    "weighted_engagement_rate",
    "brand_relative_engagement_index",
    "page_url",
    "author_follower_count",
    "product_categories",
    "brand_styles",
    "social_mechanic",
    "visual_format",
    "visual_setting",
    "visual_format_margin",
    "visual_setting_margin",
    "has_purchase_cta",
    "has_engagement_cta",
    "has_discovery_traffic_cta",
    "has_promo_language",
    "translation_status",
    "crawl_at",
    "is_official_brand",
    "creator_tier",
    "video_duration_sec",
    "post_date",
    "caption_lang",
]

# A CLIP frame label is kept only when its lead over the runner-up label reaches this margin
# (accuracy per margin band from a visual audit: docs/03-data-dictionary.md, "Frame label
# confidence"). A dropped label is unknown everywhere: page, similarity, AI evidence.
FRAME_LABEL_MIN_MARGIN = {"visual_format": 0.02, "visual_setting": 0.04}
FRAME_LABEL_POLICY = "frame-labels/v2"

BRAND_LABELS = {"nike": "Nike", "adidas": "Adidas"}

PRODUCT_LABELS = {
    "af1": "Air Force 1",
    "air_max": "Air Max",
    "tech_fleece": "Tech Fleece",
    "originals_apparel": "Originals Apparel",
    "f50": "F50",
}

CONTENT_TYPE_LABELS = {
    "vibe_ootd": "Vibe / OOTD",
    "product_showcase": "Product Showcase",
    "collaboration": "Collaboration",
    "product_promo": "Product Promo",
    "product_review": "Product Review",
    "tutorial_utility": "Tutorial / Utility",
    "official_campaign": "Official Campaign",
    "story_heritage": "Story / Heritage",
    "community_impact": "Community Impact",
}

# "uncategorized" is deliberately absent so it is never shown.
CATEGORY_LABELS = {"shoes": "Shoes", "apparel": "Apparel", "accessories": "Accessories"}

# Display policy only; the feature table and taxonomy are unchanged (see docs/04-taxonomy.md,
# "Known limitations"). originals_apparel is assigned from the #adidasoriginals sub-brand tag
# alone, which creators also use on footwear posts, so it is not shown as a product line.
BROAD_PRODUCT_LINES = {"originals_apparel"}
# A line owned by the other brand is treated as a conflicting label.
LINE_BRANDS = {
    **dict.fromkeys(("tech_fleece", "jordan", "air_max", "dunk", "af1"), "nike"),
    **dict.fromkeys(
        ("samba", "gazelle", "spezial", "ultraboost", "forum", "originals_apparel",
         "f50", "predator", "superstar", "adizero"),
        "adidas",
    ),
}

MOCK_VIDEOS = [
    {"video_id": "mock-1", "brand": "adidas", "product_lines": ["samba"],
     "content_type": ["vibe_ootd"], "caption_en": "Samba street style"},
    {"video_id": "mock-2", "brand": "nike", "product_lines": ["jordan"],
     "content_type": ["product_review"], "caption_en": "Jordan 4 on feet"},
    {"video_id": "mock-3", "brand": "adidas", "product_lines": ["spezial"],
     "content_type": ["product_showcase"], "caption_en": "Spezial colourways"},
    {"video_id": "mock-4", "brand": "nike", "product_lines": ["tech_fleece"],
     "content_type": ["tutorial_utility"], "caption_en": "How to style Tech Fleece"},
]


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    try:
        return [str(v) for v in value if v is not None and str(v)]
    except TypeError:
        return []


def _line_categories() -> dict[str, str]:
    if not TAXONOMY.exists():
        return {}
    with TAXONOMY.open() as f:
        return (yaml.safe_load(f) or {}).get("line_to_category_map", {})


def _resolve_products(
    lines: list[str], categories: list[str], brand: str, line_categories: dict[str, str]
) -> tuple[list[str], list[str], bool]:
    """(lines, categories, unresolved) to display. Broad-only and other-brand lines are
    held back; categories then follow the remaining lines. unresolved = the pipeline
    assigned lines but none can be shown with confidence."""
    brand = brand.lower()
    kept = [
        line
        for line in lines
        if line not in BROAD_PRODUCT_LINES and LINE_BRANDS.get(line, brand) == brand
    ]
    if kept == lines:
        return lines, categories, False
    shown = list(dict.fromkeys(line_categories[l] for l in kept if l in line_categories))
    return kept, shown, not kept


def product_label(key: str) -> str:
    return PRODUCT_LABELS.get(key, key.replace("_", " ").title())


def content_type_label(key: str) -> str:
    return CONTENT_TYPE_LABELS.get(key, key.replace("_", " ").title())


def brand_label(key: str) -> str:
    return BRAND_LABELS.get(str(key).lower(), str(key).title())


_HASHTAG_RE = re.compile(r"[#@]\S+")
_TRAILING_HASHTAGS_RE = re.compile(r"(?:\s*#\S+)+\s*$")
_NON_TEXT_RE = re.compile(r"[^\w\s'&\-.,!?]")


UNTITLED = "Untitled post"


def _has_caption(row: pd.Series) -> bool:
    return short_title(row["caption_en"] or row["caption_raw"], "") != ""


def display_caption(caption_raw: str) -> str:
    """The creator's original caption, minus the hashtag block at the end (shown
    separately). Inline hashtags stay, since removing them can break the sentence."""
    text = (caption_raw or "").replace("\xa0", " ")
    text = _TRAILING_HASHTAGS_RE.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def short_title(caption: str, fallback: str, max_chars: int = 34) -> str:
    text = _HASHTAG_RE.sub(" ", (caption or "").replace("’", "'"))
    text = _NON_TEXT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,-!?")
    if len(text.split()) < 2:
        return fallback
    if len(text) > max_chars:
        cut = text[:max_chars].rsplit(" ", 1)[0].rstrip(" .,-!?")
        text = f"{cut}…"
    return text[0].upper() + text[1:]


def post_url(video_id: str, author: str, page_url: str = "") -> str:
    if page_url:
        return page_url
    if video_id.isdigit() and author:
        return TIKTOK_POST_URL.format(author=author, video_id=video_id)
    return ""


def cover_path(video_id: str) -> Path | None:
    for ext in COVER_EXTS:
        path = COVERS_DIR / f"{video_id}{ext}"
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False, max_entries=256)
def _cover_data_uri(path: str, mtime: float) -> str:
    mime = {".webp": "image/webp", ".png": "image/png"}.get(Path(path).suffix, "image/jpeg")
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def tiktok_post_id(video_id: str, page_url: str = "") -> str:
    """Post ID for the embed player, taken from the original URL, else the video ID."""
    match = _POST_ID.search(page_url)
    if match:
        return match.group(1)
    return video_id if video_id.isdigit() else ""


def media_source(row: pd.Series) -> dict:
    """Where a video's visual comes from: a cover in assets/covers/ if one was added,
    otherwise a placeholder. The embed URL is kept for the Analysis page player."""
    vid = row["video_id"]
    post_id = tiktok_post_id(vid, row.get("page_url") or "")
    embed_url = TIKTOK_PLAYER_URL.format(post_id=post_id) if post_id else ""
    path = cover_path(vid)
    if path:
        src = _cover_data_uri(str(path), path.stat().st_mtime)
        return {"kind": "image", "src": src, "embed_url": embed_url}
    return {"kind": "placeholder", "embed_url": embed_url}


@st.cache_data(show_spinner=False)
def load_library() -> pd.DataFrame:
    if FEATURE_TABLE.exists():
        df = pd.read_parquet(FEATURE_TABLE)
        df = df[[c for c in COLUMNS if c in df.columns]].copy()
        is_mock = False
    else:
        df = pd.DataFrame(MOCK_VIDEOS)
        is_mock = True

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["video_id"] = df["video_id"].astype(str)
    for col in (
        "hashtags",
        "product_lines",
        "content_type",
        "product_categories",
        "brand_styles",
        "social_mechanic",
    ):
        df[col] = df[col].map(_as_list)
    for col, min_margin in FRAME_LABEL_MIN_MARGIN.items():
        label = df[col].where(df[col].notna(), None)
        low = pd.to_numeric(df[f"{col}_margin"], errors="coerce") < min_margin
        df[f"{col}_status"] = [
            "missing" if value is None else "low_confidence" if is_low else "ok"
            for value, is_low in zip(label, low)
        ]
        # Page display only (shown as "Likely: …"); never used for similarity or AI evidence.
        df[f"{col}_likely"] = label.where(low, None)
        df[col] = label.where(~low, None)
    for col in (
        "has_purchase_cta",
        "has_engagement_cta",
        "has_discovery_traffic_cta",
        "has_promo_language",
    ):
        df[col] = df[col].fillna(False).astype(bool)
    df["author_follower_count"] = pd.to_numeric(df["author_follower_count"], errors="coerce")
    for col in ("view_count", "like_count", "comment_count", "share_count", "collect_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in (
        "caption_raw",
        "caption_en",
        "author_username",
        "brand",
        "page_url",
        "translation_status",
        "creator_tier",
        "caption_lang",
    ):
        df[col] = df[col].fillna("").astype(str)
    df["crawl_at"] = pd.to_datetime(df["crawl_at"], errors="coerce")
    df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
    df["is_official_brand"] = df["is_official_brand"].fillna(False).astype(bool)
    df["video_duration_sec"] = pd.to_numeric(df["video_duration_sec"], errors="coerce")

    # Performance is defined by WER, as in the notebooks (Top-10% / Top-Q models);
    # BRI is context only: how the video compares with its own brand's median.
    wer = pd.to_numeric(df["weighted_engagement_rate"], errors="coerce")
    if wer.notna().any():
        rank = wer.rank(ascending=False, method="min")
        df["top_pct"] = (rank / wer.notna().sum() * 100).map(
            lambda x: max(1, math.ceil(x)) if pd.notna(x) else None
        )
    else:
        df["top_pct"] = [4, 6, 9, 12][: len(df)] + [None] * max(0, len(df) - 4)
    df["wer"] = wer
    df["bri"] = pd.to_numeric(df["brand_relative_engagement_index"], errors="coerce")

    line_categories = _line_categories()
    resolved = [
        _resolve_products(lines, cats, brand, line_categories)
        for lines, cats, brand in zip(df["product_lines"], df["product_categories"], df["brand"])
    ]
    df["display_lines"] = [r[0] for r in resolved]
    df["display_categories"] = [r[1] for r in resolved]
    df["product_unresolved"] = [r[2] for r in resolved]

    df["search_blob"] = df.apply(_search_blob, axis=1)
    df.attrs["is_mock"] = is_mock
    return df


def _search_blob(row: pd.Series) -> str:
    parts = [
        row["caption_raw"],
        row["caption_en"],
        row["author_username"],
        row["brand"],
        brand_label(row["brand"]),
        " ".join(f"#{h}" for h in row["hashtags"]),
        " ".join(row["display_lines"]),
        " ".join(product_label(p) for p in row["display_lines"]),
        " ".join(row["content_type"]),
        " ".join(content_type_label(c) for c in row["content_type"]),
    ]
    return " ".join(parts).lower()


def to_card(row: pd.Series) -> dict:
    products = row["display_lines"]
    types = row["content_type"]
    product = product_label(products[0]) if products else ""
    ctype = content_type_label(types[0]) if types else ""
    return {
        "video_id": row["video_id"],
        "title": short_title(row["caption_en"] or row["caption_raw"], UNTITLED),
        "brand": brand_label(row["brand"]),
        "product": product,
        "content_type": ctype,
        "top_pct": int(row["top_pct"]) if pd.notna(row["top_pct"]) else None,
        "url": post_url(row["video_id"], row["author_username"], row["page_url"]),
        "media": media_source(row),
    }


def _diversified(df: pd.DataFrame, n: int, pinned: pd.DataFrame | None = None) -> pd.DataFrame:
    """Best-first, alternating brands, one card per product line until lines run out.
    Pinned rows come first and count toward the brand rotation and seen lines."""
    pinned = pinned if pinned is not None else df.iloc[:0]
    df = df.drop(index=pinned.index, errors="ignore")
    picks: list[int] = []
    seen_lines = {(lines or [""])[0] for lines in pinned["display_lines"]}
    queues = {b: list(g.index) for b, g in df.groupby("brand", sort=False)}
    order = sorted(queues, key=lambda b: -df.loc[queues[b][0], "wer"])
    if len(pinned) and pinned["brand"].iloc[-1] in order:
        last = order.index(pinned["brand"].iloc[-1])
        order = order[last + 1 :] + order[: last + 1]
    n -= len(pinned)
    while len(picks) < n and any(queues.values()):
        for brand in order:
            queue = queues[brand]
            for i, idx in enumerate(queue):
                line = (df.at[idx, "display_lines"] or [""])[0]
                if line not in seen_lines:
                    picks.append(queue.pop(i))
                    seen_lines.add(line)
                    break
            else:
                if queue:
                    picks.append(queue.pop(0))
            if len(picks) >= n:
                break
    return pd.concat([pinned, df.loc[picks]])


# Values with fewer videos are too thin to explore (e.g. story_heritage, community_impact).
EXPLORE_MIN_VIDEOS = 20


@st.cache_data(show_spinner=False)
def explore_options() -> dict[str, list[tuple[str, str]]]:
    """Options per exploration dimension, taken from values present in the data.
    Content types are ordered by frequency, product lines alphabetically."""
    df = load_library()
    brands = [b for b in BRAND_LABELS if b in set(df["brand"].str.lower())]
    types = df["content_type"].explode().dropna().value_counts()
    types = types[types >= EXPLORE_MIN_VIDEOS]
    products = df["display_lines"].explode().dropna().value_counts()
    products = products[products >= EXPLORE_MIN_VIDEOS]
    return {
        "brand": [(b, brand_label(b)) for b in brands],
        "content": [(t, content_type_label(t)) for t in types.index],
        "product": sorted(
            ((p, product_label(p)) for p in products.index), key=lambda item: item[1].lower()
        ),
    }


FACET_KEYS = ("brand", "content", "product")


def _apply_facets(df: pd.DataFrame, facets: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    """OR within a facet, AND across facets."""
    if facets.get("brand"):
        df = df[df["brand"].str.lower().isin(facets["brand"])]
    if facets.get("content"):
        wanted = set(facets["content"])
        df = df[df["content_type"].map(lambda values: bool(wanted & set(values)))]
    if facets.get("product"):
        wanted = set(facets["product"])
        df = df[df["display_lines"].map(lambda values: bool(wanted & set(values)))]
    return df


def _cards(rows: pd.DataFrame, facets: dict[str, tuple[str, ...]]) -> list[dict]:
    cards = []
    selected = facets.get("product") or ()
    for _, row in rows.iterrows():
        card = to_card(row)
        match = next((p for p in selected if p in row["display_lines"]), None)
        if match:
            card["product"] = product_label(match)
        cards.append(card)
    return cards


@st.cache_data(show_spinner=False)
def facet_combinations() -> list[list]:
    """Distinct (brand, content types, product lines) combinations in the library.
    Lets the browser work out which facet values are still possible without a round trip."""
    df = load_library()
    combos = {
        (
            str(row.brand).lower(),
            tuple(sorted(row.content_type)),
            tuple(sorted(row.display_lines)),
        )
        for row in df[["brand", "content_type", "display_lines"]].itertuples(index=False)
    }
    return [[brand, list(content), list(products)] for brand, content, products in sorted(combos)]


def _by_performance(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["wer", "view_count"], ascending=[False, False], na_position="last")


@st.cache_data(show_spinner=False)
def featured_videos(n: int = FEATURED_COUNT) -> list[dict]:
    df = load_library()
    pool = df
    if not df.attrs.get("is_mock"):
        # Homepage curation only (search keeps everything): high performers whose brand,
        # product line and content type are all known, so every card has the same structure.
        eligible = df[
            df["brand"].map(bool)
            & (df["display_lines"].map(bool) | df["video_id"].isin(FEATURED_SLOTS))
            & df["wer"].notna()
            & (df["view_count"] >= FEATURED_MIN_VIEWS)
            & ~df["video_id"].isin(FEATURED_EXCLUDED)
        ]
        eligible = eligible[eligible.apply(_has_caption, axis=1)]
        pinned_ids = (*FEATURED_DEMO_IDS, *(v for v in FEATURED_SLOTS if v not in FEATURED_DEMO_IDS))
        pinned = eligible[eligible["video_id"].isin(pinned_ids)]
        pool = eligible[eligible["content_type"].map(bool) & eligible["display_lines"].map(bool)]
    else:
        pinned_ids = ()
        pinned = pool.iloc[:0]
    pool = pool.sort_values("wer", ascending=False, na_position="last")
    pinned = pinned.iloc[pinned["video_id"].map(pinned_ids.index).argsort()]
    # Variety only among the strongest candidates, so it never costs much performance.
    picks = _diversified(pool.head(n * 3), n, pinned=pinned)
    cards = [to_card(row) for _, row in picks.iterrows()]
    slotted = {card["video_id"]: card for card in cards if card["video_id"] in FEATURED_SLOTS}
    cards = [card for card in cards if card["video_id"] not in slotted]
    for video_id, position in sorted(FEATURED_SLOTS.items(), key=lambda item: item[1]):
        if video_id in slotted:
            cards.insert(position, slotted[video_id])
    return cards


@st.cache_data(show_spinner=False)
def search_videos(
    query: str = "",
    brand: tuple[str, ...] = (),
    content: tuple[str, ...] = (),
    product: tuple[str, ...] = (),
    limit: int = SEARCH_LIMIT,
) -> tuple[list[dict], int]:
    """Keyword (all words must match) combined with facets. Ordered by SHOWCASE_TIER, then
    videos with real reach first (by WER), so small-audience rate outliers never lead."""
    facets = {"brand": brand, "content": content, "product": product}
    df = _apply_facets(load_library(), facets)
    tokens = [t for t in query.lower().replace("#", " #").split() if t]
    if tokens:
        mask = pd.Series(True, index=df.index)
        for token in tokens:
            mask &= df["search_blob"].str.contains(token, regex=False)
        df = df[mask]
    reach = df["view_count"] >= FEATURED_MIN_VIEWS
    ranked = pd.concat([_by_performance(df[reach]), _by_performance(df[~reach])])
    tier = ranked["video_id"].map(SHOWCASE_TIER).fillna(_UNRATED)
    ranked = ranked.iloc[tier.argsort(kind="stable")]
    return _cards(ranked.head(limit), facets), len(df)


def get_video(video_id: str) -> dict | None:
    df = load_library()
    match = df[df["video_id"] == str(video_id)]
    if match.empty:
        return None
    row = match.iloc[0]
    card = to_card(row)
    card.update(
        caption=display_caption(row["caption_raw"]),
        translation=row["caption_en"] if row["translation_status"] == "done" else "",
        collected_on=row["crawl_at"].date() if pd.notna(row["crawl_at"]) else None,
        creator=row["author_username"],
        views=int(row["view_count"]),
        likes=int(row["like_count"]),
        comments=int(row["comment_count"]),
        shares=int(row["share_count"]),
        saves=int(row["collect_count"]),
        wer=float(row["wer"]) if pd.notna(row["wer"]) else None,
        bri=float(row["bri"]) if pd.notna(row["bri"]) else None,
        url=post_url(row["video_id"], row["author_username"], row["page_url"]),
        products=[product_label(p) for p in row["display_lines"]],
        product_unresolved=bool(row["product_unresolved"]),
        content_types=[content_type_label(c) for c in row["content_type"]],
        followers=(
            int(row["author_follower_count"]) if pd.notna(row["author_follower_count"]) else None
        ),
        hashtags=list(row["hashtags"]),
        categories=[CATEGORY_LABELS[c] for c in row["display_categories"] if c in CATEGORY_LABELS],
    )
    return card


def library_size() -> int:
    return len(load_library())
