"""Content analysis page.

Selected Video → Performance → Content Profile → AI Content Analyst (with the Next Content
Brief) → Similar High Performers. This is presentation order only; AI analysis may use the
similar-video retrieval as evidence. Modules without a real data source show an explicit
empty state.
"""

from __future__ import annotations

from urllib.parse import quote

import streamlit as st

import analysis_data
import ai_service
import catalog
from components.ai_insights import analyst_section, render_copy_script
from components.html import esc, load_css, render
from components.media import (
    add_official_previews,
    card_media,
    player,
    render_embed_script,
    click_to_play_url,
)
from components.navbar import render_navbar

st.set_page_config(
    page_title="Analysis · TikTok AI Content Analyst",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_css()
render_navbar()

video_id = st.query_params.get("video") or st.session_state.get("selected_video")
if video_id:
    st.session_state["selected_video"] = video_id
    st.query_params["video"] = video_id

video = catalog.get_video(video_id) if video_id else None

INFO_ICON = (
    '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">'
    '<circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1"/>'
    '<path d="M8 7v4.5M8 4.6v.1" stroke="currentColor" stroke-width="1.2" '
    'stroke-linecap="round"/></svg>'
)
HASHTAG_LIMIT = 8


def _fmt(n: int) -> str:
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.1f}".rstrip("0").rstrip(".") + unit
    return str(n)


def _pct(rate: float | None) -> str:
    if rate is None:
        return "–"
    value = rate * 100
    return f"{value:.1f}%" if value >= 1 else f"{value:.2f}%"


def _info(text: str, label: str) -> str:
    return (
        f'<span class="info" tabindex="0" aria-label="{esc(label)}">{INFO_ICON}'
        f'<span class="info-tip" role="tooltip">{esc(text)}</span></span>'
    )


if video is None:
    render(
        """
        <section class="section analysis-empty">
          <p class="eyebrow">Analysis</p>
          <h2 class="section-title">No video selected</h2>
          <p class="section-note">Choose a video from the homepage to open its analysis.</p>
          <a class="text-link" href="/" target="_self">Back to discover</a>
        </section>
        """
    )
    st.stop()


# ---------- Selected video (left) ----------

badge = f'<span class="card-badge">Top {video["top_pct"]}%</span>' if video["top_pct"] else ""
creator = f"@{esc(video['creator'])}" if video["creator"] else "Original creator"
followers = (
    f'<span class="sv-followers">{_fmt(video["followers"])} followers</span>'
    if video["followers"]
    else ""
)
product_line = esc(" · ".join(x for x in (video["brand"], ", ".join(video["products"])) if x))
if video["product_unresolved"]:
    product_line += " · " if product_line else ""
    product_line += '<span class="sv-unresolved">Product line unresolved</span>' + _info(
        "The taxonomy matched only a broad or conflicting tag for this post (e.g. "
        "#adidasoriginals), so no specific product line is shown.",
        "Why the product line is unresolved",
    )
category_line = esc(" · ".join(video["categories"]))
product = "".join(
    f'<p class="sv-product">{line}</p>' for line in (product_line, category_line) if line
)
product = f'<p class="sv-label">Product</p>{product}' if product else ""
hashtags = "".join(
    f'<span class="sv-tag">#{esc(t)}</span>' for t in video["hashtags"][:HASHTAG_LIMIT]
)
hashtags = (
    f'<p class="sv-label">Hashtags</p><div class="sv-tags">{hashtags}</div>' if hashtags else ""
)
translation = (
    f'<p class="sv-translation"><span>Machine translation</span>{esc(video["translation"])}</p>'
    if video["translation"]
    else ""
)
caption = (
    f'<p class="sv-label">Caption</p><p class="sv-caption">{esc(video["caption"])}</p>'
    f"{translation}"
    if video["caption"]
    else ""
)
if video["url"]:
    source = (
        f'<a class="sv-source" href="{esc(video["url"])}" target="_blank" rel="noopener">'
        "Original post on TikTok ↗</a>"
    )
else:
    source = ""

selected_html = f"""
  <div class="selected">
    <p class="eyebrow">Selected video</p>
    <div class="card-media player-frame">{player(video, url=video["url"])}{badge}</div>
    <p class="player-unavailable">The TikTok player couldn't load this post. It may have
    been removed or made private.</p>
    <div class="sv-creator"><span class="sv-handle">{creator}</span>{followers}</div>
    {caption}
    {product}
    {hashtags}
    {source}
  </div>
"""


# ---------- Similar high performers (full width, below the analysis) ----------


def _sim_card(card: dict) -> str:
    metrics = " · ".join(
        m
        for m in (
            f"WER {_pct(card['wer'])}" if card.get("wer") is not None else "",
            f"BRI {card['bri']:.1f}×" if card.get("bri") is not None else "",
        )
        if m
    )
    metrics_html = f'<span class="sim-metrics">{metrics}</span>' if metrics else ""
    shared = card.get("shared") or []
    shared_html = (
        f'<span class="sim-shared"><span>Similar in</span>{esc(" · ".join(shared))}</span>'
        if shared
        else ""
    )
    href = f"/?video={quote(card['video_id'])}"
    embed_url = click_to_play_url(card)
    if embed_url:
        thumb = f"""
        <div class="sim-thumb">
          {card_media(card)}
          <button class="sim-play" type="button" data-embed="{esc(embed_url)}"
                  aria-label="Play video: {esc(card['title'])}">
            <span class="sim-play-icon"></span><span class="sim-play-label">Play video</span>
          </button>
        </div>"""
    else:
        thumb = f'<a class="sim-thumb" href="{href}" target="_self">{card_media(card)}</a>'
    return f"""
      <div class="sim-card">
        {thumb}
        <a class="sim-body" href="{href}" target="_self"
           aria-label="Analyze this video: {esc(card['title'])}">
          <span class="sim-top">Top {card['top_pct']}%</span>
          <span class="sim-product">{esc(' · '.join(x for x in (card['brand'], card.get('product')) if x))}</span>
          <span class="sim-type">{esc(card.get('content_type') or '—')}</span>
          {metrics_html}
          {shared_html}
        </a>
      </div>
    """


with st.spinner("Finding comparable high performers…"):
    similar = add_official_previews(
        [dict(card) for card in ai_service.similar_cards(video["video_id"])]
    )
if similar:
    tip = _info(
        "Selected from high-performing Nike and Adidas videos based on semantic, visual, and "
        "content-strategy similarity, then checked for substantive relevance. Only videos "
        "that pass the check are shown, so there may be fewer than three.",
        "How similar videos are selected",
    )
    similar_body = f'<div class="sim-list">{"".join(_sim_card(c) for c in similar)}</div>'
elif analysis_data.similarity_active():
    tip = ""
    similar_body = (
        '<p class="module-empty">No substantively comparable high performers were found '
        "for this video.</p>"
    )
else:
    tip = ""
    similar_body = '<p class="module-empty">Similar-content retrieval is not available.</p>'

similar_html = f"""
  <section class="module similar">
    <p class="eyebrow module-title">Similar high performers{tip}</p>
    {similar_body}
  </section>
"""


# ---------- Performance (right) ----------

bri = video["bri"]
performance = [
    (
        f"Top {video['top_pct']}%" if video["top_pct"] else "–",
        "Engagement percentile",
        f"Overall, across {catalog.library_size():,} videos",
    ),
    (
        _pct(video["wer"]),
        "Weighted engagement rate",
        "Likes, comments, shares and saves relative to views",
    ),
    (
        f"{bri:.1f}×" if bri is not None else "–",
        "Brand-relative index",
        f"{bri:.1f}× {video['brand']}'s typical engagement" if bri is not None else "",
    ),
]
perf_html = "".join(
    f'<div class="perf-item"><span class="perf-value">{esc(v)}</span>'
    f'<span class="stat-label">{esc(k)}{_info(n, k) if n else ""}</span></div>'
    for v, k, n in performance
)
counts = [
    ("Views", video["views"]),
    ("Likes", video["likes"]),
    ("Comments", video["comments"]),
    ("Shares", video["shares"]),
    ("Saves", video["saves"]),
]
stat_html = "".join(
    f'<div class="stat"><span class="stat-value">{_fmt(v)}</span>'
    f'<span class="stat-label">{esc(k)}</span></div>'
    for k, v in counts
)
collected = video["collected_on"]
counts_note = (
    f'<p class="counts-note">Counts as collected on {collected:%b} {collected.day}, '
    f"{collected.year}. The TikTok player shows live counts.</p>"
    if collected
    else ""
)


# ---------- Content profile (right) ----------

PROFILE_GROUPS = {
    "frames": (
        "Based on video frames",
        "Classified from sampled video frames with an image model (CLIP), one label per "
        "field. “Likely” marks the model's top guess when it wasn't clearly ahead of the "
        "runner-up; those guesses are shown here only and aren't used for similar videos or "
        "the AI analysis. “Not identified” means no usable guess, and “Other” means none of "
        "the defined categories fit.",
    ),
    "caption": (
        "Based on caption & hashtags",
        "Classified from the creator's caption and hashtags with rule-based taxonomies. "
        "Describes how the post is framed, not what the frames show.",
    ),
}


def _profile_group(title: str, tip: str) -> str:
    return f'<div class="profile-group">{esc(title)}{_info(tip, f"About {title.lower()}")}</div>'


profile_html = ""
for key, rows in analysis_data.content_profile(video["video_id"]).items():
    title, tip = PROFILE_GROUPS[key]
    profile_html += _profile_group(title, tip) + "".join(
        f'<div class="profile-item"><dt>{esc(label)}</dt>'
        f'<dd class="{"is-empty" if value in ("—", "Not identified") or value.startswith("Likely: ") else ""}">'
        f"{esc(value)}</dd></div>"
        for label, value in rows
    )

sentiment = analysis_data.audience_sentiment(video["video_id"])
sentiment_group = _profile_group(
    "Audience sentiment",
    "Share of this video's collected comments scored positive or negative (VADER; "
    "non-English comments machine-translated first). Net = positive minus negative, "
    "in percentage points.",
)
if sentiment:
    net = round((sentiment["positive"] - sentiment["negative"]) * 100)
    sentiment_stats = "".join(
        f'<div class="senti-stat"><span class="senti-value">{value}</span>'
        f'<span class="senti-label">{label}</span></div>'
        for value, label in (
            (f"{sentiment['positive']:.0%}", "Positive"),
            (f"{sentiment['negative']:.0%}", "Negative"),
            (f"{net:+d}".replace("-", "−"), "Net sentiment"),
        )
    )
    sentiment_body = (
        f'<dd><div class="senti-stats">{sentiment_stats}</div>'
        f'<p class="senti-note">{sentiment["comments"]} comments analyzed</p></dd>'
    )
else:
    sentiment_body = '<dd class="is-empty">Not available: no comments were collected for this video</dd>'
profile_html += f'{sentiment_group}<div class="profile-item senti">{sentiment_body}</div>'


# ---------- AI content analyst + next content brief (right) ----------

analyst_html = analyst_section(video["video_id"])


render(
    f"""
    <section class="section analysis">
      <a class="back-link" href="/" target="_self">← Discover</a>
      <div class="analysis-grid">
        <aside class="analysis-left">
          {selected_html}
        </aside>
        <div class="analysis-right">
          <section class="module">
            <p class="eyebrow module-title">Performance</p>
            <div class="perf">{perf_html}</div>
            <div class="stats engagement">{stat_html}</div>
            {counts_note}
          </section>
          <section class="module">
            <p class="eyebrow module-title">Content profile</p>
            <dl class="profile">{profile_html}</dl>
          </section>
          {analyst_html}
        </div>
      </div>
      {similar_html}
    </section>
    """
)
render_embed_script()
render_copy_script()
