"""Prototype: local Drive-mount frame thumbnails for 20 videos (not the full site).

Open via the sidebar page list, or:
    streamlit run app/streamlit_app.py
then choose "frame_preview_proto".

Uses Google Drive for Desktop path (or ``FRAMES_ROOT``). No Drive API, no oEmbed,
no card iframes. Click a card → Analysis page (player / TikTok link only there).
"""

from __future__ import annotations

from urllib.parse import quote

import streamlit as st

import catalog
from components.html import esc, load_css, render
from components.media import card_media
from components.navbar import render_navbar
from frame_thumbs import (
    ThumbStats,
    attach_frame_previews,
    frames_root,
    mapping_report,
)

PROTO_N = 20
PAGE_SIZE = 16  # ~12–20 per page; 20 videos → 2 pages


st.set_page_config(
    page_title="Frame preview prototype",
    layout="wide",
    initial_sidebar_state="collapsed",
)

selected = st.query_params.get("video")
if selected:
    st.session_state["selected_video"] = selected
    st.switch_page("pages/analysis.py")

load_css()
render_navbar()


@st.cache_data(show_spinner=False)
def _proto_cards(n: int = PROTO_N) -> list[dict]:
    """Prefer library videos that already have frame_000 on the Drive mount."""
    root = frames_root()
    df = catalog.load_library()
    ids = []
    for vid in df["video_id"].astype(str):
        if (root / vid / "frame_000.jpg").is_file():
            ids.append(vid)
        if len(ids) >= n:
            break
    # Fall back with whatever we have if coverage is thin.
    if len(ids) < n:
        for vid in df["video_id"].astype(str):
            if vid not in ids:
                ids.append(vid)
            if len(ids) >= n:
                break
    cards = []
    for vid in ids[:n]:
        card = catalog.get_video(vid)
        if card:
            cards.append(card)
    return cards


def _card_html(card: dict) -> str:
    badge = (
        f'<span class="card-badge">Top {card["top_pct"]}%</span>' if card.get("top_pct") else ""
    )
    media = card_media(card, allow_embed=False)
    meta = " · ".join(x for x in (card.get("brand"), card.get("product")) if x)
    return f"""
        <a class="card" href="?video={quote(card['video_id'])}" target="_self">
          <div class="card-media">{media}{badge}</div>
          <div class="card-body">
            <h3 class="card-title">{esc(card.get("title") or "")}</h3>
            <p class="card-meta">{esc(meta)}</p>
            <p class="card-type">{esc(card.get("content_type") or "")}</p>
          </div>
        </a>
    """


cards = _proto_cards()
report = mapping_report([c["video_id"] for c in cards])

page = int(st.query_params.get("page") or 1)
page = max(1, page)
n_pages = max(1, (len(cards) + PAGE_SIZE - 1) // PAGE_SIZE)
page = min(page, n_pages)
slice_cards = cards[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]

stats = ThumbStats()
attach_frame_previews(slice_cards, stats=stats)

nav = ""
if n_pages > 1:
    links = []
    for p in range(1, n_pages + 1):
        if p == page:
            links.append(f'<span class="proto-page is-current">{p}</span>')
        else:
            links.append(f'<a class="proto-page" href="?page={p}">{p}</a>')
    nav = '<nav class="proto-pager">' + " ".join(links) + "</nav>"

grid = "".join(_card_html(c) for c in slice_cards)
render(
    f"""
    <section class="results" id="featured">
      <p class="eyebrow">Prototype · frame thumbs</p>
      <h1 class="section-title" style="margin-top:0">Local Drive frames (max {PROTO_N})</h1>
      <p class="section-note">
        Root: <code>{esc(report["frames_root"])}</code><br>
        Mapping: {report["with_frame_000"]}/{report["n"]} have frame_000.jpg ·
        page {page}/{n_pages} · size {PAGE_SIZE} ·
        thumbs {stats.hits}/{stats.requested} in {stats.elapsed_ms:.0f} ms ·
        cache hits {stats.cache_hits} · disk reads {stats.disk_reads} ·
        Drive API calls {stats.drive_api_calls}
      </p>
      {nav}
      <div class="card-grid">{grid}</div>
      {nav}
    </section>
    """
)

with st.expander("Prototype diagnostics", expanded=False):
    st.json(
        {
            "frames_root": report["frames_root"],
            "root_exists": report["root_exists"],
            "mapping": {
                "n": report["n"],
                "with_frame_000": report["with_frame_000"],
                "missing": report["missing"],
            },
            "page": {"page": page, "page_size": PAGE_SIZE, "shown": len(slice_cards)},
            "thumbs": {
                "hits": stats.hits,
                "misses": stats.misses,
                "elapsed_ms": round(stats.elapsed_ms, 1),
                "cache_hits": stats.cache_hits,
                "disk_reads": stats.disk_reads,
                "drive_api_calls": stats.drive_api_calls,
            },
            "notes": [
                "No Google Drive API credentials in this repo; reads via Drive Desktop mount.",
                "Cache: .cache/frame_thumbs/ (gitignored).",
                "Card iframes disabled (allow_embed=False).",
            ],
        }
    )
