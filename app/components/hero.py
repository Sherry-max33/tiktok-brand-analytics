from __future__ import annotations

import base64
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st

from .builder import search_form
from .explore import explore_controls
from .html import esc, render

# Hero media must be our own original or licensed asset, never TikTok or brand campaign footage.
ASSETS = Path(__file__).resolve().parent.parent / "assets"
HERO_VIDEO = ASSETS / "hero.mp4"
HERO_IMAGE = ASSETS / "hero.jpg"
# Inlined as a data URI, so keep the loop short and compressed.
MAX_VIDEO_MB = 40

# Try links run the same tag search as Explore, so results land with tags in the bar.
SUGGESTIONS = [
    ("product", "samba"),
    ("product", "jordan"),
    ("content", "vibe_ootd"),
    ("content", "tutorial_utility"),
    ("product", "tech_fleece"),
]

@st.cache_data(show_spinner=False)
def _data_uri(path: str, mime: str, mtime: float) -> str:
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _media() -> tuple[str, bool]:
    if HERO_VIDEO.exists() and HERO_VIDEO.stat().st_size <= MAX_VIDEO_MB * 1024 * 1024:
        src = _data_uri(str(HERO_VIDEO), "video/mp4", HERO_VIDEO.stat().st_mtime)
        return (
            f'<video class="hero-video" src="{src}" autoplay muted loop playsinline '
            'preload="auto"></video>',
            True,
        )
    if HERO_IMAGE.exists():
        src = _data_uri(str(HERO_IMAGE), "image/jpeg", HERO_IMAGE.stat().st_mtime)
        return f'<div class="hero-image" style="background-image:url({src})"></div>', True
    return '<div class="hero-plain"></div>', False


def render_hero(explore_options: dict) -> None:
    media, has_asset = _media()
    labels = {(dim, value): text for dim in explore_options for value, text in explore_options[dim]}
    hints = "".join(
        f'<a href="/?{urlencode({dim: value})}" target="_self">{esc(labels[(dim, value)])}</a>'
        for dim, value in SUGGESTIONS
        if (dim, value) in labels
    )
    render(
        f"""
        <section class="hero{' has-video' if has_asset else ''}">
          <div class="hero-media">{media}</div>
          <div class="hero-overlay"></div>
          <div class="hero-content">
            <h1 class="hero-title"><span>Discover what makes</span> <span>content work</span></h1>
            <p class="hero-sub">Explore high-performing TikTok content<br>
            and uncover the patterns behind performance.</p>
            {search_form("hero-search", {})}
            {explore_controls(explore_options)}
            <div class="hero-hints"><span>Try</span>{hints}</div>
          </div>
          <a class="hero-scroll" href="#featured" target="_self">View featured</a>
        </section>
        """
    )
