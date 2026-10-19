"""Video media slot shared by cards and the analysis page.

Cards prefer TikTok's oEmbed thumbnail when the response still includes thumbnail_url.
If that field is missing (TikTok's current oEmbed often only returns an iframe), the card
falls back to the official Embed Player, muted and pointer-events-free so the card link
still works. Local covers in assets/covers/ are used when present. Designed placeholders
are last resort. The Analysis page uses the official Embed Player
(TIKTOK_EMBED=0 / TIKTOK_OEMBED=0 turn each off). TikTok video files and images are never
downloaded, cached or re-hosted.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from .html import esc

TIKTOK_EMBED_ENABLED = os.getenv("TIKTOK_EMBED", "1") != "0"
TIKTOK_OEMBED_ENABLED = os.getenv("TIKTOK_OEMBED", "1") != "0"
TIKTOK_OEMBED_URL = "https://www.tiktok.com/oembed?url={url}"
OEMBED_TIMEOUT_S = 4


# TikTok's CDN preview links expire (6-hour TTL), so only the URL string is kept, in memory,
# for well under that. Failures raise and are therefore not cached.
@st.cache_data(ttl=3600, show_spinner=False, max_entries=512)
def _oembed_thumbnail(post_url: str) -> str:
    request = urllib.request.Request(
        TIKTOK_OEMBED_URL.format(url=urllib.parse.quote(post_url, safe="")),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=OEMBED_TIMEOUT_S) as response:
        thumbnail = json.load(response).get("thumbnail_url") or ""
    if not thumbnail.startswith("https://"):
        raise ValueError("oEmbed response has no thumbnail_url")
    return thumbnail


def _safe_thumbnail(post_url: str) -> str:
    try:
        return _oembed_thumbnail(post_url)
    except Exception:
        return ""


def add_official_previews(cards: list[dict]) -> list[dict]:
    """Set card["preview"] to TikTok's current official thumbnail, where one is available."""
    urls = [card.get("url") or "" for card in cards]
    if not TIKTOK_OEMBED_ENABLED or not any(urls):
        return cards
    with ThreadPoolExecutor(max_workers=min(8, len(cards))) as pool:
        previews = list(pool.map(lambda url: _safe_thumbnail(url) if url else "", urls))
    for card, preview in zip(cards, previews):
        if preview:
            card["preview"] = preview
    return cards


def placeholder(card: dict, *, large: bool = False, note: str = "Preview") -> str:
    brand = card.get("brand", "")
    product = card.get("product") or brand
    tone = "tone-" + (brand.lower() or "neutral")
    size = " is-large" if large else ""
    return (
        f'<div class="media-placeholder {tone}{size}">'
        f'<span class="ph-brand">{esc(brand)}</span>'
        f'<span class="ph-product">{esc(product)}</span>'
        f'<span class="ph-note">{esc(note)}</span>'
        "</div>"
    )


def card_media(card: dict) -> str:
    if card.get("preview"):
        # Empty alt: if the link has expired the image renders nothing and the placeholder
        # underneath shows through.
        return (
            f'<div class="preview-stack">{placeholder(card)}'
            f'<img class="official-preview" src="{esc(card["preview"])}" alt="" '
            'loading="lazy" referrerpolicy="no-referrer"></div>'
        )
    media = card.get("media") or {}
    if media.get("kind") == "image" and media.get("src"):
        return f'<img src="{esc(media["src"])}" alt="{esc(card.get("title"))}" loading="lazy">'
    embed = media.get("embed_url") or ""
    if TIKTOK_EMBED_ENABLED and embed:
        # TikTok oEmbed no longer consistently returns thumbnail_url; use the official
        # player as the card visual instead of downloading or caching a cover.
        src = embed + ("&" if "?" in embed else "?") + "autoplay=1&muted=1"
        return (
            f'<div class="preview-stack">{placeholder(card)}'
            f'<iframe class="card-embed" src="{esc(src)}" title="" loading="lazy" '
            f'allow="autoplay; encrypted-media; picture-in-picture" '
            f'referrerpolicy="strict-origin-when-cross-origin"></iframe></div>'
        )
    return placeholder(card)


def click_to_play_url(card: dict) -> str:
    """Player URL for a card's play control. Nothing is loaded until the click, and the
    click itself is the user's request to play, hence autoplay."""
    embed_url = (card.get("media") or {}).get("embed_url") or ""
    if not (TIKTOK_EMBED_ENABLED and embed_url):
        return ""
    return embed_url + ("&" if "?" in embed_url else "?") + "autoplay=1"


def player(card: dict, *, url: str = "") -> str:
    """Selected Video area.

    The official player sits on top of the editorial fallback. `render_embed_script`
    loads it (data-src) only after listening for its messages, and reveals the fallback,
    which links to the original post, if the player reports an error such as a removed
    or private post.
    """
    media = card.get("media") or {}
    if media.get("kind") == "image" and media.get("src"):
        fallback = f'<img src="{esc(media["src"])}" alt="{esc(card.get("title"))}">'
    else:
        fallback = placeholder(card, large=True, note="Open on TikTok ↗" if url else "Preview")
    if url:
        fallback = (
            f'<a class="player-link" href="{esc(url)}" target="_blank" rel="noopener" '
            f'aria-label="Open the original TikTok post">{fallback}</a>'
        )

    if TIKTOK_EMBED_ENABLED and media.get("embed_url"):
        return (
            f'<div class="player-stack">{fallback}'
            f'<iframe class="player-embed" data-src="{esc(media["embed_url"])}" '
            'allow="encrypted-media; fullscreen; picture-in-picture" allowfullscreen '
            f'title="TikTok video: {esc(card.get("title"))}"></iframe></div>'
        )
    return fallback


_EMBED_SCRIPT = """
<script>
(function () {
  const win = window.parent;
  const doc = win.document;
  if (win.__tiktokPlayerListener) win.removeEventListener("message", win.__tiktokPlayerListener);
  win.__tiktokPlayerListener = (event) => {
    let data = event.data;
    if (typeof data === "string") {
      try { data = JSON.parse(data); } catch (_) { return; }
    }
    if (!data || !data["x-tiktok-player"] || data.type !== "onPlayerError") return;
    doc.querySelectorAll(".player-stack").forEach((stack) => {
      const frame = stack.querySelector("iframe.player-embed");
      if (frame && frame.contentWindow === event.source) stack.classList.add("is-fallback");
    });
  };
  win.addEventListener("message", win.__tiktokPlayerListener);

  // Click-to-load: a card's preview becomes the official player only when its play
  // control is clicked. Other players on the page are paused so only one plays.
  if (win.__tiktokPlayClick) doc.removeEventListener("click", win.__tiktokPlayClick);
  win.__tiktokPlayClick = (event) => {
    const button = event.target.closest && event.target.closest(".sim-play[data-embed]");
    if (!button) return;
    event.preventDefault();
    doc.querySelectorAll("iframe.player-embed[src]").forEach((frame) => {
      frame.contentWindow.postMessage({ "x-tiktok-player": true, type: "pause" }, "*");
    });
    const thumb = button.closest(".sim-thumb");
    const frame = doc.createElement("iframe");
    frame.className = "player-embed";
    frame.src = button.dataset.embed;
    frame.allow = "autoplay; encrypted-media; fullscreen; picture-in-picture";
    frame.allowFullscreen = true;
    frame.title = button.getAttribute("aria-label") || "TikTok video";
    const stack = doc.createElement("div");
    stack.className = "player-stack";
    stack.append(...thumb.childNodes);
    stack.append(frame);
    button.remove();
    thumb.classList.add("is-playing");
    thumb.append(stack);
  };
  doc.addEventListener("click", win.__tiktokPlayClick);

  function start(tries) {
    const frames = doc.querySelectorAll("iframe.player-embed[data-src]");
    if (!frames.length) {
      if (tries > 0) setTimeout(() => start(tries - 1), 100);
      return;
    }
    frames.forEach((frame) => {
      frame.src = frame.dataset.src;
      frame.removeAttribute("data-src");
    });
  }
  start(50);
})();
</script>
"""


def render_embed_script() -> None:
    """Load TikTok players on the page (the main player now, card players on click) and fall
    back to the preview when one reports an error."""
    with st.container(key="page_script"):
        st.iframe(_EMBED_SCRIPT, height=1)
