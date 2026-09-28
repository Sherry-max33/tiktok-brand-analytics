"""Compact results layout used once a search has been submitted."""

from __future__ import annotations

from .builder import search_form
from .explore import explore_controls
from .featured import grid
from .html import esc, render

INFO_ICON = (
    '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">'
    '<circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1"/>'
    '<path d="M8 7v4.5M8 4.6v.1" stroke="currentColor" stroke-width="1.2" '
    'stroke-linecap="round"/></svg>'
)


def _views(n: int) -> str:
    return f"{n // 1000}K" if n < 1_000_000 else f"{n / 1_000_000:g}M"


def _count(total: int, shown: int, library_size: int, min_views: int) -> str:
    noun = "video" if total == 1 else "videos"
    text = f"{total:,} {noun} found"
    if total > shown:
        text += f" · showing top {shown}"
    tip = (
        f"Videos with {_views(min_views)}+ views appear first, ranked by weighted engagement "
        "rate (WER); lower-reach videos follow, also by WER. “Top X%” is each video's WER "
        f"percentile across all {library_size:,} videos."
    )
    return (
        f'{esc(text)}<span class="info" tabindex="0" aria-label="How results are ranked">'
        f'{INFO_ICON}<span class="info-tip" role="tooltip">{esc(tip)}</span></span>'
    )


def render_results(
    state: dict,
    cards: list[dict],
    total: int,
    options: dict,
    *,
    library_size: int,
    min_views: int,
) -> None:
    empty = (
        "Nothing matched. Try removing a tag, a broader keyword, or a product line such as "
        "Samba or Jordan."
    )
    render(
        f"""
        <section class="results" id="featured">
          <p class="eyebrow">Search results</p>
          {search_form("compact-search", state, clear_link=True)}
          {explore_controls(options)}
          <p class="results-count">{_count(total, len(cards), library_size, min_views)}</p>
          {grid(cards, empty)}
        </section>
        """
    )
