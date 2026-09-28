from __future__ import annotations

from urllib.parse import quote

from .html import esc, render
from .media import card_media


def _card(card: dict) -> str:
    badge = (
        f'<span class="card-badge">Top {card["top_pct"]}%</span>' if card.get("top_pct") else ""
    )
    media = card_media(card)
    meta = " · ".join(x for x in (card["brand"], card.get("product")) if x)
    # Every card keeps the same rows (title, brand · product, content type); a missing
    # field leaves its row empty rather than borrowing another field, so cards stay aligned.
    return f"""
        <a class="card" href="/?video={quote(card['video_id'])}" target="_self">
          <div class="card-media">{media}{badge}</div>
          <div class="card-body">
            <h3 class="card-title">{esc(card["title"])}</h3>
            <p class="card-meta">{esc(meta)}</p>
            <p class="card-type">{esc(card.get("content_type") or "")}</p>
          </div>
        </a>
    """


def grid(cards: list[dict], empty_message: str) -> str:
    if not cards:
        return f'<p class="empty-state">{empty_message}</p>'
    return f'<div class="card-grid">{"".join(_card(c) for c in cards)}</div>'


def render_featured(cards: list[dict], library_size: int) -> None:
    render(
        f"""
        <section class="section featured" id="featured">
          <header class="section-head">
            <h2 class="section-title">Featured High Performers</h2>
            <p class="section-note">Curated from {library_size:,} videos · ranked by
            weighted engagement rate</p>
          </header>
          {grid(cards, "No featured videos available.")}
        </section>
        """
    )


HOW_IT_WORKS = [
    ("01", "Discover", "Find relevant content."),
    ("02", "Benchmark", "Measure engagement and compare against brand benchmarks."),
    ("03", "Find patterns", "Identify patterns across comparable high performers."),
    ("04", "Generate", "Turn insights into an AI-assisted content brief."),
]


def render_how_it_works(library_size: int) -> None:
    steps = "".join(
        f'<div class="how-step"><span class="how-num">{num}</span>'
        f'<h3 class="how-title">{title}</h3><p class="how-text">{text}</p></div>'
        for num, title, text in HOW_IT_WORKS
    )
    render(
        f"""
        <section class="section how" id="how-it-works">
          <header class="section-head">
            <h2 class="section-title">How It Works</h2>
          </header>
          <div class="how-grid">{steps}</div>
          <details class="more" id="about">
            <summary>About this project</summary>
            <div class="more-body">
              <div class="more-col">
                <p class="eyebrow">About</p>
                <p>A research tool for exploring short-form brand content. It covers
                {library_size:,} public TikTok videos from Nike and Adidas, spanning official
                accounts and creators, and helps identify high-performing content across the
                library.</p>
              </div>
              <div class="more-col">
                <p class="eyebrow">Methodology</p>
                <p>Performance ranking is based on Weighted Engagement Rate (WER), which
                combines likes, comments, shares, and saves relative to views. “Top X%”
                indicates a video's WER percentile across the full library. A Brand-Relative
                Engagement Index (BRI) adds context by comparing each video's WER with its own
                brand's median. Product lines are identified from hashtags, while content types
                are classified using caption-based rule taxonomies.</p>
              </div>
              <div class="more-col">
                <p class="eyebrow">Data &amp; limitations</p>
                <p>Videos were collected from TikTok search results using brand- and
                product-related seed terms, along with public videos from official brand
                accounts. Because search results are algorithmically ranked, the sample may
                over-represent content that is more visible in TikTok search. Findings describe
                this sample and are not platform-wide statistics or official brand reports. The
                app links to original posts and does not host TikTok media.</p>
              </div>
            </div>
          </details>
        </section>
        """
    )


def render_footer() -> None:
    render(
        """
        <footer class="site-footer">Independent portfolio project for educational and analytical
        purposes. Not affiliated with or endorsed by TikTok, Nike, or Adidas.</footer>
        """
    )
