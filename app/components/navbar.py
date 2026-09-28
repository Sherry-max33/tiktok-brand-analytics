from __future__ import annotations

from .html import render


def render_navbar() -> None:
    render(
        """
        <nav class="topnav">
          <a class="topnav-brand" href="/" target="_self">TikTok AI Content Analyst</a>
          <div class="topnav-links">
            <a href="/#how-it-works" target="_self">How it works</a>
            <a href="/#about" target="_self">About</a>
          </div>
        </nav>
        """
    )
