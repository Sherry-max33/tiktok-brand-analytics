from __future__ import annotations

import html
import re
from pathlib import Path

import streamlit as st

STYLES = Path(__file__).resolve().parent.parent / "styles" / "main.css"

_BREAKS = re.compile(r"\n\s*")


def render(markup: str) -> None:
    """Render raw HTML. Newlines and indentation are collapsed so Markdown never
    mistakes indented markup for a code block."""
    st.markdown(_BREAKS.sub(" ", markup).strip(), unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value or ""), quote=True)


@st.cache_data(show_spinner=False)
def _css(mtime: float) -> str:
    return STYLES.read_text(encoding="utf-8")


def load_css() -> None:
    st.markdown(f"<style>{_css(STYLES.stat().st_mtime)}</style>", unsafe_allow_html=True)
