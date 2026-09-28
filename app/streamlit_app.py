"""TikTok AI Content Analyst: homepage.

Two states on one page:
- Landing (nothing submitted): full hero with the search builder, featured high performers,
  how it works.
- Results (a submitted keyword and/or Brand / Content / Product tags): compact header and cards.
Clicking a card opens the Analysis page.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

import catalog
from components.builder import render_builder_script
from components.featured import render_featured, render_footer, render_how_it_works
from components.hero import render_hero
from components.html import load_css
from components.media import add_official_previews
from components.navbar import render_navbar
from components.results import render_results

st.set_page_config(
    page_title="TikTok AI Content Analyst",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Cards and submitted searches are plain URLs, so state arrives as query parameters.
selected = st.query_params.get("video")
if selected:
    st.session_state["selected_video"] = selected
    st.switch_page("pages/analysis.py")

options = catalog.explore_options()
state: dict = {"q": (st.query_params.get("q") or "").strip()}
for key in catalog.FACET_KEYS:
    allowed = {value for value, _ in options.get(key, [])}
    raw = (st.query_params.get(key) or "").split(",")
    state[key] = tuple(dict.fromkeys(v for v in raw if v in allowed))

load_css()
render_navbar()

if state["q"] or any(state[key] for key in catalog.FACET_KEYS):
    cards, total = catalog.search_videos(
        state["q"], **{key: state[key] for key in catalog.FACET_KEYS}
    )
    render_results(
        state,
        add_official_previews(cards),
        total,
        options,
        library_size=catalog.library_size(),
        min_views=catalog.FEATURED_MIN_VIEWS,
    )
else:
    render_hero(options)
    render_featured(add_official_previews(catalog.featured_videos()), catalog.library_size())
    render_how_it_works(catalog.library_size())

render_builder_script(catalog.facet_combinations())
render_footer()
