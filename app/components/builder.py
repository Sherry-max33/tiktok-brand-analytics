"""Search-condition builder shared by the hero and the compact results bar.

Choosing a Brand / Content / Product value adds a tag to the search field instead of
searching. Options that can no longer match are hidden (cascading facets, computed in the
browser from the distinct facet combinations). Searching only happens on submit, with
OR within a facet and AND across facets.
"""

from __future__ import annotations

import json

import streamlit as st

from .html import esc

PLACEHOLDER = "Search videos, products, creators, or keywords"
PLACEHOLDER_WITH_TAGS = "Add a keyword (optional)"

SUBMIT_ICON = (
    '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">'
    '<path d="M4 12h15M13 6l6 6-6 6" fill="none" stroke="currentColor" '
    'stroke-width="1.2" stroke-linecap="square"/></svg>'
)
CLEAR_ICON = (
    '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">'
    '<path d="M6 6l12 12M18 6L6 18" fill="none" stroke="currentColor" stroke-width="1.2"/></svg>'
)


def search_form(css_class: str, state: dict, *, clear_link: bool = False) -> str:
    facets = {key: list(state.get(key) or ()) for key in ("brand", "content", "product")}
    clear = (
        f'<a class="compact-clear" href="/" target="_self" aria-label="Clear search">{CLEAR_ICON}</a>'
        if clear_link
        else ""
    )
    # Tags and the typed keyword are restored client-side from data-state / data-q;
    # a React-rendered input value would be read-only.
    return f"""
        <form class="{css_class} search-builder" method="get" action="/" target="_self"
          role="search" data-state="{esc(json.dumps(facets))}" data-q="{esc(state.get("q") or "")}">
          <div class="search-field">
            <span class="chips"></span>
            <input type="search" name="q" autocomplete="off" placeholder="{PLACEHOLDER}"
              aria-label="{PLACEHOLDER}">
          </div>
          {clear}
          <button type="submit" aria-label="Search">{SUBMIT_ICON}</button>
        </form>
    """


_SCRIPT = """
<script>
(function () {
  const COMBOS = __COMBOS__;
  const DIMS = ["brand", "content", "product"];
  const PLACEHOLDER = __PLACEHOLDER__;
  const PLACEHOLDER_WITH_TAGS = __PLACEHOLDER_WITH_TAGS__;
  const doc = window.parent.document;
  const valuesOf = (combo, dim) => dim === "brand" ? [combo[0]] : dim === "content" ? combo[1] : combo[2];

  function init(tries) {
    const form = doc.querySelector("form.search-builder");
    const explore = doc.querySelector(".explore");
    if (!form || !explore) { if (tries > 0) setTimeout(() => init(tries - 1), 100); return; }
    if (form.dataset.bound) return;
    form.dataset.bound = "1";

    const input = form.querySelector("input[name=q]");
    const chips = form.querySelector(".chips");
    let state;
    try { state = JSON.parse(form.dataset.state || "{}"); } catch (e) { state = {}; }
    DIMS.forEach(d => { state[d] = Array.isArray(state[d]) ? state[d] : []; });
    if (form.dataset.q && !input.value) input.value = form.dataset.q;

    const labels = {};
    explore.querySelectorAll(".explore-option").forEach(a => {
      labels[a.dataset.dim + ":" + a.dataset.value] = a.textContent;
    });

    const matchesOthers = (combo, skip) => DIMS.every(dim => {
      if (dim === skip || !state[dim].length) return true;
      const vals = valuesOf(combo, dim);
      return state[dim].some(v => vals.includes(v));
    });

    function toggle(dim, value) {
      const i = state[dim].indexOf(value);
      if (i >= 0) state[dim].splice(i, 1); else state[dim].push(value);
      refresh();
    }

    function refresh() {
      chips.textContent = "";
      DIMS.forEach(dim => state[dim].forEach(value => {
        const chip = doc.createElement("span");
        chip.className = "chip";
        chip.appendChild(doc.createTextNode(labels[dim + ":" + value] || value));
        const x = doc.createElement("button");
        x.type = "button";
        x.className = "chip-x";
        x.setAttribute("aria-label", "Remove " + (labels[dim + ":" + value] || value));
        x.textContent = "\\u00d7";
        x.addEventListener("click", e => { e.preventDefault(); toggle(dim, value); input.focus(); });
        chip.appendChild(x);
        chips.appendChild(chip);
      }));
      DIMS.forEach(dim => {
        const available = new Set();
        COMBOS.forEach(c => { if (matchesOthers(c, dim)) valuesOf(c, dim).forEach(v => available.add(v)); });
        explore.querySelectorAll('.explore-option[data-dim="' + dim + '"]').forEach(a => {
          const selected = state[dim].includes(a.dataset.value);
          a.classList.toggle("is-selected", selected);
          a.hidden = !selected && !available.has(a.dataset.value);
        });
      });
      const hasTags = DIMS.some(d => state[d].length);
      input.placeholder = hasTags ? PLACEHOLDER_WITH_TAGS : PLACEHOLDER;
      form.classList.toggle("has-tags", hasTags);
    }

    explore.addEventListener("click", e => {
      const option = e.target.closest(".explore-option");
      if (!option) return;
      e.preventDefault();
      toggle(option.dataset.dim, option.dataset.value);
    });

    input.addEventListener("keydown", e => {
      if (e.key !== "Backspace" || input.value) return;
      for (let i = DIMS.length - 1; i >= 0; i--) {
        if (state[DIMS[i]].length) { state[DIMS[i]].pop(); refresh(); break; }
      }
    });

    // Submitted natively (this frame may not navigate the page); empty fields are
    // disabled so they stay out of the URL.
    const hidden = {};
    DIMS.forEach(d => {
      hidden[d] = doc.createElement("input");
      hidden[d].type = "hidden";
      hidden[d].name = d;
      form.appendChild(hidden[d]);
    });
    form.addEventListener("submit", () => {
      input.value = input.value.trim();
      input.disabled = !input.value;
      DIMS.forEach(d => {
        hidden[d].value = state[d].join(",");
        hidden[d].disabled = !state[d].length;
      });
      setTimeout(() => { input.disabled = false; }, 0);
    });

    refresh();
  }
  init(50);
})();

(function () {
  const doc = window.parent.document;
  // Streamlit renders after load, so jump to #how-it-works / #about ourselves;
  // #about is the collapsed "About this project" panel and opens on arrival.
  function go(hash, tries) {
    const target = hash && doc.getElementById(hash.slice(1));
    if (!target) { if (hash && tries > 0) setTimeout(() => go(hash, tries - 1), 100); return; }
    if (target.tagName === "DETAILS") target.open = true;
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function bind(tries) {
    const links = doc.querySelectorAll('.topnav-links a[href^="/#"]');
    if (!links.length) { if (tries > 0) setTimeout(() => bind(tries - 1), 100); return; }
    links.forEach(a => {
      if (a.dataset.bound) return;
      a.dataset.bound = "1";
      a.addEventListener("click", e => {
        const hash = a.getAttribute("href").slice(1);
        if (!doc.getElementById(hash.slice(1))) return;
        e.preventDefault();
        window.parent.history.replaceState(null, "", "/" + hash);
        go(hash, 0);
      });
    });
  }
  bind(50);
  go(window.parent.location.hash, 50);
})();
</script>
"""


def render_builder_script(combos: list[list]) -> None:
    """Attach the builder to the page. Only our own dataset values enter the script;
    anything from the URL is read from data attributes on the page."""
    payload = json.dumps(combos, separators=(",", ":")).replace("</", "<\\/")
    script = (
        _SCRIPT.replace("__COMBOS__", payload)
        .replace("__PLACEHOLDER_WITH_TAGS__", json.dumps(PLACEHOLDER_WITH_TAGS))
        .replace("__PLACEHOLDER__", json.dumps(PLACEHOLDER))
    )
    with st.container(key="page_script"):
        st.iframe(script, height=1)
