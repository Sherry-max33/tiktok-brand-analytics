from __future__ import annotations

from urllib.parse import urlencode

from .html import esc

DIMENSIONS = [("brand", "Brand"), ("content", "Content"), ("product", "Product")]


def explore_controls(options: dict) -> str:
    """Explore by Brand / Content / Product. Tabs toggle option rows with CSS only.
    Options add tags to the search builder (see builder.py); the href is a no-JS fallback."""
    radios = '<input class="ex-radio" type="radio" name="explore-tab" id="ex-none">' + "".join(
        f'<input class="ex-radio" type="radio" name="explore-tab" id="ex-{dim}">'
        for dim, _ in DIMENSIONS
    )
    # The overlay label targets "ex-none", so clicking an expanded tab collapses it.
    tabs = "".join(
        f'<span class="explore-tab-wrap tab-{dim}">'
        f'<label class="explore-tab" for="ex-{dim}">{label}</label>'
        f'<label class="explore-tab-collapse" for="ex-none" aria-label="Collapse {label}"></label>'
        "</span>"
        for dim, label in DIMENSIONS
    )
    panels = []
    for dim, _ in DIMENSIONS:
        links = "".join(
            f'<a class="explore-option" href="/?{urlencode({dim: value})}" target="_self" '
            f'data-dim="{dim}" data-value="{esc(value)}">{esc(text)}</a>'
            for value, text in options.get(dim, [])
        )
        panels.append(
            f'<div class="explore-panel panel-{dim}">{links}'
            '<label class="explore-close" for="ex-none">Close</label></div>'
        )
    return (
        f'<div class="explore">{radios}'
        f'<div class="explore-bar"><span class="explore-label">Explore by</span>{tabs}</div>'
        f'{"".join(panels)}</div>'
    )
