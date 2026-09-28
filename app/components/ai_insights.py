"""AI Content Analyst insight cards and the Next Content Brief panel.

Renders a stored AI record (approved curated output or cached generation) exactly as saved:
no text is generated or rewritten here, and internal fields (evidence paths, comparable IDs,
insight IDs, validation, versions, QA) are never shown.
"""

from __future__ import annotations

import streamlit as st

import ai_service
from components.html import esc

_COPY_SCRIPT = """
<script>
(function () {
  const win = window.parent;
  const doc = win.document;
  if (win.__briefCopyClick) doc.removeEventListener("click", win.__briefCopyClick);
  win.__briefCopyClick = (event) => {
    const button = event.target.closest && event.target.closest(".brief-copy[data-copy]");
    if (!button) return;
    const text = button.dataset.copy;
    const done = (label) => {
      button.textContent = label;
      clearTimeout(button.__reset);
      button.__reset = setTimeout(() => { button.textContent = "Copy brief"; }, 1600);
    };
    const fallback = () => {
      const area = doc.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      doc.body.appendChild(area);
      area.select();
      const ok = doc.execCommand("copy");
      area.remove();
      done(ok ? "Copied" : "Copy failed");
    };
    if (win.navigator.clipboard && win.isSecureContext) {
      win.navigator.clipboard.writeText(text).then(() => done("Copied"), fallback);
    } else {
      fallback();
    }
  };
  doc.addEventListener("click", win.__briefCopyClick);
})();
</script>
"""

OBSERVATION_LABELS = {
    "creative": "Creative observation",
    "narrative": "Narrative observation",
    "visual": "Visual observation",
    "strategy": "Strategy observation",
    "commerce": "Commerce observation",
}

NO_BRIEF = (
    "No grounded brief available — the available evidence supports analysis, but not a "
    "sufficiently grounded creative brief."
)


def _card(kicker: str, item: dict) -> str:
    tag = '<span class="insight-tag">Cross-brand</span>' if item.get("cross_brand") else ""
    worth = (
        f'<p class="insight-worth"><span>Worth testing</span>{esc(item["worth_testing"])}</p>'
        if item.get("worth_testing")
        else ""
    )
    return f"""
      <article class="insight">
        <p class="insight-kicker">{esc(kicker)}{tag}</p>
        <h3 class="insight-title">{esc(item["title"])}</h3>
        <p class="insight-finding">{esc(item["finding"])}</p>
        {worth}
      </article>
    """


def insight_cards(analysis: dict) -> str:
    cards = [_card("Pattern across similar high performers", p) for p in analysis["cross_content_patterns"] or []]
    if analysis["shared_characteristic"]:
        cards.append(_card("Shared with a similar high performer", analysis["shared_characteristic"]))
    cards += [
        _card(OBSERVATION_LABELS.get(o.get("type"), "Observation"), o)
        for o in analysis["single_video_observations"]
    ]
    if analysis["audience_signal"]:
        cards.append(_card("Audience signal", analysis["audience_signal"]))
    return f'<div class="insights">{"".join(cards)}</div>'


def _field(label: str, value: str, wide: bool = False) -> str:
    return (
        f'<div class="profile-item{" wide" if wide else ""}"><dt>{esc(label)}</dt>'
        f"<dd>{esc(value)}</dd></div>"
    )


def _what_to_test(plan: dict) -> str:
    why_test = (plan.get("variable_source") or {}).get("evidence_statement")
    why_html = f'<p class="test-why">{esc(why_test)}</p>' if why_test else ""
    return f"""
      <div class="brief-test">
        <p class="brief-test-label">What to test</p>
        <p class="test-variable">{esc(plan["variable_to_test"])}</p>
        {why_html}
        <p class="test-measure"><span>Measure</span>{esc(" · ".join(plan["measure"]))}</p>
      </div>
    """


def _plain_text(brief: dict) -> str:
    plan = brief["test_plan"]
    why_test = (plan.get("variable_source") or {}).get("evidence_statement")
    lines = [
        "Next Content Brief",
        "",
        f"Creative idea: {brief['concept']}",
        f"Opening hook: {brief['opening_hook']}",
        f"Creative direction: {brief['creative_direction']}",
        f"Engagement approach: {brief['engagement_approach']}",
        f"Why this direction: {brief['why_this_direction']}",
        "",
        f"What to test: {plan['variable_to_test']}",
        *([f"Why: {why_test}"] if why_test else []),
        f"Measure: {' · '.join(plan['measure'])}",
    ]
    return "\n".join(lines)


def _copy_attr(text: str) -> str:
    """Attribute-safe text whose line breaks survive `render`, which collapses newlines."""
    return esc(text).replace("\n", "&#10;")


def brief_panel(brief: dict) -> str:
    return f"""
      <details class="brief">
        <summary class="brief-cta">Next Content Brief</summary>
        <div class="brief-panel">
          <p class="eyebrow">Creative idea</p>
          <h3 class="brief-concept">{esc(brief["concept"])}</h3>
          <dl class="profile brief-main">
            {_field("Opening hook", brief["opening_hook"], wide=True)}
            {_field("Creative direction", brief["creative_direction"], wide=True)}
            {_field("Engagement approach", brief["engagement_approach"], wide=True)}
            {_field("Why this direction", brief["why_this_direction"], wide=True)}
          </dl>
          {_what_to_test(brief["test_plan"])}
          <button class="brief-copy" type="button" data-copy="{_copy_attr(_plain_text(brief))}">Copy brief</button>
        </div>
      </details>
    """


def render_copy_script() -> None:
    """Copy the displayed Brief as plain text, in the browser (no rerun, no API call)."""
    with st.container(key="brief_script"):
        st.iframe(_COPY_SCRIPT, height=1)


def analyst_section(video_id: str) -> str:
    """Read-only: serves the stored record for this video and never calls the API."""
    found = ai_service.lookup(video_id)
    record = found["record"]
    if record is None:
        body = '<p class="module-empty">AI analysis isn\'t available for this video.</p>'
    else:
        brief = ai_service.servable_brief(record)
        body = insight_cards(record["analyst"]) + (
            brief_panel(brief) if brief else f'<p class="brief-unavailable">{esc(NO_BRIEF)}</p>'
        )
    return f"""
      <section class="module analyst">
        <p class="eyebrow module-title">✦ AI Content Analyst</p>
        <h2 class="module-heading">What stands out in this content?</h2>
        <p class="ai-qualifier">Based on this video's captured features and its similar high
        performers. Shared patterns show association, not cause.</p>
        {body}
      </section>
    """
