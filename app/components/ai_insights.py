"""AI Content Analyst insight cards and the Next Content Brief panel.

Renders a stored AI record (approved curated output or cached generation) exactly as saved:
no text is generated or rewritten here, and internal fields (evidence paths, comparable IDs,
insight IDs, validation, versions, QA) are never shown. The typing effect is presentation
only: the full saved text is in the page and is revealed progressively in the browser.
"""

from __future__ import annotations

import streamlit as st

import ai_service
from components.html import esc

GENERATE_BRIEF_KEY = "brief_generate"

_SCRIPT = """
<script>
(function () {
  const win = window.parent;
  const doc = win.document;

  // Typing reveal: blank each unseen .ai-reveal block, then type it out once it's visible
  // (a closed Brief starts typing when opened). Played once per block per browser tab;
  // click to skip.
  const BLOCKS = ".insight, .brief-panel > *, .brief-main > .profile-item";
  const reduce = win.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const seen = (key) => {
    try { return win.sessionStorage.getItem("ai-reveal:" + key) === "1"; } catch (e) { return false; }
  };
  const remember = (key) => {
    try { win.sessionStorage.setItem("ai-reveal:" + key, "1"); } catch (e) {}
  };

  function prepare(el) {
    const walker = doc.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    const parts = [];
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      if (!node.nodeValue.trim() || node.parentElement.closest("button")) continue;
      parts.push({ node, text: node.nodeValue });
    }
    parts.forEach((part) => { part.node.nodeValue = ""; });
    el.querySelectorAll(BLOCKS).forEach((block) => block.classList.add("ai-wait"));
    el.__aiParts = parts;
    el.dataset.revealed = "ready";
  }

  function type(el) {
    const parts = el.__aiParts;
    el.__aiParts = null;
    el.dataset.revealed = "typing";
    remember(el.dataset.revealKey);
    const caret = doc.createElement("span");
    caret.className = "ai-caret";
    let index = 0, pos = 0, frame = null;
    // Paced by elapsed time, not by tick count (timers run late), so a whole block takes
    // about TARGET_MS however long it is.
    const TARGET_MS = 900;
    const total = parts.reduce((sum, part) => sum + part.text.length, 0);
    let shown = 0, started = null;
    const unhide = (node) => {
      for (let b = node.parentElement; b && b !== el; b = b.parentElement) b.classList.remove("ai-wait");
    };
    const finish = () => {
      win.cancelAnimationFrame(frame);
      parts.forEach((part) => { part.node.nodeValue = part.text; });
      el.querySelectorAll(".ai-wait").forEach((block) => block.classList.remove("ai-wait"));
      caret.remove();
      el.dataset.revealed = "done";
      el.removeEventListener("click", finish);
    };
    const step = (now) => {
      if (!el.isConnected) return;
      if (started === null) started = now;
      const goal = Math.ceil(total * Math.min(1, (now - started) / TARGET_MS));
      let budget = goal - shown;
      while (budget > 0 && index < parts.length) {
        const part = parts[index];
        if (pos === 0) {
          unhide(part.node);
          part.node.parentNode.insertBefore(caret, part.node.nextSibling);
        }
        let end = Math.min(part.text.length, pos + budget);
        const space = part.text.indexOf(" ", end);
        if (space !== -1 && space - end < 6) end = space + 1;
        part.node.nodeValue = part.text.slice(0, end);
        budget -= end - pos;
        shown += end - pos;
        pos = end;
        if (pos >= part.text.length) {
          index += 1;
          pos = 0;
        }
      }
      if (index >= parts.length) return finish();
      frame = win.requestAnimationFrame(step);
    };
    el.addEventListener("click", finish);
    frame = win.requestAnimationFrame(step);
  }

  if (win.__aiReveal) {
    win.__aiReveal.mutations.disconnect();
    win.__aiReveal.visibility.disconnect();
  }
  const visibility = new win.IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting || !entry.target.__aiParts) return;
      visibility.unobserve(entry.target);
      type(entry.target);
    });
  }, { threshold: 0.15 });
  const scan = () => {
    doc.querySelectorAll(".ai-reveal").forEach((el) => {
      if (!el.dataset.revealed) {
        if (reduce || seen(el.dataset.revealKey)) { el.dataset.revealed = "done"; return; }
        prepare(el);
      }
      if (el.__aiParts) visibility.observe(el);
    });
  };
  let queued = false;
  const mutations = new win.MutationObserver(() => {
    if (queued) return;
    queued = true;
    win.requestAnimationFrame(() => { queued = false; scan(); });
  });
  mutations.observe(doc.body, { childList: true, subtree: true });
  win.__aiReveal = { mutations, visibility };
  scan();

  if (win.__briefGenerateClick) doc.removeEventListener("click", win.__briefGenerateClick);
  win.__briefGenerateClick = (event) => {
    const button = event.target.closest && event.target.closest(".brief-generate");
    if (!button || button.disabled) return;
    const trigger = doc.querySelector(".st-key-__KEY__ button");
    if (!trigger) return;
    button.disabled = true;
    button.textContent = "Writing the brief…";
    trigger.click();
  };
  doc.addEventListener("click", win.__briefGenerateClick);

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
""".replace("__KEY__", GENERATE_BRIEF_KEY)

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


NO_INSIGHTS = "The available evidence doesn't support any AI insights for this video."
LIVE_NOTE = "Generated on demand for this video and not reviewed by a person."


def _thinking(text: str) -> str:
    return (
        f'<div class="ai-thinking" role="status"><span class="ai-thinking-dot"></span>'
        f'<span class="ai-thinking-text">{esc(text)}</span></div>'
    )


def insight_cards(analysis: dict, reveal_key: str) -> str:
    cards = [_card("Pattern across similar high performers", p) for p in analysis["cross_content_patterns"] or []]
    if analysis["shared_characteristic"]:
        cards.append(_card("Shared with a similar high performer", analysis["shared_characteristic"]))
    cards += [
        _card(OBSERVATION_LABELS.get(o.get("type"), "Observation"), o)
        for o in analysis["single_video_observations"]
    ]
    if analysis["audience_signal"]:
        cards.append(_card("Audience signal", analysis["audience_signal"]))
    return (
        f'<div class="insights ai-reveal" data-reveal-key="{esc(reveal_key)}">'
        f'{"".join(cards)}</div>'
    )


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


def brief_panel(brief: dict, reveal_key: str, *, open_: bool = False) -> str:
    return f"""
      <details class="brief"{" open" if open_ else ""}>
        <summary class="brief-cta">Next Content Brief</summary>
        <div class="brief-panel ai-reveal" data-reveal-key="{esc(reveal_key)}">
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


def render_ai_script() -> None:
    """Browser-side behavior (no API call): the typing reveal, the Brief button, which
    clicks the hidden Streamlit trigger, and Copy brief."""
    with st.container(key="brief_script"):
        st.iframe(_SCRIPT, height=1)


def _brief_part(video_id: str, found: dict, *, pending: bool, notice: str | None, open_: bool) -> str:
    record = found["record"]
    brief = ai_service.servable_brief(record)
    if brief:
        return brief_panel(brief, f"{video_id}:brief", open_=open_)
    if found["source"] != "cache" or not ai_service.brief_requestable(record):
        return f'<p class="brief-unavailable">{esc(NO_BRIEF)}</p>'
    if pending:
        return f'<div class="brief">{_thinking("Writing the Next Content Brief")}</div>'
    notice_html = f'<p class="brief-unavailable">{esc(notice)}</p>' if notice else ""
    return (
        '<div class="brief"><button class="brief-cta brief-generate" type="button">'
        f"Next Content Brief</button>{notice_html}</div>"
    )


def analyst_section(
    video_id: str,
    found: dict,
    *,
    pending: str | None = None,
    notice: str | None = None,
    brief_notice: str | None = None,
    brief_open: bool = False,
) -> str:
    """Markup for `found` (ai_service.lookup); never calls the API. `pending` is the stage
    ("analysis" | "brief") the page is about to generate; `notice` / `brief_notice` explain
    why a stage can't run."""
    record = found["record"]
    if pending == "analysis":
        body = _thinking("Analyzing this video and its similar high performers")
    elif record is None:
        body = f'<p class="module-empty">{esc(notice or ai_service.REFUSALS["disabled"])}</p>'
    else:
        analysis = record["analyst"]
        insights = (
            insight_cards(analysis, f"{video_id}:analysis")
            if analysis["status"] == "generated"
            else f'<p class="module-empty">{esc(NO_INSIGHTS)}</p>'
        )
        live_note = f'<p class="ai-live-note">{esc(LIVE_NOTE)}</p>' if found["source"] == "cache" else ""
        body = insights + _brief_part(
            video_id, found, pending=pending == "brief", notice=brief_notice, open_=brief_open
        ) + live_note
    return f"""
      <section class="module analyst">
        <p class="eyebrow module-title">✦ AI Content Analyst</p>
        <h2 class="module-heading">What stands out in this content?</h2>
        <p class="ai-qualifier">Based on this video's captured features and its similar high
        performers. Shared patterns show association, not cause.</p>
        {body}
      </section>
    """
