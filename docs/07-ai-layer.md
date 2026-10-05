# AI layer: Content Analyst + Next Content Brief (frozen spec v1)

**Core principle.** The AI Content Analyst discovers incremental insight from reliable
evidence; the Next Content Brief turns those insights into an executable, testable creative
hypothesis.

**Framework:** Measure → Retrieve → Validate → Synthesize → Generate → Test.

```
Selected video → structured evidence → similar high performer retrieval → relevance
validation → AI Content Analyst → evidence-backed insights → Next Content Brief →
testable creative hypothesis → controlled experiment
```

Page questions, in order: Performance (how did it perform?) → Content Profile (what is it?) →
Audience Signals (how did the observed audience respond, when available?) → Similar High
Performers (what comparable high-performing content exists?) → AI Content Analyst (what can we
learn from the reliable evidence?) → Next Content Brief + Test Plan (how could those learnings
become a creative hypothesis, and how would we test it?).

External description: *The AI layer combines structured content features, engagement
benchmarks, optional audience signals, and multimodal retrieval of comparable high performers.
Retrieved examples are relevance-checked before cross-content synthesis. The system then
generates evidence-backed creative insights and translates them into a testable content brief
with explicit experimental variables and controls.*

## 1. AI Content Analyst

Answers: *What can we learn from the selected content and the available comparable evidence?*

Analytical questions (not fixed UI fields):

1. **What stands out?** A creative / visual / narrative / strategy framing with explanatory
   value (e.g. "culture-led rather than product-led"), not "this is a basketball video".
2. **What repeats across high performers?** Shared subject/product context, narrative
   approach, visual presentation, creative strategy, framing or social/commerce approach
   between the selected video and *relevant* comparables. This is retrieval's main value.
3. **What is worth testing?** "This pattern may be worth testing", never "this is why it
   succeeded". How to test belongs to the Brief.

Must not:

- restate performance metrics already on the page as an insight (metrics may support an
  interpretation, not be the interpretation);
- claim a feature caused performance, or turn correlation into causation;
- speculate about creator intent, audience demographics or brand strategy;
- fill in missing data;
- pad with weak insights to reach a count;
- turn false-positive retrievals into a "recurring pattern".

Audience sentiment is an optional evidence source. With comment data it may support an
`audience_signal` insight when meaningful; without it, no insight is generated (the page
already shows that data is unavailable).

## 2. Evidence layer (`analysis_context`)

Built by `app/analysis_context.py` from existing app data only:

- `selected_video`: id, brand, creator/account (username, official vs creator, tier,
  followers), caption (original, language, machine translation), hashtags, product (status,
  lines, categories), duration, post date.
- `performance`: Top X%, WER, BRI, views, likes, comments, shares, saves, collection date.
- `content_profile`: frames (visual format, visual setting) and caption & hashtags (content
  type, brand style, social mechanic, CTA signals).
- `audience_signals`: comments analyzed, positive %, negative %, net sentiment, or `null`.
- `similar_high_performers`: identity, performance, content profile, audience signals,
  similarity components (text, visual, strategy overlap, category conflict) and shared labels.

`null` = not available (never inferred); `[]` = the rule-based classifier ran and found
nothing. The context also carries plain-language conventions for these and for metric
definitions, so the generators read the same rules.

## 3. Retrieval relevance guardrail

Retrieve → **Validate** → Synthesize. Multimodal similarity is not semantic relevance (e.g.
Jordan sneakers vs. travel to Jordan via `#jordan`).

Before any cross-content synthesis, each comparable is classified relevant / excluded by
substantive alignment in subject/product context, content or narrative intent, visual
presentation, and creative strategy. Lexical overlap, brand-name ambiguity or superficial
visual similarity alone are not sufficient.

- Selected video + **≥ 2 relevant comparables** → a *recurring cross-content pattern* may be
  synthesized.
- Selected + 1 comparable → only "similarity / shared characteristic", never "recurring".
- < 2 relevant → `cross_content_synthesis_allowed = false`; fall back to single-video creative
  profiling plus other available evidence. This is normal behaviour, not a failure.

Internal (not prominent in the UI), for guardrails, debugging, retrieval QA and evaluation:

```json
{"retrieval_assessment": {"relevant_comparables": 2, "excluded_comparables": 1,
                          "cross_content_synthesis_allowed": true}}
```

### Implementation (`app/relevance.py`)

- The same judgment drives both the **Similar High Performers cards** and **pattern
  eligibility**; only the minimum counts differ. Cards show the relevant candidates in
  retrieval order, up to 3, never padded: one or two cards (or an empty state) is correct
  when that is all that is relevant. Patterns need ≥ 2.
- Retrieval returns the top 10 candidates (one per creator). Excluded candidates are skipped
  and the next relevant one backfills. `excluded_comparables` counts the rejections passed
  over before the cards were filled. Rejected candidates and their rationale stay in
  `relevance.assess()` for debugging and are never rendered or sent to the Analyst.
- **Judge**: one batched LLM call (`gpt-4.1`, temperature 0, strict JSON schema; override
  with `OPENAI_MODEL`). For each candidate it gives evidence and an aligned yes/no per
  dimension (`subject_product`, `narrative_intent`, `visual_presentation`,
  `creative_strategy`), then a `comparable` verdict and a one-sentence rationale. Brand is
  not a criterion. Hashtag-derived product lines must be checked against the caption.
- **Code-side guard**: relevant = the verdict **and** ≥ 2 aligned dimensions including
  `subject_product` or `narrative_intent`. `visual_presentation` is dropped when either
  video lacks frame evidence (no visual score and no frame labels).
- **Fallback** (no key or API error): deterministic rules. `subject_product` = shared
  product line and text similarity ≥ 0.40. `narrative_intent` = text similarity ≥ 0.55, or
  a shared content type with text similarity ≥ 0.40. `visual_presentation` = visual
  similarity ≥ 0.75 or the same frame format. `creative_strategy` = a shared content type,
  social mechanic or brand style. It is deliberately conservative.
- **Cache**: LLM judgments persist via `ai_cache.py` (Supabase when configured, see §7;
  otherwise the gitignored `data/processed/ai/`), keyed by prompt version, model, video and candidate IDs. Curated videos serve the
  judgment frozen in their registry record instead (§7). Borderline candidates can flip
  between fresh LLM runs, so persisting judgments is also what keeps the cards stable (see
  Known limitations below).

Findings on the Phase 1 sample set: the AJ1 promo's Jordan travel post (via `#jordan`) is
excluded, leaving 1 card and no pattern. The same travel post is also excluded for the Jordan
demo and the ice-bucket clip. A caption-less video with no frame data keeps 1 comparable.
`gpt-4.1-mini` produced contradictory calls (it accepted the travel post in one case) and
treated brand as a veto, so the judge uses `gpt-4.1`.

## 4. Analyst output (dynamic schema)

The schema fixes the insight *structure*, not its content. 0–3 insights; target 2–3 when the
evidence supports them, 1 or an explicit insufficient-evidence state otherwise.

```json
{
  "retrieval_assessment": {...},
  "insights": [
    {"title": "SPORT CULTURE AS THE CREATIVE FRAME",
     "finding": "The selected video and multiple relevant comparables embed Jordan within basketball culture rather than presenting the product in isolation.",
     "evidence": ["Selected video uses athlete-led basketball content",
                  "Two relevant comparables share sport/action framing",
                  "Selected video ranks in the top 4% by WER"],
     "type": "cross_content_pattern"}
  ]
}
```

Types: `creative_pattern`, `visual_pattern`, `cross_content_pattern`, `cross_brand_pattern`,
`audience_signal`, `commerce_signal`. None is required.

UI: at most 3 insights, each as title, finding, and "BASED ON: Selected video · N comparable
high performers". No fixed "Relative performance / Creative pattern / Audience response"
headings; no raw JSON.

### Implementation (`app/analyst.py`)

The Analyst returns structured JSON only; Phase 4 turns it into prose. It reads
`analysis_context` after validation, minus what it must not use: rejected candidates never
reach it, and retrieval scores, excluded counts and creator usernames are stripped. It does
not re-judge relevance. Phase 2 owns that decision.

Output (`analyst_output/v1`) keeps single-video observations apart from cross-content
findings:

| Field | Present when | Otherwise |
|---|---|---|
| `single_video_observations` | always (may be `[]`) | |
| `shared_characteristic` | exactly 1 validated comparable | `null` |
| `cross_content_patterns` | ≥ 2 validated comparables | `null` (`[]` = allowed, none found) |
| `audience_signal` | comment sentiment was collected | `null` |

Plus `status` (`generated` / `insufficient_evidence` / `unavailable`), `evidence_basis`,
`evidence_notes` (internal) and a `validation` log. Each item has `title`, `finding`,
`evidence_refs`, `worth_testing` ("… may be worth testing", or `null`), and `comparable_ids`
for the comparison items. Patterns also get a code-computed `cross_brand` flag.

**Enforced by the schema.** The response schema is built per video. Fields the evidence
doesn't allow are absent from it, so the model can't fill them. `evidence_refs` is an enum of
the paths that actually hold a value (e.g. `content_profile.frames.visual_format`,
`comparables[<id>].caption.text`), and `comparable_ids` is an enum of the validated IDs. A
claim can't cite missing data or an unvalidated video.

**Enforced by code after generation:**
- a pattern cites ≥ 2 comparables and the selected video's own fields;
- every structured label a comparison item cites (content profile, product line) is actually
  shared by each cited comparable;
- an observation cites only the selected video;
- performance fields are never the sole evidence;
- no causal wording (cause / drive(n) / led to / due to / because / boost / attribute /
  explain …) and no retrieval talk;
- a single-comparable finding is never called recurring, consistent or a pattern;
- an audience item cites `audience_signals` and does not interpret beyond the numbers.

Violations get one repair call. Items still violating are dropped and logged, and at most 3
items are kept (priority: patterns, shared characteristic, observations, audience).

**Prompt rules** (for what code can't check): null means unknown, never negative evidence;
frame labels are one zero-shot label each, so don't describe beyond them; product lines are
hashtag-derived, so check them against the caption; describe correlation, never causation;
no speculation about creator intent, demographics or brand strategy; no padding.

**Model:** `gpt-4.1`, temperature 0, strict JSON schema. The Analyst itself doesn't cache;
`ai_service.py` persists the whole pipeline result (§7).

**Validation on the eight QA cases (prompt v5):**
- The mechanical checks passed in all eight: no pattern field with fewer than 2 comparables,
  no shared characteristic unless exactly 1, no audience item without data, no more than 3
  items.
- Graceful degradation: 3 comparables gave 1 pattern + observation + audience; 1 comparable
  gave a shared characteristic + observation; 0 comparables gave observations only.
- Stability (a fresh, uncached second run per case): 6 of 8 were structurally identical. The
  other two had one fewer optional observation. Pattern availability, cited comparables and
  shared characteristics matched in all eight; wording varies.

### Known limitations (keep for final QA)

- **Relevance judgments are not deterministic.** Borderline candidates can flip between
  fresh LLM runs, even at temperature 0. Persisting and freezing outputs makes what the app
  shows stable; it does not make the judgment itself deterministic. The UI must never
  present a relevance decision as more certain than it is.
- **Analyst wording varies between runs**, and occasionally so does the number of optional
  observations. The cache pins what is displayed.
- **Ambiguous brand hashtags.** A bare `#jordan` on a skit where Jordan is a person
  (`7504834979172584735`) is labeled Jordan Brand by the taxonomy. Both the relevance judge
  and the Analyst accept that label despite prompt instructions to check the caption, so the
  Analyst still mentions a "product association". The fix belongs in the data layer
  (taxonomy V2, or a display policy that treats a bare ambiguous tag as unresolved), not
  in more prompting.
- **Cross-brand comparables are rare after LLM validation.** The rules fallback accepts
  cross-brand candidates that the LLM judge usually rejects, so curation only counts
  cross-brand coverage from LLM judgments.

## 5. Next Content Brief

Answers: *How could the observed learnings be translated into a creative direction worth
testing next?* An evidence-informed creative hypothesis, not a performance prediction or a
guaranteed best practice.

Inputs: the original `analysis_context` + the validated comparables + the Analyst insights
(so a second generation can re-check the evidence instead of amplifying an interpretation
error).

```json
{
  "concept": "...",
  "opening_hook": "...",
  "creative_direction": "...",
  "engagement_approach": "...",
  "why_this_direction": "...",
  "test_plan": {
    "variable_to_test": "...",
    "variant_a": "...",
    "variant_b": "...",
    "what_to_keep": ["..."],
    "measure": ["..."]
  }
}
```

- **Concept**: one specific creative idea ("Basketball Legacy, Reframed for Today"), not
  "create an engaging Jordan video".
- **Opening hook**: how the first seconds enter.
- **Creative direction**: the Analyst insight turned into execution.
- **Engagement approach**: grounded in existing CTA / social mechanic / commerce evidence; no
  unsupported claims such as "no CTA performs better".
- **Why this direction**: explicit provenance, framed as a hypothesis to test.
- **Test plan**: one primary variable; Variant A vs B change only that variable where
  practical. **What to keep** = experimental controls held constant so the tested variable can
  be interpreted, *not* proven success factors ("keep the visual setting consistent across
  variants", never "keep it because it drove engagement"). **Measure** with existing project
  metrics (WER, share rate, save rate), plus secondary metrics only when the objective needs
  them.
- One primary experiment by default, optionally one follow-up. Not a brainstorm list.

Every major recommendation must trace to the supplied evidence or Analyst insights; no
unsupported product, audience, creator or brand claims.

### Implementation (`app/brief.py`)

Generated only when the Analyst produced at least one insight. The Brief sees the Analyst
input plus the insights, each with an ID (`pattern_n`, `shared_1`, `observation_n`,
`audience_1`). The schema makes `based_on_insights` an enum of those IDs, `evidence_refs`
the Analyst's enum of non-null paths, and `measure` an enum of project metrics
(`Net comment sentiment` only when comments were analyzed).

**Test-variable grounding.** `test_plan.variable_source` records where the tested variable
comes from: `source_type` (`analyst_insight` with an `insight_id`, or `observed_evidence`
with `evidence_refs` into the selected video or the comparables) plus an
`evidence_statement`, the plain "why test it" sentence shown to readers. Code checks that:
- the variable's wording appears in the cited insight or in the cited fields' values, so
  a test can't be invented and attached to an unrelated insight;
- a statement about the comparables cites comparable fields, and those are the same field
  as the selected video's (or the comparable's caption or hashtags, for absence claims);
- the statement doesn't count more comparables than the source covers;
- the catch-all `Other` label is never a variant.
If no source supports a variable, the Brief can't test it.

Other code checks: at least one evidence ref; no causal, predictive ("will increase",
"outperform", "proven") or retrieval language; no performance used as justification
("strong performance"); no evidence paths pasted into the text; 2–4 controls in
`what_to_keep`, none framed as a success factor (quoted taxonomy labels such as
`'Performance'` brand style are ignored); `measure` starts with WER, at most 3 metrics; the
two variants differ. One repair call, with the first-round violations kept in the record; a
brief that still fails is `failed_validation` and is not shown.

**Cross-brand signal (Analyst).** Each comparable carries `same_brand_as_selected`, and
comparison items get a code-computed `cross_brand` flag. The Analyst may name the other
brand when that strengthens a finding. Claims that a characteristic generalizes across
brands, or cross-brand wording with no other-brand comparable cited, are rejected.
Speculation about how viewers respond ("may encourage", "viewers seeking") is rejected.

**Observability.** Both steps know which dimensions the dataset captures. Uncaptured
modalities (on-screen text, voiceover, narration, speech, music, audio) are never
evidence, present or absent. The Analyst never mentions them. The Brief keeps its test
variable, "why test it" and rationale within captured dimensions. Its execution fields may
suggest uncaptured elements only when explicitly framed as new ideas.

## 6. Implementation phases

1. **Evidence layer**: `analysis_context`, no LLM. Inspect 5–10 real videos. *(done)*
2. **Retrieval relevance validation**: `retrieval_assessment`, relevant/excluded per
   comparable, the ≥ 2 rule and fallback. *(done)*
3. **AI Content Analyst**: LLM over the validated context with the rules above. *(done)*
4. **Render Analyst**: up to 3 insight cards with "Based on" provenance.
5. **Next Content Brief**: separate generation step with the test plan. *(generation done;
   rendering with Phase 4)*
6. **Final QA** across: good retrieval + sentiment; good retrieval, no sentiment;
   cross-brand comparables; an obvious false positive; fewer than 2 relevant comparables;
   missing product line; missing content type; missing visual embedding; official account;
   UGC creator. The pass criteria: it never says more than the evidence supports, and it
   degrades correctly when the evidence gets weaker.

## 7. Curated generation and public serving

### Pipeline and versioning (`app/ai_service.py`)

One pipeline run = relevance (LLM) → `analysis_context` → Analyst → Brief → automated QA.
The record stores `video_id`, `model`, `pipeline_version`, per-step prompt versions,
`generation_attempt`, `generated_at` (UTC), the relevance picks and assessment, the Analyst
and Brief outputs, and the automated QA result. `pipeline_version` joins the pipeline
revision with every prompt version (e.g. `ai-pipeline/1+relevance/v4+analyst/v5+brief/v2`),
so any prompt or validator change invalidates earlier results.

Automated QA fails a record when the Analyst or Brief wasn't generated, when a field
appears that the evidence doesn't allow (pattern with < 2 comparables, shared
characteristic without exactly 1, audience item without sentiment), or when more than 3
items survive. Dropped items and repaired briefs are warnings.

### Curated showcase set (`app/curated.py`, `scripts/curate_ai.py`)

About 20–25 showcase videos with pre-generated, human-approved outputs, registered in the
committed `app/frozen_ai/curated.json`. They are showcase examples, not a statistical
sample.

Selection (`curate_ai.py propose`, no API calls): all Homepage Featured videos, then the top
2 suitable results for each Homepage "Try" keyword, deduplicated by video and creator, then
gap filling for coverage: both brands, Vibe/OOTD, Product Showcase, Collaboration, key
product lines, audience sentiment and cross-brand retrieval. A top-ranked video is skipped
when it has an obvious data problem (no caption, unresolved product line, no frame labels,
no validated comparables).

Workflow: `add` → `generate` (full pipeline, never hand-written) → `show` → `select` →
`qa approve|reject`. Only an approved record is served. Every attempt is kept.
Regeneration policy:
- **Systematic failure** (prompt, validator, retrieval or data problem): fix the system,
  which bumps `pipeline_version`, and regenerate.
- **Model variance**: at most `curated.MAX_ATTEMPTS` (3) attempts per pipeline version;
  select the strongest fully grounded attempt.

Offline generation retries rate-limit errors twice and pauses between videos (the account's
tokens-per-minute limit is close to one full pipeline run).

### Serving

Order of lookups, none of which calls the API: approved curated record → curated but not
approved (shows "not available yet"; never generates live) → persistent cache of live
generations → not generated.

Live generation for non-curated videos runs in two stages, each cached separately:
- **Analysis** (`generate_analysis`: relevance + Analyst), cached under
  `(video_id, analysis_version, model)`. It starts automatically when the analysis page
  opens, after the rest of the page has rendered; the AI section shows an "Analyzing"
  state meanwhile.
- **Brief** (`generate_brief`), cached under `(video_id, pipeline_version, model)`. It runs
  only when the user clicks "Next Content Brief", over the cached analysis and the same
  cached relevance judgment. A Brief that fails validation is cached too (and withheld), so
  it isn't regenerated.

Both stages pass the same admission: cache check → live generation enabled? → per-session
limit (`AI_SESSION_LIMIT`, default 2) → server-side daily ceiling (`AI_DAILY_LIMIT`, default
10, UTC days) → one run with no retries → persist → display.
The limits count videos: within a session, one video's analysis and Brief take a single
slot. Cached results never consume quota. It fails closed: if the cache or the quota state
can't be read or written, or the limits are misconfigured, nothing is generated (an
unreadable cache is never treated as a miss), and a failed stage isn't retried in the same
session. A refusal is shown in place of the AI output (e.g. "Today's AI
generation limit has been reached."). `AI_LIVE_GENERATION=off` disables live generation
entirely. Live outputs are labeled as generated on demand and not reviewed by a person.

Note: links between app pages are full page loads, and each load starts a new Streamlit
session, so the per-session limit mostly bounds repeated generations within one page view;
the daily ceiling is the effective cost bound.

Storage: with `SUPABASE_URL` and `SUPABASE_SECRET_KEY` set, the live cache and the daily
counter live in Supabase (`supabase_store.py`; tables and the atomic `reserve_generation`
function from `scripts/supabase_setup.sql`), so they survive restarts of a host with an
ephemeral disk and hold across server instances. The tables hold only AI outputs keyed by
video ID, version and model, plus one count per day; no visitor data. Row Level Security
is on with no policies, so only the server-side secret key can read or write. Without
Supabase, the cache is local JSON in `data/processed/ai/` and the counter a file-locked
JSON file (`AI_QUOTA_FILE`).

Presentation: saved insights and Briefs (curated or cached) are revealed with a typing
effect in the browser, once per block per browser tab, starting when the block scrolls into
view (a closed Brief starts when opened); clicking skips it, and it is off under
`prefers-reduced-motion`. The full saved text is already in the page; nothing is generated
or altered by the effect.

The API key is read server-side only (environment, Streamlit Secrets or local `.env`, all
gitignored) and never reaches the browser. All non-AI features work with no key or when
generation is unavailable.

### Pilot (5 + 1 videos, pipeline `…+brief/v2`)

Run before scaling to the full set. Cases: strong standard retrieval (Adidas tracksuit
OOTD), audience sentiment (Jordan athlete), cross-brand comparables, missing frame labels
(Samba OOTD), retrieval-noise risk (AJ1 promo, 1 card).
- The first run exposed a validator false positive: `'Performance'` (a brand-style label) in
  a control was flagged as a success claim. Fixed and versioned as `brief/v2`.
- The intended cross-brand case lost its cross-brand candidates under LLM relevance
  (1 same-brand card). An Adidas athlete-signing post whose validated comparable is a Nike
  signing post was added instead.
- One Brief failed on "results in" inside a "Tests whether…" sentence. That's conservative,
  but it fails closed as intended.
- Review found Brief test variables detached from the insights (an on-screen prompt test
  with no evidence behind it). This led to test-variable grounding (`brief/v3`+), then
  checks for performance-as-justification, the `Other` label, uncited or unrelated comparable
  fields, and hyphenated "-led" false positives (through `analyst/v8`, `brief/v7`). At that
  version all six pass automated QA; the grounding check rejected an untraceable
  "on-screen prompt" variable in the tutorial case and the repair chose a grounded one.
- Observability rule (`analyst/v9`, `brief/v8`): the dataset has no on-screen text,
  voiceover, narration, speech, music or audio data (`analyst.CAPTURED_DIMENSIONS`,
  `analyst.UNCAPTURED`). The Analyst never mentions them. In the Brief, the test variable,
  the "why test it" sentence and `why_this_direction` must stay within captured dimensions.
  Execution fields may suggest uncaptured elements only in a sentence framed as a new idea
  (e.g. "As a new idea not drawn from the data, …") and never "as seen in". At this version
  five of six pass on the first run.
- Catch-all frame labels (`analyst/v10`): the AJ1 promo's only visual contrast with its
  comparable was `visual_format` `Other` vs `Archival / retro`, and the Brief kept building
  the test on the catch-all label (failing closed twice). `Other` frame labels (89 formats,
  152 settings in the library) are now given to both steps as null (unknown), and the Brief
  is told that a null field can't be one side of a contrast.
- `brief/v11` fixed two validator false positives: "engagement prompt" in a control, and a
  caption test whose label ("Tutorial / Utility") lives in the cited caption's derived
  labels, not its words.
- Frozen version (`analyst/v12 + brief/v13`), the last pipeline change:
  - The generalization check was widened. No text may generalize beyond this video and its
    similar high performers to a broader population ("common among high-performing Adidas
    creator content", "typical of Nike content", "top-performing videos"). Scoped wording
    ("recurs within this comparable set") is fine.
  - A Brief that fails validation is withheld rather than failing the record, and the
    Analyst insights are served alone (`ai_service.servable_brief`). Human QA can also
    withhold a Brief at approval (`qa ID approve --withhold-brief`); this changes serving,
    not generation.
  - Results:
    - The six-video pilot passed on its only run.
    - All 25 curated videos are approved; 5 are insights only (Brief withheld).
    - Four first attempts were regenerated within the attempt cap for factual slips: a
      setting miscount, an overstated caption absence, and two cases that inferred on-screen
      colorways the data doesn't capture.
    - Total cost of the curated generation: about $1.5.
- Frame-label policy (`ai-pipeline/2`, prompts unchanged): review of a served Brief found it
  counting two comparables as studio videos when their frames showed a home interior. The
  CLIP setting labels were near random, and low-margin format labels were unreliable. Frame
  labels below a confidence margin are now null in the evidence
  (`docs/03-data-dictionary.md`, “Frame label confidence”), and the relevance cache key
  includes the policy so similar-video judgments were redone.
  - All 25 curated videos were regenerated within the attempt cap. 23 serve a Brief and 2
    are insights only. The first run withheld more Briefs than before, mostly because a
    Brief tried to contrast a comparable's format with the selected video's now-unknown
    format and the validator refused.
  - Review checked caption and hashtag claims against the raw captions. Every remaining
    visual-format claim (22 videos) was checked against the sampled frames. One Brief that
    read the unknown setting as "the frames do not show a setting" was regenerated.
  - Cost of the regeneration: about $2.1.
- Final pre-release version (`analyst/v11 + brief/v12`):
  - Caption language is removed from the evidence (a known upstream error source), and
    language or translation mentions are rejected.
  - Every catch-all taxonomy value (Other, Unknown, Unclassified …) in labels and product
    fields is unknown.
  - Titles may not restate performance.
  - Prose is written for a strategist ("this video", "similar high performers"), and the
    Brief's comparable-count checks cover those phrasings.
  - Five of six passed on the first attempt. AJ1's Brief failed all three attempts, each
    time contrasting the comparable's archival format with the video's unknown format; it
    fails closed. Semantic review found one repeated unsupported generalization in the Samba
    Analyst pattern ("common among high-performing Adidas creator content").
  - Measured cost: about 16K input and 1.1K output tokens per video with cached relevance
    judgments.
- Result at `analyst/v10 + brief/v11`: all six have an output that passes automated QA
  (one to two attempts each, within the 3-attempt cap). Human review still catches what
  the checks can't: a Brief that says a comparable "lacks a caption" when it has one, and a
  short English caption that the dataset's language detection labels `fr`, which the Analyst
  repeats.
