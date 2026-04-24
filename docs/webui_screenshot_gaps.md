# webui screenshot walkthrough — gaps vs. LlamaChat

Pass of `imagery/screenshots/*.png` (captured 2026-04-24 against llama-server
webui at `b8893`) cross-referenced against LlamaChat's current UI. Items that
are **already done** or **already filed** in the existing checklists
(`docs/webui_feature_checklist.md`, `docs/llama_server_parity_checklist.md`)
are omitted here — this doc is strictly about gaps surfaced by the
screenshots that weren't obvious before.

## State legend

- `[ ]` **TODO** — actionable, small.
- `[~]` **needs design** — the webui version makes a UX assumption we'd have to
  decide about (in-chat system message rendering, auto-title flow, etc.).
- `[!]` **won't implement** — webui-specific (HTTP-client-of-server affordance
  that doesn't translate, or duplicates something we already handle differently).

---

## 1. Welcome / compose area (`new_chat.png`)

- [ ] **Model pill near the compose box** — webui shows `Qwen3.6 | 35B-A3B | UD | IQ4_XS.gguf`
  as small badges right above the send button. LlamaChat surfaces the same
  info in the top toolbar (`ProfileDisplayName` + `ModelSummary`), so it's
  visible but spatially far from where the user's eyes are. Moving (or
  mirroring) a compact pill next to the compose box would match webui's
  more-immediate feedback.
- [ ] **"Type a message or upload files to get started"** hero text on empty
  conversations — we have `"Start the conversation"` centred, but no mention
  of file upload as an option. Small copy tweak.

## 2. Settings — General tab (`settings_general.png`)

Webui has a dedicated General tab. LlamaChat scatters some of these across
the Display tab and the per-profile System Message; others don't exist.

- [!] **API Key field** — N/A for an in-process client.
- [~] **Show system message in conversations** — checkbox that inlines the
  system prompt as a read-only bubble at the top of the transcript. Would
  also double as a reminder of which persona/profile is active mid-chat.
- [ ] **Paste-long-text-to-file threshold** — when the user pastes text larger
  than N chars, webui converts it to a virtual attachment (a text file chip)
  instead of dumping it into the compose box. Saves scroll-hell. Would pair
  with the existing clipboard-paste interception in `OnComposeKeyDownTunnel`.
- [ ] **Send message on Enter** toggle — right now Enter always sends; the
  webui lets you invert so Enter inserts a newline and Ctrl+Enter sends.
  Motor-accessibility affordance for people who often hit Enter mid-thought.
- [!] **Copy text attachments as plain text** — relevant only when we support
  text-file attachments (which we don't). Revisit alongside the paste-to-file
  feature above.
- [~] **Enable "Continue" button toggle** — lets the user hide the
  per-assistant-bubble Continue action. We always show it when the last reply
  is extendable; hiding it is purely a preference. Low value.
- [!] **Parse PDF as image** — PDF attachment path doesn't exist yet; revisit
  if we ever add PDF support.
- [~] **Ask for confirmation before changing conversation title** — fires a
  prompt when the auto-title heuristic would overwrite a title the user
  manually set. Depends on the auto-title feature below.
- [ ] **Use first non-empty line for conversation title** (auto-title from
  first user message) — webui derives an initial title from the first user
  turn instead of leaving it as "New chat". Nice default; pairs with the
  existing inline-rename flow so the user can still override.

## 3. Settings — Display tab (`settings_display.png`)

We already have the four core toggles (AutoScroll, ShowMessageStats,
ShowReasoningInProgress, HighAccessibilityMode). Gaps:

- [ ] **Keep stats visible after generation** — webui hides the tok/s +
  token-count stats once a reply finishes. LlamaChat always keeps them
  visible. Adding a toggle (default: keep visible, matching our current
  behaviour) is ~5 lines.
- [~] **Show microphone on empty input** — ours shows the mic button based
  on `CanAttachAudio` (model capability). Webui's version is empty-input
  triggered (mic appears when the compose box is empty). Different UX
  assumption; I'd argue ours is better because the affordance is
  discoverable even while you're typing. Document as intentional divergence.
- [ ] **Render user content as Markdown** toggle — we currently render user
  bubbles as plain preformatted text (via `SelectableTextBlock`). Webui lets
  the user opt into markdown rendering for their own messages. Cheap once
  the renderer is already plumbed; just needs a toggle + branch in the
  bubble template.
- [ ] **Use full-height code blocks** toggle — webui truncates tall code
  blocks with an expand button by default; this disables that. We already
  have a code-expand dialog (`CodePreviewDialog`) but don't truncate inline
  code blocks. If we ever truncate by default, this is the escape hatch.
- [ ] **Auto-show sidebar on new chat** — when the user starts a fresh
  conversation from a collapsed-sidebar state, webui re-opens the sidebar
  so they can see the new entry appear. Minor UX polish.
- [ ] **Show raw model names** toggle — flip between the friendly profile
  name ("Qwen3-ASR") and the raw GGUF filename + quant info
  ("Qwen3-ASR-1.7B-Q8_0.gguf"). We show both sides in the toolbar, so this
  is mostly irrelevant; could be a density/cleanliness preference.

## 4. Settings — Developer tab (`settings_developer.png`)

LlamaChat has no Developer tab. Most of the contents are server-mediated
affordances that don't translate cleanly, but two are real:

- [!] **Pre-fill KV cache after response** — webui re-sends the conversation
  to the server right after a response so the prompt sits warm in the
  server's KV cache. LlamaChat's KV is already warm (in-process, persistent
  across turns). N/A.
- [!] **Disable server-side thinking extraction** — our reasoning extractor
  is client-side; the toggle controls what the llama.cpp server does with
  `reasoning_format`. Our equivalent is `ExtractReasoning` on the sampler
  panel, which we already expose.
- [ ] **Strip thinking from message history** — this is load-bearing for
  context budget: webui optionally drops prior `<think>` blocks from the
  transcript it sends back to the server, so only the final reasoning-free
  answers travel as context. LlamaChat currently sends the full transcript
  including reasoning (via `BuildTranscriptFor`). Worth exposing — on a long
  conversation with verbose `<think>` output, this could recover a
  meaningful fraction of the context window.
- [ ] **Enable raw output toggle** — per-bubble button that swaps the
  rendered markdown for the raw model output string, useful for debugging
  template artifacts. We have an "edit" button that does something similar
  (shows editable text) but not a read-only "show me the raw bytes" toggle.
- [!] **Custom JSON params** — HTTP-request payload override; N/A for
  in-process use. Our equivalent is `SamplerSettings` on the profile.

## 5. Settings — MCP tab (`settings_mcp.png`)

We have the server management surface. Gaps are in the loop-behaviour knobs:

- [ ] **Agentic loop max turns** user-configurable — hardcoded as
  `ToolCallMaxRounds = 6` in `MainWindowViewModel`. Webui defaults to 10 and
  exposes it as a numeric input. Lifting the constant to `AppSettings` is
  trivial.
- [ ] **Max lines per tool preview** — webui truncates tool-result previews
  after N lines by default (25). Ours uses character + line thresholds on
  `MessageViewModel.ShouldCollapseTool` but the thresholds are constants
  (`ToolCollapseLineThreshold=3`, `ToolCollapseCharThreshold=240`). A
  user-tunable line cap would make long JSON dumps less obnoxious.
- [ ] **Show tool call in progress** toggle — when on, auto-expands the tool
  bubble while the call is executing so you can watch it stream. Default
  off. We currently always collapse if the result would be long, with no
  special-case for in-flight.
- [~] **"Always show agentic turns in conversation"** toggle — I'm not sure
  what the off-state hides here (the whole tool turn? only tool-chips?);
  would want to experiment with webui to nail down the semantics before
  implementing.

## 6. Settings — Import/Export tab (`settings_import_export.png`)

- [ ] **Delete all conversations** (destructive bulk action) — we have
  individual delete with confirmation, but no bulk wipe. Would reuse
  `DialogService.ConfirmAsync` with a destructive choice; one-shot clears
  the store.

## 7. Sampling / Penalties tabs (`settings_sampling.png`, `settings_penalties.png`)

Feature parity here — we have every param shown (Temperature, dynatemp,
Top K/P, Min P, XTC, Typical P, DRY multiplier/base/allowed-length/last-n,
repeat/presence/frequency penalties, repeat-last-n). **No new gaps.**

Minor polish:

- [ ] **"(default: N)" placeholder in numeric inputs** — webui shows the
  default value as muted placeholder text inside each input so the user
  sees what the code default is without a tooltip. Our inputs show the
  persisted value only, which makes "has the user customised this?"
  non-obvious.

## 8. Global / navigation

- [~] **Settings icon in the top-right of every conversation view** — webui
  keeps the gear icon always visible in the corner, collapsed to the
  sidebar-toggle-and-gear pair on mobile. We have it on the toolbar; same
  access, slightly different muscle memory.

---

## Summary counts

| Category | TODO | Needs design | Won't |
|---|---|---|---|
| Compose / empty state | 2 | 0 | 0 |
| General tab | 2 | 3 | 2 |
| Display tab | 4 | 1 | 0 |
| Developer tab | 2 | 0 | 3 |
| MCP loop tuning | 3 | 1 | 0 |
| Import/Export | 1 | 0 | 0 |
| Sampling polish | 1 | 0 | 0 |
| **Total** | **15** | **5** | **5** |

## Highest-payoff next items

Ordered by "clicks-to-land × visible-daily-use":

1. **Auto-title from first user message** — first impression of every new
   conversation improves substantially; trivial implementation (set Title
   from Preview on first send if still the default).
2. **Strip thinking from message history** — biggest context-budget win for
   reasoning-heavy workflows (DeepSeek, Qwen3-Thinking, etc.).
3. **Agentic loop max turns** configurable — promote the hardcoded constant
   to `AppSettings`. 5 lines.
4. **Delete all conversations** — small, but users with accumulated test
   conversations can't easily reset today.
5. **Model pill near the compose box** — visual polish; makes the active
   model obvious at a glance without scanning the toolbar.
6. **Max lines per tool preview** — same constants-to-settings promotion as
   item 3; fixes the "tool dumped 500-line JSON into my chat" problem.
7. **Render user content as Markdown** toggle — small feature, useful for
   users who paste structured prompts.
