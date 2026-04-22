# llama-server webui parity investigation

Tracking the work to bring `src/LlamaCpp.Bindings.LlamaChat` — an Avalonia desktop
chat UI built on the in-process bindings — up to feature parity with
`llama.cpp/tools/server`'s webui. The existing `samples/LlamaChat` remains the
small, minimum-viable demo; this new project is the "serious" client.

## Scope decisions

**v1 target: text-only, single-model, single-conversation chat with the sampler
knobs a power user actually turns.** Multimodal, LoRA hotswap, and speculative
decoding are phase 2 — all three require new native surface area and are easier
to slot in once the app structure is stable.

### In scope (v1)

- Streaming chat with full message history
- Full sampler UI: temperature (static + dynatemp), top-k/p, min-p, typical,
  XTC, DRY, repetition/presence/frequency penalties, mirostat v1/v2, seed
- Grammar + JSON-schema-driven output (user-provided GBNF; schema→GBNF
  conversion deferred)
- `<think>` / reasoning block extraction, displayed collapsible
- Prompt-prefix reuse across turns (avoid re-prefilling on every send)
- Conversation persistence to disk (JSON, not KV snapshot — restoration
  replays history)
- Top-N logprobs panel per emitted token
- Chat template: use model's embedded template when it matches a built-in;
  fall back to curated list otherwise. Tool-use templates noted as broken
  until (2) below is resolved.
- Per-turn timing + tokens/sec display

### Out of scope for v1 (tracked below)

- Multimodal input (vision, audio, PDF-as-image)
- Tool calling (needs full Jinja)
- LoRA adapter hotswap
- Speculative decoding with a draft model
- KV cache save/restore to disk (we rebuild from transcript instead)
- Reranking, infill helper APIs
- MCP integration

## Native binding gaps, ranked

| # | Gap | Why it blocks webui parity | v1? | Estimate |
|---|-----|----------------------------|-----|----------|
| 1 | `mtmd_*` / `clip_*` — image/audio/PDF encoding | No multimodal input at all | no | L |
| 2 | Full Jinja template renderer | Tool-use templates need it; curated list covers only plain chat | partial | L |
| 3 | Reasoning tag config per template | Separate "thinking" display | yes (hardcoded tags) | S |
| 4 | `llama_adapter_lora_*` | No runtime adapter swap | no | M |
| 5 | `llama_state_seq_{save,load}_file` | No durable KV snapshot; forces replay on resume | no | M |
| 6 | Prefix-cache detection helper in `LlamaGenerator` | Full re-prefill every turn is O(total tokens) | yes | M |
| 7 | Top-N candidate exposure from sampler | Logprobs panel needs it | yes | M |
| 8 | Draft-model coordination / speculative sampler | Perf feature, not functional | no | L |
| 9 | Infill / FIM helper | Low-level pieces exist, no high-level API | no | S |
| 10 | JSON-schema → GBNF converter | Webui accepts raw schema; we accept GBNF only | no (v2) | M |

Gaps 3, 6, 7 are the v1 work items on the bindings side. Everything else is
user-space composition against the existing native surface.

## Run log

### Run 1 — 2026-04-22 (afternoon)

**Goal:** establish scope and stand up the app skeleton.

**What ran:**

- Surveyed `tools/server/webui/` (Svelte app) and the server's HTTP surface
  (`server.cpp`, `server-context.cpp`, `server-task.cpp`) to enumerate
  user-facing features.
- Cross-referenced against the C# binding surface in
  `src/LlamaCpp.Bindings/` — specifically `NativeMethods.cs` (131 P/Invoke
  decls), the public API of `LlamaModel/Context/Generator/Sampler`, and the
  sampler builder.

**Findings:**

- Existing bindings already cover the entire sampler grid the webui exposes,
  plus grammar + lazy grammar with triggers. No P/Invokes needed for the
  sampler panel.
- `LlamaChatTemplate.Apply` routes through `llama_chat_apply_template`, which
  handles llama.cpp's curated named templates. Arbitrary Jinja templates
  shipped in custom GGUFs (especially tool-use variants) are not supported.
  Confirmed by comments in `LlamaChatTemplate.cs` and observed behavior in
  `samples/LlamaChat/ViewModels/MainWindowViewModel.cs:177-195` where the
  sample explicitly falls back to naked `role: content` concat when no
  template is embedded.
- `LlamaGenerator` currently takes a fresh prompt and has no concept of
  "what's already in the KV cache." `samples/LlamaChat` sidesteps this by
  calling `ClearKvCache()` every turn and re-decoding the full conversation
  (see note at line 197). Acceptable for demo, unacceptable for serious use
  past ~1k tokens of history.
- Logprobs: `LlamaSampler.Sample` returns the picked token only; the
  candidate array is internal. Exposing top-N requires either a new overload
  or a separate `SampleWithCandidates` returning `(token, IReadOnlyList<(int
  token, float prob)>)`.

**Decision:** build the app against the current bindings. Work items 3
(reasoning tags), 6 (prefix caching), and 7 (logprobs surfacing) will land
as small additions to the bindings as the app reaches the features that
need them — track each as its own run below. Multimodal and LoRA are
deferred to a separate investigation doc when we start phase 2.

**Next:** scaffold `src/LlamaCpp.Bindings.LlamaChat` with MVVM, wire a basic
streaming chat loop, confirm it builds and runs against a real model. Iterate
on the sampler panel and logprobs next.
