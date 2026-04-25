# llama-server parity — LlamaCpp.Bindings.Server burn-down

Reference: `llama-server --help` at the currently pinned llama.cpp
(`b8893-1-g86db42e97`, 2026-04-23), plus the runtime behaviour of its HTTP
endpoints.

This doc scopes to **the Server project** (`src/LlamaCpp.Bindings.Server`) —
what HTTP surface it exposes, which load/runtime flags it honours, and what
behaviour clients should expect. The sibling doc
[`llama_server_parity_checklist.md`](llama_server_parity_checklist.md)
covers the same territory for the LlamaChat desktop app (profile + UI
surfacing). Many binding items are shared; the states diverge because the
consumers diverge.

## Source-of-truth files

- **Endpoints**: `src/LlamaCpp.Bindings.Server/Endpoints/`
- **Services**: `src/LlamaCpp.Bindings.Server/Services/`
- **Config**: `src/LlamaCpp.Bindings.Server/Configuration/ServerOptions.cs`
- **DTOs**: `src/LlamaCpp.Bindings.Server/Models/`
- **Tests**: `src/LlamaCpp.Bindings.Tests/LlamaServerTests.cs`

## State legend

- `[x]` **done** — shipped and covered by tests.
- `[ ]` **TODO** — small lift; the binding already exposes it or the
  server just needs to surface a knob.
- `[~]` **needs binding work** — not yet in the managed wrapper, or in the
  wrapper but without the shape the server needs.
- `[#NN]` **tracked** — an open GitHub issue owns it; link in parens.
- `[!]` **won't implement** — deliberate non-goal for this project (single-
  model local server, not a multi-tenant service bus).

---

## 1. HTTP endpoints

The OpenAI-compatible surface plus llama-server's native routes.

- [x] **`GET /health`** — liveness probe. Always open even when API-key
  auth is configured.
- [x] **`GET /v1/models`** — lists the one loaded chat model. Future work
  adds the embedding model when configured.
- [x] **`GET /slots`** — pool snapshot: per-slot seq_id, in-use flag,
  cached-token count, last-used tick. Operator visibility into prompt-
  cache occupancy.
- [x] **`POST /v1/chat/completions`** — streaming (SSE) and non-streaming.
  Chat template applied from GGUF metadata.
- [x] **`POST /completion`** — llama-server's raw-text endpoint. No
  templating. Streaming + non-streaming.
- [x] **`POST /v1/embeddings`** — OpenAI-compatible. Requires a second
  (embedding) model configured via `EmbeddingModelPath`. Returns 501 when
  unconfigured so clients can tell "feature off" from "endpoint missing."
- [ ] **`POST /v1/rerank`** — reranker endpoint for models like
  bge-reranker / jina-rerank. Binding already surfaces `LlamaPoolingType.Rank`;
  this is an `EmbeddingHost`-shaped sibling with a different response
  shape (one score per query × document pair).
- [x] **`GET /metrics`** — Prometheus scrape target shipped in §11.
  Time-to-first-token histograms remain a follow-up (tracked under
  [#18](https://github.com/christopherthompson81/LlamaCpp.Bindings/issues/18)).
- [ ] **`GET /props`** — llama-server's "what model is loaded and how"
  dump. We expose most of the same info via `/v1/models` + `/slots`;
  question is whether to match the exact llama-server schema or ship our
  own shape.
- [~] **`POST /v1/completions`** — legacy (non-chat) OpenAI endpoint.
  Semantically close to our `/completion` but with OpenAI-style response
  shape. Low priority — callers that need a proper OpenAI-completions
  client almost always already use `/v1/chat/completions`.
- [!] **Tool-adjacent built-ins** (`--tools`, agentic built-in search):
  llama-server ships server-side tools that are a deliberate scope creep
  for a local runtime. Our position: tool-calling schema support in chat
  completions (so _client-side_ tool code works) is in scope; running
  tools inside the server is not.

## 2. Authentication / transport

- [x] **API keys** — `Authorization: Bearer <key>` (OpenAI idiom) and
  `X-Api-Key: <key>` (fallback) both accepted. Inline list + optional
  one-per-line file. `FixedTimeEquals` per key, no OR short-circuit.
- [x] **`/health` bypass** — probes don't ship credentials.
- [x] **Host / port binding** — `ServerOptions.Urls` → Kestrel
  `WebHost.UseUrls`. Same shape as
  `--host` / `--port` via URL.
- [x] **SSL / TLS** — set `ServerOptions.HttpsCertificatePath` (+
  `HttpsCertificatePassword`) to a PKCS#12 file; Kestrel's HTTPS
  defaults are configured to serve that cert. Pair with an
  `https://` URL in `ServerOptions.Urls` to actually listen on TLS.
- [x] **CORS** — `ServerOptions.CorsAllowedOrigins` (null/empty = off,
  `["*"]` = wildcard, otherwise exact-match list) +
  `ServerOptions.CorsAllowCredentials`. Middleware is ordered before
  the API-key auth so preflight OPTIONS requests (which don't carry
  the Authorization header) aren't 401'd.
- [!] **API-key rotation without restart** — the `ApiKeyAuth.LoadKeys`
  path reads at startup only. Rotate by redeploy. For a local server this
  is fine.

## 3. Chat-completion request features

Features clients expect from the OpenAI chat-completions API.

- [x] **`messages[]`** — user / assistant / system roles; chat template
  applied via `LlamaChatTemplate.Apply`.
- [x] **Streaming** (`stream: true`) — SSE with `delta.content` chunks,
  terminated by `data: [DONE]`.
- [x] **`temperature`**, **`top_p`**, **`top_k`**, **`seed`** — through
  `SamplerFactory.Build`.
- [x] **`logit_bias`** — OpenAI-style `{"tokenId": bias}` dict.
  Non-numeric keys / out-of-range ids reject with 400.
- [x] **`max_tokens`** — clamped to `ServerOptions.MaxOutputTokens`.
- [x] **Prompt cache hit** — `X-Cached-Tokens` response header; the
  session pool finds the longest common prefix against idle slots.
- [x] **`min_p`**, **`typical_p`**, **`top_n_sigma`**, **`xtc_*`**,
  **`dry_*`**, **`mirostat`/`mirostat_v2`**, **`repeat_penalty` /
  `frequency_penalty` / `presence_penalty` / `repeat_last_n`** — wired
  through `SamplerParams` into `SamplerFactory`. Mirostat overrides
  truncation + temperature (llama-server parity). Invalid mirostat
  values reject with 400.
- [x] **`stop` sequences** — accepts a single string or array (max 4,
  matching OpenAI). `StopMatcher` does streaming-safe partial-match
  detection with hold-back; stop string is stripped from returned
  content and `finish_reason="stop"`.
- [x] **`grammar`** / **`response_format`** / **`json_schema`** — all
  three shapes supported, precedence is raw `grammar` &gt;
  `json_schema` (llama-server shortcut) &gt; `response_format` (OpenAI
  envelope). `response_format.type` supports `"text"` / `"json_object"`
  (any-JSON grammar) / `"json_schema"` (compiled via
  `JsonSchemaToGbnf`). Malformed schema or unknown type returns 400
  before a pool slot is taken; GBNF parse errors likewise.
- [ ] **`logprobs`** / **`top_logprobs`** — needs per-token probability
  snapshot surfaced from the sampler. Binding touches the logits via
  `llama_get_logits_ith`; the sampler currently applies and selects but
  doesn't expose the per-token probability of the selected token. Small
  binding extension.
- [x] **Tool-calling schema** (`tools[]`, `tool_choice`) — DTOs
  accepted, tools rendered into the chat-template via the
  Jinja-side `tools` argument. `tool_choice = {"type":"function",
  "function":{"name":"X"}}` compiles a grammar from X's parameters
  schema and wraps the output as a single `tool_calls` entry on the
  response with `finish_reason="tool_calls"`. `tool_choice="required"`
  (any-tool) returns 400 for now — a GBNF union across every tool's
  schema isn't supported yet. `tool_choice="auto"` / unset passes
  tools through to the prompt but doesn't parse output for tool
  calls; clients have to do that themselves in V1.
- [x] **Multi-part content blocks** (text + image-url parts) — wired
  in §7. The text-only multipart path works regardless of mmproj
  availability (it flattens to a plain string); image parts require
  mmproj and otherwise reject with 400.
- [!] **`n > 1`** — multiple completions per request. Low priority; any
  client can issue N concurrent requests to the same endpoint and get
  the same result with better use of the session pool.

## 4. Completion (`/completion`) request features

- [x] **`prompt`**, **`n_predict`**, **`temperature`**, **`top_p`**,
  **`top_k`**, **`seed`**, **`logit_bias`**, and the full extended
  sampling set from §3 (min_p, typical_p, top_n_sigma, XTC, DRY,
  mirostat, penalties) — all covered via the shared `SamplerFactory`.
- [x] **Streaming** — SSE with `{content, stop}` chunks; terminates with
  a final `{content: "", stop: true, stop_reason, model, tokens_cached}`.
- [x] **`stop` sequences** — shares the `StopMatcher` with the chat
  endpoint; streaming path reports `stop_reason="stop"` in its final
  data chunk.
- [x] **`grammar`**, **`json_schema`**, **`response_format`** — share
  the `GrammarFactory` + `SamplerFactory` path with the chat endpoint.
- [ ] **`cache_prompt`** — currently always on (prompt cache is
  unconditional). llama-server lets clients opt out per request. Needs a
  bool field; when false, skip the LCP match and claim a fresh-state slot.
- [~] **`image_data`** — multimodal side of `/completion`. V1 ships
  images through the chat endpoint only; the raw `/completion` path
  has no template to splice a media marker into. Lower priority.

## 5. Embeddings (`/v1/embeddings`)

- [x] **`input` as string or array** — `JsonElement`-based parse normalises.
- [x] **Usage reporting** — `usage.prompt_tokens` / `total_tokens`.
- [x] **Model alias** — `EmbeddingModelAlias` surfaced in responses.
- [#15] **`encoding_format: "base64"`** — tracked in GH #15.
- [#16] **Batched multi-input decode** — tracked in GH #16. Current loop
  is sequential through a `SemaphoreSlim`.
- [#17] **`dimensions` truncation** — tracked in GH #17. OpenAI-style
  Matryoshka truncation + L2 renormalise.
- [ ] **`/v1/rerank`** — listed under §1 endpoints; shares most of this
  section's infrastructure.

## 6. Model loading

- [x] **`-m, --model`** — `ServerOptions.ModelPath`.
- [x] **`-c, --ctx-size`** — `ServerOptions.ContextSize`.
- [x] **`-b, --batch-size`** — `ServerOptions.LogicalBatchSize`.
- [x] **`-ub, --ubatch-size`** — `ServerOptions.PhysicalBatchSize`.
- [x] **`-np, --parallel`** — `ServerOptions.MaxSequenceCount`.
- [x] **`-ngl, --gpu-layers`** — `ServerOptions.GpuLayerCount`.
- [x] **`--mmap` / `--no-mmap`** — `ServerOptions.UseMmap`.
- [x] **`--mlock`** — `ServerOptions.UseMlock`.
- [x] **`-kvo / -nkvo, --kv-offload`** — `ServerOptions.OffloadKqv`.
- [ ] **`-ctk` / `-ctv` KV cache types** — binding has
  `LlamaKvCacheType` enum; just needs `EmbeddingKvCacheType` /
  `KvCacheTypeK` / `KvCacheTypeV` on `ServerOptions`.
- [ ] **`--threads` / `--threads-batch`** — binding exposes them; V1
  relied on llama.cpp defaults. Small add.
- [ ] **`-fa, --flash-attn`** — `LlamaFlashAttention` enum on
  `LlamaContextParameters`; add to `ServerOptions`.
- [ ] **`-sm, --split-mode`** / **`-mg, --main-gpu`** — on
  `LlamaModelParameters`; add to `ServerOptions`.
- [ ] **`--swa-full`** — on `LlamaContextParameters`; add to
  `ServerOptions`.
- [ ] **`--check-tensors`** — on `LlamaModelParameters`; add.
- [~] **`--rope-scaling`, `--rope-scale`, `--rope-freq-base`, all YARN
  knobs** — mostly needs managed property on `LlamaContextParameters`
  (binding row). Load-time-only; no per-request shape.
- [~] **`-dio, --direct-io`**, **`--numa`**, **`--no-host`**,
  **`--repack`**, **`-dev, --device`**, **`-ts, --tensor-split`**,
  **`-ot, --override-tensor`**, **`--cpu-moe`** — all block on binding
  work.
- [~] **`-mu, --model-url`** / **`-hf, --hf-repo`** — download manager.
  Significant design work (resumable, hash-verified, cache directory).
- [!] **`--embd-gemma-default`, `--fim-qwen-*-default`, etc.** — CLI
  shortcut presets. Our config system replaces these.

## 7. Multimodal

- [x] **`--mmproj FILE`** — `ServerOptions.MmprojPath`. Optional
  `MmprojHost` loads the projector on startup; chat completions
  accept OpenAI multi-part content with `image_url` parts (data:
  URLs only in V1; remote-URL fetch is a follow-up). The endpoint
  detects image presence, leases a fresh pool slot with its cache
  invalidated (image tokens in KV aren't recoverable from text),
  runs `MtmdContext.EvalPromptAsync` to prefill, then streams via
  `StreamFromCurrentStateAsync`.
- [x] **`--mmproj-offload`**, **`--image-min-tokens`**,
  **`--image-max-tokens`** — exposed as `MmprojOnCpu`,
  `MmprojImageMinTokens`, `MmprojImageMaxTokens`.
- [ ] **`--mmproj-auto`** — probe for a sibling mmproj file next to
  the chat model. Cheap filename convention check; not yet wired.
- [ ] **Remote-URL image fetch** — `image_url.url` fields with
  `http(s)://` schemes currently 400. Needs a download manager with
  size caps + content-type sniffing; follow-up to #19.

## 8. Speculative decoding

- [~] **`-md, --model-draft`** — binding ships
  `LlamaSpeculativeGenerator` (closed in GH #4). The server side hasn't
  wired it in: `ModelHost` would need an optional draft model +
  generator selection in the chat-completions endpoint, plus per-request
  opt-in (`speculative: true`?) so greedy-sensitive callers can disable
  it.
- [#14] **Full DeepMind rejection-sampling variant** — tracked in
  GH #14. Current path is greedy verification.

## 9. LoRA adapters

- [~] **`--lora` / `--lora-scaled`** — binding has
  `LlamaLoraAdapter` + `LlamaContext.AttachLoraAdapter`. Server config
  could accept a list of `{path, scale}` pairs attached at startup.
  Per-request adapter selection (different LoRA per caller) needs more
  design — the current binding attaches adapters to the context, not the
  session, so concurrent requests on one context share adapters.
- [~] **`--control-vector`** — separate binding work.

## 10. Sampling knobs not yet in chat/completion requests

- [x] min_p, typical_p, top_n_sigma, XTC, DRY (§3 / §4)
- [x] mirostat, mirostat_v2 (§3 / §4)
- [x] repeat_penalty / frequency_penalty / presence_penalty, repeat_last_n
- [ ] adaptive_p terminal — binding exposes `WithAdaptiveP`; parity
  would be a third terminal-selection branch in `SamplerFactory`.
- [ ] Dynamic temperature (`dynatemp_range`, `dynatemp_exponent`) —
  binding has `WithExtendedTemperature`; low priority.
- [ ] Custom sampler ordering (`samplers`, `sampler_seq`) — current
  builder applies a fixed order; exposing it requires a sampler-
  builder change.

## 11. Observability + lifecycle

- [x] **`X-Cached-Tokens` header** — per-request cache-hit count.
- [x] **Cancellation on client disconnect** — `HttpContext.RequestAborted`
  propagates to the generator loop; release returns the slot. Regression
  test in `Cancelled_Stream_Releases_Pool_Slot`.
- [x] **Startup logging** — model path, GPU layers, pool size, loaded
  embedding model (when applicable).
- [x] **`/metrics`** — Prometheus scrape target. Counters:
  `llama_requests_total{endpoint,status}`, `llama_tokens_generated_total`,
  `llama_tokens_prompt_total`, `llama_tokens_cached_total`. Gauges:
  `llama_slot_in_use{slot_id}`, `llama_slot_cached_tokens{slot_id}`.
- [x] **Per-request timing in response** — `timings` sub-object on
  chat + completion responses (and in the final SSE chunk on
  streaming): `prompt_n`, `prompt_ms`, `prompt_per_token_ms`,
  `predicted_n`, `predicted_ms`, `predicted_per_token_ms`, `cached_n`.
- [x] **Request access log** — ASP.NET `UseHttpLogging` wired with
  method + path + status + duration on every request.
- [!] **Built-in dashboard / web UI** — LlamaChat exists; it can be
  pointed at this server. Not duplicating that inside the server binary.

## 12. Server-side safety / ops

- [ ] **Max prompt tokens per request** — `MaxOutputTokens` guards the
  generation budget but nothing bounds prompt length. A bad client could
  send a prompt that doesn't fit in the context and get an unhelpful
  error. A pre-tokenise length check with a clear 413 would be nicer.
- [ ] **Request timeouts** — long generations currently ride purely on
  client-side timeouts. A server-side upper bound (e.g. 5 minutes) would
  prevent stuck requests.
- [ ] **Graceful shutdown** — SIGTERM should drain in-flight requests
  rather than drop them mid-stream.

---

## Open GitHub issues

| # | Title | Section |
|---|---|---|
| 14 | Speculative decoding: implement full DeepMind rejection-sampling protocol | §8 |
| 15 | /v1/embeddings: support encoding_format="base64" | §5 |
| 16 | /v1/embeddings: batch multiple inputs into one forward pass | §5 |
| 17 | /v1/embeddings: support OpenAI's `dimensions` truncation parameter | §5 |

## Summary counts

| State | Count | Meaning |
|---|---|---|
| `[x]` done | 47 | shipped, tested |
| `[ ]` TODO | 11 | binding already exposes; server-side wiring only |
| `[~]` needs binding | 14 | binding work first |
| `[#NN]` tracked | 4 | dedicated issue |
| `[!]` won't | 6 | explicit non-goal |

## Recommended order of attack

Weighed by user-visible impact per unit of work, with the understanding
that we've already hit the big items (multi-session, prompt caching,
embeddings, auth, observability, cancellation, extended sampling).

1. **Logprobs / top_logprobs** (§3) — useful for evals; tracked in
   [#20](https://github.com/christopherthompson81/LlamaCpp.Bindings/issues/20).
2. **Remote-URL image fetch** (§7) — the multimodal follow-up; needs
   a size-capped download manager. Tracked alongside #19.
3. **`tool_choice="required"` (any-tool)** (§3) — needs a GBNF union
   across every tool's parameters schema. Follow-up to #21.
4. **Auto-mode tool-call parsing** (§3) — detect Qwen3
   `<tool_call>...</tool_call>` / Llama-3.1 `<|python_tag|>` /
   Mistral `[TOOL_CALLS]` wrappers in plain output. Follow-up to #21.

Everything under `[~]` is binding-side work of varying size. Speculative
decoding (§8) and LoRA (§9) are the two most feature-complete on the
binding side — they could be wired into the server whenever we decide
the shape of "how does a client opt in."
