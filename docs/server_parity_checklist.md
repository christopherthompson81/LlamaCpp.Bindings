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
- [x] **`POST /v1/rerank`** — Cohere/Jina-style reranker endpoint
  for models like bge-reranker. Loads a third optional model via
  `RerankHost` (parallel to `EmbeddingHost`); request takes a
  query + array of documents + optional `top_n`, returns indices
  sorted by `relevance_score` descending. Used `LlamaContextParameters.PoolingType
  = Rank` (newly exposed binding-side knob — BGE rerankers don't
  always advertise rank pooling in metadata) and the explicit
  `RunEncoder` path because BGE-reranker is XLMRoberta encoder-only,
  not the encoder-decoder branch `EncodeForEmbedding` falls through.
- [x] **`GET /metrics`** — Prometheus scrape target shipped in §11.
  Time-to-first-token histograms remain a follow-up (tracked under
  [#18](https://github.com/christopherthompson81/LlamaCpp.Bindings/issues/18)).
- [!] **`GET /props`** — out of scope. The only consumer is llama.cpp's
  bundled web UI, which we don't ship; OpenAI-compatible frontends
  (Continue, Open WebUI, openai-python) never touch it. Matching the
  exact schema would also force us to chase upstream's drift every
  release. We surface what callers actually need via `/v1/models` and
  `/slots` and leave it at that.
- [!] **`POST /v1/completions`** — out of scope. The legacy non-chat
  OpenAI endpoint is functionally redundant with our `/completion` (raw
  prompt in, text out); the only thing missing is the OpenAI response
  envelope, and callers that want OpenAI compatibility already speak
  `/v1/chat/completions`. Adding a second endpoint with its own DTO
  layer for marginal client coverage isn't worth the surface area.
- [!] **Tool-adjacent built-ins** (`--tools`, agentic built-in search):
  llama-server ships server-side tools that are a deliberate scope creep
  for a local runtime. Our position: tool-calling schema support in chat
  completions (so _client-side_ tool code works) is in scope; running
  tools inside the server is not.
- [x] **`POST /tokenize`** — text → token IDs using the loaded vocab.
  Optional `with_pieces` returns each token's rendered text alongside
  its id (useful for debug UIs).
- [x] **`POST /detokenize`** — token IDs → text. Round-trips
  `/tokenize` output.
- [#28] **`POST /v1/audio/transcriptions`** — Whisper-compatible audio
  transcription. Tracked as GH #28; needs multipart parsing + audio
  mmproj integration.
- [#29] **`POST /infill`** — code-completion fill-in-the-middle for
  Qwen-Coder / DeepSeek-Coder / similar. Tracked as GH #29; reuses
  the existing completion pipeline once the FIM prompt is formatted.
- [#33] **`POST /v1/messages`** (Anthropic) — re-promoted from "won't"
  after honest review. The external schema doesn't fully overlap
  with OpenAI (system field, content blocks, typed streaming events,
  tool-use/tool-result block shape), so Anthropic-SDK clients can't
  transparently speak our existing `/v1/chat/completions`. Tracked
  in GH #33.
- [!] **`POST /v1/responses`** (OpenAI Responses) — newer OpenAI
  endpoint with low adoption today. Reasonable to wait until a real
  client demand surfaces.
- [!] **`GET /cors-proxy`** — experimental upstream feature for the
  bundled web UI's MCP support. Won't ship.
- [!] **`POST /apply-template`** — render a chat template against a
  given message list and return the prompt text. Niche debug tool;
  the same data is one chat-completions call away.
- [!] **`GET/POST /lora-adapters`** — runtime LoRA hotswap. We
  attach LoRA at startup; per-request adapter selection is a separate
  binding-design concern (the binding attaches LoRA to the context,
  not the session).
- [!] **`POST /slots/{id}`** — runtime save/load of slot KV state to
  disk. Niche; conflicts with our SessionPool's transparent prefix-
  matching design.

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
- [x] **`logprobs`** / **`top_logprobs`** — chat completions accept
  both fields. `LlamaGenerator.GenerateAsync` gained `logprobsTopN`
  + an `onLogprobs` callback; the generator computes a numerically-
  stable log-softmax over the full vocab under the same lock as the
  sample call (so the logits are still fresh) and emits a
  `TokenLogprobInfo` per generated token. The server collects these
  and shapes them as OpenAI's `choices[].logprobs.content[]` with
  per-token `bytes` arrays. Streaming honours the field too — each
  content-bearing chunk carries the logprobs for the tokens it
  produced. `top_logprobs` capped at 20 to match OpenAI.
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
- [#30] **`t_max_predict_ms`** — per-request generation timeout
  separate from the global `RequestTimeoutSeconds`. Tracked as GH #30.
- [#31] **`logprobs` as integer** — llama-server accepts both the
  OpenAI boolean form and an integer `n_probs` alias. Our DTO is
  boolean-only. Tracked as GH #31.
- [#34] **`return_progress`** — re-promoted from "won't" after
  honest review. Only signal a streaming client gets during prefill;
  silent-streaming gap on long prompts is a real UX hole. Tracked in
  GH #34.
- [#35] **`return_tokens`** — re-promoted from "won't" after honest
  review. Useful for KV-cache-aware clients (Continue.dev,
  speculative-decoding wrappers) that benefit from token-level
  identity across requests. Tracked in GH #35.
- [!] **`response_fields`, `n_indent`, `n_cache_reuse`,
  `post_sampling_probs`, `backend_sampling`** — niche per-request
  knobs surfaced by the code-audit; declined as not worth the DTO
  surface unless a real client asks. Each is independently trivial
  to add later if needed.
- [!] **Per-request `lora`, per-request `speculative.*`** — both ride
  on context-level state in the binding. Per-request adapter
  selection has known design surface (binding attaches at context,
  not session); per-request speculative tuning would need extending
  `LlamaSpeculativeGenerator` to take per-call lookahead. Both
  deferred behind concrete use cases.
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
- [x] **`cache_prompt`** — `cache_prompt` field on chat + completion
  requests (default `true`). When `false`, `SessionPool.LeaseAsync`
  forces match=0 across all free slots and tiebreaks to LRU, so the
  request runs against the coldest available slot and `X-Cached-Tokens`
  reports 0. Useful for determinism testing.
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
- [x] **`/v1/rerank`** — shipped (§1).
- [x] **`embd_normalize`** — request-time vector normalisation: <c>2</c>
  (default L2), <c>1</c> (L1), <c>0</c> (off / preserve encoder
  output), negative (defer to encoder).
- [x] **`return_text`** — TEI-format compat: when <c>true</c>, each
  embedding entry carries the original input string. Backwards-
  compatible extension to the OpenAI shape (clients ignore unknown
  fields).
- [#32] **TEI-format rerank** (`texts` request field, bare-array
  response, `score` instead of `relevance_score`) — tracked in GH #32.
  Our current rerank speaks Cohere/Jina shape; full TEI compat needs
  response-shape switching.

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
- [x] **`-ctk` / `-ctv` KV cache types** — `ServerOptions.KvCacheTypeK`
  / `KvCacheTypeV`. Quantised options need Flash Attention; the
  binding's `LlamaKvCacheType` enum lists every accepted variant.
- [x] **`--threads` / `--threads-batch`** — `ServerOptions.ThreadCount`
  / `BatchThreadCount`. `-1` (default) inherits llama.cpp's choice.
- [x] **`-fa, --flash-attn`** — `ServerOptions.FlashAttention`
  (`Auto` / `Enabled` / `Disabled`).
- [x] **`-sm, --split-mode`** / **`-mg, --main-gpu`** —
  `ServerOptions.SplitMode` (`None` / `Layer` / `Row`) and
  `ServerOptions.MainGpu`.
- [x] **`--swa-full`** — `ServerOptions.UseFullSwaCache`. Defaults to
  `true` to match the binding's opinionated default — full SWA lets
  the KV cache be edited (multi-turn chat with retries).
- [x] **`--check-tensors`** — `ServerOptions.CheckTensors`.
- [x] **`--rope-scaling`, `--rope-scale`, `--rope-freq-base`, all YARN
  knobs** — `LlamaContextParameters` exposes
  `RopeScalingType` (with the new public `LlamaRopeScalingType` enum:
  Unspecified / None / Linear / Yarn / Longrope), `RopeFreqBase`,
  `RopeFreqScale`, `YarnExtFactor`, `YarnAttnFactor`, `YarnBetaFast`,
  `YarnBetaSlow`, `YarnOriginalContext`. `ServerOptions` mirrors the
  whole set; `ModelHost` and `DraftHost` route through the same
  `BuildContextParameters` helper so both contexts pick up identical
  values. Defaults inherit GGUF metadata.
- [x] **`-dio, --direct-io`** — `ServerOptions.UseDirectIo`.
- [x] **`--no-host`** — `ServerOptions.NoHost`.
- [x] **`--repack`** — `ServerOptions.UseExtraBufts` (true by default,
  matches `llama_model_default_params`).
- [x] **`-dev, --device`** — `ServerOptions.Devices` (list of names
  resolved against `LlamaHardware.EnumerateDevices` at startup; bad
  names fail fast with the available list). The binding's
  `LlamaComputeDevice` now exposes its underlying
  `ggml_backend_dev_t` so callers can pin a specific device list.
- [x] **`-ts, --tensor-split`** — `ServerOptions.TensorSplit` (per-
  device proportions; padded to `llama_max_devices()` internally).
- [x] **`--numa`** — `ServerOptions.NumaStrategy`. ModelHost calls
  `LlamaBackend.InitializeNuma` after `Initialize()` but before any
  model load, so NUMA placement is consistent across every loaded
  model (main + embedding + rerank + draft). No-op on non-NUMA
  hardware and at the default <c>Disabled</c>.
- [x] **`-ot, --override-tensor`** —
  `ServerOptions.TensorBuftOverrides` (list of
  `{Pattern, Device, Host}`). Resolved at startup against
  `LlamaHardware.EnumerateDevices`; bad device names fail fast. Pairs
  with the new public `LlamaBufferType` wrapper +
  `LlamaTensorBuftOverride` record on the binding side.
- [x] **`--cpu-moe`** — `ServerOptions.CpuMoe` (bool). Implemented as
  a preset that appends llama.cpp's canonical regex
  `\.ffn_(up\|down\|gate\|gate_up)_(ch\|)exps` mapped to the CPU
  device's primary buft.
- [!] **`-mu, --model-url`** / **`-hf, --hf-repo`** — out of scope.
  Treated as a client concern: getting a GGUF onto disk is a separate
  problem from serving it, and operators have well-understood tools
  (huggingface-cli, wget, rclone, container-image bakes) that already
  do this better than an ad-hoc in-server downloader. The server's
  surface stays "load this path" — fewer moving pieces, no resume /
  hash / cache-directory state to maintain inside the inference
  process.
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
- [x] **`--mmproj-auto`** — `ServerOptions.MmprojAuto`. When
  <c>MmprojPath</c> is unset and this is <c>true</c>, look for a
  single sibling matching <c>mmproj-*.gguf</c> or
  <c>*.mmproj.gguf</c>. Multiple matches log a warning and skip
  (forcing the operator to disambiguate); zero matches falls through
  to "no mmproj loaded".
- [#27] **Remote-URL image fetch** — tracked in GH #27 with the SSRF /
  size-cap / content-type-sniff design considerations spelled out.

## 8. Speculative decoding

- [x] **`-md, --model-draft`** — `ServerOptions.DraftModelPath` (with
  `DraftContextSize`, `DraftLogicalBatchSize`, `DraftPhysicalBatchSize`,
  `DraftGpuLayerCount`, `DraftLookahead`). Optional `DraftHost`
  loads the draft model + a dedicated speculative main context on
  startup; chat completions accept a per-request `speculative: true`
  flag that routes through `LlamaSpeculativeGenerator`. Falls back to
  the standard path when a request also uses images, forced tool
  calls, or per-token logprobs (none of which the speculative path
  carries in V1). Speculative requests serialize through a dedicated
  semaphore (concurrency = 1) and bypass the SessionPool — no prefix
  caching for these requests.
- [#14] **Full DeepMind rejection-sampling variant** — tracked in
  GH #14. Current path is greedy verification.

## 9. LoRA adapters

- [x] **`--lora` / `--lora-scaled`** —
  `ServerOptions.LoraAdapters` is a list of `{Path, Scale}` entries
  loaded against the main model and attached to the shared context at
  startup. Bad paths / shape mismatches surface eagerly (operators see
  the failure at boot, not on the first request). Speculative
  requests inherit the same adapters because the dedicated
  speculative main context shares the underlying `LlamaModel`. The
  draft model is intentionally left untouched. Per-request adapter
  selection (different LoRA per caller) is deferred — the binding
  attaches adapters to the context, not the session, so concurrent
  requests on one context share adapters; per-call switching needs
  more design.
- [x] **`--control-vector`** /
  **`--control-vector-scaled`** /
  **`--control-vector-layer-range`** —
  `ServerOptions.ControlVectors` (list of `{Path, Scale}` entries),
  `ControlVectorLayerStart`, `ControlVectorLayerEnd`. ModelHost loads
  each file, sums them element-wise, and attaches the merged vector
  to the shared context with the configured layer range. DraftHost's
  speculative main context mirrors the same data so speculative
  requests inherit the steering. Binding row: a public
  `LlamaControlVector` (with `LoadFromFile` + `Combine`) plus
  `LlamaContext.SetControlVector` / `ClearControlVector`. The loader
  uses `gguf_init_from_file(no_alloc=true)` to parse the header +
  tensor offsets, then reads F32 bytes directly from the file —
  avoids dragging in the ggml allocator API.

## 10. Sampling knobs not yet in chat/completion requests

- [x] min_p, typical_p, top_n_sigma, XTC, DRY (§3 / §4)
- [x] mirostat, mirostat_v2 (§3 / §4)
- [x] repeat_penalty / frequency_penalty / presence_penalty, repeat_last_n
- [x] adaptive_p terminal — `adaptive_p_target` + `adaptive_p_decay` on
  chat + completion. When `target ≥ 0`, replaces the
  greedy/distribution terminal with `WithAdaptiveP`. Mirostat (when
  set) still wins.
- [x] Dynamic temperature — `dynatemp_range` + `dynatemp_exponent` on
  chat + completion. When `range > 0`, the temperature stage uses
  `WithExtendedTemperature(temp, range, exponent)` instead of plain
  `WithTemperature(temp)`.
- [x] Custom sampler ordering — `samplers` (a string list) on chat +
  completion. When non-null, only the named stages run, in the order
  given. Allowed names: `dry`, `top_k`, `top_p`, `min_p`, `typical_p`,
  `top_n_sigma`, `xtc`, `temperature`. Unknown names → 400. Stages
  whose params are absent silently skip. Penalties + logit bias stay
  at the head; terminal selection stays at the tail.

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
- [x] **`--endpoint-metrics` / `--no-slots`** —
  `ServerOptions.ExposeMetricsEndpoint` and `ExposeSlotsEndpoint`,
  both default `true`. Operators running multi-tenant deployments
  behind a TLS-terminating proxy that doesn't forward the API key
  can disable these so per-slot cache state and request counters
  aren't visible to unauthenticated observers.

## 12. Server-side safety / ops

- [x] **Max prompt tokens per request** — `ServerOptions.MaxPromptTokens`.
  Pre-tokenise check on chat + completion endpoints rejects oversize
  prompts with HTTP 413 and an explanatory body before any pool slot is
  taken. `0` derives the cap from `ContextSize - MaxOutputTokens`.
- [x] **Request timeouts** — `ServerOptions.RequestTimeoutSeconds`
  (default 300). Each request gets a linked CTS combining the client's
  abort token with a server-side timer; non-streaming requests return
  HTTP 504 on timeout, streaming requests close the connection so
  clients observe end-of-stream early.
- [x] **Graceful shutdown** — `ServerOptions.ShutdownDrainSeconds`
  (default 30) wired into `HostOptions.ShutdownTimeout` so SIGTERM
  drains in-flight requests rather than dropping them mid-stream.

---

## Open GitHub issues

Tracked as separate items because each has enough scope (or external
dependency) to deserve its own thread:

| # | Title | Section |
|---|---|---|
| 14 | Speculative decoding: full DeepMind rejection-sampling protocol | §8 |
| 15 | /v1/embeddings: encoding_format="base64" | §5 |
| 16 | /v1/embeddings: batch multiple inputs into one forward pass | §5 |
| 17 | /v1/embeddings: `dimensions` truncation parameter | §5 |
| 22 | Server: SSL/TLS end-to-end integration test | §2 |
| 23 | Server: streaming format for forced tool calls | §3 |
| 24 | Server: detect tool-call wrappers in auto-mode chat output | §3 |
| 25 | Server: render assistant tool_calls in history through the chat template directly | §3 |
| 26 | Server: tool_choice="required" (any-tool) needs GBNF union across schemas | §3 |
| 27 | Server: remote-URL image fetch for /v1/chat/completions | §7 |
| 28 | Server: /v1/audio/transcriptions endpoint (Whisper-compatible) | §1 |
| 29 | Server: POST /infill endpoint (FIM code completion) | §1 |
| 30 | Server: per-request t_max_predict_ms timeout | §3 |
| 31 | Server: accept logprobs as integer (n_probs alias) | §3 |
| 32 | Server: TEI response shape on /v1/rerank | §5 |
| 33 | Server: POST /v1/messages (Anthropic Messages API compat) | §1 |
| 34 | Server: return_progress for prefill-phase progress events | §3 |
| 35 | Server: return_tokens for KV-cache-aware clients | §3 |

## Summary counts

| State | Count | Meaning |
|---|---|---|
| `[x]` done | 84 | shipped, tested |
| `[ ]` TODO | 0 | binding already exposes; server-side wiring only |
| `[~]` needs binding | 0 | binding work first |
| `[#NN]` tracked | 18 | dedicated issue |
| `[!]` won't | 11 | explicit non-goal |

## Remaining `[ ]` TODO items

None. Every "binding already exposes; server-side wiring only" row is
shipped. What's left lives in the binding-blocked, tracked-issues, and
non-goal columns above.

Deliberate skips: none currently. Earlier-skipped rows
(`cache_prompt`, `--mmproj-auto`) shipped after re-evaluation;
remote-URL image fetch was reclassified as a real protocol gap and
moved to GH #27.

## Recommended order of attack

Big features (multi-session, prompt caching, embeddings, auth,
observability, cancellation, extended sampling, multimodal, tool
calling, logprobs, rerank) are all shipped. What's left is operator-
ergonomic polish + binding-blocked features.

Every row outside the dedicated-issue and won't-implement columns is
shipped. The current set of tracked issues is the result of two
complementary surveys: the original `--help`-derived checklist
captured the documented surface, and a follow-up code-level audit of
`tools/server/` caught endpoints (audio transcription, /infill,
/tokenize, /detokenize) and request fields (`embd_normalize`,
`return_text`, integer-form `logprobs`, `t_max_predict_ms`) that
weren't in `--help`. The audit's findings are split across rows in
this checklist and dedicated GH issues #28-#32.

Largest remaining items: #14 (DeepMind rejection-sampling for
speculative), #28 (audio transcriptions, needs binding work for the
multipart + audio mmproj path), #27 (remote-URL image fetch with SSRF
hardening). Smallest: #22 (SSL e2e test), #30 (per-request timeout).

Parity, in the sense of "any feature llama-server has that we don't
deliberately decline," is done at the surface layer; the open issues
are scope-limited follow-ups rather than gaps.
