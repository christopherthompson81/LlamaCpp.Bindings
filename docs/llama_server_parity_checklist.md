# llama-server model-loading parity — LlamaChat burn-down

Reference: `llama-server --help` as of llama.cpp `b8893-1-g86db42e97` (2026-04-23).
Sampling-side parity lives in `docs/webui_feature_checklist.md` §6 — this
doc covers **loading, runtime, and model-source** parameters only.

## Source-of-truth files

- **Binding surface**: `src/LlamaCpp.Bindings/LlamaModelParameters.cs`, `LlamaContextParameters.cs`
- **Profile persistence**: `src/LlamaCpp.Bindings.LlamaChat/Models/ModelLoadSettings.cs`
- **UI**: `src/LlamaCpp.Bindings.LlamaChat/Views/ProfileEditorView.axaml`

## State legend

- `- [x]` **done** — wired through binding → profile → UI, visible to the user.
- `- [ ]` **TODO** — binding already exposes it (or trivially can); just needs a profile field + UI control. Small scope.
- `- [~]` **needs binding** — llama.cpp native struct or function isn't mirrored yet; pull into the binding first.
- `- [!]` **won't implement** — deliberate out-of-scope for a single-user desktop chat app (see per-item note).
- `- [-]` **N/A** — server-only concept (HTTP, API keys, slots, router, CLI shortcuts, logging flags) that has no analog in an in-process Avalonia client.

---

## 1. CPU / threading

- [ ] **`-t, --threads`** — `LlamaContextParameters.ThreadCount` exists; needs a profile field + UI spinner.
- [ ] **`-tb, --threads-batch`** — same (`BatchThreadCount`).
- [~] **`-C, --cpu-mask`** — needs binding for `cpuparams_t.mask` + ops to apply it.
- [~] **`-Cr, --cpu-range`** — same cpuparams plumbing as above.
- [~] **`--cpu-strict`** — same.
- [~] **`--prio`** — process/thread priority; needs `ggml_threadpool_params` bridge.
- [~] **`--poll`** — polling level; same plumbing.
- [~] **`-Cb --cpu-mask-batch` / `-Crb --cpu-range-batch` / `--cpu-strict-batch` / `--prio-batch` / `--poll-batch`** — all batch-side variants of the above; land together once the cpuparams binding exists.

## 2. Context + batch sizing

- [x] **`-c, --ctx-size`** — `ModelLoadSettings.ContextSize`, NumericUpDown in the profile editor.
- [x] **`-n, --predict`** — `SamplerPanelViewModel.MaxTokens`, exposed on the Sampling tab.
- [x] **`-b, --batch-size`** — `ModelLoadSettings.LogicalBatchSize`.
- [x] **`-ub, --ubatch-size`** — `ModelLoadSettings.PhysicalBatchSize`.
- [~] **`--keep N`** — "tokens to keep from initial prompt"; needs `LlamaGenerator` to accept a keep count before prefill truncation.
- [ ] **`--swa-full`** — binding exposes `LlamaContextParameters.UseFullSwaCache` (defaults to true); profile + UI toggle.

## 3. Attention + RoPE

- [x] **`-fa, --flash-attn`** — `ModelLoadSettings.FlashAttention` (ComboBox Auto/Off/On).
- [ ] **`--rope-scaling {none,linear,yarn}`** — binding has `llama_rope_scaling_type` enum but no public property on `LlamaContextParameters`; add + surface in UI.
- [ ] **`--rope-scale`** — same — `rope_freq_scale` field already on native struct.
- [ ] **`--rope-freq-base`** — same — `rope_freq_base`.
- [ ] **`--rope-freq-scale`** — duplicate of `--rope-scale`; wire once.
- [~] **`--yarn-orig-ctx`** — native struct has `yarn_orig_ctx`; binding-side public API doesn't expose it yet.
- [~] **`--yarn-ext-factor`** — same (`yarn_ext_factor`).
- [~] **`--yarn-attn-factor`** — same (`yarn_attn_factor`).
- [~] **`--yarn-beta-slow` / `--yarn-beta-fast`** — same (`yarn_beta_fast/slow`).

## 4. Memory + file I/O

- [x] **`--mlock`** — `ModelLoadSettings.UseMlock`.
- [x] **`--mmap, --no-mmap`** — `ModelLoadSettings.UseMmap`.
- [~] **`-dio, --direct-io`** — native `llama_model_params.use_direct_io` exists but the managed wrapper doesn't surface it; add.
- [~] **`--numa TYPE`** — `ggml_numa_strategy` enum is mirrored; `NativeMethods.llama_numa_init` needs to be bound + called at backend init. Load-time, not per-model.
- [~] **`--no-host`** — native `use_host_buffer` flag; not in wrapper.

## 5. GPU / offload

- [x] **`-ngl, --gpu-layers N`** — `ModelLoadSettings.GpuLayerCount`; integer only today.
- [ ] **`-ngl auto | all`** — sentinel values on top of the numeric path; decide on a UI shape (ComboBox "Auto / All / N layers" with a conditional spinner).
- [ ] **`-sm, --split-mode`** — binding has `LlamaSplitMode` on `LlamaModelParameters.SplitMode`; needs profile + UI.
- [ ] **`-mg, --main-gpu`** — binding has `MainGpu`; needs profile + UI.
- [~] **`-ts, --tensor-split`** — native `tensor_split` float array; wrapper doesn't surface it yet. Add as `float[] TensorSplit`.
- [~] **`-fit, --fit [on|off]`** — auto-fit to device memory; needs whatever llama.cpp helper backs this (check llama-server source).
- [~] **`-fitt, --fit-target` / `-fitc, --fit-ctx`** — companion knobs; land with `-fit`.
- [x] **`-kvo / -nkvo, --kv-offload`** — `ModelLoadSettings.OffloadKQV`.
- [~] **`--repack, -nr, --no-repack`** — weight repacking; native flag not in wrapper.
- [~] **`-dev, --device <list>`** — device list; needs `llama_model_default_params` devices ptr + managed list wrapper.
- [-] **`--list-devices`** — CLI-only discovery; for the UI, we'd roll a "Devices" ComboBox by enumerating `ggml_backend_dev_*` directly.
- [~] **`-ot, --override-tensor`** — per-tensor placement override; needs binding for `llama_model_params.tensor_buft_overrides`.
- [~] **`-cmoe, --cpu-moe` / `-ncmoe, --n-cpu-moe`** — MoE CPU-keep; needs binding for the MoE-related model params.
- [~] **`--op-offload, --no-op-offload`** — op-level offload flag; native flag not in wrapper.

## 6. KV cache

- [ ] **`-ctk, --cache-type-k TYPE`** — native struct already mirrors `type_k`; just needs a public `LlamaKvCacheType` enum on `LlamaContextParameters` + profile + UI ComboBox. (This is the one the user asked about.)
- [ ] **`-ctv, --cache-type-v TYPE`** — same (`type_v`).
- [!] **`-dt, --defrag-thold`** — flagged DEPRECATED in the help output; skip.

## 7. LoRA / control vectors

- [~] **`--lora FNAME`** — tracked in GH #3 (LoRA adapter support). Requires 10 native fn bindings + SafeHandle.
- [~] **`--lora-scaled FNAME:SCALE`** — lands with #3.
- [~] **`--control-vector FNAME`** — separate native API; binding not yet.
- [~] **`--control-vector-scaled`** — same.
- [~] **`--control-vector-layer-range`** — same.

## 8. Model sources

- [x] **`-m, --model FNAME`** — `ModelLoadSettings.ModelPath` + file picker.
- [~] **`-mu, --model-url URL`** — download-manager feature; larger design (resumable, hash-verified, cache dir). Not trivially bolted on.
- [~] **`-hf, --hf-repo user/model:quant`** — HuggingFace integration; depends on the download manager.
- [~] **`-hfd / -hff / -hfv / -hffv / -hft`** — HF variants (draft / explicit file / vocoder / token). Land together with `-hf`.
- [!] **`-dr, --docker-repo`** — Docker Hub integration; narrower audience than HF, skip unless demand shows up.
- [~] **`--offline`** — only relevant once we do remote-fetch; revisit alongside the download manager.
- [ ] **`--check-tensors`** — binding exposes `LlamaModelParameters.CheckTensors`; needs profile + UI toggle.

## 9. Chat / templating

- [x] **`--jinja, --no-jinja`** — we always use Jinja templates read from GGUF metadata (see `LlamaChatTemplate`).
- [~] **`--chat-template JINJA_TEMPLATE`** — override per-profile. Plumbing exists (`ChatSession` picks the template); surfacing a textarea is UI work, but pasting a Jinja template into a profile editor is a sharp edge we'd want validation for.
- [~] **`--chat-template-file FILE`** — same, just reads from disk.
- [~] **`--chat-template-kwargs JSON`** — advanced Jinja-parser params; matches webui's `--reasoning` knob behaviour.
- [x] **`-rea, --reasoning` / `--reasoning-format`** — covered by `SamplerPanelViewModel.ExtractReasoning` + the `<think>` extractor on the render side.
- [~] **`--reasoning-budget N` / `--reasoning-budget-message`** — token budget for the `<think>` phase; needs a decode-time hook in the generator.
- [!] **`--skip-chat-parsing`** — debugging flag; skip.
- [!] **`--prefill-assistant`** — server-only semantics around whether a trailing assistant message is a prefill or a complete turn; irrelevant to our send-loop.

## 10. Multimodal (mmproj)

- [x] **`-mm, --mmproj FILE`** — `ModelLoadSettings.MmprojPath`.
- [~] **`-mmu, --mmproj-url URL`** — download integration, same dependency as `-mu`.
- [~] **`--mmproj-auto`** — auto-detect sibling `mmproj-*.gguf` next to the model path. Cheap to implement (path probe) but requires a filename convention — safer as a "Detect" button.
- [x] **`--mmproj-offload` / `--no-mmproj-offload`** — inverse of `ModelLoadSettings.MmprojOnCpu`.
- [x] **`--image-min-tokens`** — `ModelLoadSettings.MmprojImageMinTokens`.
- [ ] **`--image-max-tokens`** — wire the companion max field.

## 11. Speculative / draft model

- [~] **`-md, --model-draft FNAME`** — tracked in GH #4 (speculative decoding workflow). Whole feature gated on that.
- [~] **`-cd, --ctx-size-draft`** — lands with #4.
- [~] **`-devd, --device-draft`** — same.
- [~] **`-ngld, --gpu-layers-draft`** — same.
- [~] **`-ctkd, --cache-type-k-draft` / `-ctvd, --cache-type-v-draft`** — same.
- [~] **`--draft, --draft-n, --draft-max`** — same.
- [~] **`--draft-min`** — same.
- [~] **`--draft-p-min`** — same.
- [~] **`--spec-replace`** — same.
- [~] **`--spec-type`** — same.
- [~] **`--spec-ngram-size-n / -m` / `--spec-ngram-min-hits`** — n-gram cache speculative; lands with #4.
- [~] **`-lcs, --lookup-cache-static` / `-lcd, --lookup-cache-dynamic`** — lookup-cache speculative; lands with #4.
- [!] **`--spec-default` / `--fim-qwen-*-spec`** — CLI shortcut presets for specific model combos; profile system replaces this. Users can save their own presets.

## 12. Context management / server-slot behaviour

- [~] **`-ctxcp, --ctx-checkpoints`** — SWA checkpoints per slot; load-time tunable, wrapper not exposed.
- [~] **`-cpent, --checkpoint-every-n-tokens`** — same.
- [~] **`-cram, --cache-ram`** — cache RAM cap; same.
- [~] **`-kvu, --kv-unified`** — unified KV buffer; native flag needs mirroring.
- [~] **`--context-shift, --no-context-shift`** — infinite-text context shifting; generator-side feature.
- [~] **`--warmup, --no-warmup`** — empty-run warmup at load; small wrapper addition.
- [~] **`--spm-infill`** — Suffix/Prefix/Middle order for infill models; needs generator-side support.
- [~] **`--pooling {none,mean,cls,last,rank}`** — `LlamaPoolingType` enum is mirrored but the property isn't on `LlamaContextParameters`; expose + plumb.
- [ ] **`-np, --parallel`** — maps to `LlamaContextParameters.MaxSequenceCount`; profile + UI spinner. (Value still `1` in practice until GH #5 multi-sequence lands, but surfacing the knob is cheap.)

## 13. Model metadata / override

- [~] **`--override-kv KEY=TYPE:VALUE`** — native `llama_model_kv_override`; wrapper doesn't expose it. Worth having for testing unusual chat templates without editing GGUFs.

## 14. Sampling params

Refer to `docs/webui_feature_checklist.md` §6. Snapshot:

- [x] Temperature, top-k, top-p, min-p, typical, top-n-σ, XTC, DRY, mirostat 1/2, penalties (repeat/frequency/presence), seed, dynatemp.
- [~] **`--samplers`** / **`--sampler-seq`** — custom sampler ordering. Current binding applies a fixed chain; exposing order is a sampler-builder change.
- [~] **`--ignore-eos`** — needs LlamaGenerator extension (already flagged in webui checklist).
- [~] **`-l, --logit-bias`** — needs binding + UI.
- [~] **`--adaptive-target` / `--adaptive-decay`** — newer adaptive-p sampler; native fn binding needed.
- [x] **`--grammar` / `--grammar-file`** — `SamplerSettings.Grammar`.
- [x] **`-j, --json-schema` / `-jf`** — Response Format tab with JSON-Schema → GBNF compiler.
- [!] **`-bs, --backend-sampling`** — experimental server-side backend sampling; declined.
- [!] **`--dry-sequence-breaker`** — DRY breakers; already covered by the defaults, override path is a string-parsing mini-feature; skip.

## 15. Vocoder / TTS

- [~] **`-mv, --model-vocoder`** — TTS workflow, not yet built in LlamaChat.
- [~] **`--tts-use-guide-tokens`** — same dependency.

## 16. Reranker / embedding modes

- [!] **`--embedding, --embeddings`** — server restricts to embedding-only mode. LlamaChat is a chat client; embeddings are a separate workflow.
- [!] **`--rerank, --reranking`** — same, reranker endpoint.

## 17. Server-only (all N/A)

`--host`, `--port`, `--reuse-port`, `--path`, `--api-prefix`, `--webui-config` / `--webui-config-file` / `--webui` / `--webui-mcp-proxy`, `--tools` (server built-in agent tools), `--api-key` / `--api-key-file`, `--ssl-key-file` / `--ssl-cert-file`, `--threads-http`, `--cache-prompt` / `--cache-reuse`, `--metrics`, `--props`, `--slots` / `--slot-save-path` / `--slot-prompt-similarity`, `--media-path`, `--models-dir` / `--models-preset` / `--models-max` / `--models-autoload`, `--alias` / `--tags`, `--timeout`, `--lora-init-without-apply`, `--sleep-idle-seconds`, `--completion-bash`, `--cache-list`, `--version` / `--license` / `-h`.

## 18. Logging (N/A)

`--log-disable`, `--log-file`, `--log-colors`, `-v / --verbose`, `-lv / --verbosity`, `--log-prefix`, `--log-timestamps`. LlamaChat's `ErrorBoundary` + `ErrorLog` + toast system covers the equivalent scope.

## 19. Escape / CLI oddities (N/A)

`-e / --escape / --no-escape`, `-r / --reverse-prompt`, `-sp / --special`. Interactive-CLI concepts.

## 20. Built-in model presets

`--embd-gemma-default`, `--fim-qwen-*-default`, `--gpt-oss-*-default`, `--vision-gemma-*-default`. Profile system replaces these with user-defined presets (duplicate + edit flow in Settings → Profiles).

- [!] **all built-in preset flags** — non-goal.

---

## Summary counts

| State | Count | Meaning |
|---|---|---|
| `[x]` done | 17 | Wired through binding + profile + UI |
| `[ ]` TODO | 13 | Binding already exposes it; small scope |
| `[~]` needs binding | 52 | Binding work precedes UI |
| `[!]` won't | 9 | Deliberate non-goals |
| `[-]` / group-declined N/A | ~35 (server-only + logging + CLI) | Server-only |

## Recommended order of attack

Given what the user has in front of them today, the short-list that moves the needle most for daily use:

1. **KV cache type (`-ctk` / `-ctv`)** — asked for, trivial binding-side lift, big memory win on long contexts. Do first.
2. **RoPE tuning (`--rope-scaling` / `--rope-freq-base` / `--rope-scale`)** — binding plumbing already there; unlocks long-context extension for models that didn't ship with it baked in. A dozen lines of profile + UI.
3. **Threading (`--threads` / `--threads-batch`)** — `ThreadCount` / `BatchThreadCount` already on `LlamaContextParameters`; users often want to cap this. One numeric field each.
4. **Split-mode + main-gpu (`-sm` / `-mg`)** — already on `LlamaModelParameters`; the only reason they're not surfaced is that the profile editor wasn't grown beyond single-GPU assumptions. ComboBox + spinner.
5. **`--check-tensors`** — single toggle, useful when chasing GPU OOM / bad-weights failures.

Everything in `[~]` is a meaningful binding-side commitment (new native fn mirrors, new SafeHandles, or new struct fields). Worth fencing those off into per-feature plans (GH #3, #4, #5 are already written up for LoRA / speculative / multi-seq).
