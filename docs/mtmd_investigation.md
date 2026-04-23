# Multimodal (mtmd) investigation

Running log of bugs, quirks, and workarounds discovered while building the
Phase B multimodal path. Chronological per run.

## Run 1 — 2026-04-22 19:32 — SIGSEGV encoding Qwen3-VL image on CUDA backend

**Question:** First real-world test of the desktop app's image-attach path
against Qwen3.6-35B-A3B + `mmproj-BF16.gguf`. Load succeeded, attached
`test-1.jpeg` (640×488, NYT front page), asked "describe this image." Process
died immediately — no managed exception (`last-error.log` stale from earlier).

**Environment:**
- GPU: RTX 3090, 24 GB VRAM, ~22 GB free at crash time (nvtop confirms).
- Text model: `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, all layers on GPU.
- mmproj: `mmproj-BF16.gguf` from unsloth/Qwen3.6-35B-A3B-GGUF, 860.98 MiB
  model size, projector type `qwen3vl_merger`.
- Native build: local CUDA build at `/home/chris/Programming/llama.cpp/build_cuda`,
  pinned `b8610-10-g1d6d4cf7a` in `third_party/llama.cpp/VERSION`.
- Image: 640×488 JPEG → mtmd reports `image_tokens nx=20 ny=15` (300 tokens).

**Evidence — coredumpctl thread 1 (crash thread):**

```
#0 ggml_gallocr_alloc_graph          (libggml-base.so)
#1 ggml_backend_sched_alloc_graph    (libggml-base.so)
#2 clip_image_batch_encode           (libmtmd.so)
#3 mtmd_encode                       (libmtmd.so)
#4 mtmd_helper_eval_chunk_single     (libmtmd.so)
#5 mtmd_helper_eval_chunks           (libmtmd.so)
```

SIGSEGV inside ggml's graph-compute allocator, called from the CLIP image
encoder. Not an OOM (22 GB VRAM free); not our P/Invoke marshalling
(mtmd_tokenize already succeeded, we're mid-way through eval_chunks).

**Reproduced via new CLI harness** (`samples/LlamaChat.Cli --mmproj ... --image ...`):
deterministic SIGSEGV, exit 139. Full native log captured at
`/tmp/mtmd-run.log` during the debug session.

**Key native log snippets before the crash:**

```
[Warn] load_hparams: Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks
[Warn] load_hparams: if you encounter problems with accuracy, try adding --image-min-tokens 1024

[Info] warmup: warmup with image size = 1472 x 1472
[Info] alloc_compute_meta: graph splits = 55, nodes = 823

[Error] ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4657.81 MiB on device 0: cudaMalloc failed: out of memory
[Error] ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 4884066560
(falls back to non-FA path, succeeds)

[Warn] warmup: WARNING: the CLIP graph uses unsupported operators by the backend
[Warn] warmup:          the performance will be suboptimal
[Warn] warmup:          list of unsupported ops (backend=CUDA0):
[Warn] warmup:         SOFT_MAX: type = f32, ne = [8464 8464 16 1]
[Warn] warmup:             ROPE: type = f32, ne = [72 16 8464 1]
...
[Warn] warmup: please report this on github as an issue
[Warn] warmup: ref: https://github.com/ggml-org/llama.cpp/pull/16837#issuecomment-3461676118

(warmup succeeds)
...
[Info] encoding image slice...
<SIGSEGV>
```

**Interpretation:** Qwen3-VL's CLIP graph emits ops at unusual shapes
(`ne=[8464, 8464, 16, 1]` softmax is ~1.1 B f32 elements — that's M-RoPE
attention across all image patches). The CUDA backend doesn't support them
and splits the graph so those ops run on CPU while surrounding ops run on
GPU. The warmup at 1472×1472 happens to survive this path; the real-image
encode at 640×488 (20×15 patch grid, different split shape) segfaults in the
graph allocator. This matches the warning text: the referenced llama.cpp
PR/issue asks users to report these exact warnings because the fallback path
is known to be fragile.

**Fix validated:** Running the same CLI with `--cpu-encode` (sets
`mtmd_context_params.use_gpu = false`) succeeds. Image encoded in 4350 ms;
model correctly described the image ("physical copy of a newspaper", "The
New York Times", "MEN WALK ON MOON"). Exit 0.

**Actions taken:**
- Added `ModelLoadSettings.MmprojOnCpu` (default `false` — matches native
  default) + `MmprojImageMinTokens` (optional, for the Qwen-VL grounding
  warning). Threaded through `ChatSession.Load` into
  `MtmdContextParameters { UseGpu, ImageMinTokens }`.
- Exposed both as controls in `ProfileEditorView.axaml` with tooltips
  pointing at this doc.
- CLI grew `--mmproj / --image / --prompt / --cpu-encode / --image-max-tokens`
  flags so we can reproduce crashes outside the desktop app.

**Not doing (yet):**
- Defaulting to CPU encode globally. SmolVLM-256M (our test fixture) works
  fine on GPU and is fast there; penalising the common case for the
  Qwen3-VL bug is the wrong tradeoff. Users who hit the crash get a clear
  toggle.
- Catching/recovering from the SIGSEGV. Not portable, not safe in-process.
- Warning proactively based on arch metadata. mmproj arch detection isn't
  exposed via our current bindings, and the bug will likely be fixed
  upstream before we'd finish wiring it.

**Follow-ups to watch:**
- llama.cpp PR #16837 and the referenced issue — when this lands upstream,
  bump `third_party/llama.cpp/VERSION` and re-test GPU encode.
- Non-consecutive KV position warnings (`find_slot: non-consecutive token
  position 4 after 3 ...`) during image decode. Expected for M-RoPE image
  tokens but worth re-visiting if we see output degradation.
- 20×15 = 300 image tokens is below the 1024 minimum the Qwen-VL warning
  recommends. Output was coherent anyway for this image, but grounding
  tasks (localizing objects in scenes) may need the min-tokens knob set.

## Run 2 — 2026-04-23 21:19 — bump pin to b8893/b8894, GPU encode works

**Question:** Upstream shipped 22 days of mtmd + CUDA fixes between our pin
(b8620, 2026-04-01) and HEAD (b8893, 2026-04-23). No single PR was clearly
"this fixes Qwen3-VL" — the obvious candidate #21103 was closed unmerged
— but several CUDA graph/equality commits looked like they could
incidentally fix it. Worth bumping to find out and exercise the version
upgrade flow.

**Process:**
1. `git pull` + rebuild in `/home/chris/Programming/llama.cpp/build_cuda`.
   New soversion `0.0.8894` (libllama.so / libmtmd.so).
2. Swapped `runtimes/linux-x64/native/*` to the new libs. Pinned new
   `llama.h`, `mtmd.h`, `mtmd-helper.h` into `third_party/llama.cpp/include`.
3. Diffed upstream public headers against our old pin:
   - `mtmd_decode_use_non_causal` gained a `const mtmd_input_chunk*` arg.
   - `mtmd_decoder_pos` struct (16 B, 4× uint32) added + `mtmd_image_tokens_get_decoder_pos`
     + `mtmd_helper_image_get_decoder_pos`.
   - `LLAMA_SPLIT_MODE_TENSOR = 3` added to `llama_split_mode`.
   - `LLAMA_FTYPE_MOSTLY_Q1_0 = 40` added (we don't mirror the ftype enum).
   - `llama_params_fit()` and `llama_memory_breakdown_print()` **removed**.
     We had a P/Invoke + public `LlamaContext.LogMemoryBreakdown()` wrapper
     + one test for the latter; all removed.
4. Extended `tools/dump-struct-sizes.c` for `mtmd_decoder_pos`, regenerated
   `tools/struct-sizes.json` (all existing entries unchanged, new entry
   matches the C# mirror).
5. Updated `third_party/llama.cpp/VERSION` with the new commit + full
   delta summary.
6. Rebuild, full test suite: 211 pass (was 212 — dropped one for the removed
   function).

**Evidence the Qwen3-VL bug is fixed:** Re-ran the CLI harness:

```
samples/LlamaChat.Cli --model Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
                      --mmproj mmproj-BF16.gguf --image test-1.jpeg
                      --ctx 4096 --max 128
```

Result: exit 0, model generated the expected description.

Before (b8620):          After (b8893):
- GPU encode: SIGSEGV    - GPU encode: `image slice encoded in 177 ms`
- FA disabled w/ warns   - FA enabled, no unsupported-op warnings
- CPU fallback works     - CPU fallback (`--cpu-encode`) still works (4252 ms)

**GPU encode is ~24× faster than the workaround**, no-opt-in required.

**Keeping the CPU-encode knob anyway** — CLIP backends on exotic arches or
future regressions may still need the escape hatch. Low cost to keep the
checkbox; revised tooltip wording already generic ("buggy CUDA CLIP graphs").

**No root cause identified** — we didn't bisect. One or more of the CUDA
fixes between b8620 and b8893 closed the hole (likely candidates:
#21736 `CUDA: also store node->src ne/nb for graph equality`,
#21566 `CUDA: check for buffer overlap before fusing`,
#21271 `CUDA: fix FA kernel selection logic`). Not worth bisecting —
symptom gone, future regressions will surface in the smoke test.

**Bindings-update flow validated.** The mechanical parts (binary swap,
header pin, struct-sizes regen) were straightforward; the API delta
(~3 binding changes, 1 public-API removal, 1 test cascade) was the only
real work. End-to-end took <30 min including the rebuild.
