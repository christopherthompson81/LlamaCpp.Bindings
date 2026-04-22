# ASan / UBSan harness investigation

Goal: stand up an AddressSanitizer + UndefinedBehaviorSanitizer build of
`llama.cpp` and run our test suite against it, to surface latent
memory-safety bugs in the binding before they get caught the way the
Phase-3 double-accept bug got caught (i.e., the hard way, by a strict
downstream consumer).

Premise per the prior session's handoff: implementing grammar revealed a
double-`Accept` bug that had lived quietly since Phase 3. Plain stochastic
sampling tolerated it because penalty-history skew didn't crash anything;
grammar's strict state machine did. The hypothesis being tested here is
that more bugs of that class — silent in normal usage, fatal under strict
consumers — exist, and that ASan + differential testing is a cheaper way
to surface them than building Tier-3 features one at a time as probes.

ASan first; differential test second.

## Run 1 — 2026-04-22 — environment recon

**Question:** What does the host actually have, and what does the ASan
build look like before we stage anything?

**Commands:**

```
ls ~/Programming/llama.cpp/build_asan/bin/
readelf -d ~/Programming/llama.cpp/build_asan/bin/libllama.so | grep -E "RPATH|RUNPATH|NEEDED"
gcc -print-file-name=libasan.so
ldconfig -p | grep -E "libasan|libubsan"
dotnet test --nologo
```

**Findings:**

- ASan llama.cpp build is present at `~/Programming/llama.cpp/build_asan/`,
  built CPU-only (no `libggml-cuda.so` in the output), SOVERSION
  `0.0.8620` — same source pin as the staged CUDA build.
- ASan `libllama.so` has `RUNPATH=[/home/chris/Programming/llama.cpp/build_asan/bin:]`
  baked in. Same caveat as the CUDA build: moving or deleting the build
  dir silently breaks sibling resolution. Acceptable for a local dev
  harness.
- Direct deps include `libasan.so.8` and `libubsan.so.1`. Both are present
  on the system (`/lib/x86_64-linux-gnu/`). GCC's own libasan path is
  `/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so`.
- Baseline: 156/156 tests passing in 12 s with the staged CUDA libs. This
  is the post-swap regression target — anything we change should still
  pass against CUDA when restored.

**Implications for next step:**

- We need `libasan.so.8` to be loaded *before* `libllama.so`, otherwise the
  loader rejects ASan-instrumented libs with "ASan must be the first DSO
  loaded". Use `LD_PRELOAD=/lib/x86_64-linux-gnu/libasan.so.8`.
- `dotnet` itself is a moving target under ASan: the CLR allocates
  aggressively, has its own SEGV handlers, and leaks on shutdown by
  design. We need `ASAN_OPTIONS=detect_leaks=0` from the start to avoid
  drowning in CLR noise; only the binding's native-side allocations are
  in scope.
- We do NOT want `halt_on_error=1` for early runs — one report per test
  run isn't enough; let it keep going so we get a full picture.
- Toggle between CUDA and ASan native libs via `tools/fetch-binaries.py
  --from-local <dir>`. Two commands, no script needed yet:
  - `python tools/fetch-binaries.py --from-local ~/Programming/llama.cpp/build_asan/bin --platform linux-x64`
  - `python tools/fetch-binaries.py --from-local ~/Programming/llama.cpp/build_cuda/bin --platform linux-x64`

**Next:** stage ASan libs, run a single smoke test (`Backend_Initializes`)
under the ASan harness, see whether the .NET process even survives
backend init. If yes, scale up to the full suite. If no, debug the
loader/CLR conflict before going further.

## Run 2 — 2026-04-22 — first ASan pass, scaling up

**Question:** Does .NET survive ASan-instrumented `libllama.so` at all,
and does the binding's full execution path produce any ASan/UBSan
findings?

**Setup:**

- Staged ASan libs via
  `python tools/fetch-binaries.py --from-local ~/Programming/llama.cpp/build_asan/bin --platform linux-x64`.
- Used TinyLlama (`/mnt/data/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`,
  668 MB, LLaMA-family, fast on CPU) as `LLAMACPP_TEST_MODEL`. Qwen3
  35B-A3B at IQ4_XS on CPU under ASan would be unusably slow and may
  exhaust RAM with ASan's ~3× shadow overhead. TinyLlama trades
  Qwen3-specific test pass-rate for tractable runtime.
- All runs via `tools/run-tests-asan.sh [filter]`.

**Sub-runs (incremental scale-up):**

| Filter | Tests | Pass | Fail | Wall | ASan findings |
|---|---|---|---|---|---|
| `Backend_Initializes` | 1 | 1 | 0 | 0.05 s | 0 |
| `StructLayoutTests` + `SmokeTests` | 57 | 57 | 0 | 2 s | 0 |
| `GenerationTests` | 15 | 13 | 2 | 34 s | 0 |
| `Grammar*` | 7 | 7 | 0 | 8 s | 0 |
| **full suite (no filter)** | **156** | **149** | **7** | **67 s** | **0** |

**Findings:**

- **Zero ASan/UBSan reports across the entire 156-test suite.** No
  `/tmp/asan.*.log` files were produced at any scale. The hypothesis
  that more memory-safety bugs of the Phase-3 double-accept class are
  hiding in the binding is, for now, *not supported by ASan evidence*.
- ASan slowdown was ~5.5× wall (67 s vs 12 s baseline), well inside the
  10–100× envelope the handoff warned about. The harness is cheap enough
  to run on every commit if we want.
- The .NET runtime + `LD_PRELOAD=libasan.so.8` interaction works without
  the `DOTNET_DbgEnableMiniDump=0` workaround being load-bearing in
  practice — but kept it in the script defensively, because removing it
  would only matter if SEGV-handler conflict re-emerges, and the cost of
  having it set is zero.
- All 7 test failures under TinyLlama are model-mismatch artifacts (test
  assertions that hardcode Qwen3 expectations), not memory bugs:
  - `VocabAdvancedTests.AddsBos_Matches_Previously_Observed_Qwen_Behaviour`
    — Qwen-specific by name.
  - `ChatTemplateTests.Model_Exposes_Embedded_Chat_Template` — TinyLlama's
    template is `{% for ... %}` Jinja, not Qwen's ChatML.
  - `TokenizationTests.Tokenize_Roundtrip_Matches_Input` /
    `..._Long_Input_Needing_Retry` — LLaMA tokenizer prepends a leading
    space on detokenization; Qwen doesn't. Reproduces in the actual
    output diff (`" Hello, world!"` vs `"Hello, world!"`).
  - `MultiTurnChatTests.Multi_Turn_History_Is_Retained_Without_Reprocessing`
    — 1.1B model can't reliably remember a codeword across turns.
  - `GenerationTests.Distribution_Generation_With_Fixed_Seed_Is_Reproducible`
    — text differs because the model is different. The reproducibility
    property the test guards (same seed → same output) is presumably
    intact, just compared against the wrong reference string.
  - `AdvancedSamplerGenerationTests.Xtc_In_Chain_Produces_Output` —
    `string.IsNullOrWhiteSpace(output) == true`. XTC's probability/threshold
    pair is tuned for larger vocabularies; on TinyLlama's distribution it
    apparently filters all candidates. Worth verifying under Qwen3 to
    rule out a code regression, but most likely a tuning artifact.

**Implications:**

- ASan is not paying down the bug-hunting hypothesis we set out to test.
  The binding is clean against this tool. That's a real result — it
  raises confidence that the surface area we control (marshaling,
  SafeHandle lifetimes, struct layout, sampler chain interactions) does
  not have the obvious class of memory bug.
- The Phase-3 double-accept bug would have been undetectable by ASan
  anyway, in retrospect: it was a logic / state-machine error in the
  managed sequencing of native API calls, not a memory-safety violation.
  ASan was always going to be a complementary probe, not a duplicate.
- Therefore: **shift weight to differential testing** for the next round.
  Differential is the tool that catches the actual class of bug we're
  worried about — divergence between our binding's output and the
  upstream `llama-cli` reference, given identical model + sampler config
  + seed.

**Test-hygiene side findings (file as follow-ups, do not fix in this
investigation):**

- Several tests bake Qwen3-specific expectations into assertions without
  guarding on the loaded model. They pass with the default fixture but
  fail under any model substitution, which limits their usefulness for
  exactly this kind of harness work. A `[SkipIfModelIsNot("qwen")]`
  helper, or splitting into model-agnostic vs model-specific test
  classes, would let us run the full suite under ASan without noise.
- `Xtc_In_Chain_Produces_Output` asserts only on output non-emptiness,
  but its parameters (`probability=0.1, threshold=0.05`) appear
  tuned for Qwen3. If the test's intent is to verify XTC plumbing
  rather than exercise a specific tuning, it should ratchet down
  `threshold` or skip when the active model is small.

**Next:** restore CUDA libs to baseline (one fetch-binaries command), then
build the differential-test harness against `llama-cli`. The handoff
sketches this as Run 1 of that probe; we'd start with TinyLlama on CPU
(deterministic) and a fixed seed.

## Run 3 — 2026-04-22 — model-agnostic test refactor

**Question:** Can the suite be run against any reasonable GGUF without
spurious failures? Specifically: is the current 7-test failure under
TinyLlama all model brittleness (test problem) or some of it real
divergence (binding problem)?

**Why this run:** the user's read of Run 2's findings — that the suite
should adapt to model substitution rather than brittle-fail — is
strategically correct. We'll need this every time we want to run a
harness against something other than Qwen3 (ASan, differential test,
perf comparison, future frontier model). One investment now beats
re-doing this every quarter.

**Approach (chosen by user from two options):**

1. New `ModelCapabilities` class (single file, ~110 LoC) that probes the
   loaded model's `general.architecture`, parameter count, vocab size,
   chat-template presence, BOS-add behavior, RoPE type. Exposes
   `SkipUnlessFamily(...)`, `SkipUnlessMinParameters(...)`,
   `SkipUnlessMinVocab(...)` helpers that follow the existing
   `Console.WriteLine("SKIP: ...") + return` convention (xUnit 2.x has no
   built-in dynamic skip; sticking with the established pattern avoids a
   new package dependency).
2. Wire `Capabilities` into `ModelFixture` and `GpuGenerationFixture`
   (both probe the model in their constructor).
3. Surgery on each of the 7 failing tests:
   - **Intentional family guard:** `AddsBos_Matches_Qwen_Behaviour` →
     `SkipUnlessFamily("qwen2", "qwen3")`. The pinned Qwen behavior is
     valuable (would catch upstream changes); just gate it cleanly.
   - **Universal vs family-specific split:** the chat-template test
     became two — one asserts "model has a parseable template", the new
     `Qwen_Chat_Template_Uses_ChatML_Markers` keeps the ChatML pin
     guarded. (+1 test in the suite count.)
   - **Property-based assertion:** the tokenize-roundtrip tests no longer
     compare detok output against the literal input string. The universal
     property is "roundtrip preserves content modulo a possible single
     leading space" (LLaMA SentencePiece encodes the leading space into
     the first token; Qwen BPE does not). Initially attempted "tokenize
     stability under repeated round trips" but that is NOT universal:
     `tokenize("X") ≠ tokenize(" X")` on LLaMA, so the second round trip
     diverges by design.
   - **Min-capability gate:** `Multi_Turn_History_Retained` keeps its
     binding-level position-counter assertion universal but gates the
     qualitative codeword-recall check on `>= 3B params` (TinyLlama 1.1B
     hallucinates here even with perfectly preserved KV state).
   - **Real bug fix:** `Distribution_..._Reproducible` was missing
     `ResetContextFor` before `run1` — only between runs. Under Qwen3
     the model converged regardless; under any other model the leftover
     KV state from prior tests diverged the two outputs. This was a real
     test-reliability defect surfaced by the model swap.
   - **Binding-level reframe:** `Xtc_In_Chain_Produces_Output` now asserts
     "at least one token emitted" instead of "non-whitespace output". XTC
     plus a small-vocab model can legitimately emit only whitespace; the
     binding-level property is "the chain wires up and tokens come out".

**Results:**

| Run | Model | Tests | Pass | Fail | Wall |
|---|---|---|---|---|---|
| Pre-refactor | Qwen3 35B (CUDA) | 156 | 156 | 0 | 12 s |
| Pre-refactor | TinyLlama 1.1B (CPU) | 156 | 149 | 7 | 67 s |
| Post-refactor | Qwen3 35B (CUDA) | 157 | 157 | 0 | 11 s |
| Post-refactor | TinyLlama 1.1B (CPU) | 157 | 157 | 0 | 1 s |

The +1 test is the chat-template split. The drop to 1 s on TinyLlama is
because the run was against the staged CUDA libs (CUDA off-loads the
generation work that dominated the 67 s wall); TinyLlama on CUDA is
faster than TinyLlama on the ASan CPU build, as expected.

**Findings:**

- All 7 model-substitution failures resolved without weakening any
  binding-level assertion. The Qwen-specific assertions are preserved as
  guarded tests so upstream regressions still surface.
- One genuine reliability bug found and fixed (`Distribution_..._Reproducible`'s
  missing pre-run KV reset). Worth highlighting separately: this was the
  user's strategic point in microcosm — model substitution surfaced a
  latent defect that Qwen3's robustness was masking.
- The `ModelCapabilities` infrastructure is ~110 LoC of new test code,
  adds zero package dependencies, and is reusable for any future
  capability gate (encoder-only, recurrent, MRope, etc.).

**Side findings (NOT acted on, file as future work):**

- The `EmbeddingModelFixture` does not yet expose `Capabilities`. Not
  needed for the current 7 failures, but if any embedding test becomes
  family-specific later, mirror the pattern from `ModelFixture`.
- The MaxParameterCount-based skip threshold of 3B in
  `Multi_Turn_History_Retained` is empirical (TinyLlama 1.1B fails;
  Qwen3 35B-A3B passes). If a 2-3B model gets used for testing later,
  re-tune from observed behavior rather than guess.

**Next:** the original "next" — differential test against `llama-cli` —
remains the right call. With this refactor in place, that probe can
freely use TinyLlama on CPU (deterministic, fast) without dragging
seven false positives behind it.
