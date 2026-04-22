# Differential test investigation

Goal: catch logic-level divergences between our binding and llama.cpp's
own reference consumer (`llama-cli`), by running both with identical
model + sampler + seed and comparing the produced token sequence
byte-for-byte. The class of bug being targeted is what the prior
session's handoff called the "Phase-3 double-accept" pattern: a state
sequencing error in the managed code that produces *different* output
from what llama-cli would produce, with no native crash and no memory
report.

The companion ASan probe (see [asan_investigation.md](asan_investigation.md))
already reported clean. ASan was always going to be complementary;
this is the tool that targets the actual class.

Prerequisite: deterministic CPU-only inference. Per
`memory/cuda_moe_determinism.md`, CUDA + MoE is *not* reproducible
across runs; CPU + greedy + small single-sequence model *is*. So:
- Model: `/mnt/data/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
  (1.1 B, LLaMA family, fast on CPU).
- Inference: CPU only, both sides (`--ngl 0` for `llama-cli`,
  `GpuLayerCount = 0` for the binding).

## Run 1 — 2026-04-22 — recon: what does llama-cli output, and how do we
make it deterministic-enough to compare?

**Finding:** `llama-cli` no longer supports `-no-cnv` (it now points users
at `llama-completion` instead). With `llama-completion` + `-no-cnv -st
--no-display-prompt -ngl 0 --no-warmup --no-perf`, we get clean raw
generated text on stdout with no UI cruft. Default sampler chain
(`edskypmxt`) reduces to top-k → top-p → min-p → temperature → terminal
once the dry/penalty/xtc/top-n-sigma defaults (all "off") are factored
out — exactly what our binding's `LlamaSamplerBuilder` constructs.

A subtle alignment detail: `llama-completion` auto-prepends BOS to the
prompt in raw mode. The binding doesn't (with `addSpecial=false`). For
parity, the runner must use `addSpecial=true`. With that single tweak
the prompt token sequence matches exactly (5 → 6 tokens for "The
capital of France is", first token = BOS = 1).

## Run 2 — 2026-04-22 — first byte-for-byte comparison

**Setup:** new console tool [tools/diff-runner](../tools/diff-runner/)
mirrors llama-completion's CLI surface (model, prompt, seed, temp,
top-k/p, min-p, penalties, max-tokens). Driver script
[tools/diff-against-llama-completion.sh](../tools/diff-against-llama-completion.sh)
runs both with the same flags and diffs `GEN_TEXT` against
`llama-completion`'s stdout byte-for-byte.

**Coverage matrix:**

| Config | Result |
|---|---|
| greedy, "The capital of France is", n=20 | **MATCH** (54 bytes) |
| greedy, n=60 | **MATCH** (168 bytes) |
| greedy, "Once upon a time", n=40 | **MATCH** (173 bytes) |
| greedy, "def fibonacci(n):", n=40 | **MATCH** (103 bytes) |
| greedy, n=100 | **MATCH** (291 bytes) |
| dist, temp=0.7 top-k=40 top-p=0.95 min-p=0.05 seed=42 | **MATCH** (124 bytes) |
| dist, temp=1.0 top-k=20 top-p=0.9 seed=7 | **MATCH** (102 bytes) |
| dist, temp=0.5 top-k=40 top-p=0.95 min-p=0.05 seed=42 | **MATCH** (106 bytes) |
| greedy + repeat=1.3 freq=0.5 pres=0.5 | **DIVERGENCE** at byte 13 |
| dist + repeat=1.2 temp=0.7 seed=42 | **DIVERGENCE** at byte 38 |
| dist + repeat=1.5 freq=0.3 pres=0.3 temp=0.8 seed=17 | **DIVERGENCE** at byte 16 |

**Greedy without penalties: identical.** Distribution sampling without
penalties: identical. **Any configuration with penalties active:
divergence.**

**Diagnosis (the divergence shape gives it away):** prompt is "The
capital of France is". Reference continuation: " Paris.\nThe **city**
has a population…". Binding continuation: " Paris.\nThe **capital** of
France is located…". The reference is *avoiding* "capital" — a word
*from the prompt* — exactly what a repeat-penalty would do. The binding
isn't penalizing it. So the binding's penalty sampler doesn't know the
prompt tokens existed.

Reading [LlamaGenerator.cs:80-101](../src/LlamaCpp.Bindings/LlamaGenerator.cs#L80-L101):
the prompt batch is decoded via `DecodePromptBatch(promptArray)`, then
the generation loop begins. Nothing calls `_sampler.Accept(tok)` on the
prompt tokens. The penalty sampler's history buffer is populated only
by `Accept()` calls. During generation, `llama_sampler_sample`
internally calls accept on each generated token (per the existing
double-accept comment at line 115-119), so generation tokens DO enter
the history. Prompt tokens do not.

**This is the bug.** Class: state-machine sequencing error. ASan-invisible
(no memory issue, just wrong inputs to a stateful computation). Test-
assertion-invisible (output is plausible English, just different from
the reference). Surfaced in <30 minutes by the differential probe — the
exact tool the prior session's handoff predicted would catch this class.

**Reference behavior cross-check:** llama.cpp's `common_sampler_accept`
in [common/sampling.cpp](../../llama.cpp/common/sampling.cpp) takes an
`accept_grammar` flag. For prompt tokens, llama-completion calls it
with `accept_grammar=false` — accept into the chain (priming penalties)
but NOT into the grammar (which applies only to generation). Our fix
should mirror this: the prompt accept loop should hit `_sampler.Accept`
(the chain) but never touch the grammar handle. As of the e1423ef
grammar refactor our `LlamaSampler.Accept(int)` only operates on the
chain handle (grammar is a separate field), so a plain `for (var t in
promptTokens) _sampler.Accept(t);` is the correct fix shape.

**Next:** propose the fix to user, then implement + add a regression
test (penalty-active diff against reference) + re-run the full
coverage matrix and confirm all rows match.

## Run 3 — 2026-04-22 — fix, regression test, validation

**Fix:** [LlamaGenerator.cs](../src/LlamaCpp.Bindings/LlamaGenerator.cs)
gained one line plus a comment block after `DecodePromptBatch`:

```csharp
foreach (var t in promptArray) _sampler.Accept(t);
```

This mirrors llama.cpp's `common_sampler_accept(.., accept_grammar=false)`
for prompt tokens — accept into the chain so penalties (and any future
history-aware sampler) see the prompt; do not touch the grammar handle
(grammar is held outside the chain since e1423ef and only constrains
generation, not the prompt).

**Coverage re-run, post-fix:**

| Config | Pre-fix | Post-fix |
|---|---|---|
| greedy, n=20 | MATCH | MATCH (54 b) |
| greedy, n=60 | MATCH | MATCH (168 b) |
| greedy, "Once upon a time", n=40 | MATCH | MATCH (173 b) |
| greedy, "def fibonacci(n):", n=40 | MATCH | MATCH (103 b) |
| greedy, n=100 | MATCH | MATCH (291 b) |
| dist, temp=0.7 top-k=40 top-p=0.95 seed=42 | MATCH | MATCH (124 b) |
| dist, temp=1.0 top-k=20 top-p=0.9 seed=7 | MATCH | MATCH (102 b) |
| dist, temp=0.5 top-k=40 top-p=0.95 seed=42 | MATCH | MATCH (106 b) |
| greedy + repeat=1.3 freq=0.5 pres=0.5 | DIVERGE @13 | **MATCH (101 b)** |
| dist + repeat=1.2 temp=0.7 seed=42 | DIVERGE @38 | **MATCH (60 b)** |
| dist + repeat=1.5 freq=0.3 pres=0.3 temp=0.8 seed=17 | DIVERGE @16 | **MATCH (422 b)** |

All 11 config rows now match `llama-completion` byte-for-byte. The diff
script also picked up a small auxiliary fix: `llama-completion` writes a
literal " [end of text]" to stdout when EOS fires; the binding stops
without that sentinel. Both behaviors are correct; the script now strips
the sentinel before the byte comparison.

**Regression test:** new `Penalty_Sampler_Sees_Prompt_Tokens_Tinyllama_Reference`
in [GenerationTests.cs](../src/LlamaCpp.Bindings.Tests/GenerationTests.cs).
Captures the `llama-completion`-verified reference output for one
specific config (TinyLlama, seed=42, repeat=1.5, max=15) and asserts
byte equality. Gated to LLaMA-family ≤2B param models via
`ModelCapabilities.SkipUnlessFamily("llama")` + parameter ceiling, so it
skips cleanly on the default Qwen3 fixture; skips informatively on
larger LLaMA-family models that would tokenize differently.

Verified the test catches the bug: temporarily reverting the fix and
re-running showed the canonical "model regurgitates the prompt" symptom
in the assertion failure: expected `"...Answer: Paris, the French
capital."`, actual `"...Answer: The capital of France..."`. With the
fix restored, the test passes in 90 ms.

The single canary in the unit suite is for CI signal in the LLaMA case;
the differential harness is the comprehensive check across the
configuration space and across model families. Both should be run before
release.

**Earlier attempts that didn't work (kept here so it doesn't get
re-tried):** an `Assert.NotEqual(unpenalized, penalized)` test using
seeded distribution and max=15 *passes even with the bug active*. The
reason: generated-token history accumulates fast enough that within 15
tokens the penalty starts having different effects between repeat=1.0
and repeat=1.5 even with empty initial history. The captured-reference
approach is necessary — checking that the binding produces the
reference byte sequence, not just "some difference exists".

**Full test suite:** 158/158 (was 157, +1 from the new regression test)
on Qwen3 (CUDA) and TinyLlama (CUDA, deterministic non-MoE). No
regressions.

**Bug-found cost summary:**
- ASan probe (Run 2 of [asan_investigation.md](asan_investigation.md)): 0 findings.
- Differential probe: 1 real shipped bug, found in <30 minutes from first
  matrix run to root cause.
- Bug class: state-machine sequencing error. Identical to the Phase-3
  double-accept the user's strategic premise was named after — same
  shape, different spot. ASan-invisible (no memory corruption), test-
  invisible (output is plausible English just shifted by a few words),
  surfaced only by a strict reference consumer.

**Next:** the differential harness and the regression test are the
ongoing protections. There may be other history-aware samplers (DRY,
others) where the same prompt-prime is needed but isn't tested here;
worth a quick audit once the basic version is committed. Also: the
diff-runner is a prime candidate for a CI step that runs against
TinyLlama on every PR.

## Run 4 — 2026-04-22 — audit other history-aware samplers; second bug found

**Question:** Are there other history-aware samplers in the chain
(DRY, Mirostat, AdaptiveP) where the same prompt-prime fix should
matter, or where comparable parameter-default mismatches with upstream
might lurk?

**Audit:**

| Sampler | History-aware? | Needs prompt-prime? | Notes |
|---|---|---|---|
| Penalties (repeat/freq/pres) | Yes | Yes (FIXED) | The Run 1-3 bug. |
| DRY | Yes | Yes | Fixed by the Run 3 chain-prime; needs separate verification (this run). |
| Mirostat / MirostatV2 | Internal state only | No | Tracks selected-token surprise post-sample; reference impl doesn't feed prompt. |
| AdaptiveP | Internal EMA only | No | Tracks selected-token probability; reference impl doesn't feed prompt. |
| TopK/TopP/MinP/Typical/TopNSigma | Stateless | No | Pure logit transformations. |
| Temperature/ExtendedTemperature | Stateless | No | Pure transformations. |
| XTC | Per-step RNG | No | Reset per call. |
| LogitBias / BannedTokens | Stateless | No | Pure logit modification. |
| Grammar | Yes (state machine) | **No** by design | `accept_grammar=false` for prompt; Run 3 fix correctly avoids it. |
| Infill | Special-token logic | No | FIM-specific. |

**DRY differential coverage:**

| Config | Result |
|---|---|
| greedy + DRY mult=1.0 | MATCH (157 b) |
| dist + DRY mult=0.8 + seed=42 | MATCH (98 b) |
| dist + DRY=0.8 + repeat=1.2 | MATCH (60 b) |
| dist + repeat=2.0 + freq=0.5 + pres=0.5 + DRY=2.0 + seed=17, n=100 | **DIVERGE @ 300** |
| dist + repeat=1.3 + freq=0.5 + pres=0.5 + DRY=1.0 + seed=17, n=100 | **DIVERGE** |
| dist + repeat=1.3 + DRY=1.0 + freq=0.5 + seed=17, n=100 | **DIVERGE** |

**Bisection:** the divergent configs all involve DRY plus at least one
of the chain penalties, run for >=100 tokens. Removing DRY → MATCH.
Increasing `--repeat-last-n` from 64 to 256 → MATCH at the same length.
This pointed at the rolling-window region for repeat-penalty
intersecting with DRY.

But the actual root cause turned out to be different: **DRY's default
sequence breakers differ between the binding and llama-completion**.

- llama-completion's `--dry-sequence-breaker` documentation:
  *"add sequence breaker for DRY sampling, clearing out default breakers
  ('\n', ':', '"', '*') in the process; use 'none' to not use any"*.
- Our binding's `WithDry` default `sequenceBreakers = null` resolved to
  `Array.Empty<string>()` — equivalent to llama-completion's `"none"`.

So a user calling `WithDry(...)` without specifying breakers got
*different* DRY behavior than they'd get from llama-completion's
defaults. Verified by:

```
llama-completion -p ... --dry-multiplier 1.0 --dry-sequence-breaker none
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
diff-runner    -p ... --dry-multiplier 1.0
=> MATCH
```

**Fix:** [LlamaSampler.cs](../src/LlamaCpp.Bindings/LlamaSampler.cs) —
`WithDry`'s null `sequenceBreakers` now defaults to
`["\n", ":", "\"", "*"]`. XML doc updated to note that
`Array.Empty<string>()` is the way to opt out (mirroring
llama-completion's "none").

This is a behavioral change for any caller currently relying on the
empty default. Acceptable: the binding's job is to mirror llama.cpp
defaults; any user who explicitly wanted no breakers can pass an empty
array. Most users won't have noticed because, until DRY+penalties were
combined, the two defaults produced similar-enough output that no
visible bug surfaced.

**Coverage post-fix:**

| Config | Result |
|---|---|
| greedy + penalties | MATCH |
| dist + penalties + seed | MATCH |
| greedy + DRY only | MATCH |
| dist + DRY only | MATCH |
| dist + penalties + DRY | MATCH |
| **dist + repeat+freq+pres+DRY, n=100, extreme** | **MATCH (407 b)** |
| **dist + repeat+freq+pres+DRY, n=100, moderate** | **MATCH (358 b)** |
| **triplet penalty + DRY + freq, n=100** | **MATCH (335 b)** |

All 9 stress configs match.

**Regression test:** new
`Dry_Default_Sequence_Breakers_Match_Llama_Completion_Tinyllama_Reference`
in [GenerationTests.cs](../src/LlamaCpp.Bindings.Tests/GenerationTests.cs).
Same gating shape as the penalty test (LLaMA-family ≤2B). The reference
captured here is the binding's GPU-fixture output, not the CPU
diff-runner output: CUDA and CPU paths in llama.cpp produce different
(but each deterministic) FP results, so a CPU reference would fail on
the GPU fixture even though both are correct on their own backend. The
upstream byte-equivalence is the diff harness's job; this test pins the
GPU output so the WithDry default can't silently regress in the unit
suite.

**Test count:** 159/159 (was 158, +1 from the new DRY regression test).

**Bug-found cost summary, updated:**
- ASan probe: 0 findings.
- Differential probe: 2 real shipped bugs found in <90 minutes total.
  1. Penalty sampler not seeing prompt tokens (fixed, regression test).
  2. DRY sampler default breakers mismatch llama-completion's
     (fixed, regression test).
- Both bugs are state-/parameter-mismatches, not memory issues. Both
  produce plausible output, neither would surface in a normal unit
  test. Both surface immediately in differential testing.

**Process note:** the bisection nearly took me down a wrong path
(rolling-window theory). The breakthrough was reading
llama-completion's help text carefully and noticing the *default*
breaker set. Useful template for future probes: when a config divergence
shows up, check `--help` for *defaults* the binding might be missing,
not just for *flags* the binding might mismap.

**Followups (still NOT acted on):**
- The diff-runner is a CI candidate. Wiring it into a make target or
  GitHub Action that runs on every PR would prevent re-introduction
  of any default mismatches at zero ongoing cost.
- A property-based diff that varies prompt + seed + sampler combos
  randomly would have higher coverage than the hand-picked matrix here.
- Mirostat / AdaptiveP weren't differentially tested — the audit table
  argues they don't need prompt-prime, but that's reasoning, not
  empirical verification. Worth running through the diff harness with
  config flags wired up if either gets used in production.
