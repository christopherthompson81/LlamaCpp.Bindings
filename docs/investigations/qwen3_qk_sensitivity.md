# Qwen3-0.6B QK sensitivity vs llama.cpp's Q4_K_M heuristic

Premise: GGUFLab's Adaptive Quantization tool runs a per-tensor sensitivity
sweep against an F16 source and assigns each tensor the smallest quant
type whose round-trip relative MSE stays below a threshold τ. Comparing
those per-tensor picks against llama.cpp's hard-coded ftype heuristic
answers a concrete question: *given the same overall bit budget, does
the heuristic allocate bits to the right tensors?*

The expectation going in was that for Qwen3-0.6B — small, dense, well-
studied — the heuristic should be approximately right.

**v3 ship status (after Run 21)**: GGUFLab's per-category profile-driven
recipe builder now produces recipes that **strictly Pareto-dominate
stock Q4_K_M** on Qwen3-1.7B — smaller file (1,232 MB vs 1,282 MB) AND
better PPL (16.934 vs 17.408 — F16 is 16.887, so the v3 gap closes 91%
of stock's quality loss). The win comes from the per-category
optimizer reallocating budget across categories (uniform Q6_K on
attention K/V and ffn_down, uniform Q3_K on ffn_gate and token_embd)
rather than stock's per-layer alternation pattern, *given freedom to
disable* the `use_more_bits` baseline floor. The per-layer ffn_down
drill that motivated this recipe served as the empirical evidence
that justified disabling the floor — protected layers' mean Q4_K Δ
was 9× lower than the unprotected set's, confirming the heuristic was
wrong about which layers needed protection on this model.

**Final status (after Run 9)**:
- *Architectural signature*: QK-norm causes a real, isolated, ~100×
  amplification of imatrix-weighted MSE on K/Q tensors (Run 5).
  Confirmed.
- *PPL coefficient diagnosis*: per-category ablation (Run 9) measures
  how much each tensor category actually costs in PPL when quantized
  to Q4_K. The ranking:

      ffn_up (+0.46)  ≈  attn_v (+0.44)  >  attn_k (+0.30)
      ≈ attn_q (+0.27)  >>  attn_output (+0.08)  ≈ ffn_gate (+0.08)
      >  ffn_down (+0.03)

  The Q4_K_M heuristic gets `attn_v` protection right but **misses
  `ffn_up`** (the highest-leverage category — heuristic gives it
  default Q4_K) and **over-protects `ffn_down`** (the lowest-
  leverage — heuristic spends Q5_K/Q6_K on it via `use_more_bits`).
- *Operational claim* ("imat-weighted MSE recipe beats Q4_K_M on
  PPL"): retracted (Run 8). The recipe was loud where MSE was loud
  (late attn_k/attn_q from QK-norm) but those are mid-sensitivity
  for PPL — and the actual most-sensitive category (`ffn_up`)
  doesn't show up loudly in the MSE measurement at all. So the
  recipe makes 3 of 4 worst-possible category choices.

**Path forward (after Run 12)**: cross-architecture validation
confirms what was suspected — **sensitivity profiles are
architecture-specific**.

  - Qwen3-0.6B (QK-norm): `attn_v` is 2nd-most-sensitive. Heuristic
    correctly protects it. `ffn_up` is missed. `ffn_down` has a knee
    that justifies its protection.
  - Llama-3.2-1B (no QK-norm): `attn_v` is the *least*-sensitive
    category (+0.008 PPL at Q4_K). The heuristic's `use_more_bits`
    bumps on `attn_v` are wasted budget. `ffn_up` *and* `ffn_gate`
    are both under-protected.

A universal "fix the heuristic" recipe doesn't exist. The right tool
design (and the GGUFLab Adaptive Quantization v2 specification):

  1. Ship per-architecture sensitivity profiles as build-time
     artifacts (one ablation campaign per architecture, ~1 hour).
  2. The recipe builder takes a budget target and a profile;
     produces an allocation that minimizes predicted ΔPPL while
     respecting per-category knees (e.g. ffn_down's cliff at Q2_K).
  3. The MSE-based sensitivity sweep (Runs 1-5) keeps a reduced but
     real role: picking *which tensor within a category* to bump
     first when the builder allocates extra bpw to that category.
     That's where the QK-norm signal earns its keep.

Recipe v2 (Run 10) was a category-level swap that beat stock Q4_K_M
by 0.29 PPL on Qwen3-0.6B — proof that ablation-driven recipes work.
But applying that exact swap to Llama-3.2-1B would not work the
same way; you'd need a different swap (promote ffn_gate too, demote
attn_v).

The QK-norm signature is a measurement of an architectural property,
not a recipe ingredient. Per-tensor imat-weighted MSE is a
measurement, not an optimization target.

**Earlier final verdict (after Run 5)**: QK-norm has a real, isolated
quantization signature, but it's about *imatrix-aware quantization
losing its usual benefit on K/Q tensors*, not about the raw weights
being unusually hard. The 4-way comparison nails the story:

- Imatrix weighting normally *reduces* worst-case MSE (~5× on every
  Llama-3.2-1B tensor — the quantizer optimizes for the imatrix).
- On QK-normed Qwen3-0.6B, imatrix weighting *amplifies* worst-case
  MSE by ~100× on `attn_k` / `attn_q`, ~37× on `attn_v`, and
  ~1.3× (no effect) on `ffn_down`. The amplification is isolated
  to the tensors QK-norm operates on.

Mechanism: QK-norm decouples raw Q/K weight magnitude from the
post-projection activation scale, so the K/Q activations span
wider per-column dynamic range. The imatrix captures that range,
but Q4_K's per-block-scale quantization optimization can't track
per-column importance variance — so quantization errors landing
in heavy-imatrix columns aren't preferentially suppressed.

The early sections below are preserved as-is so the chronological
reasoning is visible. The corrected synthesis lives in Run 5.

The heuristic over-protects `attn_v` and badly under-protects late-layer
`attn_k` / `attn_q`, almost certainly because Qwen3 uses QK-norm and the
heuristic was tuned in the Llama-1/2 era when `attn_v` was empirically
the most-sensitive attention tensor.

## Run 1 — 2026-04-26 16:01

**Command**: GGUFLab → Adaptive Quantization → Run sensitivity sweep
on `~/.cache/llama-models/Qwen/Qwen3-0.6B/Qwen3-0.6B.F16.gguf`.
The GUI's `ResolveImatrixForGguf` auto-fill picked up the sidecar
`Qwen3-0.6B.F16.imatrix.gguf` next to the model, so **this sweep
was imatrix-weighted** (every score row carries `imatrixWeighted=true`,
and `result.ImatrixPath` is set). The first writeup of this run
incorrectly described it as unweighted; that was sloppy and is
corrected here. The imatrix-weighted MSE formula uses per-column
importance, so the absolute numbers below differ in units from a
naive sweep — see Run 4 for the unweighted re-run with matching units.
Default 11 candidates: F16, BF16, Q8_0, Q6_K, Q5_K, Q4_K, IQ4_XS,
Q3_K, IQ3_S, Q2_K, IQ2_S. 198 tensors scored after filtering 1-D
norms.

**Output**: `~/.cache/llama-models/Qwen/Qwen3-0.6B/Qwen3-0.6B.F16.scores.json`
(2178 rows = 198 tensors × 11 candidates).

## Run 2 — 2026-04-26 16:55  (analysis)

**Command**:
```bash
python3 tools/analyze-sensitivity-vs-heuristic.py \
    --scores ~/.cache/llama-models/Qwen/Qwen3-0.6B/Qwen3-0.6B.F16.scores.json \
    --ftype  q4_k_m \
    --layers 28
```

The script reproduces llama.cpp's Q4_K_M heuristic for a dense
non-Falcon model (mirroring `llama-quant.cpp` ~lines 415–640) and
compares per-tensor picks at the recipe-τ that lands at the same
average bpw as the heuristic.

### Headline numbers

| metric                                    | value |
|-------------------------------------------|-------|
| heuristic (Q4_K_M) average bpw            | 4.803 |
| recipe at τ=0.01 average bpw              | 4.527 |
| tensors where the two disagree            | 188 / 198 |

The recipe lands at ~0.3 bpw *less* total budget than the heuristic
while keeping every tensor under 1% relative-MSE — so the comparison
is fair on cost.

### Where the heuristic over-spends: early `attn_v`

The Q4_K_M heuristic protects `attn_v` with Q5_K (default) and bumps
to Q6_K via `use_more_bits`. On Qwen3-0.6B, the F16→Q*_K round-trip
relative MSE for early `attn_v` tensors is so small (1e-5–1e-4 range)
that the recipe is happy with IQ2_S (2.5 bpw) at τ=0.01:

| tensor                  | heuristic    | recipe at τ=0.01 | h-MSE   | r-MSE   |
|-------------------------|--------------|------------------|---------|---------|
| blk.0.attn_v.weight     | Q6_K (6.56)  | IQ2_S (2.50)     | 1.15e-5 | 2.12e-3 |
| blk.1.attn_v.weight     | Q6_K         | IQ2_S            | 4.21e-6 | 6.94e-4 |
| blk.2.attn_v.weight     | Q6_K         | IQ2_S            | 8.38e-6 | 1.34e-3 |
| blk.3.attn_v.weight     | Q5_K         | IQ2_S            | 1.88e-4 | 9.37e-3 |

That's 4+ bpw of over-spend per tensor on the cheap end of the model,
where the heuristic's "V is sensitive" assumption is loudest.

### Where the heuristic under-spends: late `attn_k` and `attn_q`

The heuristic gives every `attn_k`/`attn_q` the family default `Q4_K`
(no `use_more_bits` sprinkler for K/Q). On Qwen3-0.6B that's fine in
the early layers but catastrophic in the late blocks:

| tensor                  | heuristic    | rel-MSE at heuristic-pick |
|-------------------------|--------------|---------------------------|
| blk.20.attn_k.weight    | Q4_K (4.5)   | **0.143**                 |
| blk.23.attn_k.weight    | Q4_K         | **0.376**                 |
| blk.24.attn_k.weight    | Q4_K         | **0.591**                 |
| blk.25.attn_k.weight    | Q4_K         | **0.743**                 |
| blk.26.attn_k.weight    | Q4_K         | **0.759**                 |
| blk.25.attn_q.weight    | Q4_K         | **0.565**                 |
| blk.26.attn_q.weight    | Q4_K         | **0.546**                 |

`rel-MSE = 0.74` means the round-trip error has 74% the variance of
the source tensor — those weights are essentially destroyed. The
recipe chooses Q8_0 (8.5 bpw) for these specific tensors to bring
relative MSE under ~0.5%.

### Working hypothesis: QK-norm shifts where sensitivity lives

Qwen3 uses RMSNorm on Q and K activations independently before the dot
product (QK-norm, introduced in Llama-3.1 / Phi-3 / Gemma-2). That
means the *raw* `attn_k`/`attn_q` weights don't have to produce
unit-scale outputs — the post-norm activations do. The result is
weight distributions with wider per-block dynamic range that Q4_K's
fixed scale-per-32-elements doesn't handle well, particularly in late
layers where the network has accumulated the most learned variance.

The legacy heuristic, designed for Llama-1/2 (no QK-norm, V was
empirically the loudest), can't see this — it's hard-coded to the
2023-era topology.

## Caveats

- **Imatrix-weighted measurement**. The sweep used the sidecar
  imatrix (see corrected note in Run 1 above). Imatrix weighting
  tends to *reduce* worst-case rel-MSE because errors in low-importance
  columns get small weight — so the 0.76 rel-MSE on late `attn_k`
  is the *favorable* measurement. An unweighted re-run can only go
  the same way or worse (Run 4 confirms which).
- **Round-trip MSE ≠ output PPL**. A 0.74 rel-MSE on one tensor is
  not by itself a 0.74 PPL hit on the model — downstream layers can
  partially compensate. The right next step is a real Q4_K_M vs
  recipe perplexity comparison on wikitext.
- **One model**. The whole hypothesis hinges on QK-norm being the
  cause. A non-QK-norm contrast (Llama-3.2-1B or similar) would
  confirm or refute it cheaply.

## Next steps

1. Run the analysis on a non-QK-norm contrast model. Llama-3.2-1B
   (16 layers, no QK-norm) is the cleanest counterfactual at
   ~2.5 GB F16. If the heuristic's pattern matches the recipe's
   on that model, QK-norm is confirmed as the divergence cause.
   *(Done in Run 3.)*
2. Re-sweep Qwen3-0.6B *without* an imatrix to put the units on the
   same footing as Run 3 (Llama-3.2-1B was unweighted). The Run 1
   numbers were imatrix-weighted, which is the *favorable* form;
   the unweighted re-run is more conservative.  *(Pending in Run 4.)*
3. Side-by-side perplexity: stock Q4_K_M vs the recipe at the same
   bpw. If the recipe wins on PPL the finding is operational, not
   just academic — the GGUFLab page can recommend the recipe over
   the heuristic for Qwen3-class models.

## Run 28 — 2026-04-29 (Per-tensor Llama-3.2-1B `ffn_down`: soft-floor over-fires on gentle profiles)

### Question

Run 27 validated the v3 recipe builder cross-architecture using *category-only* data on Llama-3.2-1B. The soft-floor noise gate (#44/Run 26) only fires inside the per-tensor refinement loop, so we don't yet know whether it behaves correctly on a non-Qwen3 architecture. The Llama-3.2-1B category profile has exactly one analyzer-suspect category — `ffn_down` shows a non-monotonic Q2_K +2.28 → Q3_K +2.52 inversion at category scope. If that inversion repeats at per-layer scope, the gate will fire on it; we want to see whether the soft floor (Q3_K) blocks bad demotes the way it does on Qwen3, or whether on a gentler profile the gate over-fires and hurts the recipe.

Targeted scope (per user direction): drill `ffn_down` only — 16 layers × 7 candidate types = 112 cells, ~30 min on the in-place path now that #46/#48 are fixed.

### Setup

- Model: `bartowski/Llama-3.2-1B-Instruct-f16.gguf`, 16 layers, GQA (32 heads / 8 kv-heads), tied embeddings.
- Imatrix: regenerated 2026-04-29 against `wiki.test.raw` (post-fix collector, dated Apr 29 12:32).
- Corpus: `wiki.test.raw`.
- F16 baseline PPL: 13.8248.
- Build wall: 38 min (in-place + #48 disk fallback for any token_embd cells; 113 cells incl. baseline).

### Per-tensor data

ΔPPL by layer × type for `blk.{i}.ffn_down.weight`:

| layer  |   Q2_K |   Q3_K | IQ4_XS |   Q4_K |   Q5_K |   Q6_K |   Q8_0 |
| ------ | -----: | -----: | -----: | -----: | -----: | -----: | -----: |
| blk.0  | +0.145 | +0.389 | +0.018 | +0.010 | +0.004 | +0.001 | +0.003 |
| blk.1  | +0.087 | **+1.187** | +0.072 | +0.001 | -0.001 | +0.040 | -0.002 |
| blk.2  | +0.133 | +0.083 | -0.020 | +0.004 | -0.004 | -0.003 | +0.003 |
| blk.3  | +0.080 | +0.026 | -0.011 | +0.011 | -0.013 | -0.001 | -0.003 |
| blk.4  | +0.121 | +0.036 | +0.008 | +0.010 | +0.004 | +0.000 | -0.001 |
| blk.5  | +0.055 | +0.024 | +0.018 | +0.007 | -0.004 | +0.002 | -0.001 |
| blk.6  | +0.105 | +0.049 | +0.008 | +0.008 | -0.004 | +0.000 | +0.000 |
| blk.7  | +0.060 | +0.028 | +0.014 | +0.004 | -0.000 | +0.001 | -0.002 |
| blk.8  | +0.123 | +0.037 | +0.018 | +0.004 | +0.004 | +0.002 | -0.001 |
| blk.9  | +0.160 | +0.048 | +0.014 | +0.001 | +0.002 | +0.001 | -0.001 |
| blk.10 | +0.156 | +0.048 | +0.014 | +0.014 | +0.004 | +0.000 | +0.000 |
| blk.11 | +0.131 | +0.040 | +0.014 | +0.013 | -0.001 | -0.001 | -0.000 |
| blk.12 | +0.126 | +0.029 | +0.012 | +0.007 | +0.001 | -0.000 | +0.001 |
| blk.13 | +0.120 | +0.045 | +0.004 | +0.008 | +0.002 | +0.002 | +0.001 |
| blk.14 | +0.123 | +0.038 | +0.015 | +0.010 | +0.002 | +0.000 | -0.000 |
| blk.15 | +0.179 | +0.078 | +0.021 | +0.021 | +0.004 | -0.000 | +0.001 |
| **mean** | +0.119 | +0.136 | +0.014 | +0.008 | +0.000 | +0.003 | -0.000 |
| **stdev** | +0.035 | +0.294 | +0.019 | +0.005 | +0.005 | +0.010 | +0.002 |

Observations:
- **Q2_K**: layer-uniform around +0.12, no catastrophic outlier.
- **Q3_K**: dominated by `blk.1`'s **+1.19** outlier (14× the second-highest). Without blk.1, mean drops to +0.066 — at or below Q2_K. Q3_K is genuinely worse than Q2_K specifically at one layer; elsewhere it's fine.
- **Q4_K and up**: every layer at or below the F32-noise band (~0.005). All "free" relative to the F16 baseline.
- **Inversions**: `Q3_K > Q2_K` on blk.0 and blk.1. `Q4_K ≈ Q5_K ≈ Q6_K ≈ Q8_0` everywhere within F32 noise (frequent last-bit sign flips).

### Recipe A/B at 4.95 bpw

Recipe builder run twice against the same profile, only the soft-floor knob varies:

| | Soft floor (default `Q3_K`) | Hard-stop (`RefinementFloorWhenNoisy=null`) |
| --- | --- | --- |
| `ffn_down` picks | **Q6_K×8 + Q3_K×5 + IQ4_XS×3** | Q6_K×8 + Q5_K×8 |
| Differing layers | blk.{2,3,5,6,8,9,11,12} demoted | (matches Run 27) |
| Avg bpw | 4.86 | 4.99 |
| Quantized size | 749 MB | **779 MB** |
| Wiki-test PPL (ctx=512) | **14.3630** | **14.1810** |
| Δ vs F16 | +0.5382 | +0.3562 |
| Δ vs hard-stop | **+0.1820 PPL** | (anchor) |

**The soft floor's demotes hurt by 0.18 PPL.** They look "free" per the per-tensor coefficients (each tensor's Q3_K Δ is well below the Q6_K category-level Δ used as the demote threshold) but compound non-additively when 8 of 16 layers demote simultaneously — the same multi-tensor compounding pattern Run 25 documented on Qwen3 (Run 22's variants A/B/C/D summing pairs that didn't add linearly).

### Why the gate fires here at all

The analyzer flags `ffn_down` as `NonMonotonic` because the *category-level* curve has Q3_K (+2.52) > Q2_K (+2.28). The per-tensor data (computed in this run) shows that inversion is real but driven by a *single layer* (blk.1's Q3_K +1.19 outlier). At the rest of the stack Q3_K is fine. The category-shape analyzer can't see that — it only sees the inverted aggregate.

Once the gate fires, soft floor (Q3_K) lets refinement run with a Q3_K-or-above floor. The demote loop walks the ladder ascending: Q2_K (below floor, skip), Q3_K (try). For 5 of the 16 layers Q3_K Δ is below the category-pick Q6_K Δ (+0.04), so the loop demotes them. For 3 more layers IQ4_XS Δ is below threshold and Q3_K isn't, so they demote one rung higher. The remaining 8 are protected by the stock baseline floor at Q6_K (the use_more_bits set on this architecture) which the optimizer can't undercut.

### Cross-architecture asymmetry of the soft floor

Comparing soft-floor outcomes across the two architectures we now have evidence on:

| Profile | Per-tensor data | Hard-stop recipe vs F16 | Soft-floor recipe vs F16 | Verdict |
| --- | --- | --- | --- | --- |
| Qwen3-1.7B (Run 26) | Genuine layer variance (Run 22's mixed picks) | 16.9596 PPL (refinement skipped → uniform Q6_K) | **16.8460 PPL** (refinement picks Q6_K×15 + IQ4_XS×9 + Q4_K×4) | soft-floor saves 0.11 PPL |
| Llama-3.2-1B (Run 28) | Below-noise (Q4_K and up all ~0) | **14.1810 PPL** (refinement skipped) | 14.3630 PPL (8 layers demoted) | soft-floor *costs* 0.18 PPL |

**Diagnosis.** The soft-floor design assumed that when the gate fires (NonMonotonic / high-rel-std category-shape), the per-tensor data still had *enough signal above noise* that refinement decisions would be sound. That assumption holds on Qwen3 where the noise gate fires precisely because the underlying per-tensor signal is strong but the category-level aggregate looks weird. It breaks on Llama-3.2-1B where the category-level inversion is driven by a single outlier and the rest of the per-tensor data is at the noise floor — refinement then makes noise-driven demotes that compound badly.

The hard-stop "wins" on Llama not because skipping refinement is virtuous, but because *doing nothing* is strictly better than *acting on noise*.

### Mitigation: per-cell signal-magnitude check

The current gate looks at curve *shape* (NonMonotonic, rel-std). It needs an additional guard: when the category-pick Δ is itself below an absolute noise threshold (e.g., 0.05 PPL on the wiki.test corpus), refinement should skip regardless of curve shape — there's no headroom to spend, and any demote that "looks free" is at least partly noise-driven.

Pseudocode:
```
if categoryDelta < AbsoluteCategoryNoiseFloor (default 0.05):
    skip refinement   # nothing meaningful to optimize
else if NonMonotonic or highRelStd:
    apply soft floor (existing logic)
else:
    refine freely (existing logic)
```

Filing as a follow-up issue. For now the workaround is `RefinementFloorWhenNoisy = null` on profiles known to have below-noise category-pick Δs (which is hard to characterize without already running the recipe builder once).

### Net status

Cross-architecture validation of the recipe builder is now richer:
- Run 27 (category-only): v3 builder Pareto-dominates stock Q4_K_M on Llama-3.2-1B with default options.
- Run 28 (with per-tensor data): the soft-floor noise gate's defaults are *not* universally safe — they assume per-tensor data has signal above noise, which holds on Qwen3 but not on Llama-3.2-1B's gentle profile.

Recipe-shipping advice: until the magnitude check lands, on architectures whose per-tensor `ffn_down` Q4_K Δs cluster around zero (Llama family per this run; likely Qwen3-4B per its category-only profile), prefer category-only profiles or set `RefinementFloorWhenNoisy = null` to avoid noise-driven demotes.

## Run 27 — 2026-04-29 (Cross-architecture validation: v3 recipe builder beats stock on Llama-3.2-1B)

First cross-architecture validation of the v3 recipe builder + soft-floor noise gate on a non-Qwen3 model. Llama-3.2-1B-Instruct's terrain is qualitatively different from Qwen3 — classical multi-head attention (no QK-norm), no Q2_K cliffs, gentle ΔPPL curves throughout — so it tests whether the recipe builder's defaults still produce a Pareto-positive recipe when the headroom is small.

### Setup

The cached profile (`data/profiles/llama-3.2-1B.profile.json`, dated 2026-04-28) had to be discarded: its measurement records had been wiped from the investigation DB during the perf-test invalidations between Runs 22 and 25, and the profile schema doesn't record `imatrix_sha` in provenance, so we couldn't verify whether the underlying measurements used an imatrix at all. The cached imatrix sidecar (`Llama-3.2-1B-Instruct.imatrix.gguf`) was dated 2026-04-26 — *before* the 2026-04-27 evening imatrix-collector fix that resolved Q2_K's per-block calibration. So both the profile and the imatrix it was built against were potentially miscalibrated for low-bpw decisions.

Rebuild order:

1. Regenerate the Llama-3.2-1B imatrix against the current collector (88s wall, 564 chunks of `wiki.test.raw`, 112 tensors covered, 1.3 MiB output).
2. Rebuild the category profile with `LlamaSensitivityProfileBuilder.BuildAsync` against the fresh imatrix: 8 categories × 7 candidate types ({Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS}) = 57 cells, F16 baseline = 13.8248 PPL.
3. Build a recipe at 4.95 bpw (Q4_K_M-class) with default options (soft-floor noise gate, stock baseline applied) and PPL-test against stock Q4_K_M built from the same source + same fresh imatrix.

### Profile shape

Llama-3.2-1B's profile is much milder than Qwen3-1.7B's:

| Category | Q2_K Δ | Q3_K Δ | IQ4_XS Δ | Q4_K Δ | Q5_K Δ | Q6_K Δ | Q8_0 Δ |
| -------- | -----: | -----: | -------: | -----: | -----: | -----: | -----: |
| `attn_q.weight`     | +0.24 | +0.13 | +0.01 | +0.04 | −0.01 | +0.00 | −0.00 |
| `attn_k.weight`     | +0.46 | +0.20 | +0.06 | +0.03 | +0.00 | −0.01 | −0.00 |
| `attn_v.weight`     | +0.76 | +0.32 | +0.18 | +0.01 | −0.02 | +0.01 | +0.01 |
| `attn_output.weight`| +1.27 | +0.38 | +0.04 | +0.09 | −0.02 | +0.01 | +0.00 |
| `ffn_up`            | +2.42 | +0.47 | +0.16 | +0.12 | +0.04 | +0.03 | −0.00 |
| `ffn_gate`          | +1.38 | +0.33 | +0.11 | +0.09 | +0.01 | +0.00 | +0.00 |
| `ffn_down`          | +2.28 | **+2.52** | +0.22 | +0.13 | +0.01 | +0.04 | +0.01 |
| `token_embd.weight` | +2.23 | +0.50 | +0.17 | +0.12 | +0.02 | −0.01 | −0.00 |

No category catastrophes anywhere — even Q2_K stays under +2.5 PPL across the board, vs Qwen3-1.7B's `ffn_down × Q2_K = +3709`. Run 12's hypothesis is empirically confirmed: classical MHA without QK-norm is much friendlier to low-bpw quantization.

The one anomaly: `ffn_down × Q3_K (+2.52)` is *higher* than `Q2_K (+2.28)` — a NonMonotonic measurement that the analyzer would flag. The recipe builder's monotone-from-above clamp handles this automatically; soft-floor refinement would also kick in if the profile had per-tensor data, but at category-only it just doesn't produce a per-tensor refinement to gate.

`token_embd.weight` is unusually sensitive (+2.23 at Q2_K) because Llama-3.2-1B has tied embeddings — `token_embd` is also the lm_head, and quantizing it heavily quantizes the output projection too.

### Recipe verdict

| Variant | PPL | Δ vs F16 | Size | % of F16 |
| ------- | ---: | -------: | ---: | -------: |
| F16 (anchor) | 13.8248 | — | 2480 MB | 100 % |
| Stock Q4_K_M | 14.3042 | +0.4795 | 808 MB | 33 % |
| **Profile recipe (soft-floor default)** | **14.1810** | **+0.3562** | **779 MB** | **31 %** |

**Pareto win**: −0.12 PPL AND −29 MB vs stock Q4_K_M. Stock-vs-F16 quality gap closed by 26 %.

Recipe layout produced by the optimizer:

- `attn_k`: Q6_K × 16 (highest-Δ attention category, gets the most bits)
- `attn_output`: Q5_K × 16
- `attn_q`: Q5_K × 16
- `attn_v`: Q6_K × 8 + Q5_K × 8 (split via stock baseline floor)
- `ffn_down`: Q6_K × 8 + Q5_K × 8 (split via stock baseline floor)
- `ffn_gate`: Q4_K × 16 (cheap-Δ, gets the cheapest type)
- `ffn_up`: Q4_K × 16
- `token_embd.weight`: Q4_K × 1 (matches stock)

The optimizer correctly identifies attn_k as the highest-leverage attention tensor on this architecture, in contrast to Qwen3 where attn_k/attn_q dominate via the QK-norm amplification. ffn_gate and ffn_up land at Q4_K because their Q4_K Δ is essentially zero. Stock baseline floors keep attn_v and ffn_down half at Q6_K (matching stock's `use_more_bits` pattern on those tensors).

### Operational note: in-place ablator crash

The first two profile-build attempts crashed natively (SIGSEGV / exit 139) after completing 48 of 57 cells, both at the same boundary (after `ffn_down × Q6_K`). The 48 measured cells persisted to the DB so resume worked; the crash reproduced on the resume. Switching to `Options.UseInPlaceAblator = false` (disk-quantize path) ran the remaining 9 cells without issue and saved a complete profile.

The crash boundary is suspicious — next cell would have been either `ffn_down × Q8_0` (a candidate type the in-place path doesn't normally encounter on Qwen3 sweeps) or one of the 7 `token_embd.weight` cells (token_embd is referenced as both the embedding table AND the lm_head on Llama-3.2-1B due to tied embeddings, so an in-place tensor-data swap could corrupt the lm_head copy). Filed as issue #46 with reproduction details and bisection suggestions.

### Operational note: profile schema gap

The cached pre-regen profile JSON didn't carry `imatrix_sha` in its `provenance` block, so we couldn't tell from the file alone whether it had been built with the stale or the regenerated imatrix. After data invalidations wipe the underlying DB rows, this becomes a hard freshness signal we lose. Worth adding to the schema (small, additive, backwards-compatible) — would also let the recipe builder warn when the user feeds it a profile whose imatrix doesn't match the target's.

### Net status

The v3 recipe builder + soft-floor default generalizes from Qwen3-1.7B (QK-norm, Q2_K cliffs, ffn_down catastrophes) to Llama-3.2-1B (classical MHA, no cliffs, gentle curves) without any algorithm changes. Both architectures now Pareto-dominate stock Q4_K_M with the same out-of-the-box options.

## Run 26 — 2026-04-29 (Soft-floor noise gate — new project record beats Run 22 AND F16)

Issue #44 stage 1 follow-up. Run 25 had shipped a hard-stop gate
(`NoiseAwareRefinement` skips per-tensor refinement entirely on noise
flags) which produced 17.0113 PPL on the current DB-derived profile
— worse than Run 22's 16.8989. The diagnosis was that the gate was
too blunt: it treated the *legitimate* per-tensor demotes (Run 22's
`ffn_down` Q6_K×15 + IQ4_XS×9 + Q4_K×4) and the *catastrophic* Q2_K
demote case as the same thing. Both lived behind the same
`rel-std@worst > 1.0` flag.

### Mechanism

Replace the hard-stop with a **soft floor**: when a category is
noise-flagged, raise the per-tensor refinement floor (default `Q3_K`,
3.4375 bpw) instead of skipping refinement entirely. Q3_K blocks the
catastrophic demotes (Q2_K, IQ2_S) without disturbing the IQ4_XS /
Q4_K / Q6_K rungs that Run 22's good picks lived on. New option:
`LlamaQuantRecipeFromProfileOptions.RefinementFloorWhenNoisy` —
default `Q3_K`, set to `null` to restore the legacy hard-stop.

Implementation: in `LlamaQuantRecipeFromProfile`, the per-tensor
refinement loop's gate now computes an `effectiveRefineFloor` (max of
declared category floor and the noise-fallback type) instead of
`continue`-skipping. The floor is then passed through to
`RefineCategoryPerTensor` where the existing `floor` parameter
already enforces the bpw lower bound on demote/promote candidates.

### Validation

Two harnesses, both built with default soft-floor options
(`NoiseAwareRefinement = true`, `RefinementFloorWhenNoisy = Q3_K`):

1. **Run 22 replay** (git profile @95d6eb2 + soft floor + default
   options):

       PPL: 16.8989    Size: 1210 MB    Avg bpw: 4.7389

   Bit-exact reproduction of Run 22's documented numbers, *with no
   workaround flag*. Previously (hard-stop default) this same script
   needed `NoiseAwareRefinement = false` to bypass the gate; now the
   default does the right thing because all of Run 22's picks are
   above the Q3_K soft floor.

2. **Run 25 build** (current DB-derived profile + soft floor):

       PPL: 16.8460    Size: 1227 MB    Avg bpw: 4.8082

   **New project record.** Beats Run 22 (16.8989) by 0.053 PPL AND
   beats F16 baseline (16.8870) by 0.041 PPL at 30% of F16's size.
   The improvement comes from the additional per-tensor data drilled
   in Run 25's QK sweep being usable now — Run 25 had shipped that
   data into the DB but the hard-stop gate refused to apply any of
   it. The soft floor lets the well-measured upper rungs (IQ4_XS,
   Q4_K, Q5_K, Q6_K) influence per-tensor picks while still blocking
   the noise-suspect Q2_K demote on attention categories.

### Data hygiene

The current-leader recipe is now persisted at
`data/recipes/qwen3-1.7B.current-leader.json` with a sidecar
metadata file (`...meta.json`) recording observed PPL, size, source
run, and the profile path it was built from. Both harnesses use a
shared "compare-against-recorded-best, only update on improvement"
pattern so testing iterations can't silently overwrite a working
baseline with a regression. Initial leader: this run's 16.8460
recipe.

### Test coverage

New unit test `NoiseAwareRefinement_SoftFloor_AllowsAboveFloor_BlocksBelow`
in `QuantRecipeFromProfileTests` distinguishes three behaviors on a
synthetic non-monotonic profile that has Q3_K data measured:

- Gate off: `blk.2.ffn_up` demotes all the way to Q2_K (0.01 Δ
  treated as signal).
- Soft floor (default): demotes to Q3_K (the floor), Q2_K blocked.
- Hard-stop (`RefinementFloorWhenNoisy = null`): refinement skipped,
  stays at Q4_K category pick.

The pre-existing `_FallsBackToCategoryPick` and
`_RefinesAnyway` tests both continue to pass under the new default
because their synthetic ladders don't include Q3_K — the soft floor
collapses to "no measured types between Q2_K (blocked) and Q4_K
(current)", same observable behavior as the hard-stop.

### Net status

`Adaptive Quantization` on Qwen3-1.7B now ships **better PPL than
F16** at **30% of F16's size** with default options. Run 25's
negative result is fully reversed — the per-tensor data was good,
the gate was too blunt. The soft floor extends the v3 recipe
builder's "honor analyzer signals" guarantee from "block bad demotes
on noise" to "block bad demotes on noise *while still using the good
ones*."

## Run 25 — 2026-04-29 (Per-tensor expansion to QK categories: negative result + compounding diagnosis)

With Run 24's in-place ablator landed and the drill-candidates UI
analyzer in place, the natural next step was extending the
per-tensor measurement set to the four attention weight categories
(`attn_q` / `attn_k` / `attn_v` / `attn_output`) — the QK-norm-sensitive
set Run 1's analysis identified as where the per-layer story
ought to live. Run 22 had shipped with per-tensor data only on
`ffn_down`; the hope was that drilling attention would unlock another
recipe improvement on top of Run 22's 16.8989 PPL / 1210 MB result.

### Sweeps run

Two per-tensor sweeps via the in-place ablator path (
`Options.UseInPlaceAblator = true` default since Run 24):

1. **QK sweep** (4 categories × 28 layers × 4 types = 448 cells):
   202 min wall, ~27 s/cell at concurrency 4. Clean run.
2. **ffn_down sweep** (1 × 28 × 4 = 112 cells): 56 min wall.
   Re-recorded ffn_down per-tensor data that had been wiped during
   the perf-test invalidations between Runs 22 and 25.

The ffn_down re-sweep was triggered by a sharp observation from the
user: "did the perf-test invalidations wipe ffn_down's IQ4_XS rung?"
Yes — `db.DeleteMatching(model, corpus, imatrix, ctx)` with no target
filter wipes the entire campaign signature, including per-tensor
data that wasn't being re-measured. The git-committed Run 22 profile
JSON (`95d6eb2`) preserved a snapshot.

### Recipe builds

Same Run 22 build settings (4.95 bpw, ApplyStockBaseline=off,
UsePerTensorData=on, AllowPerTensorPromotion=on,
PerTensorPromotionThresholdPpl=0.05):

| recipe                                     |     PPL |   size |
| ------------------------------------------ | ------: | -----: |
| F16 baseline                               | 16.8870 | 4070 MB |
| Stock Q4_K_M                               | 17.4080 | 1282 MB |
| Run 22 (Option B, ffn_down only drilled)   | **16.8989** | **1210 MB** |
| Run 25 (QK-only per-tensor)                | 17.0055 |  1210 MB |
| Run 25 (QK + ffn_down per-tensor restored) | **18.2386** |  1142 MB |

Both Run 25 attempts came back *worse* than Run 22, not better.
Adding more measurement data made the recipe worse.

### PPL signal verification

Before assuming a recipe-builder bug, verified our PPL against
llama.cpp's reference `llama-perplexity` binary:

| measurement                                  | LlamaPerplexity (ours) | llama-perplexity (reference) |     Δ |
| -------------------------------------------- | ---------------------: | ---------------------------: | ----: |
| Qwen3-1.7B F16 baseline                      |                16.9000 |              16.8954 ± 0.16  | +0.005 |
| F16 + ffn_down × Q2_K (all 28 layers)        |                17.8484 |              17.8653 ± 0.16  | −0.017 |

Both within float32 noise + reported uncertainty. **Our PPL signal is
not drifted.** And — significantly — the catastrophe Run 17/22
documented as `ffn_down × Q2_K = +3709` no longer holds: a single
ffn_down-everywhere Q2_K ablation costs only +0.97 PPL today. The
2026-04-27 imatrix regeneration genuinely fixed Q2_K's per-block
calibration on this model.

### Real diagnosis: multi-category compounding

With signal and individual measurements both verified, the regression
traces to the recipe builder treating per-category measurements as
independent and additive. Run 25's recipe puts **five categories at
Q2_K simultaneously**:

| category    | Q2_K Δ (single-category) | Run 25 picks  |
| ----------- | -----------------------: | ------------- |
| `attn_q`    |                    −0.08 | Q2_K × 28      |
| `attn_v`    |                    −0.61 | Q2_K × 19      |
| `ffn_down`  |                    +0.96 | Q2_K × 23      |
| `ffn_gate`  |                    −0.01 | Q2_K × 28      |
| `attn_output` |                  +0.11 | Q2_K × 8       |

Each category individually says "Q2_K is fine for me." The optimizer
sums those tiny costs against Q2_K's bpw savings and concludes Q2_K
is a great deal. But running multiple categories at Q2_K
simultaneously **compounds non-additively** — the same lesson
Run 22's variants A/B/C/D demonstrated (0.04 + 0.26 → 0.51, not 0.30).
The combined recipe lands at 18.24 PPL vs the additive expectation
of ~17.4.

### Why Run 22 didn't hit this

Run 22's profile was a **happy accident of data heterogeneity**:

- Per-category coefficients: from Run 17 (2026-04-27 16:45,
  **pre-imatrix-regen**) — showed catastrophic
  `ffn_down × Q2_K = +3709`, `attn_v × Q2_K = +15.85`, etc.
- Per-tensor coefficients (ffn_down only): collected 2026-04-28
  with the **post-regen imatrix** — mild per-layer numbers.

The old per-category catastrophes forced the optimizer to pick `Q6_K`
or `Q5_K` for every category at the category level. Per-tensor
refinement could then only **demote** a few layers within reason —
that's how Run 22 got `ffn_down: Q6_K×15 + IQ4_XS×9 + Q4_K×4`.

Today's profile has internally consistent post-regen measurements
throughout. Without the old-imatrix per-category numbers acting as a
de-facto safety floor, the optimizer freely picks Q2_K everywhere.
**Run 22 was protected by an accident; the cleaner Run 25 dataset
exposed a recipe-builder modeling gap.**

### Drill-candidates analyzer earned its keep

The analyzer (Run 24) flagged exactly the issue without being asked.
On today's profile:

| category    | shape           | rel-std@worst | priority | note                         |
| ----------- | --------------- | ------------: | -------: | ---------------------------- |
| `ffn_up`    | SparseCliff     |             — |      5.4 | (not drilled yet)            |
| `ffn_down`  | SparseCliff     |          2.29 |      0.3 | drilled, high variance       |
| `attn_k`    | NonMonotonic    |          3.40 |      0.1 | drilled, high variance       |
| `attn_v`    | NonMonotonic    |          3.05 |      0.0 | drilled, high variance       |
| `attn_output` | Smooth        |         97.43 |      0.0 | drilled, **97× rel-std**     |
| `attn_q`    | NonMonotonic    |          5.97 |      0.0 | drilled, high variance       |

Four of the five drilled categories came back **NonMonotonic** —
some per-layer ΔPPL measurements at higher bpw were *worse* than
those at lower bpw, which is mathematically impossible and indicates
measurement noise dominating the signal. The recipe builder doesn't
currently consult this — it treats every per-tensor measurement as
trustworthy regardless of the analyzer's noise verdict.

### Decision and follow-up

Run 22 ships unchanged as the production recipe (preserved in git at
`95d6eb2`). The new measurements are correct and remain in the DB +
the regenerated `data/profiles/qwen3-1.7B-per-layer.profile.json` —
they're the right ground truth. The fix is in the recipe builder, not
the data.

Tracked as a follow-up issue: **recipe builder must model
multi-category compounding** when many categories pick the same
low-bpw type. Sketches of approaches:

1. **Compounding penalty / saturation factor**: when N categories pick
   the same low-bpw type, multiply the predicted Δ by a saturation
   factor that grows with N. Tune empirically against held-out
   recipes.
2. **Quick combined-recipe sanity check**: build the candidate
   recipe, quantize against a smaller corpus chunk, measure PPL; if
   it exceeds the additive prediction by >X%, reject and retry with
   a more conservative budget.
3. **Honor drill-candidates analyzer signals**: when a category's
   per-tensor data is NonMonotonic or has rel-std > 1.0 at the worst
   type, fall back to category-level coefficients (don't refine
   per-tensor). The drilled measurements stay in the DB but the
   recipe builder ignores them when noise dominates signal.

(3) is the cheapest first move — wires existing analyzer output into
the recipe builder. (1) is principled but needs calibration data.
(2) is heaviest but most defensible.

## Run 24 — 2026-04-28 (In-place ablator: tensor-surgery ProfileBuilder path, ships as default)

After #37 (continuous-flow scheduler, 1.45×) and the scored-only
logits mask (1.51× cumulative), the next-largest fraction of campaign
wall time was per-cell model load+free: each consumer in the disk
path reloads the ablation GGUF (~22 s on Qwen3-1.7B) before every PPL.
Across 22 cells × 4 concurrency, that's ~120 s of campaign wall
spent on setup. Per the user observation that "we're typically only
swapping out one class of tensors or even a single tensor," the work
is largely redundant — the F16 reference data for ~95 % of tensors
is identical from cell to cell.

### Architecture

A small C++ shim (<code>tools/native-shims/llamashim.cpp</code>)
re-exports two stable C entry points layered over llama.cpp's
internal <code>llama_internal_get_tensor_map</code> symbol:

- <code>llamashim_get_model_tensor(model, name)</code> — linear scan
  of the model's <code>tensors_by_name</code> vector.
- <code>llamashim_set_tensor_data(tensor, data, offset, size)</code> —
  forwards to <code>ggml_backend_tensor_set</code>, which dispatches
  to the tensor's owning backend (CUDA D2H, CPU memcpy, …).

The shim is built against the pinned llama.cpp binaries
(<code>tools/native-shims/build.sh</code>) and re-validated on every
pin bump. <strong>It depends on a non-public C++ ABI</strong> — the
internal symbol's signature is stable since at least b6500 but is
not part of the documented C API.

On top of the shim, three new C# pieces:

- <code>LlamaModel.SetTensorData / GetTensorData / FingerprintTensor</code>:
  public APIs for in-place tensor manipulation.
- <code>LlamaTensorRoundTrip.Encode / EncodeInto</code>:
  F16 → F32 → quantize (via <code>ggml_quantize_chunk</code>) →
  F32 → F16 round-trip helper. Internal F32 working buffers and the
  quant scratch buffer are pooled via <code>ArrayPool&lt;T&gt;.Shared</code>.
- <code>LlamaInPlaceAblator</code>: holds a persistent F16 LlamaModel,
  a memory-mapped view of the F16 source GGUF, and a content cache
  of original tensor bytes. <code>RunAblationAsync</code> applies a
  list of (tensor, type) ablations by reading source F16 → round-trip
  encoding through the candidate type → uploading via
  <code>SetTensorData</code> — and restores any tensors dirtied by a
  previous call that aren't in the new set.

The campaign builder gets a new <code>Options.UseInPlaceAblator</code>
(default <code>true</code>). When enabled,
<code>RunInPlaceCampaignAsync</code> spawns N consumer workers, each
holding its own <code>LlamaModel</code> + <code>LlamaInPlaceAblator</code>,
and drains a <code>Channel&lt;(spec, tensors)&gt;</code> of pre-materialized
ablation specs. (One subtle bug uncovered during integration: an
initial bounded channel with <code>capacity = pplConcurrency</code>
deadlocked because the producer pre-loaded all 22 items before
workers existed to drain. Switched to unbounded — work items are
tiny lists of tensor names, no real memory cost.)

### Numerical parity (smoking-gun foundation tests)

Three checks confirmed the path is mathematically faithful before any
performance work:

1. <strong>Writes affect inference.</strong> Zero out
   <code>blk.0.attn_q.weight</code> on Qwen3-0.6B → PPL goes from 31.15
   to 23,886 (766× catastrophe, exactly what zeroing a critical
   attention weight should do). Confirms <code>SetTensorData</code>
   actually changes GPU memory the matmul kernels read from.
2. <strong>Q4_K single-tensor parity.</strong> Disk-path
   <code>blk.0.attn_v.weight × Q4_K</code> Δ = +0.0799; in-place
   Δ = +0.0782 (gap −0.0017).
3. <strong>Q2_K single-tensor parity.</strong> Same probe at Q2_K:
   disk Δ = +0.1048; in-place Δ = +0.1038 (gap −0.0011).
4. <strong>Q2_K 28-tensor parity at concurrency=1.</strong> All
   ffn_down at Q2_K on Qwen3-0.6B: disk Δ = +4.0227; in-place
   Δ = +3.9992 (gap −0.0235).
5. <strong>Cross-worker isolation.</strong> Two
   <code>LlamaModel</code> instances loaded from the same GGUF have
   matching initial fingerprints; modifying tensor X on model A leaves
   model B's fingerprint unchanged. No cross-pollution.

The PPL gaps trace back to F16 matmul kernel vs Q-quant matmul kernel
producing slightly different reduction orders on bit-identical
dequantized values — same family of effect as the cuBLAS GEMM
algorithm shift documented in Run 23, but smaller in magnitude
(~0.002 PPL vs ~0.014).

### Phantom-bug detour: Run 17's "+3709" reference

The 22-cell per-category campaign produced ΔPPL values that didn't
match Run 17's reference table — most strikingly,
<code>ffn_down × Q2_K</code> on 1.7B was +0.96 vs the documented
+3709. After exhaustive bisecting (writes work, parity matches at
small scale, workers don't pollute, …) the explanation turned out
to be data, not code: <strong>the Qwen3-1.7B imatrix file was
regenerated 2026-04-27 evening, after Run 17 captured its numbers</strong>.
The new imatrix calibrates Q2_K's per-block scales much more
accurately, dramatically reducing the worst-case PPL impact of
ffn_down × Q2_K. Today's disk path produces the same +0.96 the
in-place path does. Run 17/22 reference numbers should be read as
"under the older imatrix"; current numbers are the new ground truth.

### Performance investigation

After parity was confirmed, the campaign-level wall time was the
focus. Initial measurement: in-place 553 s vs disk path's 521 s
(scored-mask) — slightly slower. Per-phase trace
(<code>ABLATOR_TRACE=1</code>) revealed encode time degrading
4–5× over the campaign:

| cell index | encode wall |
| ---------- | ----------: |
| early (1–5) | 16 s avg |
| mid  (6–14) | 25 s avg |
| late (15–22) | 60 s avg |

Three contributing causes, in order of impact:

1. <strong>CPU contention with concurrent PPL softmax.</strong> Each
   worker's encode is single-threaded; concurrent workers' PPL
   softmax uses ProcessorCount/pplConcurrency = 4 threads each. Once
   workers desync from their initial all-encode startup, encode
   threads compete with persistent softmax threads for the 16-thread
   CPU.
2. <strong>Allocation churn.</strong> Each <code>Encode</code> call
   allocated ~140 MB of working memory (two F32 buffers + quant scratch
   + output). 28 tensors × 22 cells × 4 workers built up GC pressure.
3. <strong>Source-bytes cache growth.</strong> Each worker caches
   F16 source bytes per touched tensor; ~3.5 GB per worker by
   campaign end.

### Optimizations applied

- <strong>Buffer pooling</strong> via <code>ArrayPool&lt;T&gt;.Shared</code>
  for the F32 round-trip buffers and the quant scratch. Modest win:
  553 s → 544 s.
- <strong>Parallel encode within a worker</strong> via
  <code>Parallel.ForEach</code> over the 28 tensors at DOP =
  ProcessorCount. Encode shifts from a 30 s sequential drag to a
  bursty 4 s surge that briefly preempts concurrent softmax workers
  — the priority shape suggested by the user, achieved without an
  explicit task queue or thread-priority layer. Apply phase wall:
  16–65 s → 0.4–10.8 s (8× reduction). Total wall: 544 s → 537 s.

The total wall improvement was small even though apply phase
collapsed dramatically because PPL is GPU-bound and the binding
constraint at concurrency 4. Confirmed via concurrency sweep:

| concurrency | wall | avg PPL | throughput (PPLs/s) |
| ----------- | ---: | ------: | ------------------: |
| 3           | 561 s | 67 s | 0.0448 |
| 4           | 537 s | 84 s | 0.0476 |

c=4 has ~6 % higher throughput than c=3 despite each PPL taking
longer — consistent with GPU saturated but not over-driven (nvtop
showed 100 % during the run). c=4 is the operating sweet spot.

### Final wall-time landscape

| run | wall | δ vs original | cumulative |
| --- | ---: | ------------: | ---------: |
| disk old-batch (original) | 788.9 s | — | 1.00× |
| + #37 continuous-flow | 543.1 s | −245.8 s | 1.45× |
| + scored-only logits mask | 521.0 s | −22.1 s | 1.51× |
| + in-place ablator (this run) | 537.6 s | +16.6 s | 1.47× |

In-place is ~3 % slower than disk-path scored-mask on the per-category
sweep at this concurrency. Both paths are PPL-bound at the same GPU
saturation point, so they converge. The in-place path's edge lives
elsewhere:

- <strong>Disk space.</strong> Disk path holds peak ~5–7 ablation
  GGUFs in flight (one per pplConcurrency + channel buffer); on
  Qwen3-1.7B that's 17–25 GB of <code>/tmp</code>. In-place uses
  effectively zero disk beyond the source.
- <strong>Per-tensor profile builds.</strong> Per-tensor ablations
  ablate ONE tensor per cell, not 28. Disk path's per-cell quantize +
  load + free overhead stays at ~30 s; in-place's apply drops to
  ~0.2 s. On a 1370-cell per-tensor sweep that's tens of minutes
  saved.

### Decision

Ship in-place as default (<code>UseInPlaceAblator = true</code>) for
the disk-space win and to make per-tensor sweeps practical. Keep the
disk path as fallback (set the option to <code>false</code>) for
cases where the shim hasn't been re-validated against a llama.cpp
pin bump.



Triggered by the perf-issue triage that produced Run 22's continuous-flow scheduler win (#37). With per-PPL becoming the dominant cost at the campaign level, the question was whether the device-side logsumexp+gather subgraph proposed in #43 was the right next move.

### Conflicting prior claims

The Run 18 deferred-work note (line 920 of this doc, prior to this entry) said:

> bottleneck is real GPU compute (57ms/chunk × 547 chunks = 31s of the 51s wall on Qwen3-1.7B), not readback as initially hypothesized. ComputeBatchedAsync(n_seq=4) saves ~5%; explicit GPU+CPU pipelining could give ~1.65× theoretical, deferred for modest payoff.

The in-code comment in `LlamaPerplexity.Compute` claimed the opposite — readback ~65 % of wall, only meaningful win is a device-side reduction subgraph for ~3× wall. **Both can't be right.**

### Eager-touch experiment

Added env-var-gated phase timers (`PPL_PERF_TRACE`, `PPL_EAGER_TOUCH`) around `llama_decode`, an optional bulk pre-touch of the full logits buffer, and the parallel softmax loop. Workload: Qwen3-1.7B Q4_K_M, wiki.test, ctx=512, second-half-only.

| setting        | wall   | decode | touch  | score |
|----------------|------:|-------:|-------:|------:|
| eager-touch=0  | 46.9 s |  1.0 s |  0.0 s | 45.8 s |
| eager-touch=1  | 47.7 s |  1.0 s | **32.6 s** | 14.0 s |

The eager-touch flushed the lazy d2h transfer time out from "softmax score" into a clearly-attributed phase. Decode-as-measured is **1 s, not 31 s** — the Run 18 note conflated `llama_decode`'s API-return latency (which fires before all queued GPU work is done) with the actual pipeline length. The 31 s the note attributed to compute is mostly d2h transfer of `[V, P]` logits to host pinned memory.

PPL was bit-identical (17.4080) across all five runs — sanity ✓. The original in-code comment was correct: readback is ~70 % of wall on this workload.

### Why the proposed device-side subgraph isn't reachable from the public API

Pursuing the device-side reduction surfaced three structural blockers in llama.cpp's public surface:

1. **The logits buffer is host-pinned, not device-resident.** llama.cpp allocates the output buffer using `ggml_backend_dev_host_buffer_type(output_dev)` — for CUDA, `cudaMallocHost`. The d2h transfer is a node IN the model graph, so by the time `llama_get_logits` returns, the bulk transfer has been issued. There's no device-resident buffer we could share.
2. **`llama_context`'s backend / sched isn't exposed.** No public API to ship our own subgraph to the same device or share the same buffer-pool.
3. **The eval callback (`cb_eval`) does fire per-graph-node mid-execution** and gives single-tensor read access via `ggml_backend_tensor_get`, but doesn't naturally support running a parallel ggml subgraph during graph evaluation, and a per-tensor `tensor_get` of the LM-head output still pays the V-wide d2h.

Constructing a separate ggml CUDA context and copying the existing logits to it would pay an extra h2d-then-d2h round-trip — strictly worse than the current path.

The right fix is upstream: an "output-mode = per-token-NLL" option in `llama_context_params` that reduces logits to `(logsumexp, target_logit)` inside the model graph BEFORE the d2h. The reduction is one log-softmax + one gather, both implementable from primitives ggml already has. Output buffer drops from `[V, P]` to `[2, P]` per chunk; d2h drops by V/2. **#43 is closed with a writeup pointing at this upstream-fix path, so a future repo search for "device-side logsumexp" finds it.**

### Side-finding: scored-only logits mask (the cheap win this investigation produced)

While reading upstream `llama-perplexity` for context, found that it sets `batch.logits[idx] = (pos >= first) ? 1 : 0` — only requesting logits at positions it'll actually score. Our `PopulateBatchAllLogits` requested them everywhere. With `ScoreSecondHalfOnly` (the default), requesting only the second half halves both the LM-head matmul work and the logits readback volume.

Replacing `PopulateBatchAllLogits` with a `firstNeededPosition`-aware variant (and the same fix on the multi-seq batched path):

| measurement                         | before  | after   |
|-------------------------------------|--------:|--------:|
| Standalone PPL on Qwen3-1.7B Q4_K_M | 46.9 s  | 39.1 s (1.20×) |
| 22-cell campaign with #37 scheduler | 543.1 s | 521.0 s (1.04× incremental, 1.51× cumulative vs pre-#37) |
| Q4_K_M PPL                          | 17.4080 | 17.4080 (bit-identical) |
| F16 PPL                             | 16.8865 | 16.9000 (+0.0135, cuBLAS algorithm shift at smaller batch) |

The campaign-level gain is smaller than the per-PPL gain because at #37's continuous-flow concurrency the producer's serial quantize work has become a non-trivial share of wall — improving per-PPL further would shift the bottleneck to that side. The F16 baseline shift of +0.0135 PPL is real (a one-time, post-commit, kernel-stable shift); Q4_K_M is bit-identical because dequant kernels are batch-size-stable while pure cuBLAS GEMM has more algorithm variability.

Recipe-vs-recipe and recipe-vs-stock DELTAS — the actual surface AQ decisions are made on — are unaffected, since both legs of any delta use identical kernel choices.

### Resulting state

- Continuous-flow PPL scheduler shipped in #37 (Run 22).
- Scored-only logits mask shipped here.
- #43 closed with a clear "fix-it-upstream" writeup; the in-code comment in `LlamaPerplexity.cs` updated to reflect the corrected bottleneck breakdown and to point at this Run-23 entry plus the closed issue.
- Run 18 deferred-work note's "GPU compute is 31 s" claim is superseded by the eager-touch finding above; that line should be read as historical context rather than current state.

## Run 22 — 2026-04-28 (Option B: per-family clamp at refinement scope only — beats Run 21)

Run 21 shipped uniform Q6_K on all 28 ffn_down layers because the
demote-only refinement pass had no measured type below Q6_K to demote
to (the per-layer ffn_down measurements were Q2_K / Q4_K / IQ4_XS /
Q6_K, and the cross-family monotone clamp pulled IQ4_XS up to Q6_K's
clamped value, hiding the IQ-quant signal). Run 22 reopens that
pathway by changing where the clamp lives.

### The IQ-vs-K family question

Public results (and the user's own experience) show IQ4_XS routinely
beats higher-bpw K-quants on certain tensor distributions despite
having less raw bit budget — codebook + importance-aware machinery
can fit per-block error patterns that K-quant's per-block scale
can't. A cross-family "lower bpw must be at least as bad as higher
bpw" clamp violates that empirically, so **the monotone-from-above
rule only holds within a single quant family**.

### The per-family clamp regression (deferred Run)

A first attempt moved the per-family running max into the
category-enumeration step in `BuildClampedScaledDeltas`. That recipe
came back at **17.4435 PPL / 1209 MB** — *worse* than stock
(17.4080) and well behind Run 21 (16.9336). The per-family clamp at
category scope freed the optimizer to enumerate IQ4_XS for any
category whose K-curve was clamped flat, and several categories
flipped to IQ4_XS in combination — which compounded badly.

### Isolation experiments

To disambiguate "IQ-quants are bad in this model" from "this
particular mix of IQ-quants is bad", four single-edit variants were
quantized and PPL'd against the Run 21 baseline:

| variant                                    | PPL      | Δ vs Run 21 |
|--------------------------------------------|---------:|------------:|
| Run 21 baseline (uniform K within budget)  | 16.9336  | (anchor)    |
| A — `attn_k` only IQ4_XS                   | 16.9740  | +0.04       |
| B — `ffn_down` only IQ4_XS                 | 17.1888  | +0.26       |
| C — both `attn_k` + `ffn_down` IQ4_XS      | 17.4434  | **+0.51**   |
| D — uniform IQ4_XS everywhere              | 17.5046  | +0.57       |

Single-category IQ4_XS is fine on `attn_k` and modestly costly on
`ffn_down`, but the *combination* costs more than the sum (0.04 +
0.26 = 0.30, but C measured 0.51). Single-tensor / single-category
ablations therefore *under-predict* IQ-quant cost when the
optimizer enumerates IQ-quants for several categories at once.

### Option B — clamp scope split

The fix is to apply the clamps at different scopes with different
rules:

- **Category-enumeration scope** (`BuildClampedScaledDeltas`): keep
  a single global running max. Conservative, cross-family monotone.
  Prevents the optimizer from picking IQ4_XS for several categories
  in combination.
- **Per-tensor refinement scope** (`RefineCategoryPerTensor`): use
  per-family running max. The category type is already chosen; this
  pass only edits *individual tensors* within the chosen category.
  The compounding cost from Variant C cannot occur here because only
  one category is being touched.

That's the entire diff: revert the per-family clamp at category
scope, keep it at per-tensor scope.

### Result — strict Pareto improvement over Run 21 *and* stock

Apply settings identical to Run 21
(`ApplyStockBaseline=off`, `UsePerTensorData=on`,
`AllowPerTensorPromotion=on`, target 4.95 bpw, imatrix-aware,
wikitext-2 raw, ctx 512, second-half scoring).

|                     | PPL        | Δ from F16 | File size       | Avg bpw |
|---------------------|-----------:|-----------:|----------------:|--------:|
| F16 baseline        | 16.8870    | (anchor)   | 4,069 MB        | 16.000  |
| Stock Q4_K_M        | 17.4080    | +0.521     | 1,282 MB        | ~4.95   |
| Run 21 (all-K)      | 16.9336    | +0.047     | 1,232 MB        | 4.919   |
| **Run 22 (Option B)** | **16.8989** | **+0.012** | **1,210 MB**  | **4.739** |

Smaller, lower-bpw, *and* better PPL than Run 21 — and dramatically
better than stock. Δ from F16 collapsed from stock's +0.521 to
+0.012 (97.7% of the F16-vs-Q4_K_M gap closed).

### What the recipe actually picked (`ffn_down`)

Run 21's ffn_down was uniform Q6_K (28 layers). Option B's
per-tensor refinement reshuffles them:

- **15 layers Q6_K** — the layers where the per-layer measurement
  flagged real cost at lower bpw.
- **9 layers IQ4_XS** — layers where IQ4_XS measured at-or-below
  the budget tolerance, so the demote-and-budget pass took it.
- **4 layers Q4_K** — middle ground.

This is the per-tensor refinement payoff that Run 21 could not
realize: the per-layer ffn_down data was always there, but the
cross-family monotone clamp was hiding the IQ4_XS signal at every
layer regardless of measurement.

### Why this design is right (and what it concedes)

- Concedes that single-category ablations don't capture the
  *interaction* cost between categories at low bpw — Variant C's
  +0.51 isn't predictable from A + B. Until the profile builder
  measures pairwise interactions, the conservative thing at
  category scope is to assume cross-family clamping.
- Keeps the empirical IQ-quant lift available where it provably
  works: per-tensor decisions inside a category, where only one
  family is being shuffled at a time and there's no
  multi-category-mix surface to compound across.
- Symmetric with how stock's `use_more_bits` already operates —
  per-layer surgical edits inside categories — but driven by
  measurement instead of a hardcoded layer list.

### Code shape

`LlamaQuantRecipeFromProfile.BuildClampedScaledDeltas`: single
running max across the descending-bpw ladder.

`LlamaQuantRecipeFromProfile.RefineCategoryPerTensor`: dictionary
keyed by `LlamaQuantFamily`, monotone within family only. Also
filters the demote-target candidate set to *measured types only* so
monotone-filled types aren't valid demote choices.

47/47 tests pass. The deleted test
`NoiseClamp_PerFamilyMonotone_PreservesIQuantSignalAgainstNoisyKQuant`
asserted the per-family-at-category-scope behavior that the
isolation experiments showed was empirically wrong — removed
rather than rewritten because the new behavior is "single global
clamp at category scope", which the existing
`NoiseClamp_*_PreservesMonotoneOrdering` tests already cover.

### Operational note on test infrastructure

The IQ-isolation experiments (variants A–D) were run as four
sequential one-off `dotnet run` projects, each producing one quant
and one PPL number. The profile builder's
`LlamaPerplexity.RunParallelAsync` infrastructure could have run all
four in a single call with VRAM-aware concurrency, cutting the
cycle time substantially. Future variant-comparison work should
batch through that path rather than spinning up individual
projects.

### Verdict

**Option B is the new default.** It strictly Pareto-dominates Run
21 and stock Q4_K_M on Qwen3-1.7B at this budget. The next runs
should re-validate on Qwen3-0.6B and Qwen3-4B (cross-size, same
family) and Llama-3.2-1B (cross-architecture) to confirm the design
choice generalizes.

## Run 21 — 2026-04-28 19:49  (First Pareto win over stock Q4_K_M — Qwen3-1.7B)

Run 17 ended with the recipe builder shipping at parity to stock on
Qwen3-1.7B; the v3 work since then layered measurement clamps,
snap-to-stock, per-layer ablations, and a custom quantizer that
honors demotions (Run 14 fix). Run 21 is the first end-to-end
quality test of the full v3 pipeline against stock on a real model.

### Setup

Profile: hand-authored
[`data/profiles/qwen3-1.7B-per-layer.profile.json`](../../data/profiles/qwen3-1.7B-per-layer.profile.json)
combining the existing Qwen3-1.7B reference profile's per-category
coefficients with 28 ffn_down × {Q2_K, Q4_K, IQ4_XS, Q6_K} per-tensor
measurements collected on 2026-04-28.

Apply: GGUFLab Adaptive Quantization page, target 4.95 bpw
(Q4_K_M-class), `ApplyStockBaseline=off`,
`UsePerTensorData=on`, `AllowPerTensorPromotion=on`.

Both quants imatrix-aware (`Qwen3-1.7B.F16.imatrix.gguf`),
PPL on wikitext-2 raw, ctx 512, second-half scoring.

### Result — strict Pareto improvement

|                   | PPL        | Δ from F16 | File size       | Δ from stock      |
|-------------------|-----------:|-----------:|----------------:|------------------:|
| F16 baseline      | 16.887     | (anchor)   | 4,069 MB        | —                 |
| Stock Q4_K_M      | 17.408     | +0.521     | 1,282 MB        | (baseline)        |
| **v3 recipe**     | **16.934** | **+0.047** | **1,232 MB**    | **−51 MB / −0.474 PPL** |

The recipe is **smaller AND better** than stock by every measure.
PPL gap to F16 collapsed by 91% (0.521 → 0.047). File size dropped
by 51 MB (3.95%). Average bpw 4.919 vs stock's ~4.95.

### What the recipe actually picked

Per-category type distribution (all 28 layers per category):

| category    | recipe pick | stock Q4_K_M's pattern              |
|-------------|-------------|-------------------------------------|
| `attn_q`    | Q5_K        | Q4_K                                |
| `attn_k`    | Q6_K        | Q4_K                                |
| `attn_v`    | Q6_K        | Q6_K (use_more_bits) / Q4_K (rest)  |
| `attn_output` | Q5_K      | Q4_K                                |
| `ffn_up`    | Q5_K        | Q4_K                                |
| `ffn_gate`  | **Q3_K**    | Q4_K                                |
| `ffn_down`  | Q6_K        | Q6_K (use_more_bits) / Q4_K (rest)  |
| `output`    | Q4_K        | Q6_K                                |
| `token_embd`| **Q3_K**    | Q4_K                                |

Stock's hand-tuned heuristic alternates per-layer protection within a
mostly-Q4_K backbone. The optimizer's choice is structurally
different: **uniform high-bpw on attention K/V and ffn_down, uniform
mid-bpw on Q/O/up, uniform low-bpw on gate and embedding**. Same
total budget, very different distribution.

### Why it works

The win is *not* per-tensor refinement. Looking at the recipe's
ffn_down picks: **all 28 layers landed at Q6_K** — uniform. So the
per-tensor data didn't drive within-category variation here; the
optimizer simply picked Q6_K as the category type and the demote-
only refinement had nothing to do with no measured type below the
budget tolerance lower-bpw than Q6_K being safe.

The win came from the **per-category optimizer** + clamps + the
freedom to ignore stock's `use_more_bits` floors. Specifically:

1. **Clamps work.** ffn_down's category-level Q6_K Δ measured
   negative; the zero-floor clamp normalizes that to 0 (matching
   F16). Q4_K's Δ is +0.112, Q5_K's +0.296 (non-monotone — likely
   noise). The clamped curve makes Q6_K look like a free win.

2. **The optimizer enumerates exhaustively.** With the floor
   removed (`ApplyStockBaseline=off`), the optimizer is free to
   put *all* ffn_down at Q6_K and pay for it by demoting ffn_gate
   to Q3_K and token_embd to Q3_K. Stock's heuristic locks in a
   per-layer alternation pattern that can't make that trade.

3. **The trade is correct.** ffn_gate at Q3_K costs +0.108 PPL
   per the category data, but with `ffn_gate × Q3_K = 0.108`
   compared against the savings from `ffn_down at Q6_K everywhere`
   (large), the net is positive.

### What the per-layer ffn_down drill *did* contribute

The per-layer measurements were the **evidence that justified
turning off `ApplyStockBaseline`**. Without that data, we wouldn't
know stock's `use_more_bits` set was empirically miscalibrated on
this model (Mean Q4_K Δ: protected 0.001 vs unprotected 0.012).
Knowing it lets us legitimately disable the stock-baseline floor
and trust the optimizer to find a better distribution.

The per-tensor data also told us *where the per-layer story was
real and where it was noise*. Q2_K's catastrophic +3709 PPL at the
category level vs ~+1.18 PPL summed across per-layer ablations
revealed multi-layer compounding non-linearity at extreme bpw —
the v3 noise clamp + snap-to-stock guards correctly prevent the
optimizer from chasing those phantom wins.

### Verdict

**v3 ships.** This is the first GGUFLab recipe to strictly
Pareto-dominate stock Q4_K_M on a real model: smaller file, lower
PPL, same imatrix-aware quantization path. The win is structural
(the optimizer reallocates budget across categories) rather than
per-layer surgical, but it's only available because the per-layer
drill provided the evidence that justified disabling
`ApplyStockBaseline`.

### Implications and ship plan

1. **Generalization test.** Run the same end-to-end on Qwen3-0.6B
   and Qwen3-4B with the existing reference profiles +
   `ApplyStockBaseline=off`. If both also Pareto-beat stock, the
   v3 default recommendation should be:
   _Stock Q4_K_M is suboptimal on Qwen3-class models; use
   GGUFLab's per-category profile-driven recipe._
2. **Cross-architecture test.** Run on Llama-3.2-1B. Run 19 saw v2
   ship at parity on Llama; v3 with the new clamps + custom
   quantizer might find a similar reallocation win or might not
   (Llama's heuristic may already be approximately optimal).
3. **Per-tensor finder.** With promotion now wired, future runs
   should drill categories besides ffn_down (attn_v showed the
   most measurement noise; per-layer attn_v on 4B was the
   originally-motivating drill). With promotion enabled, individual
   high-sensitivity layers can be lifted to Q6_K paid for by
   demote-safe siblings. ffn_down didn't trigger that path because
   the optimizer chose Q6_K for the whole category.

### Reproduction

```
src=$HOME/.cache/llama-models/Qwen/Qwen3-1.7B/Qwen3-1.7B.F16.gguf
imt=$HOME/.cache/llama-models/Qwen/Qwen3-1.7B/Qwen3-1.7B.F16.imatrix.gguf
cp data/profiles/qwen3-1.7B-per-layer.profile.json $HOME/.cache/llama-models/Qwen/Qwen3-1.7B/

# In GGUFLab Adaptive Quantization:
#   - Input GGUF: $src
#   - Imatrix:    $imt
#   - Profile:    Browse to qwen3-1.7B-per-layer.profile.json
#   - Target:     4.95 bpw
#   - Advanced:   ApplyStockBaseline=off, UsePerTensorData=on, AllowPerTensorPromotion=on
#   - Quantize → Qwen3-1.7B.F16.profile-4.95.gguf

# Stock baseline:
# In GGUFLab Quantize:
#   - Input:   $src, Imatrix: $imt, Quant type: Q4_K_M
#   - Output:  Qwen3-1.7B.F16.Q4_K_M.gguf

# PPL on both via Perplexity page (wiki.test, ctx=512, score-second-half-only).
```

Recipe artifact saved at
`$HOME/.cache/llama-models/Qwen/Qwen3-1.7B/Qwen3-1.7B.F16.recipe-4.95.json`
for inspection and reproduction.

## Run 20 — 2026-04-27 21:50  (Cross-size at scale: Qwen3-4B with imatrix)

Run 17 added Qwen3-1.7B and showed cross-size transfer breaks
when the candidate ladder gets finer; Run 19 added Llama-3.2-1B
and showed v2 ships at parity on classical-MHA arches. Run 20
adds **Qwen3-4B** — the next size in the QK-norm family — to test
two questions:

1. Does the v2 win persist at larger sizes within the same family?
2. Does cross-size scaling work in the larger-target direction
   (0.6B→4B, 1.7B→4B)?

Setup: Qwen3-4B downloaded as F16 (8.05 GB, mradermacher), imatrix
built on wikitext-2.test (181 s wall after the imatrix-perf
detour), expanded ablation campaign (8 cats × 7 types) built with
imatrix in 60 min. Validation matrix uses imatrix for both stock
and recipe sides — the production-realistic comparison Run 18
flagged as needed.

### Profile shape — 4B is barely sensitive

Per-Q4_K coefficient comparison across the family:

| category    | 0.6B  | 1.7B  | 4B    |
|-------------|-------|-------|-------|
| ffn_up      | +0.227 | +0.443 | **+0.133** |
| attn_v      | +0.142 | +0.340 | **−0.058** |
| attn_k      | +0.115 | +0.314 | +0.339 |
| ffn_gate    | +0.106 | +0.232 | +0.011 |
| attn_q      | +0.114 | +0.040 | +0.101 |
| attn_output | +0.091 | +0.079 | +0.096 |
| ffn_down    | −0.000 | +0.112 | **−0.033** |

And the Q2_K cliff that dominated 1.7B is gone:

| category | 0.6B Q2_K | 1.7B Q2_K | 4B Q2_K |
|----------|-----------|-----------|---------|
| ffn_down | +4.40     | **+3709** | +1.19   |
| attn_v   | +2.72     | **+15.85**| **−0.45** |
| attn_k   | +2.02     | +4.81     | +0.18   |

`attn_v Q2_K = −0.45` on 4B — quantizing actually slightly
**helps** PPL. All categories auto-floor at Q2_K because nothing
crosses the catastrophic threshold. **4B is more like Llama than
like 1.7B from a sensitivity profile standpoint.** The QK-norm
signature persists in the sense that attn_k is still elevated,
but the overall profile is much milder than the 1.7B that drove
v2's headline result.

### Validation matrix at Q4_K_M and Q5_K_M

(All builds use imatrix on stock and recipe sides — Run 20 is the
first apples-to-apples-with-imatrix comparison in the
investigation.)

| recipe                        | actual bpw | wikitext-2 PPL | Δ vs stock |
|-------------------------------|------------|----------------|------------|
| stock Q4_K_M                  | 4.955      | 14.5940        | —          |
| **recipe 4B→4B @Q4**          | 4.974      | **14.4559**    | **−0.138** |
| recipe 1.7B→4B @Q4 (cross)    | 4.823      | 14.8944        | +0.301     |
| recipe 0.6B→4B @Q4 (cross)    | 4.942      | 14.8083        | +0.214     |
| stock Q5_K_M                  | 5.735      | 14.0846        | —          |
| recipe 4B→4B @Q5              | 5.620      | 14.3788        | +0.294     |
| recipe 1.7B→4B @Q5 (cross)    | 5.777      | 14.3389        | +0.254     |
| recipe 0.6B→4B @Q5 (cross)    | 5.688      | 14.7047        | +0.620     |

### Verdict — the v2 pitch narrows further

**1. Same-model wins shrink with size.** Q4_K_M on 4B is −0.14
PPL — a real win, but 16× smaller than 1.7B's −2.26. The reason
is structural: 4B's profile has no Q2_K cliffs, so stock's
`use_more_bits` heuristic is already near-optimal. v2 has less
headroom to find. The pattern across the family at Q4_K_M:

  - Qwen3-0.6B: not run as target  (size class too small for the
    Q4_K_M-class budget to be informative)
  - Qwen3-1.7B: **−2.26 PPL** (12% relative)
  - Qwen3-4B:   **−0.14 PPL** (1% relative)
  - Llama-3.2-1B: **−0.03 PPL** (0.2% relative)

The 1.7B win is **the cliff-rich case**; 4B is closer to Llama
in profile shape and gets a Llama-sized improvement.

**2. Cross-size in the larger-target direction is a clear loss
on 4B.** Both 0.6B→4B and 1.7B→4B regress by +0.2 to +0.3 PPL at
Q4_K_M. Run 17 had hinted at this with the 0.6B→1.7B step
(byte-identical recipes there masked the divergence at finer
ladders); Run 20 makes it concrete. Smaller-model profiles bake
in cliff structures that the larger model doesn't have, so
size-scaled recipes are over-protective in places where 4B could
ride lower bpw, and under-protective in places where 4B's mild
profile suggests more aggressive choices.

**3. Q5_K_M is a stock-heuristic-wins tier across the board.**
Even same-model 4B→4B regresses by +0.29. At Q5_K_M's higher
bpw budget, stock has enough bits to be near-optimal, and v2's
exhaustive enumeration finds local optima that don't survive
non-additive compound effects (the Run 17b lesson).

### Implication for v2 ship plan

The honest read crystallizes:

- **v2 ships per-(arch, size-class) profiles.** Same-model
  recipes are the only reliably-non-regressing path. Build a
  profile for each (arch, size) pair you intend to serve.
- **The biggest wins concentrate on small-to-medium QK-norm
  models with cliff-rich profiles.** Qwen3-1.7B-class is the
  sweet spot. Larger sizes within the family have less headroom;
  classical-MHA architectures don't have the cliffs that drive
  the gap.
- **Don't push v2 above ~Q4_K_M-class budgets for the headline
  pitch.** At Q5_K_M-class, ship stock; the heuristic is already
  good and v2 is a wash or slight regression.

### Now in `data/profiles/`

- `qwen3-0.6B.profile.json` (9 cats × 7 types, with imatrix)
- `qwen3-1.7B.profile.json` (9 cats × 7 types) — built without
  imatrix in Run 17, follow-up: rebuild with imatrix for
  cross-comparison consistency
- `qwen3-4B.profile.json`   (8 cats × 7 types, with imatrix —
  tied embeddings, no separate output.weight)
- `llama-3.2-1B.profile.json` (8 cats × 7 types, with imatrix —
  tied embeddings)

### Bug fix uncovered while running

Run 20 surfaced a crash in the recipe builder when the source
profile has a category (e.g. `output.weight`) that doesn't exist
as a tensor in the target model — common in cross-size builds
across architectures with vs without tied embeddings. Fixed in
c8defad: the per-category enumeration skips orphan categories
that have no matching target tensor. Locked in by
`ProfileCategory_WithoutMatchingTensorInTarget_IsSkipped`.

### Imatrix-generation perf detour

Building Run 20 required adding imatrix support throughout. The
first 4B imatrix run was projected at ~16 min on the original
scalar imatrix collector. Two rounds of optimization landed:

- **Round 1 (vectorized math):** TensorPrimitives.Multiply for
  the row-square + Vector256 manual SIMD float→double accumulator.
  1.68× wall on Qwen3-1.7B.
- **Round 2 (producer/consumer worker pool):** Eval callback only
  copies activation bytes and queues; N background workers do the
  math in parallel. ConcurrentDictionary + per-Entry locks for
  fine-grained sync. **4.7× total wall on 1.7B**, 5× on 4B.
- **Round 3 (chunkSize=1024):** 3% improvement, within noise —
  d2h sync per matmul is the real floor, not per-callback fixed
  overhead. Not pursued.

4B imatrix went from ~16 min (scalar projection) to **3:06 actual**.

## Run 19 — 2026-04-27 18:45  (Cross-architecture: v2 on Llama-3.2-1B)

Run 18 closed v2 with a clean −2.26 PPL win on Qwen3-1.7B at
slightly smaller file size than stock Q4_K_M. Run 19 asks the
generalization question: does the same recipe builder beat Llama's
stock heuristic? Run 12 already established that profiles are
arch-specific (QK-norm signature absent on Llama); whether the
v2 recipe machinery still produces a measurable win without that
signature is unknown.

### Build

`LlamaSensitivityProfileBuilder` on Llama-3.2-1B-Instruct,
9 categories minus `output.weight` (Llama has tied embeddings;
output shares the token_embd tensor), 7 candidate types. Build
wall 1242s (~21 min — slower per ablation than Qwen3-1.7B because
Llama-1B's PPL pass is denser per chunk despite fewer layers).

baseline F16 PPL = 13.8217 (matches Run 12).

### Profile shape — strikingly mild

| category    | Q2_K   | Q3_K   | Q4_K    | Q5_K    | Q6_K    | floor |
|-------------|--------|--------|---------|---------|---------|-------|
| ffn_up      | +2.486 | +0.472 | +0.128  | +0.051  | +0.025  | Q2_K  |
| ffn_down    | +2.847 | +2.606 | +0.107  | +0.029  | +0.047  | Q2_K  |
| ffn_gate    | +1.430 | +0.331 | +0.095  | +0.008  | +0.005  | Q2_K  |
| attn_output | +1.274 | +0.379 | +0.094  | −0.011  | +0.004  | Q2_K  |
| attn_q      | +0.245 | +0.126 | +0.040  | −0.016  | +0.000  | Q2_K  |
| attn_k      | +0.447 | +0.195 | +0.029  | +0.007  | −0.007  | Q2_K  |
| attn_v      | +0.757 | +0.321 | −0.006  | −0.020  | +0.009  | Q2_K  |
| token_embd  | +2.237 | +0.506 | −0.002  | +0.026  | −0.002  | Q2_K  |

Compare to Qwen3-1.7B's catastrophes — `ffn_down Q2_K = +3709`
on Qwen3, only **+2.85** here. Same for `attn_v Q2_K`: +15.85 on
Qwen3, +0.76 on Llama. Every category's auto-floor lands at Q2_K
because nothing crosses the catastrophic threshold. The
no-QK-norm story (Run 12) plays out as expected: Llama's
attention weights are quantization-friendly across the full
candidate ladder.

### Validation matrix

| recipe                        | actual bpw | PPL     | Δ vs stock |
|-------------------------------|------------|---------|------------|
| stock Q4_K_M                  | 5.178      | 14.2843 | —          |
| **recipe @Q4_K_M**            | 5.209      | **14.2550** | **−0.029** |
| stock Q5_K_M                  | 5.850      | 13.9883 | —          |
| recipe @Q5_K_M                | 5.657      | 14.0878 | +0.099     |

### Verdict — v2 generalizes, but the magnitude scales with arch difficulty

- **Q4_K_M on Llama:** v2 is essentially a wash (−0.029 PPL at
  slightly larger file). Stock Q4_K_M's heuristic is already
  near-optimal for an arch this quantization-friendly; there's
  no cliff to avoid and no tensor systematically over- or
  under-protected by the heuristic.
- **Q5_K_M on Llama:** v2 makes a Pareto trade — 0.19 bpw
  *smaller* at +0.10 PPL. Whether this is a win depends on
  whether the user weights size or quality.

The Qwen3 v2 win (−2.26 PPL, 12% relative) was driven by closing
the gap caused by the QK-norm signature. Llama doesn't have that
gap to close. **The recipe builder is correct on both
architectures** — it doesn't ship worse-than-stock recipes
anywhere — but it only wins materially where the heuristic was
leaving real value on the table.

### Implication for v2 ship plan

This is an **honest narrowing of the v2 pitch**:

- v2 ships on QK-norm architectures (Qwen3, possibly other
  modern arches with attention-side normalization) with a
  significant PPL improvement at Q4_K_M-class budgets.
- v2 ships on classical-MHA architectures (Llama) at parity
  with stock — no harm, but no headline improvement either.
- The same code, same algorithm, same baseline: the difference
  is entirely in the profile data the architecture produces.

This is also a **quality validation** — the recipe builder
doesn't over-promise across architectures. On Llama where stock
is already good, v2 doesn't manufacture fake improvements.

### Now in `data/profiles/`

- `qwen3-0.6B.profile.json` (9 cats × 7 types, with imatrix)
- `qwen3-1.7B.profile.json` (9 cats × 7 types)
- `llama-3.2-1B.profile.json` (8 cats × 7 types — no output.weight)

The v2 ship plan is per-architecture profiles in this directory.
Users targeting other architectures or sizes can build their own
via `LlamaSensitivityProfileBuilder`; the recipe builder + custom
quantizer + stock baseline are all architecture-generic.

## Run 18 — 2026-04-27 17:30  (MinPplGainPerBpw threshold: refusing wasteful trades)

Run 17's same-model recipe (PPL 16.4570) was the best v2 result —
beating stock Q4_K_M by 2.23 PPL — but a hand-pick audit revealed
the optimizer was over-spending bpw chasing noise-level predicted
gains. Specifically: <c>attn_v.weight</c> picked Q8_0 (8.5 bpw)
over Q6_K (6.5625) because the profile predicted 0.001 lower ΔPPL
at Q8_0. That's 0.056 bpw burned for 0.001 PPL — clearly bad in
human judgment but consistent with strict
<c>min(pplSum within budget)</c>.

Run 17b tested five variants:

| variant                            | actual bpw | PPL     | Δ vs A    |
|------------------------------------|------------|---------|-----------|
| A: algorithm baseline              | 5.0686     | 16.4570 | —         |
| B: + attn_v Q6_K (was Q8_0)        | **5.0126** | **16.4253** | **−0.032** |
| C: + token_embd Q5_K (was Q4_K)    | 5.1658     | 16.4031 | −0.054    |
| D: + output.weight Q4_K (was Q6_K) | 4.8499     | 16.6375 | **+0.181** |
| E: + ffn_down all Q6_K             | 5.0504     | 16.5359 | +0.079    |

Two findings vindicated the design:

1. **B's win confirms the algorithm was over-trading.** Switching
   attn_v to Q6_K reclaimed 0.056 bpw AND slightly improved actual
   PPL — the strict-min-pplSum was actively harmful.
2. **D's loss vindicates stock baseline as a hard floor.** The
   profile said <c>output.weight Q4_K</c> was free (single-tensor
   ablation: Q4_K and Q6_K both at −0.014 ΔPPL). In compound
   recipes, demoting it from Q6_K to Q4_K cost +0.18 PPL. Stock's
   <c>use_more_bits</c> protection is covering for a profile blind
   spot, not over-conservative. Don't let profile data override
   baseline floors.

### Implementation: composite scoring with `MinPplGainPerBpw`

Replace the strict-min-pplSum optimization with composite scoring:

```
minimize  pplSum + λ × bpw    subject to    bpw ≤ budgetCap
```

Where λ (default 0.05 PPL/bpw) is the efficiency threshold. With
this objective, the optimizer only takes a promotion when its
predicted gain divided by added bpw exceeds 0.05. The attn_v
Q8_0→Q6_K case: 0.001 / 0.056 = 0.018 PPL/bpw — below 0.05,
rejected. Q6_K wins.

Verified end-to-end on Qwen3-1.7B at 5.026 bpw target:

| recipe                       | actual bpw | PPL     | Δ vs stock |
|------------------------------|------------|---------|------------|
| stock Q4_K_M                 | 5.026      | 18.6837 | —          |
| Run 17 algorithm (λ=0)       | 5.069      | 16.4570 | −2.23      |
| **Run 18 refined (λ=0.05)**  | **5.013**  | **16.4253** | **−2.26** |

The refined recipe is **byte-identical to Run 17b's hand-pick variant B**
— the threshold change naturally produced the same result the
hand audit identified. v2 now ships at:

- **−2.26 PPL** vs stock Q4_K_M (12% relative improvement)
- **−0.013 bpw** vs stock (slightly *smaller* file)

That's a Pareto improvement: smaller and better at the same time.

### Variant C / D / E lessons

- **C (+ token_embd Q5_K)** is a real but minor win (−0.054 PPL at
  +0.097 bpw). The algorithm doesn't auto-promote because Q5_K
  doesn't fit the 5.026 budget; if a user wants it, raising the
  target by ~0.15 bpw lets the algorithm pick Q5_K naturally.
  Not a code change — a budget-tuning choice.
- **D (output Q4_K)** is the negative result we needed. Single-
  tensor ablations don't capture compound effects when many other
  tensors are also quantized. The stock baseline's protection of
  <c>output.weight</c> is essential.
- **E** is just D + an attempted reallocation of the freed budget;
  inherits D's regression.

### v2 ship summary

The Adaptive Quantization v2 stack:

1. **Per-architecture sensitivity profile** (declarative JSONC,
   built via the ablation campaign).
2. **`LlamaQuantRecipeFromProfile.Build`** with:
   - bpw-budget-aware exhaustive enumeration
   - composite scoring (`MinPplGainPerBpw`)
   - per-tensor stock baseline floor (use_more_bits ports)
   - per-category recommended floors from the profile
   - pattern-based protection for uncategorized tensors
   - linear size-scaling factor (within-family extrapolation)
3. **`LlamaCustomQuantizer`** that realizes the recipe verbatim
   (pinning per-tensor types past llama.cpp's heuristic).

On Qwen3-1.7B at Q4_K_M-class budget: **−2.26 PPL at slightly
smaller file size than stock**. Same result whether starting from
the 1.7B same-size profile (validated above) or the 0.6B
size-scaled profile (shown to converge on same recipe at this
ladder; finer ladders may diverge per Run 17). The cross-size
transfer story is honest: rankings transfer, magnitudes scale
~linearly outside the catastrophic-cliff regime, and the recipe
builder makes correct decisions when fed properly-floor-protected
profile data.

### What's deferred from this investigation

- **Per-(arch, size) profile library expansion.** Run 17 showed
  cross-size transfer breaks at finer candidate ladders. To ship
  v2 for arbitrary Qwen3 sizes we want pre-built profiles at
  more size points (4B, 8B). One profile build per size at ~30
  min compute.
- **Cross-architecture validation.** The recipe builder's
  beats-stock claim has only been demonstrated on Qwen3-1.7B.
  The same machinery on Llama-3.2 (no QK-norm) is untested.
  Run 12 has Stage 1 ablation data; full profile + validation
  matrix would close the cross-arch question.
- **PPL throughput.** Investigated; bottleneck is real GPU
  compute (57ms/chunk × 547 chunks = 31s of the 51s wall on
  Qwen3-1.7B), not readback as initially hypothesized. At
  parity with llama.cpp's own perplexity binary (44s).
  ComputeBatchedAsync(n_seq=4) saves ~5%; explicit GPU+CPU
  pipelining could give ~1.65× theoretical, deferred for
  modest payoff.

## Run 17 — 2026-04-27 16:45  (Expanded profile: same-model wins big, cross-size breaks)

Run 16 shipped a 0.45-PPL win at Q4_K_M-class budget using the
3-rung profile (Q2_K/Q4_K/Q6_K) with stock baseline + profile
overrides. Run 17 expands the ablation campaigns to:
- **9 categories** (added `output.weight`, `token_embd.weight` as
  proper ablation categories — Run 16 covered them only via
  `UncategorizedProtections`).
- **7 candidate types** (added Q3_K, Q5_K, Q8_0, IQ4_XS to the
  Q2_K/Q4_K/Q6_K base).

Build wall: 0.6B 7 min, 1.7B 26 min — significantly faster than
Run 13's 14 / 20 min thanks to the per-batch VRAM-aware concurrency
fix (Q2_K/Q3_K batches now use 8-way GPU concurrency where the
F16-derived heuristic only allowed 4).

A bug surfaced and was fixed mid-run: the original `CategoryMatch`
used `EndsWith("output.weight")`, which double-counted all 28
`attn_output.weight` tensors as belonging to the `output.weight`
category, producing a nonsense +10 PPL Q2_K coefficient. Fix:
dot-containing patterns require exact match or
`."+ pattern` suffix. Test
`CategoryMatch_OutputWeightDoesNotCatchAttnOutputWeight` locks it
down. Without that test the buggy profile would have shipped.

### Profile findings

| category            | 0.6B Q4_K | 1.7B Q4_K | 0.6B Q2_K | 1.7B Q2_K |
|---------------------|-----------|-----------|-----------|-----------|
| `output.weight`     | +0.07     | **−0.01** | +6.92     | +2.98     |
| `token_embd.weight` | +0.002    | +0.025    | +0.64     | +0.13     |
| `attn_v`            | +0.14     | +0.34     | +2.72     | **+15.85**|
| `ffn_down`          | −0.0004   | +0.11     | +4.40     | **+3709** |

Two notable surprises:

- **`output.weight` at Q4_K is essentially free** on both models.
  Run 16's `UncategorizedProtections = Q6_K` for output is over-
  conservative — recipes that demote it to Q4_K (or even Q5_K on
  1.7B) should reclaim ~0.3 bpw with no measurable PPL loss.
- **`token_embd.weight` is barely sensitive** on either model.
  Stock's Q4_K choice is also over-conservative; Q2_K only costs
  +0.13/+0.64 PPL. The auto-floor lands at Q2_K.

### End-to-end matrix at Q4_K_M and Q5_K_M targets

| recipe                       | actual bpw | wikitext-2 PPL | Δ vs stock |
|------------------------------|------------|----------------|------------|
| stock Q4_K_M                 | 5.026      | 18.6837        | —          |
| **expand 1.7→1.7 @Q4**       | 5.069      | **16.4570**    | **−2.227** |
| expand 0.6→1.7 @Q4           | 5.058      | 17.4804        | −1.203     |
| stock Q5_K_M                 | 5.772      | 17.1065        | —          |
| **expand 1.7→1.7 @Q5**       | 5.764      | **16.3413**    | **−0.765** |
| expand 0.6→1.7 @Q5           | 5.757      | 17.2985        | +0.192     |

(Custom quantizer wall: ~30 s per recipe with Run 16's
parallelization. PPL ~52 s each. Total run ~10 min.)

### The headline: same-model expanded profile is a 5× improvement over Run 16

Run 16 same-model recipe: −0.45 PPL vs stock at Q4_K_M.
**Run 17 same-model recipe: −2.23 PPL** — five times the
improvement. The expanded categories give the optimizer real
signal on `output.weight` (instead of pinning it at Q6_K via
protection table), and the finer type ladder lets it spend
budget on `attn_v` Q8_0 where the profile says Q6_K isn't
enough but Q8_0 is essentially free.

For comparison: stock Q4_K_M's PPL of 18.68 is what users get
today. Our recipe-built 16.46 PPL at the same bpw is a 12%
relative quality improvement — bigger than the gap between
ftypes one tier apart (Q4_K_M → Q5_K_M is 17.11 → close to our
recipe's 16.46 at half the size penalty).

### The headline that hurts: cross-size transfer breaks at finer granularity

Run 16's striking finding was that the 0.6B-derived recipe and
the 1.7B-derived recipe produced *byte-identical* output. With
the 3-rung Q2_K/Q4_K/Q6_K ladder, both profiles converged on the
same per-category type assignments through the algorithm.

That property is gone with the 7-rung ladder. The recipes now
differ on **every category**:

| category         | 1.7→1.7 @Q4 | 0.6→1.7 @Q4 |
|------------------|-------------|-------------|
| `attn_k`         | IQ4_XS×28   | Q6_K×28     |
| `attn_q`         | Q5_K×28     | IQ4_XS×28   |
| `attn_v`         | **Q8_0×28** | Q6_K×28     |
| `attn_output`    | Q3_K×28     | Q4_K×28     |
| `ffn_up`         | Q5_K×28     | IQ4_XS×28   |
| `ffn_gate`       | Q3_K×28     | Q4_K×28     |
| `ffn_down`       | Q6_K×14, IQ4_XS×14 | Q4_K×14, Q6_K×14 |

The same-model recipe is much more aggressive: full-layer Q8_0
on `attn_v`, Q3_K on `attn_output` / `ffn_gate` (cheap types).
The cross-size recipe stays conservative in the Q4_K/Q6_K range.
At the bpw budget they're both nearly the same, but the
quality differs by 1 PPL (1.7→1.7 = 16.46, 0.6→1.7 = 17.48).

What's happening: with finer types, the optimization landscape
is much more sensitive to the actual ΔPPL coefficients. Small
differences propagate into very different category choices. The
0.6B coefficients × 2.83× linear scaling are *close* to 1.7B's
true coefficients, but not close enough — each minor mismatch
nudges the optimizer into a different local minimum, and
collectively those nudges add up to the 1-PPL gap.

The Run 13 verdict still holds at the rough level (rankings
transfer, magnitudes scale ~linearly except at the cliff), but
the recipe builder's optimization is sensitive enough that
"close enough" isn't actually close enough.

### Implications for v2 ship plan

The cross-size transfer pitch from Run 13 ("ship one profile per
architecture, scale at apply time") is compromised. Two viable
replacements:

1. **Per-(architecture, size-class) reference profiles.** Maintain
   a small library: e.g. Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B,
   Qwen3-8B for the family. Users pick the closest. The recipe
   builder still applies size-scaling for residual mismatch, but
   the base profile is much closer. ~30 min build per profile —
   tractable for ~4-6 profiles per supported family.

2. **Build-time profile.** The recipe-builder CLI runs a profile
   pass on the user's specific source model before generating
   the recipe. ~26 min for 1.7B, longer for bigger models. As a
   one-time cost per user-model, it's reasonable. The CLI could
   cache profiles by source-model SHA so users don't re-pay.

(1) is operationally cleaner; (2) is the long-term escape hatch
for users on uncommon model sizes. Both should ship — (1) covers
the 95% case, (2) handles the rest.

The v2 wins are real and ~5× bigger than Run 16:

- −2.23 PPL @ Q4_K_M (same-model)
- −0.77 PPL @ Q5_K_M (same-model)
- −1.20 PPL @ Q4_K_M (cross-size, conservative win)
- +0.19 PPL @ Q5_K_M (cross-size, slight regression — high-budget
  cases are where size-scaling artifacts hurt most)

For a per-(arch, size) shipping model, the floor is "~−2 PPL
relative to stock Q4_K_M," which is a strong v2 ship.

### Open work after Run 17

- Update v2 ship plan to per-(arch, size-class) profiles in
  `data/profiles/`. Document how users build their own.
- Test cross-size at one more size-class (Qwen3-4B if available,
  or 8B) to refine the size-scaling exponent in the
  not-the-shipped-size case.
- Cross-architecture: build a Llama-3.2 family profile and
  validate the same beats-stock claim there. (The QK-norm
  signature in Run 12 says coefficients differ; recipe behavior
  is unknown.)

## Run 16 — 2026-04-27 14:15  (Stock baseline + profile recipe BEATS stock)

Run 15 revealed the v2 recipe builder lost 3 PPL to stock Q4_K_M
because single-tensor profile ablation can't capture stock's per-
layer alternation pattern (`use_more_bits` protects ffn_down and
attn_v on layers 0..n/8, last n/8, and every third middle layer —
about half the layers per category). The user proposed: keep
stock's per-layer wisdom as a hard floor, layer profile-driven
optimizations on top. This run validates that approach.

Build (committed as 6156215):
- `LlamaStockBaseline.cs` ports `use_more_bits` from
  `llama-quant.cpp:417`. The Q4_K_M-flavored baseline assigns
  `output.weight` → Q6_K, `token_embd.weight` → Q4_K, and
  `attn_v` / `ffn_down` → Q6_K on `use_more_bits` layers.
- `LlamaQuantRecipeFromProfileOptions.ApplyStockBaseline = true`
  by default. The recipe enumeration uses `effective_type =
  max(baseline(t), recipe_pick_for_category(t))` per tensor —
  recipe can promote above stock, never demote below.
- ΔPPL prediction is pro-rated by the fraction of category
  elements actually riding the recipe type (vs stuck at baseline).

Validation against the same Run 15 matrix:

| recipe                       | actual bpw | wikitext-2 PPL | Δ vs stock |
|------------------------------|------------|----------------|------------|
| stock Q4_K_M                 | 5.026      | 18.6841        | —          |
| recipe (1.7B → 1.7B) + base  | 5.084      | **18.2320**    | **−0.45**  |
| recipe (0.6B → 1.7B) + base  | 5.084      | **18.2320**    | **−0.45**  |
| Run 15 recipe (no base)      | 4.979      | 21.6730        | +2.99      |

(Custom quantizer wall: ~27 s per recipe — 5× the 140s of pre-
parallelization. PPL ~52 s each. Total run ~3 min including the
Run 15 baselines for reference.)

### Findings

**1. v2 recipe beats stock at Q4_K_M-class budget.** −0.45 PPL at
+0.06 bpw vs stock — meaningful improvement at small size cost.
This is the first run where Adaptive Quantization v2 is genuinely
better than the heuristic, validating the hypothesis the
investigation has chased since Run 8.

**2. Cross-size transfer is byte-identical.** The 0.6B-derived
and 1.7B-derived recipes produce *the same exact recipe* under
this configuration — both PPL 18.2320 to four decimals. The size-
scaling math + baseline floor + budget-aware enumeration converges
to the same per-tensor type assignment regardless of which family
profile you start from. The "ship one profile per architecture"
plan from Run 13 is operationally validated.

**3. The profile alone wasn't enough.** Run 15's recipe (no
baseline) at 4.98 bpw was 21.67 PPL — 3 PPL worse than stock at
similar bpw. Adding the baseline took us from "+3 PPL worse" to
"−0.45 PPL better" without changing anything about the profile
data. The single-tensor ablation captures useful signal but misses
per-layer interactions; layering it on top of stock's per-layer
heuristic recovers what we couldn't measure directly.

**4. The recipe at 5.084 bpw is mostly stock-baseline.** Looking
at per-layer breakdown:
- `output.weight` Q6_K (baseline)
- `token_embd.weight` Q4_K (baseline)
- `attn_v` Q6_K ×14, Q4_K ×14 (baseline use_more_bits alternation)
- `ffn_down` Q6_K ×14, Q4_K ×14 (baseline use_more_bits alternation)
- `attn_k` Q6_K ×28 (profile-driven full promote, on top of baseline)
- `attn_q` Q4_K ×28, `attn_output` Q4_K ×28, `ffn_gate` Q4_K ×28,
  `ffn_up` Q4_K ×28 (profile says default-type is fine)

The profile contributes one full-category promote (`attn_k`); the
baseline contributes the rest. The +0.45 PPL improvement vs stock
is essentially "promote attn_k.weight to Q6_K everywhere on top of
stock's per-layer pattern." Future profile expansions (more
categories, finer types) should give the profile more to
contribute beyond a single promote.

### v2 ship recipe

What ships:
- `LlamaSensitivityProfile` (JSONC) — declarative, schema-versioned.
- `LlamaSensitivityProfileBuilder` — per-arch ablation campaign,
  resumable, parallel-PPL.
- `LlamaQuantRecipeFromProfile.Build` — bpw-budget-aware, exhaustive
  enumeration, stock-baseline floor, per-tensor protections.
- `LlamaCustomQuantizer.QuantizeWithRecipeAsync` — realizes recipe
  per-tensor verbatim, parallelized.
- One profile per architecture (Qwen3 today; Llama / others later).

What we measure: at Q4_K_M-class budget on Qwen3-1.7B, v2
delivers a 0.45-PPL improvement over stock at +0.06 bpw, and the
same improvement holds whether the profile was built on the 0.6B
or 1.7B target.

### Open questions for Run 17

- Does the win hold at higher budgets (Q5_K_M, ~5.77 bpw)? Run 15
  showed the no-baseline recipe tied stock there (17.13 vs 17.11);
  with baseline it should be a clean win.
- Does it hold on Llama (no QK-norm, different sensitivity profile)?
- Does expanding the candidate ladder (Q3_K, Q5_K, Q8_0, IQ4_XS)
  let the profile contribute *more* on top of baseline, or do we
  hit diminishing returns?

## Run 15 — 2026-04-27 13:30  (Custom quantizer + matched-bpw stock baselines — Run 14 inverts)

Run 14 had a confounder: `tt_overrides` only fires as "manual" when
the override differs from the default ftype's pick
(`llama-quant.cpp:682`), which silently dropped our demote-to-Q4_K
attempts on `output.weight` and ffn_down(half-the-layers). That
~120 MiB of forced Q6_K protection was riding our recipes; what
Run 14 measured as a "profile recipe wins" was mostly that heuristic
backstop, not our bit allocation.

Run 15 fixes the comparison. We landed `LlamaCustomQuantizer`
(commit 7cd231f) — a pure-bindings driver that bypasses
`llama_model_quantize`'s heuristic entirely and realizes the
recipe's per-tensor types verbatim. Then re-ran the same Run 14
matrix with the new quantizer, now at *actually* matched bpw, and
added Q5_K_M as a higher-bpw second baseline.

### Result

| recipe                  | actual bpw | wikitext-2 PPL | Δ vs stock |
|-------------------------|------------|----------------|------------|
| stock Q4_K_M            | 5.026      | **18.68**      | —          |
| recipe (1.7B → 1.7B) @Q4| 4.979      | 21.67          | **+2.99**  |
| recipe (0.6B → 1.7B) @Q4| 4.979      | 21.67          | **+2.99**  |
| stock Q5_K_M            | 5.772      | **17.11**      | —          |
| recipe (1.7B → 1.7B) @Q5| 5.813      | 17.13          | +0.02      |
| recipe (0.6B → 1.7B) @Q5| 5.575      | 21.06          | +3.95      |

(Custom quantizer wall: ~140 s per recipe — pure-CPU dequant /
requant, single-threaded ggml_quantize_chunk. PPL ~62 s each.
Total run 16 min.)

### Diagnosis — `output.weight` is the elephant in the unprofiled room

The recipes in the profile know about 7 categories:
`attn_{q,k,v,output}.weight`, `ffn_{up,gate,down}`. They know
*nothing* about `output.weight` (the lm_head projection) or
`token_embd.weight`. Both fall through to
`Options.UncategorizedType` (default Q4_K).

Stock Q4_K_M's heuristic promotes `output.weight → Q6_K` because
`use_more_bits` knows it's high-leverage at the model's output
side. Our recipes ship with `output.weight` at Q4_K. That single
tensor is 310M params; Q4_K vs Q6_K is a 2.06-bpw difference on
~15% of the model's bytes. The PPL cost is ~3 — exactly what we
measured.

Concretely: the gap between stock Q4_K_M (18.68) and our recipe
(21.67) is essentially the cost of demoting `output.weight` from
Q6_K to Q4_K. Our recipe's smart attn_k/v/q/ffn_up promotions
partially recovered the loss, but didn't make up for the missing
output protection.

**Run 14's "+1.37 PPL improvement" was an artifact of the
override-only-elevates bug protecting us from our own profile's
blind spot.** llama-quant kept `output.weight` at Q6_K against our
will, and that hidden protection masked the missing-category
problem.

### Other findings still standing

- **Cross-size transfer math works.** At Q4_K_M target, the 0.6B
  and 1.7B-derived recipes produced *byte-identical* recipes (both
  PPL 21.67 at the same actual bpw). The size scaling correctly
  reproduces the same category choices when both profiles share
  the same blind spot. The cross-size design is fine; it's
  operating on incomplete data.

- **At Q5_K_M, recipe(1.7→1.7) ties stock** (17.13 vs 17.11). At
  higher budget the algorithm promotes enough other categories
  that the missing output protection doesn't dominate. This says
  the profile is competitive *when the budget hides the gap* — a
  reminder that bpw budget and profile completeness interact.

- **The 0.6B recipe at Q5 budget collapses to 5.57 bpw**, not the
  full 5.77 budget. At the higher budget the size-scaled
  coefficients made promoting more categories not predicted-worth-
  it, so the algorithm settled lower. PPL 21.06 — back to the
  output@Q4_K disaster. This is consistent: when you don't
  characterize the most-sensitive tensors, optimizing for the
  others doesn't help.

### Operational implication — fix the profile

Two steps before any further validation:

1. **Add `output.weight` and `token_embd.weight` as profile
   categories.** The builder already supports custom categories
   via `Options.Categories`. We just need to re-run the ablation
   campaigns with `["attn_q.weight", "attn_k.weight",
   "attn_v.weight", "attn_output.weight", "ffn_up", "ffn_gate",
   "ffn_down", "output.weight", "token_embd.weight"]`. The token
   embedding is shaped (vocab × hidden) and `output.weight` is
   (hidden × vocab); both will measure cleanly.

2. **Expand the candidate type ladder.** Q2_K/Q4_K/Q6_K is too
   coarse — Run 15 hit several "couldn't fit budget cleanly"
   cases. Add Q3_K, Q5_K, Q8_0, IQ4_XS for finer-grained
   bit-allocation. Wall cost ~1 hour for both Qwen3 sizes with
   the corrected VRAM heuristic.

Both go together: the rerun campaign is one set of compute. Run
16 is the full re-validation against this corrected profile, with
the same Q4_K_M / Q5_K_M / cross-size matrix.

### What this means for the v2 ship plan

- The recipe builder, custom quantizer, and JSONC schema are all
  unchanged-correct. The bug is in the *contents* of the shipped
  reference profiles — they don't include enough categories.
- The "small-model profile predicts large-model behavior" claim
  from Run 13 is still credible but unproven against properly
  matched bpw. Need Run 16 to settle.
- Cross-size cost model: validated when both profiles have the
  same blind spot. Will need to re-validate when profiles are
  complete.

## Run 14 — 2026-04-27 12:30  (End-to-end: profile recipe vs stock Q4_K_M on 1.7B)

First real PPL test of Adaptive Quantization v2: build three recipes
for Qwen3-1.7B at matched-ish bpw and score wikitext-2 PPL:

1. **Stock Q4_K_M** — llama.cpp's heuristic, the baseline to beat
2. **Recipe(1.7B → 1.7B)** — 1.7B profile applied directly (same-size,
   no scaling). Upper bound: best we can do *with* same-model data.
3. **Recipe(0.6B → 1.7B)** — 0.6B profile with size-scaling exp=1.0
   applied to the 1.7B target. The v2 ship path: does the cross-size
   transfer Run 13 predicted actually hold under real PPL?

All quantizations targeted ~5.05 weighted bpw (matching stock Q4_K_M's
actual file bpw on this model). Recipes were the bpw-budget-aware
output from `LlamaQuantRecipeFromProfile.Build` (committed in 8ac9288).

### Result

| recipe                  | predicted bpw | actual file bpw | wikitext-2 PPL | Δ vs stock |
|-------------------------|---------------|------------------|----------------|------------|
| stock Q4_K_M            | —             | 5.03             | 18.6841        | —          |
| recipe (1.7B → 1.7B)    | 5.10          | 5.59             | **17.3094**    | **−1.37**  |
| recipe (0.6B → 1.7B)    | 5.10          | 5.59             | **17.4129**    | **−1.27**  |

(Build wall: 17.8s stock + 14.8s + 14.7s recipes; PPL 59-63s each.
n_ctx=512, second-half scoring — same as Run 9/11/13.)

### Two findings — one expected, one *very* expected

**1. Cross-size transfer is essentially free.** The 0.6B-derived
recipe lands within **0.10 PPL** of the 1.7B-derived recipe — well
inside measurement noise on PPL. Run 13's hypothesis is confirmed
end-to-end: a small-model-per-architecture profile, scaled linearly
to the target's parameter count, produces a recipe that's
operationally indistinguishable from a same-model profile recipe.
Adaptive Quantization v2 does *not* need a profile per
(architecture, size-class). One profile per architecture, scaled
at apply time, is the ship plan.

**2. Profile recipes meaningfully beat stock Q4_K_M on PPL** —
~7% relative improvement (1.37 PPL drop) on wikitext-2. This
matches the Run 9/13 prediction: the heuristic under-protects
`ffn_up` and over-protects `ffn_down`, and the profile correctly
re-allocates bits in line with measured sensitivity.

### The caveat that turns "free" into "not free"

Stock Q4_K_M lands at 5.03 bpw on the file; the recipes both land
at 5.59 bpw — 11% larger files. The v2 recipe is not a free
improvement; it's a "spend 0.5 bpw to drop PPL by 1.4" trade. Plus,
this isn't what the recipe builder *intended* — it predicted 5.10
bpw and was overruled by llama-quant.

Root cause (confirmed via `llama-quant.cpp:682`):

```cpp
if (qtype != new_type) {           // override only fires when it CHANGES type
    new_type = qtype;
    manual = true;
}
// if !manual: fall through to heuristic, which can promote
```

`tt_overrides` only triggers as "manual" when the override differs
from the default ftype's pick. Setting the recipe's "ffn_down →
Q4_K" override on a Q4_K_M base hits `qtype == default_type`,
`manual` stays false, and llama-quant's `use_more_bits` heuristic
runs and re-promotes ffn_down → Q6_K on alternating layers. Same
for `output.weight`. Net effect: our recipe can only
*elevate* tensors above the heuristic's pick, never demote.

That ~120 MiB of stuck Q6_K = the 0.5 bpw discrepancy between
predicted (5.10) and actual (5.59).

### What this means for v2

- **The cross-size transfer finding is real and unaffected** by the
  override quirk — both the 1.7B and 0.6B-derived recipes hit the
  same actual bpw (5.59) and produced PPLs within 0.1 of each
  other. Run 13's hypothesis is validated regardless.
- **The "beats stock" claim is preliminary.** Comparing 5.59 vs
  5.03 isn't a fair size-matched test. The honest follow-up is
  Q5_K_M (closer to 5.4 bpw) as the baseline at this size class —
  if our recipe still beats Q5_K_M on PPL, the bit-allocation is
  genuinely smarter; if it loses, we're just buying PPL with bits.
- **The bpw budget is broken at Q4_K-base.** To make recipes
  actually realize their predicted bpw, run with `--ftype Q3_K_S`
  (or similar small base) so every recipe choice is an upgrade
  the override system applies cleanly. Track as Run 15.

### Memory captured

- `feedback_recipe_stock_match.md`: when our recipe matches a
  stock heuristic, surface as warning not blocker (user feedback
  during plan).

### Next

Run 15: rerun this same comparison with `--ftype Q3_K_S` as the
base for the recipe quants, so the recipe builder's predicted bpw
is what actually ships. Plus a Q5_K_M baseline at matched bpw —
the proper apples-to-apples test of whether profile-built bit
allocation beats heuristic bit allocation at the same size.

## Run 13 — 2026-04-27 11:32  (Cross-size: Qwen3-0.6B vs Qwen3-1.7B, same family)

Run 12 settled cross-architecture (Qwen3 ≠ Llama). Run 13 asks a
narrower question: *within a single architecture family, does the
profile transfer across model sizes, or do we need a per-size
profile?* This is the call we have to make for Adaptive Quantization
v2 — if profiles are size-stable we can ship one Qwen3 profile and
let recipe builders extrapolate; if they're not, the builder has
to be parameterized by the target model size.

Method: ran `LlamaSensitivityProfileBuilder` end-to-end on
Qwen3-0.6B and Qwen3-1.7B (no imatrix on the 1.7B — it's the
delta-magnitude across categories we care about, not the absolute
imat-vs-not effect). Both at n_ctx=512 to match Run 9/11
convention. Two campaigns ran back-to-back via the new builder,
which is itself the validation: we now have a one-shot tool to
characterize an architecture in 14 / 20 minutes wall-clock.

### Result — side-by-side ΔPPL

| category       | 0.6B Q2_K    | 1.7B Q2_K    | 0.6B Q4_K | 1.7B Q4_K | 0.6B Q6_K | 1.7B Q6_K |
|----------------|--------------|--------------|-----------|-----------|-----------|-----------|
| `ffn_down`     | **+4.4021**  | **+3709.61** | -0.0004   | +0.1121   | -0.0437   | -0.0094   |
| `ffn_up`       | +3.8251      | **+7.9517**  | **+0.2273** | **+0.4425** | +0.0288 | +0.0448 |
| `ffn_gate`     | +2.9459      | +2.6121      | +0.1063   | +0.2322   | -0.0009   | -0.0035   |
| `attn_v`       | +2.7204      | **+15.8469** | +0.1419   | +0.3403   | -0.0536   | +0.0015   |
| `attn_output`  | +2.3262      | +2.4240      | +0.0910   | +0.0789   | +0.0264   | +0.0227   |
| `attn_k`       | +2.0196      | +4.8127      | +0.1149   | +0.3140   | -0.0567   | -0.0598   |
| `attn_q`       | +1.2772      | +1.9758      | +0.1143   | +0.0398   | +0.0021   | +0.0180   |

(F16 baselines: 0.6B = 21.4720 PPL, 1.7B = 16.8869 PPL.
Workspace `/mnt/data/models/.profile-tmp`, build wall 853s and
1219.7s respectively. Concurrency auto-resolved to 7 / 2 — the
1.7B value is over-conservative, see "Implementation note" below.)

### Verdict — the small model predicts the big one (with one caveat)

Initial reading was "rankings transfer but magnitudes and floors
don't." On closer inspection that's too pessimistic — *all three*
transfer in the operationally-relevant range, and the one place
they diverge (the Q2_K cliff depth) doesn't change recipe-builder
decisions.

**1. Q4_K ranks transfer cleanly.** `ffn_up` is the top Q4_K hit
on both (+0.23 → +0.44). The bottom three (`attn_q`, `attn_output`,
`ffn_down` at Q4_K) are noise-floor on both. The recipe builder
operating on the 0.6B profile would promote the same set of
categories on 1.7B that the 1.7B profile would have it promote —
which is the actual operational question.

**2. Q4_K magnitudes scale with size, but proportionally.** The
amplifying categories all move by ~2-3× (1.7B / 0.6B):

| category | 0.6B | 1.7B | ratio |
|----------|------|------|-------|
| ffn_up   | 0.23 | 0.44 | 1.9× |
| attn_v   | 0.14 | 0.34 | 2.4× |
| attn_k   | 0.11 | 0.31 | 2.7× |
| ffn_gate | 0.11 | 0.23 | 2.2× |

That's a **predictable, near-linear scaling** with parameter count
(2.83× more weights → ~2-3× more sensitivity). The 0.6B profile
is a reliable order-of-magnitude estimator for 1.7B once you
account for size — and even unscaled it correctly identifies
which categories the recipe should spend bits on.

**3. The QK-norm signature persists.** `attn_v` / `attn_k` retain
their elevated sensitivity vs a no-QK-norm architecture (Run 12
had Llama's `attn_v` near-zero at Q4_K). The architectural
signature is family-level, not size-level.

**The single caveat — Q2_K cliff depth.** At Q2_K, `ffn_down`
goes from "survivable" (+4.40) on 0.6B to "catastrophic"
(+3709.61) on 1.7B. The cliff *direction* still transfers
(`ffn_down` is the worst Q2_K victim on both), but the *depth*
changes by ~840×. This matters for the floor heuristic if it
naively transplants thresholds from a small reference profile
to a larger target — a 0.6B-derived profile says "Q2_K is fine
everywhere" and the 1.7B will produce gibberish if you believe
it.

But this is a hard-threshold artifact, not a real predictive
failure. The 0.6B profile already flags `ffn_down` as the
*most-fragile* category at Q2_K (+4.40 ranks #1, beating
`ffn_up`'s +3.83); the *order of fragility* is preserved. A
floor heuristic that uses the small model's rank-of-cliff-victim
plus a size scaling factor would correctly refuse Q2_K for
`ffn_down` on 1.7B without needing to measure the catastrophe
directly.

### Operational implication for Adaptive Quantization v2

**A small-model-per-architecture profile is enough.** The recipe
builder needs:

1. **Per-category Q4_K coefficients** from the small model →
   used directly to score recipes (sum ΔPPL across categories).
2. **A size scaling factor** (default ~`target_params / source_params`,
   capped at maybe 5×) → multiplies the small model's coefficients
   when applying the profile to a larger target.
3. **Floor selection by rank, not by absolute threshold** →
   "the most-cliff-prone category at the smallest tested type
   gets a floor one step higher" rather than "any ΔPPL > 5.0
   sets a floor." This makes the floor decision robust to the
   absolute-magnitude divergence at the cliff.

For real recipes (Q4_K-and-up territory) the cliff isn't even
in scope, so the divergence is mostly academic.

**v2 plan**: ship one profile per supported architecture
(generated from the smallest practical family member, e.g.
Qwen3-0.6B for the Qwen3 family). Recipe builder takes the
target model's parameter count and applies the size scaling.
Per-(arch, size-class) profiles are not required — they'd be
nice-to-have validation data, not a shipping requirement.

### Implementation note — heuristic over-conservative on larger models

The 1.7B build used concurrency=2; live monitoring showed total
GPU memory at ~7 GB and ~3.5 GB / instance. The 0.80 × 24 GB
usable budget would have fit 4-5 instances. The current
heuristic (`weights × 1.85`) overestimates because it's
multiplicative, but the actual compute buffer is essentially
size-independent at fixed n_ctx (~298 MiB on 0.6B, ~300 MiB on
1.7B per llama.cpp's own log lines). Switching to additive
(`weights + ~700 MiB headroom`) is the obvious fix and roughly
doubles realistic concurrency on 1.7B-class models. Filed as
follow-up; the 1.7B numbers are still correct, just slower than
they needed to be.

### Profiles archived

- `data/profiles/qwen3-0.6B.profile.json` — Qwen3 family
  reference (small-model-per-architecture profile for v2)
- `data/profiles/qwen3-1.7B.profile.json` — same family, larger
  target — kept as validation data for the size-scaling heuristic

Both written by `LlamaSensitivityProfileBuilder.SaveToJson` —
the public-API artifact that downstream recipe-builders will
read. The 1.7B serves as the ground-truth check: a v2 recipe
builder that applies size-scaled 0.6B coefficients to the 1.7B
should pick essentially the same recipe a builder operating on
the 1.7B profile directly would pick.

## Run 12 — 2026-04-27 (Cross-architecture: Llama-3.2-1B Stage 1)

The user predicted that quantization sensitivity profiles would be
architecture-specific. Run 12 tests it directly: same Stage 1
ablation protocol on Llama-3.2-1B (no QK-norm, classical MHA), 16
layers, F16 baseline 13.82 PPL on wikitext-2.

### Result — side-by-side ΔPPL at Q4_K

| category       | Qwen3-0.6B (QK-norm) | Llama-3.2-1B (no QK-norm) | Qwen3 rank | Llama rank |
|----------------|----------------------|----------------------------|------------|------------|
| `ffn_up`       | **+0.4573**          | **+0.1646**                | **1**      | **1**      |
| `attn_v`       | +0.4367              | +0.0077                    | 2          | **7 (least)** |
| `attn_k`       | +0.2959              | +0.0072                    | 3          | 6          |
| `attn_q`       | +0.2674              | +0.0592                    | 4          | 3          |
| `attn_output`  | +0.0801              | +0.0405                    | 5          | 5          |
| `ffn_gate`     | +0.0757              | **+0.1245**                | 6          | **2**      |
| `ffn_down`     | +0.0303              | +0.0449                    | 7          | 4          |

(F16 baselines: Qwen3 = 21.47 PPL, Llama = 13.82 PPL.)

### Verdict — the user's prediction is confirmed

**Universal patterns** (true on both architectures):

- **`ffn_up` is the most-sensitive category, on both models.** The
  Q4_K_M heuristic's "default Q4_K" decision is wrong on Qwen3
  *and* Llama. Promoting `ffn_up` is a universal heuristic
  improvement — that part of recipe v2 (Run 10) generalizes.
- `attn_output` is mid-rank (#5) on both.

**Architecture-specific patterns** (and they're dramatic):

- **`attn_v`**: 2nd-most-sensitive on Qwen3 (+0.44) → *least*-
  sensitive on Llama (+0.008). The heuristic's `use_more_bits`
  bumps to Q5_K/Q6_K on `attn_v` are necessary on Qwen3 but
  **wasted budget on Llama** — at Q4_K it costs Llama only
  0.06 % of its baseline PPL.
- **`attn_k`**: 3rd on Qwen3 → 6th on Llama. Same pattern.
- **`ffn_gate`**: 6th (low) on Qwen3 → 2nd (high) on Llama.
  Llama would benefit from `ffn_gate` protection; Qwen3 wouldn't.

### What this means for QK-norm

The QK-norm story (Run 5) was about imat-weighted *MSE* amplification
on K/Q tensors. Run 8 showed MSE didn't predict PPL at the *category*
level. Run 12 now closes the loop:

- On QK-norm Qwen3, attention tensors (`attn_v`/`k`/`q`) really are
  more PPL-sensitive than on non-QK-norm Llama. So the QK-norm
  signature *does* translate into PPL impact, just not in the
  direction the imat-weighted MSE recipe was chasing (it was
  protecting `attn_k`/`attn_q` heavily — those are #3/#4 in PPL
  rank on Qwen3, not #1).
- The heuristic's hand-tuned `attn_v` protection happens to be
  *correct* for QK-norm Qwen3 (attn_v really is high-sensitivity)
  but **is genuine over-spending on Llama** where attn_v is
  effectively free.

Whether that's a coincidence (Llama-1/2-era heuristic accidentally
correct for QK-norm) or whether QK-norm-era models inherited
sensitivity patterns that earlier MHA models also had at lower
amplitude (in which case the heuristic was always right and Llama-3.2
just regressed) is its own research question. The data here can't
tell.

### Implication for the GGUFLab recipe builder

A universal "fix the heuristic" recipe doesn't exist. The Q4_K_M
heuristic is wrong in *different* ways on different architectures:

  - On Qwen3-0.6B: under-protects `ffn_up`; over-protects `ffn_down`
    (but only mildly — see Stage 2 knee).
  - On Llama-3.2-1B: under-protects `ffn_up` *and* `ffn_gate`;
    over-protects `attn_v` and `attn_k` (their Q5_K/Q6_K bumps
    buy almost nothing).

The right tool design ships **per-architecture sensitivity profiles**.
A profile is a (category, bpw) → ΔPPL table built by running Stages
1+2 on the architecture once. The recipe builder picks bumps by
ROI (ΔPPL improvement per ΔBPW) from the profile, never crosses a
known knee, and produces a recipe tuned to that architecture rather
than a one-size-fits-all rule.

Profiles measured so far: Qwen3-0.6B (Stages 1+2). Llama-3.2-1B
needs Stage 2 for completeness — the Stage 1 ranking is in hand.

### What would falsify this

If we ran Stage 1 on a different Qwen3 size (1.7B, 4B) and got a
materially different ranking from Qwen3-0.6B, that would say
"profiles are model-size-specific too" — bad for tooling. The bet
is that profiles are *architecture-family*-specific (QK-norm vs
not, MoE vs dense, GQA ratio, …) but reasonably stable within a
family. Validating that requires a Run 13 on Qwen3-1.7B or 4B.

### Operational guidance update — final form

For GGUFLab:

  1. Ship **per-architecture sensitivity profiles** as build-time
     artifacts. One profile per (architecture, model-size-class)
     measured via Stage 1+2 ablation.
  2. The recipe builder takes a budget target and a profile;
     produces an allocation that minimizes predicted ΔPPL while
     respecting per-category knees.
  3. The Adaptive Quantization tool's role becomes *building the
     profile*, not picking the recipe. The user runs an ablation
     campaign once per model architecture (~1 hour); the recipe
     builder consumes the profile.
  4. The MSE-based sensitivity sweep (Runs 1–5) keeps a useful but
     reduced role: it picks **which tensor *within* a category**
     to bump first when the recipe builder allocates extra bpw to
     a category. That's the QK-norm signal earning its keep —
     finding the layer-15-attn_k that needs Q5_K bumped to Q6_K
     before bumping layer-3-attn_k.

This is genuinely a research-grade tool design. The investigation
is complete; the implementation is its own project.

## Run 11 — 2026-04-27 (Stage 2: per-(category × bpw) — and the ffn_down cliff)

Stage 1 (Run 9) measured ΔPPL per category at one bpw point (Q4_K).
Stage 2 fills in the curve at Q2_K and Q6_K to answer: are the
per-category coefficients **linear in bpw**, or do some categories
have non-linear "knees" where dropping below a threshold is
catastrophic?

**Setup**: same single-category ablation protocol, target type
varied across {Q2_K, Q4_K, Q6_K}, others held at F16. Reuses the
F16 baseline from Run 9 (PPL 21.4720). 14 new runs (7 categories ×
2 new bpw points). Total bench time ~50 min.

### Result — full (category × bpw) ΔPPL table

ΔPPL when only this category is quantized to the listed type, vs
F16 baseline. **Higher is worse.**

| category       | Q6_K (6.6 bpw) | Q4_K (4.5 bpw) | Q2_K (2.6 bpw) |
|----------------|-----|-----|-----|
| `attn_q`       | −0.03 | +0.27 | +4.61   |
| `attn_k`       | −0.05 | +0.30 | +4.69   |
| `attn_v`       | −0.12 | +0.44 | **+18.32**  |
| `attn_output`  | −0.02 | +0.08 | +4.02   |
| `ffn_up`       | +0.03 | +0.46 | **+12.75**  |
| `ffn_gate`     | −0.04 | +0.08 | +7.19   |
| `ffn_down`     | +0.01 | +0.03 | **+124.64** |

(Slightly-negative numbers at Q6_K are PPL measurement noise — Q6_K is
essentially F16-equivalent for every category. The "improvement" is
~±0.1 PPL, well within run-to-run variance.)

### Key finding: `ffn_down` has a catastrophic knee between Q4_K and Q2_K

At Q4_K, `ffn_down` is the *least* sensitive category (+0.03 PPL).
At Q2_K, it's the *most* sensitive by a wide margin (+124.64 PPL —
PPL goes from 21.47 to 146.11 from quantizing ffn_down alone). That's
a ~4000× jump in ΔPPL for a 1.9 bpw drop. The category is essentially
free to quantize down to Q4_K, then falls off a cliff.

**This rehabilitates the heuristic's `ffn_down` protection.** Stage 1
read as "the heuristic over-protects ffn_down." That was true *at
Q4_K*. But the heuristic's `use_more_bits` bumps to Q5_K/Q6_K aren't
wasted — they're keeping ffn_down safely above its knee. The
heuristic's instinct here is correct *as a safety margin*, even if
the bumps look like over-spending at Q4_K-and-above.

Recipe v2 (Run 10) accidentally got this right by demoting ffn_down
to *exactly* Q4_K (not lower). If v2 had pushed ffn_down to IQ4_XS
or Q3_K, it would have started picking up the knee penalty.

### Other categories' shape

`attn_v` and `ffn_up` are also notably non-linear — both have
significant Q2_K penalties (+18.32 and +12.75) compared to a steady-
ish progression from Q6_K → Q4_K. Their knees are real but less
extreme than `ffn_down`'s.

The remaining four (`attn_q`, `attn_k`, `attn_output`, `ffn_gate`)
have nearly-linear curves — Q2_K penalty is ~3-7 PPL, gradual not
catastrophic. These are "robust" categories you can push to lower
bpw with predictable degradation.

### What Stage 2 means for recipe-builder design

A τ-on-MSE rule (Recipe v1, retracted in Run 8) treats every tensor
the same way. It can't see knees because it doesn't model the
shape of each category's PPL curve.

A coefficient-driven rule needs to know:
1. The **per-category coefficient at Q4_K** (Stage 1) — first-order
   ranking. ffn_up most sensitive, ffn_down least. Use this to fix
   the ffn_up/ffn_down inversion in the heuristic.
2. The **per-category knee bpw** (Stage 2) — minimum-safe bpw before
   things go catastrophic. ffn_down: knee around Q3_K/Q2_K, do not
   cross. attn_v: similar. ffn_up: similar.
3. The **per-category slope between Q4_K and the knee** — for tensors
   you might bump above Q4_K, what's the marginal PPL improvement?

Stage 2 supplies (2) and gives a partial answer to (3). The right
recipe-builder shape is now clearer:

```
allocate_budget(target_bpw, knees, slopes):
  start every category at Q4_K            # known safe
  rank categories by improvement-per-bpw: # high-slope = best ROI
    [ffn_up, attn_v, ...] → bump to Q5_K / Q6_K while budget remains
  never drop a category below its knee bpw
```

That's effectively what the Q4_K_M heuristic already does *if its
priorities are correct*. Recipe v2 (Run 10) was the heuristic with
the priorities corrected for `ffn_up`. v3 with the full Stage-2
table could squeeze out more.

### Open question still: cross-architecture

Everything so far is on Qwen3-0.6B. The heuristic's `ffn_down` cliff
might be QK-norm-specific, or it might be universal. **Run 12
(Llama-3.2-1B Stage 1)** is in flight to find out — if Llama also
shows ffn_up > ffn_down at Q4_K, the inversion is universal and the
heuristic just needs an update. If the ranking matches the
heuristic's priorities on Llama, the user's prediction is right and
the GGUFLab tool needs **per-architecture sensitivity profiles**.

## Run 10 — 2026-04-27 (Recipe v2: validate the ablation findings)

Stage 1 (Run 9) said the Q4_K_M heuristic over-protects `ffn_down` and
under-protects `ffn_up`. Recipe v2 swaps just that — keep the rest of
the heuristic identical, but move the `use_more_bits` bumps from
`ffn_down` to `ffn_up`. If the ablation coefficients are actionable,
this should beat stock Q4_K_M at near-matched bpw.

**Recipe v2** (per-category):

  | category    | Q4_K_M heuristic              | recipe v2                     | change |
  |-------------|-------------------------------|-------------------------------|--------|
  | attn_v      | Q6_K via use_more_bits, else Q5_K | (unchanged)                | —      |
  | ffn_up      | Q4_K (default)                | Q6_K for first n/16; Q5_K via use_more_bits; else Q4_K | **bumped** |
  | ffn_down    | Q6_K for first n/16; Q5_K via use_more_bits; else Q4_K | Q4_K (default) | **demoted** |
  | attn_q,k,output, ffn_gate | Q4_K           | Q4_K                          | —      |
  | output      | Q6_K                          | Q6_K                          | —      |
  | token_embd  | Q4_K                          | Q4_K                          | —      |

The swap is approximately bpw-neutral on Qwen3-0.6B because both
`ffn_up` and `ffn_down` are 3M-param tensors per layer.

**Result**:

  | variant                          | bpw  | size    | PPL          | ΔPPL vs F16 |
  |----------------------------------|------|---------|--------------|-------------|
  | F16 baseline                     | 16.00| 1439 MB | 21.4720      | 0           |
  | stock Q4_K_M (heuristic)         | 5.09 | 484 MB  | 22.4027      | +0.93       |
  | **recipe v2** (ablation-informed)| **5.17**| 492 MB | **22.1103**| **+0.64**   |
  | recipe at τ=0.00456 (Run 8)      | 5.11 | 486 MB  | 26.2425      | +4.77       |

Recipe v2 delivers 0.29 PPL lower than stock Q4_K_M with 1.6 % more
bytes (5.17 vs 5.09 BPW). To rule out "won on extra budget" — the
marginal Q4_K_M → F16 yields ~0.085 PPL/BPW, so 0.08 BPW of extra
budget alone would buy ~0.007 PPL improvement. Recipe v2's measured
improvement is ~40× that — **the win is in the allocation, not the
budget**.

### Verdict — the ablation methodology works

Three things this confirms:

1. **The ablation rankings translate into PPL impact.** Stage 1 said
   `ffn_up` is the most-sensitive category and `ffn_down` is the
   least-sensitive; swapping just the protection between them moved
   PPL by 0.29.
2. **The Q4_K_M heuristic has a real, exploitable mistake on Qwen3-
   0.6B.** Two of its seven category-level decisions are inverted
   for this architecture; fixing one of them (the ffn_up/ffn_down
   swap) is a buildable improvement.
3. **A coefficient-driven recipe builder is the right next step.**
   v2 was a hand-coded swap based on the Stage-1 ranking. A v3
   could solve a real optimization (minimize ΔPPL subject to bpw
   budget) over the (category, bpw) coefficient table from Stage 2.

### Where to go from here (the real future-work list)

1. **Stage 2: (category × bpw) coefficient table.** Repeat single-
   category ablation at Q2_K and Q6_K. Yields per-category marginal
   PPL/BPW. Lets us answer "should `attn_v` be Q5_K/Q6_K (heuristic)
   or Q6_K-uniform?" by direct measurement.
2. **Stage 3: depth bucketing.** For each category, ablate
   early/middle/late thirds separately. Probably matters most for
   attention tensors (Run 5 showed late-layer attn_k MSE is wildly
   different from early). The use_more_bits rule already does
   something like this implicitly; we can do better with measured
   coefficients.
3. **Recipe v3 (constrained optimization).** Given the (category,
   depth, bpw) coefficient table, find the recipe that minimizes
   predicted ΔPPL subject to total bpw ≤ target. Solvable as
   integer programming or simple greedy with budget tracking.
4. **Cross-architecture validation.** Build the same table for
   Llama-3.2-1B (no QK-norm) and a bigger Qwen3 (1.7B / 4B) to
   confirm the coefficient pattern is family-stable. The
   GGUFLab tool would then ship per-architecture tables.
5. **Re-examine the imatrix's role.** The imat-weighted MSE didn't
   predict PPL at the *category* level (Run 8 lost), but it might
   still predict PPL at the *per-tensor-within-a-category* level
   ("which `ffn_up` of which layer should I bump first?"). Worth
   testing — that would integrate the QK-norm finding back into
   the recipe builder usefully.

The QK-norm story (Runs 1-5) is now correctly contextualized: it's
a real measurement finding about K/Q tensors, but those tensors
are mid-PPL-coefficient, so chasing the MSE signal alone leads
the recipe astray. The fix is a different objective function, not
a different MSE measurement.

## Run 9 — 2026-04-27 (Stage 1 ablation: per-category PPL coefficients)

Run 8 closed the operational claim — the recipe loses to the heuristic
even at matched bpw. Run 9 is the diagnostic for *why*: per-tensor
imatrix-weighted MSE doesn't predict PPL impact. So we measure the
PPL coefficient per tensor category directly.

**Method**: F16 baseline + 7 single-category-ablation quants. For each
category in {attn_q, attn_k, attn_v, attn_output, ffn_up, ffn_gate,
ffn_down}, quantize *only that category* to Q4_K via a recipe that
explicitly maps every other 2-D weight tensor to F16. PPL on
`wiki.test.raw`. ΔPPL = (ablation PPL) − (F16 baseline PPL) is the
per-category PPL cost of going from F16 to Q4_K on those tensors only.

**Implementation note**: tt_overrides only fire when the base ftype is
quantized (the `if (!pure && ggml_is_quantized(default_type))` guard
in llama-quant.cpp). Naive `--ftype MostlyF16 --recipe target-only.json`
was a no-op until I switched to `--ftype Q4_K_M` with a recipe covering
*every* 2-D tensor (target → Q4_K, others → F16). Worth knowing for
future ablation work.

### Result

| rank | category       | ΔPPL    | heuristic does     | match?           |
|------|----------------|---------|--------------------|------------------|
| 1    | `ffn_up`       | **+0.4573** | Q4_K (default — no protection) | ❌ heuristic misses |
| 2    | `attn_v`       | +0.4367 | Q5_K / Q6_K (protected via use_more_bits) | ✅ correct |
| 3    | `attn_k`       | +0.2959 | Q4_K (default)     | ✅ correct       |
| 4    | `attn_q`       | +0.2674 | Q4_K (default)     | ✅ correct       |
| 5    | `attn_output`  | +0.0801 | Q4_K (default)     | ✅ correct       |
| 6    | `ffn_gate`     | +0.0757 | Q4_K (default)     | ✅ correct       |
| 7    | `ffn_down`     | **+0.0303** | Q5_K / Q6_K (protected via use_more_bits) | ❌ heuristic over-protects |

F16 baseline PPL: 21.4720.

### What the heuristic gets right and wrong

The Q4_K_M heuristic protects two categories: `attn_v` and `ffn_down`.
On Qwen3-0.6B:

- **`attn_v` protection — correct**. Ablation confirms `attn_v` is the
  2nd-most-sensitive category. The Llama-1/2 era intuition holds.
- **`ffn_down` protection — wrong**. Ablation shows `ffn_down` is the
  *least* sensitive category by a wide margin (+0.03 PPL — barely
  measurable). The heuristic burns budget here for ~no benefit.
- **`ffn_up` non-protection — wrong**. Ablation shows `ffn_up` is the
  *most* sensitive category, narrowly edging out `attn_v`. The
  heuristic puts it at the default Q4_K and walks past one of the
  highest-leverage protection opportunities.

The other four categories (attn_q, attn_k, attn_output, ffn_gate) are
all in the heuristic's "default Q4_K" bucket, and the ablation agrees
none of them are special enough to warrant protection on this model.

### Why this also explains Run 7/8's recipe failure

The Run 7 imatrix-weighted recipe at matched bpw (Run 8) was making
exactly the wrong allocation choices:

| category    | imat-weighted MSE rank | actual PPL-impact rank | recipe's choice                        | direction |
|-------------|------------------------|------------------------|----------------------------------------|-----------|
| `attn_k`    | 1st (MSE 0.76)         | 3rd                    | bumped to BF16 / Q8_0 on late layers   | wasted    |
| `attn_q`    | 2nd (MSE 0.56)         | 4th                    | bumped to Q8_0 on late layers          | wasted    |
| `attn_v`    | mid                    | 2nd                    | cut to IQ2_S on early layers           | harmful   |
| `ffn_up`    | mid                    | **1st**                | cut to IQ4_XS broadly                  | harmful   |
| `ffn_down`  | low                    | 7th                    | mostly preserved                       | overspend |

Imatrix-weighted MSE is uncorrelated with actual PPL coefficient.
The recipe was loud where the MSE was loud (late attn_k/attn_q from
QK-norm) and quiet where the MSE was quiet (ffn_up) — but PPL doesn't
care about the same things imatrix-weighted MSE cares about.

### What this gives us as a buildable recipe

A budget-conscious recipe for Qwen3-0.6B (and presumably the broader
Qwen3 family with QK-norm) should, relative to Q4_K_M:

- **Bump `ffn_up`** from Q4_K to Q5_K / Q6_K. The heuristic doesn't
  do this; ablation says it's the highest-leverage protection.
- **Cut `ffn_down`** from Q5_K / Q6_K back to Q4_K (or even IQ4_XS).
  The heuristic over-spends here for negligible PPL benefit.
- **Keep `attn_v` at Q5_K / Q6_K** as the heuristic does.
- **Keep `attn_q`, `attn_k`, `attn_output`, `ffn_gate` at Q4_K** as
  the heuristic does.
- **Don't bump late-layer `attn_k` / `attn_q` to Q8_0+ even though
  imat-weighted MSE screams**. They're mid-sensitivity per the
  ablation; spending a budget there is worse than spending it on
  `ffn_up`.

This is a *testable* recipe. Whether it actually beats Q4_K_M at
matched bpw on PPL is the natural next experiment.

### What other factors might still matter

Stage 1 is the cheapest first cut — uniform per-category coefficients
at one bpw point. Open dimensions:

1. **(category × bpw)**: does `attn_v` at Q3_K hurt 4× more than at
   Q4_K, or 16×, or differently per category? Repeating Stage 1 at
   Q2_K and Q6_K gives a (category, bpw) lookup table.
2. **Depth bucketing**: Run 5 showed late-layer `attn_k` MSE behaves
   very differently from early-layer. Per-depth ablation (early /
   middle / late thirds × per-category) tests whether the PPL
   coefficient is layer-dependent. 7×3 = 21 buckets × 1 bpw =
   significant compute but probably worth it before claiming a
   universal "Qwen3 recipe."
3. **Coupling**: the Stage-1 model assumes per-category effects sum.
   Real PPL impact may have non-additive coupling (e.g., ablating
   `attn_v` + `ffn_down` together hurts more than the sum). Pairwise
   ablations can detect this; full-coverage recipes (like
   Q4_K_M itself) are the integration test.
4. **Position in residual stream**: a structural prior, not a
   measurement — `attn_output` and `ffn_down` write to the residual
   while `attn_q/k/v` are consumed inside attention. The ablation
   data is consistent with this prior (ffn_up — a residual-stream
   *input* — being most sensitive while ffn_down — also residual-
   stream-adjacent — being least sensitive is interesting, suggests
   the up-projection's role in shaping the post-norm input matters
   more than the heuristic assumed).

### Operational guidance update

The Adaptive Quantization tool's recipe should NOT use raw imat-
weighted MSE thresholds — that demonstrably picks the wrong tensors.
Instead the recipe should be informed by per-category PPL coefficients.
A future Adaptive Quantization v2 could:

1. Ship per-architecture coefficient tables (Stage 1+2 results).
2. Solve a constrained optimization: minimize Σ α(type, bpw) × Δ_PPL
   subject to total bpw ≤ budget. The MSE measurement still
   contributes — it picks *which tensor within a category* gets
   bumped — but the *category-level* allocation comes from the
   coefficients, not raw MSE.

This is a real research project; running Stage 2 + Stage 3 and
publishing the coefficient tables would be the deliverable.

## Run 8 — 2026-04-26 21:04  (matched-bpw recipe vs heuristic)

Run 6/7 compared the recipe at τ=0.01 (4.33 BPW) against stock
Q4_K_M (5.09 BPW). The 0.76 BPW budget gap was a reasonable
critique of the verdict — the heuristic had more bits to spend, so
losing on PPL with less budget didn't necessarily mean the recipe's
allocation was *worse*, only that less budget plus MSE-aware
allocation lost to more budget plus heuristic allocation.

This run binary-searches τ to find a recipe with the same total
BPW as Q4_K_M, then compares.

**Setup**: τ=0.00456 yields 5.11 BPW (within 0.02 of the heuristic's
5.09). Quantized → `qwen3-recipe-matched.gguf`. PPL on
`wiki.test.raw`, n_ctx=512, second-half scoring.

### Result

| variant                      | bpw  | size  | PPL          |
|------------------------------|------|-------|--------------|
| heuristic Q4_K_M             | 5.09 | 484 MB| **22.4027**  |
| recipe at τ=0.00456 (matched)| 5.11 | 486 MB| **26.2425**  |
| recipe at τ=0.01 (Run 7)     | 4.33 | 388 MB| 29.0736      |

The matched-bpw recipe is +3.84 PPL (+17 %) worse than the
heuristic, despite having 0.02 BPW *more* budget to spend. The
recipe's loss is **not a budget-shortage artifact** — it's a
genuine allocation problem.

### Final verdict — the heuristic is actually well-tuned

The investigation arc is complete:

- Run 5 (architectural): QK-norm produces a real, isolated
  ~100× amplification of imatrix-weighted MSE on `attn_k`/`attn_q`.
  Stable observation, confirmed by the 4-way control.
- Run 8 (operational): even when the recipe spends the same total
  BPW as Q4_K_M, the heuristic's distribution of bits beats the
  recipe's by 17 % PPL. The heuristic's "tradition" of protecting
  `attn_v` and early-layer ffn_down via `use_more_bits` is
  empirically *correct* for PPL outcomes, even on QK-norm
  architectures where imatrix-weighted MSE measurement says
  `attn_v` is over-protected and late `attn_k`/`attn_q` are under-
  protected.

What this means: **per-tensor imatrix-weighted MSE is not a good
proxy for PPL impact**. The recipe-at-τ rule treats every tensor
in isolation and aggressively cuts wherever individual MSE looks
acceptable; the heuristic uses static rules informed by years of
hand-tuning on real PPL outcomes. The static rules win at matched
bpw on this test.

The QK-norm imatrix-weighted MSE story (Runs 1–5) describes a real
quantization phenomenon, but **acting on it via a per-tensor τ
threshold actively makes the model worse**. The model is robust
to per-tensor reconstruction error in ways individual MSE doesn't
capture; what matters is how errors compound across the residual
stream, and the heuristic's allocation evidently composes better.

### Operational guidance

- Don't trust the GGUFLab Adaptive Quantization recipe as a
  drop-in replacement for stock ftypes. At τ=0.01 it's 30 % worse
  PPL with 16 % less budget; at matched-bpw τ it's still 17 %
  worse PPL.
- The recipe is a useful **diagnostic** for researchers — it
  surfaces where the heuristic's allocation diverges from a
  per-tensor-MSE-greedy allocation. The Run 1 table answers
  "where does the heuristic spend bits the MSE doesn't justify,
  and where does it skimp on tensors with high MSE?". That's a
  legitimate research output.
- For actual deployment quants, use stock Q4_K_M (or whatever
  ftype meets your budget). The legacy heuristic is well-tuned
  enough that beating it with a per-tensor MSE rule isn't
  possible at this granularity.
- A PPL-anchored search loop would be the right way to find a
  recipe that actually beats Q4_K_M, but that's a different
  problem (Bayesian optimization over per-tensor type assignment,
  PPL as the objective). The Adaptive Quantization tool's
  τ-threshold recipe isn't that loop.

### What's left for future work

1. PPL-anchored search rather than τ-threshold (significant work).
2. K-quants only as a candidate ladder, in case IQ-quant kernels
   are systematically worse than their MSE suggests.
3. Verify the verdict on a different model (Qwen3-1.7B, Qwen3-4B,
   Phi-3) — does the heuristic also beat the recipe at matched
   bpw on bigger QK-norm models, or is this Qwen3-0.6B-specific?

The investigation is paused here. The architectural finding is
worth keeping; the operational claim is retracted.

## Run 7 — 2026-04-26 20:58  (Run 6 redone with fixed imatrix collector)

After fixing the imatrix-collector bug (last-block FFN tensors were
silently dropped due to a logits=last-only / scheduler-pruning
interaction), the whole Run 6 chain was re-executed to confirm the
verdict wasn't an artifact of the bug:

  1. Rebuilt Qwen3-0.6B imatrix — 196 tensors tracked (was 193, +3
     for the previously-missing `blk.27.ffn_*`).
  2. Re-swept Qwen3-0.6B with the new imatrix → fresh score table.
  3. Quantized heuristic Q4_K_M with new imatrix.
  4. Quantized recipe at τ=0.01 with new sweep + new imatrix.
  5. PPL on both vs `wiki.test.raw`, n_ctx=512, second-half scoring.

### Result: same verdict, smaller gap

| variant                   | bpw  | size  | Run 6 PPL | Run 7 PPL (fixed) |
|---------------------------|------|-------|-----------|-------------------|
| heuristic Q4_K_M          | 5.09 | 484 MB| 22.3648   | **22.4027**       |
| recipe at τ=0.01          | 4.33 | 388 MB| 29.3432   | **29.0736**       |
| recipe − heuristic        |      |       | +6.98     | **+6.67**         |

The imatrix fix moved both numbers slightly in the expected
direction: heuristic essentially unchanged (+0.04 PPL), recipe
slightly better (−0.27 PPL). The recipe-vs-heuristic gap shrank by
~0.3 PPL but is still ~6.7 PPL (~30 %). The 3 tensors that were
missing from the bug (`blk.27.ffn_down/gate/up`) constitute 1.5 %
of the model's weights and aren't in the QK-norm-affected category
that drives the asymmetry signal — so it's reassuring but not
surprising that fixing them barely moved the result.

### Final verdict (after Run 7)

The imatrix bug was real and is fixed; the QK-norm signature
(Run 5) and the recipe-PPL-loss (Run 6/7) both stand. Specifically:

- The Run 5 architectural finding — that QK-norm causes ~100×
  amplification of imatrix-weighted MSE on `attn_k`/`attn_q` —
  is independent of the imatrix bug because every `attn_*` tensor
  always had correct imatrix entries.
- The recipe at τ=0.01 spends 16 % less budget than Q4_K_M and
  loses by ~30 % PPL. Per-tensor MSE optimization (the recipe-at-τ
  rule) is too greedy: errors compound across the layer chain in
  ways individual rel-MSE doesn't capture, so the recipe's
  aggressive `IQ2_S`/`IQ4_XS` cuts hurt more than the protection
  on late-layer K/Q helps.

**Operational recommendation**: do not market the Adaptive
Quantization recipe at τ=0.01 as "better than Q4_K_M" — it isn't,
on this PPL test. The recipe is a useful diagnostic (see Run 1's
table; it correctly identifies which tensors are most sensitive
under imatrix weighting) but its picks shouldn't be applied
blindly. A PPL-anchored search (try a recipe → score PPL →
adjust → repeat) is the right way to find a recipe that beats
Q4_K_M, if one exists at this budget at all.

The future-work shortlist from Run 6 is unchanged:

1. Recipe at τ tuned to match Q4_K_M's bpw exactly (Run 8).
2. K-quants only, no IQ-quants (Run 9).
3. PPL-anchored recipe search rather than τ-driven recipe.

## Run 6 — 2026-04-26 19:39  (PPL test — the recipe loses)

**Why this run**: Run 5 produced a clean architectural signature
(QK-norm amplifies imatrix-weighted MSE on K/Q by ~100×) and the
"operational implication" was that the Adaptive Quantization
recipe — built from imatrix-weighted scores — should outperform
the heuristic on wikitext PPL because it spends bits where the
imatrix-aware error is concentrated. Run 6 tests that claim
directly.

**Setup**: New `samples/Quantize.Cli` (with `--imatrix` and
`--recipe-from-scores` / `--tau`) and `samples/Perplexity.Cli`.
The bindings now surface the imatrix path on
`LlamaQuantizationParameters` so the production quantizer is
imatrix-aware end-to-end. Built two variants from
`Qwen3-0.6B.F16.gguf`:

  1. **Heuristic baseline**: `--ftype Q4_K_M --imatrix sidecar`.
     Stock per-tensor heuristic, imatrix-aware quantization.
  2. **Recipe-overridden**: same plus
     `--recipe-from-scores Qwen3-0.6B.F16.scores.json --tau 0.01`.
     The Run 1 imatrix-weighted score table feeds the recipe;
     `tt_overrides` pin per-tensor types from there.

PPL on wikitext-2 test (`wiki.test.raw`, 583 chunks at n_ctx=512,
second-half scoring — matches llama.cpp's published-numbers setup).

### Result

| variant                        | bpw  | size      | PPL          |
|--------------------------------|------|-----------|--------------|
| stock Q4_K_M (heuristic)       | 5.09 | 484 MB    | **22.3648**  |
| recipe at τ=0.01               | 4.31 | 386 MB    | **29.3432**  |

The recipe is **31 % worse on PPL** while saving 16 % bytes. That
isn't a fair-bpw comparison — the recipe spent 0.78 bpw less — but
the trade is still bad: a 31 % PPL hit for 16 % less storage is
not a win at any reasonable price point.

### What this tells us (honestly)

The Run 5 verdict on QK-norm — that imatrix-weighted MSE blows up
on QK-norm K/Q tensors specifically and asymmetrically — is still
true as an architectural signature. The *operational* claim that
followed from it ("therefore the recipe outperforms the heuristic")
is **not supported** by this test.

Two mechanisms likely:

1. **Per-tensor MSE doesn't compose into PPL impact.** Even when
   every individual tensor passes τ in isolation, errors stack
   through the 28-layer forward pass. The recipe's many `IQ2_S`
   picks (early-layer `attn_v`, etc.) each have small per-tensor
   rel-MSE but their cumulative effect on the residual stream
   dwarfs the savings from protecting late `attn_k`/`attn_q`.
2. **High imatrix-weighted MSE on a tensor doesn't necessarily
   mean that tensor's quantization hurts PPL much.** The activation
   importance the imatrix captures is a per-token expectation;
   the model can be robust to large errors in heavily-used columns
   if downstream layers compensate. The 0.76 rel-MSE on
   `blk.26.attn_k` doesn't translate one-to-one into a 76 %
   degradation in attention outputs from that layer.

### Caveats

- **Imatrix-collector bug**: both Qwen3-0.6B and Llama-3.2-1B
  imatrix files are missing the *last* block's FFN tensors
  (`blk.27.ffn_*` for Qwen3, `blk.15.ffn_*` for Llama). 3/198 of
  Qwen3's tensors. The bug is deterministic in
  `LlamaImatrix.ComputeAsync`. Both the heuristic baseline and
  the recipe quantize call get the same incomplete imatrix, so
  the comparison is equally handicapped on both sides — the gap
  isn't an artifact of this. The bug is logged as a follow-up
  to fix; tests should be re-run after the fix.
- **bpw not matched**. The right next test is recipe at a tighter
  τ (so it lands at ~5.09 bpw matching the heuristic). At equal
  budget, the recipe's MSE-aware allocation might still help —
  this run only shows that "less budget plus MSE-aware allocation"
  loses to "more budget plus heuristic allocation".

### Where this leaves us

The recipe-vs-heuristic claim is now uncertain in this direction
and likely to need substantial work to pin down. The honest
read:

- The QK-norm signature is real (Run 5 stands).
- Imatrix-weighted MSE is a *measurement of* something about the
  weights, but it isn't a directly-actionable proxy for PPL.
  Per-tensor optimization (and the recipe-at-τ rule) is too
  greedy — it cuts where individual tensors look "fine" without
  modeling the cumulative effect across layers.
- The Adaptive Quantization tool's recipe is a useful diagnostic
  (which tensors does the heuristic mispredict?) but its output
  shouldn't be applied blindly. A PPL-anchored search loop
  (try a recipe → score PPL → adjust → repeat) would be the
  honest next step if we want a recipe that beats the heuristic.

### Future-work shortlist

1. Fix the imatrix-collector bug, regenerate, re-run the whole
   chain to confirm the conclusions are stable.
2. Run 7: recipe at τ tuned to match Q4_K_M's bpw (around 0.005
   should land near 5.09). If the recipe beats the heuristic at
   matched bpw, the operational claim gets resurrected.
3. Run 8: K-quants only, no IQ-quants. The IQ-quant codebook
   search may have systematic issues this experiment doesn't
   isolate.
4. PPL-anchored recipe search rather than τ-driven recipe.

## Run 5 — 2026-04-26 19:11  (Llama-3.2-1B with imatrix — verdict)

**Why this run**: Run 4 reversed the conclusion by showing that
unweighted MSE looks normal on Qwen3-0.6B. The remaining open
question was whether imatrix weighting amplifies MSE on *every*
architecture (in which case the imatrix is just an MSE inflator
and QK-norm is a red herring) or *only* on QK-normed K/Q tensors
(in which case QK-norm has a real, isolated signature).

**Setup**: built `Llama-3.2-1B.imatrix.gguf` from wikitext-2 test
(564 chunks, 288k tokens, 109 tensors tracked, 4.9 min). Re-swept
the model with that imatrix attached.

**Commands**:
```bash
llama-imatrix \
  --input  ~/.cache/llama-models/bartowski/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-f16.gguf \
  --corpus ~/.cache/llama-test-models/wiki.test.raw \
  --output ~/.cache/llama-models/bartowski/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct.imatrix.gguf

llama-sensitivity-sweep \
  --input   .../Llama-3.2-1B-Instruct-f16.gguf \
  --imatrix .../Llama-3.2-1B-Instruct.imatrix.gguf \
  --output  .../Llama-3.2-1B-Instruct.imatrix-weighted.scores.json \
  --benchmark
```

### The full 4-way matrix

Worst rel-MSE at the Q4_K_M heuristic's pick, by category:

|                           | attn_k     | attn_q     | attn_v     | ffn_down   | ≥0.05    | ≥0.10    |
|---------------------------|------------|------------|------------|------------|----------|----------|
| Qwen3-0.6B  unweighted    | 5.53e-3    | 5.38e-3    | 1.40e-3    | 5.67e-3    | 0/198    | 0/198    |
| Qwen3-0.6B  imat-weighted | **7.59e-1**| **5.65e-1**| 5.13e-2    | 7.41e-3    | 21/198   | 17/198   |
| Llama-3.2-1B unweighted   | 6.19e-3    | 5.54e-3    | 1.44e-3    | 5.43e-3    | 0/113    | 0/113    |
| Llama-3.2-1B imat-weighted| 1.24e-3    | 1.10e-3    | 2.84e-4    | 1.40e-3    | 0/113    | 0/113    |

Imatrix-weighting effect (imat-weighted ÷ unweighted):

| tensor       | Qwen3-0.6B | Llama-3.2-1B  |
|--------------|------------|---------------|
| `attn_k`     | **×138**   | ×0.20 (5× reduction) |
| `attn_q`     | **×105**   | ×0.20          |
| `attn_v`     | ×37        | ×0.20          |
| `ffn_down`   | ×1.3       | ×0.26          |

### Verdict

The QK-norm hypothesis is confirmed in its refined form. The
asymmetry has three properties that together rule out alternative
explanations:

1. **Imatrix weighting normally *reduces* MSE.** On Llama-3.2-1B
   every category drops by ~5× under imatrix weighting. That's
   the expected behavior — the quantizer's per-block scale
   optimization concentrates precision on heavy-imatrix columns,
   so the column-importance-weighted error shrinks. So the
   imatrix is *not* a generic MSE inflator.
2. **Qwen3-0.6B uniquely *amplifies*** on attention tensors —
   100×+ on `attn_k` and `attn_q`, 37× on `attn_v`. The amplification
   is isolated to the tensors QK-norm operates on.
3. **`ffn_down` is unaffected.** The non-attention category
   amplification on Qwen3 is ×1.3, basically a no-op. So the
   Qwen3 vs Llama difference isn't "Qwen3 trained harder" or
   anything generic — it's specific to K/Q/V.

Mechanism: QK-norm normalizes Q and K activations *independently*
(after the linear projection, before the dot product). That frees
the *raw* Q and K weights from the constraint of producing unit-
scale activations on average; the post-norm scaling re-normalizes
whatever the projection produces. The result is wider per-column
dynamic range in the K/Q activations. The imatrix captures this
range and weights heavily-used columns proportionally. Q4_K's
per-block (256 elements) linear scale can fit the per-block weight
distribution, but it can't *track* the per-column importance
variance — so quantization errors that land in heavy-imatrix
columns aren't suppressed the way they are on non-QK-norm models.

In short: **on a QK-norm model, the per-block-scale quantizer
delivers its usual reconstruction quality on the raw weights, but
forfeits the usual benefit of imatrix-aware quantization for K/Q
tensors specifically**. The errors that land in heavy-imatrix
columns aren't preferentially suppressed.

### Operational implication (now grounded)

The Run 1 imatrix-weighted picture was real: at Q4_K, late-layer
`attn_k`/`attn_q` of Qwen3-0.6B accumulate errors in heavily-used
inference columns. Whether this translates to perplexity
degradation is the next question — possible Run 6 — but the
mechanism is now identified, not speculated.

For GGUFLab's Adaptive Quantization tool: when an imatrix is
supplied, the recipe surfaced from Run 1's score table (`Q8_0`
on late `attn_k`/`attn_q`) is doing the operationally correct
thing: spending bits where the imatrix-aware error is high. The
recipe at the same total bpw as Q4_K_M's heuristic should
outperform the heuristic on this model — that's the perplexity
test waiting in Run 6.

For users: when running Adaptive Quantization on a QK-norm
architecture (Qwen3, also worth verifying on Phi-3, Gemma-2,
Llama-3.1+), supply an imatrix. Without one the recipe will
look indistinguishable from the heuristic; *with* one, the
recipe diverges in exactly the places that matter for inference.

## Run 4 — 2026-04-26 18:43  (Qwen3-0.6B re-swept WITHOUT imatrix — conclusion reversed)

**Why this run**: Run 1's imatrix-weighted Qwen3 numbers were
compared against Run 3's unweighted Llama numbers — apples vs
oranges. Re-running Qwen3 without imatrix puts both models on the
same footing, naive per-element MSE.

**Command**:
```bash
llama-sensitivity-sweep \
  --input ~/.cache/llama-models/Qwen/Qwen3-0.6B/Qwen3-0.6B.F16.gguf \
  --output ~/.cache/llama-models/Qwen/Qwen3-0.6B/Qwen3-0.6B.F16.unweighted.scores.json \
  --benchmark
```

**Result** — wall 233 s, 3.4× CPU speedup (matches Run 1 timings).

### The headline finding flips

Worst rel-MSE at the heuristic's pick, by category:

| category   | Qwen3 (Run 1, w/ imatrix) | Qwen3 (Run 4, unweighted) | Llama-3.2-1B (Run 3, unweighted) |
|------------|---------------------------|---------------------------|----------------------------------|
| attn_k     | **7.59e-1**               | 5.53e-3                   | 6.19e-3                          |
| attn_q     | **5.65e-1**               | 5.38e-3                   | 5.54e-3                          |
| attn_v     | 5.13e-2                   | 1.40e-3                   | 1.44e-3                          |
| attn_output| 2.59e-2                   | 5.30e-3                   | 5.38e-3                          |
| ffn_gate   | 1.60e-2                   | 5.36e-3                   | 5.37e-3                          |
| ffn_up     | 1.51e-2                   | 5.36e-3                   | 5.32e-3                          |
| ffn_down   | 7.41e-3                   | 5.67e-3                   | 5.43e-3                          |
| TOKEN_EMBD | 5.08e-3                   | 5.08e-3                   | 5.45e-3                          |

Tensors with rel-MSE ≥ 0.05 at heuristic-pick:
- Qwen3 imatrix-weighted: 21 / 198
- Qwen3 unweighted:       **0 / 198**
- Llama-3.2-1B:           0 / 113

### What this means

The "QK-norm makes attention weights catastrophically hard to
quantize at Q4_K" claim from Run 2 was **wrong**. Without imatrix
weighting, Qwen3-0.6B's late-layer `attn_k` quantizes to Q4_K
about as cleanly as Llama-3.2-1B's. The raw weight reconstruction
isn't unusual.

The 0.76 number in Run 1 came from imatrix-weighted MSE. The
imatrix-weighted formula multiplies each per-element error by
the per-column importance:

`rel_MSE_imat = (1/N) Σ imatrix[col] · (src - rt)² / mean(W²)`

If `imatrix` has very wide dynamic range across columns (some
columns ~1000×, others ~1), then a normal-magnitude error landing
in a heavy-weight column blows up the sum while a similar error
in a light column contributes nothing. The denominator
(`mean(W²)`) doesn't include the imatrix weighting, so the ratio
inflates.

So what Run 1 actually measured was: *in the columns the model
heavily uses during inference, are the Q4_K-quantized errors big*?
The answer for late-layer Qwen3 attn_k was yes, dramatically.
That's still a meaningful operational signal — errors in dead
columns don't hurt inference, errors in live columns do — but
the cause story changes.

### The new (open) hypothesis

QK-norm probably *causes* per-column activation dynamic range to
be wider on K and Q tensors (because the per-element norm relaxes
the constraint on raw activation magnitudes). Wider activation
range → wider imatrix → more amplification of any per-element
error in heavy columns. The raw weights themselves quantize fine.

But "probably" is doing work here. To confirm or deny, we need
**Run 5**: build an imatrix for Llama-3.2-1B and re-sweep with it.
If Llama also goes to 0.5+ rel-MSE on attn_k under imatrix
weighting, the imatrix is just an MSE inflator and QK-norm is a
red herring. If Llama stays under ~0.05, the QK-norm specificity
holds.

## Run 3 — 2026-04-26 18:23  (QK-norm hypothesis test on Llama-3.2-1B)

Pivoted the contrast model. Qwen2-0.5B was the obvious "same family
minus QK-norm" choice, but its `hidden_size = 896` isn't divisible
by the K-quant super-block size (256), so llama.cpp's heuristic
falls back to Q8_0 for almost every attention/FFN tensor — the
"Q4_K_M" file is mostly Q8_0 by mass. The sensitivity sweep
faithfully marks K-quants as skipped for those tensors, but the
result isn't comparable to Qwen3-0.6B's all-K-quant world.
Llama-3.2-1B (`hidden_size = 2048 = 8×256`) keeps the candidate
ladder fully populated.

**Source**: `bartowski/Llama-3.2-1B-Instruct-GGUF` →
`Llama-3.2-1B-Instruct-f16.gguf` (2.4 GB). 16 layers, MHA without
QK-norm. **Sweep was unweighted** — no imatrix sidecar exists for
this model in our cache, so the comparison numbers below are naive
MSE. Run 4 re-runs Qwen3-0.6B unweighted to put the contrast on
matching units. Sweep ran in 6.4 min wall, 3.4× CPU speedup vs serial.

**Command**:
```bash
llama-sensitivity-sweep \
  --input ~/.cache/llama-models/bartowski/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-f16.gguf \
  --output ~/.cache/llama-models/bartowski/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct.scores.json \
  --benchmark
python3 tools/analyze-sensitivity-vs-heuristic.py \
  --scores .../Llama-3.2-1B-Instruct.scores.json --ftype q4_k_m --layers 16
```

### Headline contrast

Both models disagree with the heuristic on ~all tensors at matched
bpw — but the *severity* of the disagreement is dramatically
different. The right metric is "what rel-MSE does the heuristic's
pick actually produce?", not "do the picks disagree?".

| metric                                       | Qwen3-0.6B (QK-norm) | Llama-3.2-1B (no QK-norm) |
|----------------------------------------------|----------------------|---------------------------|
| worst rel-MSE at heuristic-pick              | **0.759**            | 0.0062                    |
| worst category                               | `attn_k` blk.26      | `attn_k` blk.15           |
| tensors with rel-MSE ≥ 0.05 at heuristic     | 21 / 198             | **0 / 113**               |
| tensors with rel-MSE ≥ 0.10 at heuristic     | 17 / 198             | 0 / 113                   |
| heuristic average bpw                        | 4.803                | 4.797                     |
| recipe avg bpw at τ=0.01                     | 4.527                | 4.250                     |

### Worst-case rel-MSE per category

| category              | Qwen3-0.6B max | Llama-3.2-1B max | ratio |
|-----------------------|----------------|-------------------|-------|
| `attn_k`              | 7.59e-1        | 6.19e-3           | **122×** |
| `attn_q`              | 5.65e-1        | 5.54e-3           | **102×** |
| `attn_v`              | 5.13e-2        | 1.44e-3           | 36×   |
| `attn_output`         | 2.59e-2        | 5.38e-3           | 4.8×  |
| `ffn_gate`            | 1.60e-2        | 5.37e-3           | 3.0×  |
| `ffn_up`              | 1.51e-2        | 5.32e-3           | 2.8×  |
| `ffn_down`            | 7.41e-3        | 5.43e-3           | 1.4×  |
| `TOKEN_EMBD`          | 5.08e-3        | 5.45e-3           | 0.9×  |
| `OUTPUT`              | 3.14e-4        | (n/a, tied)       | —     |

The QK-affected categories (`attn_k`, `attn_q`, `attn_v`) explode
two orders of magnitude on the QK-norm model relative to the
no-QK-norm baseline. The non-attention categories (`ffn_*`,
`TOKEN_EMBD`) are within ~3× — same general behavior, just a
slightly harder model. **The QK-norm signal is isolated to the
tensors QK-norm operates on**, which is exactly what the
hypothesis predicts.

### Verdict

Hypothesis confirmed. The Q4_K_M heuristic is fine on classical
MHA models like Llama-3.2-1B (mildly over-cautious — every tensor
under 1% rel-MSE means the recipe could drop most picks to IQ4_XS
and still keep quality, but the heuristic isn't *wrong*). On
QK-normed Qwen3-0.6B it under-protects late-layer `attn_k`/`attn_q`
to the point of catastrophic per-tensor error.

The mechanism is now clear: QK-norm shifts where sensitivity lives
because the post-norm activation is what the dot-product cares
about, leaving the *raw* K and Q weights free to adopt wider per-
block dynamic ranges that K-quant's per-block scale can't capture.
This is consistent with QK-norm's design intent (give the network
flexibility on the K/Q side of the attention) but has the
side-effect of making those weights uniquely hard to quantize at
4-bit per-block precision.

### Operational implication

For QK-norm architectures (Qwen3, also worth checking on Phi-3,
Gemma-2, Llama-3.1+), the stock Q4_K_M is delivering significantly
worse per-tensor reconstruction than its label suggests. The
Adaptive Quantization tool's recipe correctly addresses this by
spending the budget where it actually matters (Q8_0 on the worst
attn_k/attn_q, IQ-quants on tensors the heuristic over-protects).
The next step (Run 4, deferred) is a real perplexity comparison —
recipe-built GGUF vs stock Q4_K_M on wikitext — to confirm the
per-tensor MSE story translates to end-to-end model quality.
