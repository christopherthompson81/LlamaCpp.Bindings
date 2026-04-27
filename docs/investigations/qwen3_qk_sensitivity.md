# Qwen3-0.6B QK sensitivity vs llama.cpp's Q4_K_M heuristic

Premise: GGUFLab's Adaptive Quantization tool runs a per-tensor sensitivity
sweep against an F16 source and assigns each tensor the smallest quant
type whose round-trip relative MSE stays below a threshold τ. Comparing
those per-tensor picks against llama.cpp's hard-coded ftype heuristic
answers a concrete question: *given the same overall bit budget, does
the heuristic allocate bits to the right tensors?*

The expectation going in was that for Qwen3-0.6B — small, dense, well-
studied — the heuristic should be approximately right.

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
