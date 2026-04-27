# Qwen3-0.6B QK sensitivity vs llama.cpp's Q4_K_M heuristic

Premise: GGUFLab's Adaptive Quantization tool runs a per-tensor sensitivity
sweep against an F16 source and assigns each tensor the smallest quant
type whose round-trip relative MSE stays below a threshold τ. Comparing
those per-tensor picks against llama.cpp's hard-coded ftype heuristic
answers a concrete question: *given the same overall bit budget, does
the heuristic allocate bits to the right tensors?*

The expectation going in was that for Qwen3-0.6B — small, dense, well-
studied — the heuristic should be approximately right.

**Final verdict (after Run 5)**: QK-norm has a real, isolated
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
