# Qwen3-0.6B QK sensitivity vs llama.cpp's Q4_K_M heuristic

Premise: GGUFLab's Adaptive Quantization tool runs a per-tensor sensitivity
sweep against an F16 source and assigns each tensor the smallest quant
type whose round-trip relative MSE stays below a threshold τ. Comparing
those per-tensor picks against llama.cpp's hard-coded ftype heuristic
answers a concrete question: *given the same overall bit budget, does
the heuristic allocate bits to the right tensors?*

The expectation going in was that for Qwen3-0.6B — small, dense, well-
studied — the heuristic should be approximately right. Result: it isn't.
The heuristic over-protects `attn_v` and badly under-protects late-layer
`attn_k` / `attn_q`, almost certainly because Qwen3 uses QK-norm and the
heuristic was tuned in the Llama-1/2 era when `attn_v` was empirically
the most-sensitive attention tensor.

## Run 1 — 2026-04-26 16:01

**Command**: GGUFLab → Adaptive Quantization → Run sensitivity sweep
on `~/.cache/llama-models/Qwen/Qwen3-0.6B/Qwen3-0.6B.F16.gguf`. No
imatrix supplied. Default 11 candidates: F16, BF16, Q8_0, Q6_K, Q5_K,
Q4_K, IQ4_XS, Q3_K, IQ3_S, Q2_K, IQ2_S. 198 tensors scored after
filtering 1-D norms.

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

- **No imatrix**. The sweep was unweighted. With imatrix the late
  `attn_k`/`attn_q` numbers might come down somewhat (the imatrix
  picks up activation scale), but the contrast with the heuristic's
  `attn_v` priority is unlikely to flip.
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
2. Re-sweep Qwen3-0.6B *with* an imatrix (built from wikitext-2 in
   the Imatrix tool) and re-run the analysis to see how column
   weighting shifts the picks.
3. Side-by-side perplexity: stock Q4_K_M vs the recipe at the same
   bpw. If the recipe wins on PPL the finding is operational, not
   just academic — the GGUFLab page can recommend the recipe over
   the heuristic for Qwen3-class models.
