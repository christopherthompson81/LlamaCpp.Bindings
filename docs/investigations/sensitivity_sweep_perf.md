# Sensitivity-sweep performance — where the time goes, what's left

GGUFLab's Adaptive Quantization tool runs `LlamaQuantSensitivity.MeasureAsync`,
which for each tensor and each candidate type round-trips F32 → quant → F32
and computes relative MSE. Earlier work parallelized the per-tensor candidate
loop with `Parallel.For`. This investigation uses the new
`samples/SensitivitySweep.Cli` to characterize what's actually expensive,
how parallelism scales, and what's worth optimizing next.

## Test rig

- Intel i7-10700K (8 physical cores, 16 logical via SMT, AVX2)
- NVIDIA RTX 3090 (irrelevant — quantization is CPU-only upstream)
- Qwen3-0.6B F16, layers 0–3 only (28 tensors × 11 candidates = 308 rows)
- No imatrix, default candidate ladder

## Run 1 — 2026-04-26 17:20  (parallelism scaling)

```bash
for n in 1 2 4 8 16; do
  llama-sensitivity-sweep \
    --input Qwen3-0.6B.F16.gguf \
    --include '^blk\.[0-3]\..*\.weight$' \
    --max-parallel $n \
    --benchmark
done
```

| max-parallel | wall (s) | speedup | quantize CPU (s) | total CPU (s) |
|--------------|----------|---------|------------------|---------------|
| 1            | 67.92    | 1.00×   | 62.99            | 67.83         |
| 2            | 37.40    | 1.82×   | 66.36            | 71.10         |
| 4            | 30.42    | 2.23×   | 66.49            | 71.29         |
| 8            | 23.03    | 2.95×   | 66.66            | 71.43         |
| 16           | 23.07    | 2.94×   | 99.82            | 104.71        |

### Findings

1. **Quantize is 93% of all CPU time.** Source-dequantize, candidate-
   dequantize, and the C# MSE loop together are <8%. Anything below
   that floor isn't worth optimizing.

2. **Sweet spot is 8 threads.** Matches physical-core count on this
   8C/16T machine. The default (`Environment.ProcessorCount / 2`) is
   already correct.

3. **SMT contention is real and costly.** Going from 8 → 16 threads
   gives **zero wall-clock improvement** but burns **50% more CPU
   time** (66.66 s → 99.82 s in the quantize phase alone). The
   quantize kernels are AVX-heavy; two hyperthreads on the same core
   contend for the same SIMD execution unit and serialize. Defaulting
   to `ProcessorCount / 2` was the right call.

4. **Memory bandwidth contention sets in around 4 threads.** Speedup
   drops from 1.82× (p=2) → 1.12× per added thread (p=4) → 0.36× per
   added thread (p=8). The source F32 buffer is shared across
   candidates (good — fits in L2 / streamed) but each writer streams
   gigabytes of quant output to its own destination buffer; eight of
   those concurrent writers saturate the DRAM channel.

## Where the remaining headroom is

Quantize-phase scaling at p=8 is 2.95× speedup vs. 8× theoretical —
a ~37% efficiency. The remaining 5× is split between:

- **Memory bandwidth** (~50% of the gap): each parallel worker streams
  its own quant output. K-quants are 4–8 bpw, so a worker writes
  ~0.5–1 byte for every 4 bytes read — read-dominated. Writes are
  smaller but still real on this single-channel desktop CPU.
- **Per-block iterative search in IQ-quants** (~25%): IQ2_S, IQ3_S,
  IQ4_XS run a codebook-search loop per 32-element block. The other
  10 candidates finish quickly while these three drag on.
- **Allocation churn** (~25%): each candidate allocates its own
  `float[]` round-trip buffer (155M × 4B = 620 MB for token_embd).
  At p=8 that's ~5 GB of live float arrays during a big tensor.
  GC is mostly background but page-fault and cache-line contention
  on freshly-allocated pages costs measurable wall time.

## What's worth doing next

### Sub-sample huge tensors (highest impact)

`token_embd.weight` on Qwen3-0.6B is 155M elements. At p=8 with the
full ladder, it takes ~3 minutes — most of the sweep's wall clock on
this small model. Quantization MSE per row is independent across
rows, so scoring 4 000 rows of token_embd instead of 151 936 yields
a statistically equivalent estimate at ~38× speedup on this one
tensor. Add a `MaxRowsPerTensor` option (default ~4 000 for tensors
above some element threshold). Estimated end-to-end speedup on a
typical sweep: **3–5× on small dense models, larger on bigger ones**
because embedding/output dominates more.

### Skip F16 / BF16 against an F16 source (free)

When the source is already F16, the F16 round-trip is bit-exact and
BF16 differs only in mantissa truncation — both rel_MSE numbers are
known a priori (≈0 for F16, ~1e-7 for BF16). Short-circuit them and
skip the work. Saves 2/11 ≈ **18% on tensors where K-quants are
fast**; less on tensors where IQ-quants dominate.

### Buffer reuse via a per-thread pool (small)

Replace the per-candidate `new byte[]` / `new float[]` with rented
buffers from `ArrayPool<T>.Shared`. Eliminates the GC churn for the
F32 round-trip buffer in particular. Estimated **3–8% wall-clock**
on tensors big enough for GC to matter.

### Vectorized MSE accumulation (marginal)

The C# scalar `for (int e = 0; e < source.Length; e++) sum += diff*diff;`
loop is 4.5% of CPU. Rewriting with `Vector<float>` halves it.
Maybe **2% wall-clock total** — last on the priority list.

## What's NOT worth doing

### GPU acceleration of quantization

ggml has no GPU kernel for `ggml_quantize_chunk` — it's CPU-only by
design upstream. K-quant and IQ-quant quantization run a per-block
fitting algorithm (linear regression on the K-family, iterative
codebook search on the I-family) that wasn't designed for SIMT
hardware. Porting it to CUDA / OpenCL is a serious project that
diverges from upstream's intent and exposes us to maintenance debt
on every llama.cpp pin bump.

The dequant half (`to_float`) does exist on GPU because inference
needs it, but that's 1.5% of our CPU time — not the bottleneck.

### Cross-tensor parallelism

At p=8 we already saturate the 8 physical cores per tensor. Adding
"process 2 tensors concurrently × 8 candidates each" would just
hit the same 8-core ceiling at higher memory pressure. Worth
revisiting only if we've already done sub-sampling and cross-
tensor work on tiny tensors becomes the new bottleneck.

### Going past 8 threads

Confirmed harmful on this CPU (above). The default of physical-core
count should stay; users on non-SMT chips can dial up.

## Concrete recommendation

In priority order:
1. **`MaxRowsPerTensor` sub-sampling.** Single biggest user-visible
   win, low complexity. Probably a 3–5× speedup on real models.
2. **Skip F16/BF16 against F16 source.** Few-line change, ~10–18%.
3. **`ArrayPool<T>` for round-trip buffers.** Few-line change, ~5%.
4. (defer) SIMD MSE.
5. (don't) GPU quantize kernels, cross-tensor parallelism, more threads.
