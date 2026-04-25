# Prefix-cache reuse divergence — investigation

Tracking [issue #13](https://github.com/christopherthompson81/LlamaCpp.Bindings/issues/13):
`MultiTurnChatTests.Prefix_Reuse_Produces_Same_Output_As_Full_Rebuild` fails at
the current pin (b8893) — pass B's greedy stream diverges from pass A's at
roughly position 15, well after the half-point KV trim.

## Initial hypothesis

The test asserts **byte equality** between two greedy streams that take
different compute paths into the same final KV state:

- **Pass A**: decode `tokens[0..end]` in one `GenerateAsync` call.
- **Pass B**: decode `tokens[0..half-1]` via a throwaway generator, trim via
  `RemoveSequenceRange(0, half, -1)`, then decode `tokens[half..end]` with
  `firstNewIndex=half`.

On CUDA, the logits produced by "process N tokens together" vs
"process M tokens with stored K/V for the first N−M" frequently differ in the
last few ULPs — kernel choice, reduction order, block size are batch-shape
dependent. In principle attention math is invariant; in practice GPU
implementations are not. A single 1-ULP logit difference between two
near-tied candidates is enough to flip an argmax, and greedy generation
amplifies a single flip into every subsequent token.

If that's the story, the test is effectively asserting a property GPUs
don't guarantee. Fix would be either to soften the assertion (non-empty
coherent output, terminates cleanly) or run the test on CPU.

If instead the logits at the first post-trim position are identical but the
streams still diverge later, there's a real binding-level bug (sampler
history not seeded, KV residue after partial removal, etc.) and we need to
keep digging.

## Run 1 — 2026-04-24 18:20

**Command:** `DIAG=1 dotnet test --filter "FullyQualifiedName~PrefixReuseDiagnostic"` on Qwen3-0.6B / 3090.

Three probes against the same 16-token prompt `"The rain in Spain falls mainly on the plain, and the hills are alive."` split at the 8-token midpoint.

| Probe                                            | mismatched logits | max \|Δ\| (abs) | argmax match | stream diverge @ |
|---                                               |---               |---             |---           |---               |
| Baseline (decode full prompt twice, GPU)         | 0 / 151936        | 0.000000e+00    | yes          | never            |
| Split decode, GPU, `OffloadKQV=true`             | 151936 / 151936   | **4.10e-01**    | yes          | **index 3**      |
| Split decode, GPU, `OffloadKQV=false`            | 151936 / 151936   | **4.10e-01**    | yes          | (not captured — logit ≡ KQV=true case) |
| Split decode, CPU-only (`GpuLayerCount=0`)       | 0 / 151936        | 0.000000e+00    | yes          | never            |

**What the data says:**

1. The model is deterministic when the compute path is identical — GPU
   decode of the same prompt twice produces bit-for-bit identical last-
   position logits. So there's no RNG or state-smearing bug at our level.
2. The divergence is triggered specifically by decoding in two batches
   (`[0..half)` then `[half..end)`) instead of one (`[0..end)`) — the
   resulting KV state has the same tokens at the same positions, but the
   *numerics* differ by up to 0.41 logit units, and that's enough to
   flip argmax picks a few steps into greedy generation.
3. Disabling `OffloadKQV` does not change the divergence. The K/V tensor
   storage format isn't the issue; the difference is in the attention
   *compute* path.
4. The exact same split-decode comparison on CPU is bit-identical. This
   is CUDA-specific.

**Conclusion:** not a binding bug. Upstream CUDA kernels implement
"attention over N new tokens with cached K/V for M earlier" via a
different codepath than "attention over N+M tokens in one batch", and the
two codepaths are not bit-equivalent at 0.41-logit scale. For greedy
decode on a 150k-vocab model, that's well above the threshold where
argmax picks diverge within a few steps.

The test is asserting an invariant CUDA cannot honour at the moment.
Upstream would need to either (a) reuse the same attention kernel for both
prefill and "prefill-with-KV" paths, or (b) increase numerical precision on
the latter until tie-breaking behaviour stabilises. Neither is in scope for
this binding.

**Fix direction:**

- Run the byte-equality form of the test on a CPU context, where the
  invariant holds. That keeps the correctness contract for anyone running
  the binding on CPU-only deployments.
- Keep a GPU-path smoke test that exercises `firstNewIndex`-based prefix
  reuse (ensures the code path runs and produces coherent output) but does
  not require byte equality.
- Document in `ChatSession`'s prefix-reuse docs that continuation streams
  on GPU are semantically-correct-but-not-bit-reproducible against a cold
  rebuild.

No upstream issue filed yet — worth checking whether this is a known
trade-off in llama.cpp's CUDA backend before opening one.

