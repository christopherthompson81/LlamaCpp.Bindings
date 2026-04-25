using System.Runtime.InteropServices;
using Xunit;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Diagnostic (not a correctness gate) for issue #13. Captures empirical data
/// about where the prefix-reuse divergence starts — first post-trim position
/// or later — and logs it so the investigation doc has ground truth. This
/// file should stay opt-in (<c>DIAG=1</c>) so it doesn't run in normal CI.
/// </summary>
[Collection(GpuCollection.Name)]
public class PrefixReuseDiagnostic
{
    private readonly GpuGenerationFixture _fx;
    private readonly ITestOutputHelper _log;

    public PrefixReuseDiagnostic(GpuGenerationFixture fx, ITestOutputHelper log)
    {
        _fx = fx;
        _log = log;
    }

    [Fact]
    public void Capture_Logit_Divergence_At_First_Post_Trim_Position()
    {
        if (Environment.GetEnvironmentVariable("DIAG") != "1")
        {
            Assert.Skip("Opt-in diagnostic — run with DIAG=1.");
        }

        var vocab = _fx.Model.Vocab;
        var tokens = vocab.Tokenize(
            "The rain in Spain falls mainly on the plain, and the hills are alive.",
            addSpecial: false, parseSpecial: false);

        int half = tokens.Length / 2;
        int nVocab = vocab.TokenCount;

        // ---- Pass A: full decode, snapshot logits at position (half - 1). ----
        _fx.Context.ClearKvCache();
        DecodeRange(_fx.Context, tokens, 0, tokens.Length);
        var logitsA = CopyLastLogits(_fx.Context, nVocab);
        int argmaxA = ArgMax(logitsA);

        // Pass A already decoded the whole prompt. Snapshot argmax of
        // greedy generation at every position by stepping through the
        // full greedy loop using just the last-position logits we have,
        // then decoding the picked token and repeating. This is cheap
        // because we reuse the same context.
        var streamA = GreedyStream(_fx.Context, nVocab, maxSteps: 16);

        // ---- Pass B: split decode, throwaway first half, trim, re-run. ----
        _fx.Context.ClearKvCache();
        DecodeRange(_fx.Context, tokens, 0, half);
        bool trimmed = _fx.Context.RemoveSequenceRange(0, fromPosition: half, toPosition: -1);
        Assert.True(trimmed, "partial trim must succeed for this diagnostic");

        // Now decode the suffix (tokens[half..end]) on top of the cached prefix.
        DecodeRange(_fx.Context, tokens, half, tokens.Length - half);
        var logitsB = CopyLastLogits(_fx.Context, nVocab);
        int argmaxB = ArgMax(logitsB);

        var streamB = GreedyStream(_fx.Context, nVocab, maxSteps: 16);

        // Compare last-position logits.
        int mismatches = 0;
        double maxAbsDiff = 0;
        int maxAbsDiffIdx = -1;
        float maxAbsDiffA = 0, maxAbsDiffB = 0;
        for (int i = 0; i < nVocab; i++)
        {
            var d = Math.Abs((double)(logitsA[i] - logitsB[i]));
            if (d > 0) mismatches++;
            if (d > maxAbsDiff)
            {
                maxAbsDiff = d;
                maxAbsDiffIdx = i;
                maxAbsDiffA = logitsA[i];
                maxAbsDiffB = logitsB[i];
            }
        }

        _log.WriteLine($"[PrefixReuse] prompt={tokens.Length} tokens, half={half}, nVocab={nVocab}");
        _log.WriteLine($"[PrefixReuse] last-position logits — mismatches: {mismatches}/{nVocab}");
        _log.WriteLine($"[PrefixReuse] last-position max |A-B|: {maxAbsDiff:E6} at token {maxAbsDiffIdx} (A={maxAbsDiffA:E6}, B={maxAbsDiffB:E6})");
        _log.WriteLine($"[PrefixReuse] argmax A = {argmaxA} ('{vocab.TokenToPiece(argmaxA)}'), B = {argmaxB} ('{vocab.TokenToPiece(argmaxB)}'), {(argmaxA == argmaxB ? "match" : "DIVERGE")}");

        _log.WriteLine($"[PrefixReuse] greedy stream A (first 16): [{string.Join(", ", streamA)}]");
        _log.WriteLine($"[PrefixReuse] greedy stream B (first 16): [{string.Join(", ", streamB)}]");

        // Find the first divergent index in the streams.
        int divergeAt = -1;
        for (int i = 0; i < Math.Min(streamA.Count, streamB.Count); i++)
        {
            if (streamA[i] != streamB[i]) { divergeAt = i; break; }
        }
        _log.WriteLine($"[PrefixReuse] greedy streams diverge at index: {(divergeAt < 0 ? "never" : divergeAt.ToString())}");
    }

    [Fact]
    public void Gpu_With_NoKv_Offload_Split_Decode_Divergence()
    {
        // Same split-decode comparison with GPU compute but K/V held off the
        // GPU. If divergence shrinks substantially here, the source is the
        // CUDA KV cache's storage format (e.g. lower-precision K/V, or
        // strided-tile reads). If it's unchanged, the kernel-path difference
        // is in the attention compute itself, not the cache.
        if (Environment.GetEnvironmentVariable("DIAG") != "1")
        {
            Assert.Skip("Opt-in diagnostic — run with DIAG=1.");
        }

        var path = TestModelProvider.EnsureModelPath();
        using var model = new LlamaModel(path, new LlamaModelParameters
        {
            GpuLayerCount = -1, // full GPU
            UseMmap = true,
        });
        using var ctx = new LlamaContext(model, new LlamaContextParameters
        {
            ContextSize = 512,
            LogicalBatchSize = 512,
            PhysicalBatchSize = 512,
            MaxSequenceCount = 1,
            OffloadKQV = false,
        });

        var vocab = model.Vocab;
        var tokens = vocab.Tokenize(
            "The rain in Spain falls mainly on the plain, and the hills are alive.",
            addSpecial: false, parseSpecial: false);
        int nVocab = vocab.TokenCount;
        int half = tokens.Length / 2;

        ctx.ClearKvCache();
        DecodeRange(ctx, tokens, 0, tokens.Length);
        var logitsA = CopyLastLogits(ctx, nVocab);
        int argmaxA = ArgMax(logitsA);

        ctx.ClearKvCache();
        DecodeRange(ctx, tokens, 0, half);
        DecodeRange(ctx, tokens, half, tokens.Length - half);
        var logitsB = CopyLastLogits(ctx, nVocab);
        int argmaxB = ArgMax(logitsB);

        int mismatches = 0;
        double maxAbs = 0;
        for (int i = 0; i < nVocab; i++)
        {
            var d = Math.Abs((double)(logitsA[i] - logitsB[i]));
            if (d > 0) mismatches++;
            if (d > maxAbs) maxAbs = d;
        }

        _log.WriteLine($"[GpuNoKvOffload] GpuLayerCount=-1, OffloadKQV=false");
        _log.WriteLine($"[GpuNoKvOffload] mismatches: {mismatches}/{nVocab}, max |A-B|: {maxAbs:E6}");
        _log.WriteLine($"[GpuNoKvOffload] argmax A={argmaxA}, B={argmaxB}, {(argmaxA == argmaxB ? "match" : "DIVERGE")}");
    }

    [Fact]
    public void Cpu_Only_Split_Decode_Divergence()
    {
        // Re-run the split-decode vs full-decode comparison with GPU offload
        // disabled. CUDA kernels for batched attention vs "attention over new
        // tokens with cached K/V" are different codepaths, and either can be
        // the source of the 4e-1 divergence observed on GPU. Running on CPU
        // uses a single kernel family; if it still diverges there's an
        // upstream bug; if it matches then the issue is CUDA-specific.
        if (Environment.GetEnvironmentVariable("DIAG") != "1")
        {
            Assert.Skip("Opt-in diagnostic — run with DIAG=1.");
        }

        // Separate model/context so we don't disturb the shared GPU fixture.
        var path = TestModelProvider.EnsureModelPath();
        using var model = new LlamaModel(path, new LlamaModelParameters
        {
            GpuLayerCount = 0, // force CPU
            UseMmap = true,
        });
        using var ctx = new LlamaContext(model, new LlamaContextParameters
        {
            ContextSize = 512,
            LogicalBatchSize = 512,
            PhysicalBatchSize = 512,
            MaxSequenceCount = 1,
            OffloadKQV = false,
        });

        var vocab = model.Vocab;
        var tokens = vocab.Tokenize(
            "The rain in Spain falls mainly on the plain, and the hills are alive.",
            addSpecial: false, parseSpecial: false);
        int nVocab = vocab.TokenCount;
        int half = tokens.Length / 2;

        // Pass A: single decode.
        ctx.ClearKvCache();
        DecodeRange(ctx, tokens, 0, tokens.Length);
        var logitsA = CopyLastLogits(ctx, nVocab);
        int argmaxA = ArgMax(logitsA);

        // Pass B: split decode.
        ctx.ClearKvCache();
        DecodeRange(ctx, tokens, 0, half);
        DecodeRange(ctx, tokens, half, tokens.Length - half);
        var logitsB = CopyLastLogits(ctx, nVocab);
        int argmaxB = ArgMax(logitsB);

        int mismatches = 0;
        double maxAbs = 0;
        for (int i = 0; i < nVocab; i++)
        {
            var d = Math.Abs((double)(logitsA[i] - logitsB[i]));
            if (d > 0) mismatches++;
            if (d > maxAbs) maxAbs = d;
        }

        _log.WriteLine($"[CpuSplit] GpuLayerCount=0, OffloadKQV=false");
        _log.WriteLine($"[CpuSplit] mismatches: {mismatches}/{nVocab}, max |A-B|: {maxAbs:E6}");
        _log.WriteLine($"[CpuSplit] argmax A={argmaxA}, B={argmaxB}, {(argmaxA == argmaxB ? "match" : "DIVERGE")}");
    }

    [Fact]
    public void Baseline_Decode_Twice_Should_Be_Identical()
    {
        // Sanity gate: the MODEL itself must be deterministic when we decode
        // the same token sequence twice from a fresh KV. If even this diverges
        // then the backend is non-deterministic at the kernel level and the
        // prefix-reuse test's byte-equality assertion is impossible to satisfy
        // — period. If it passes, the divergence observed in the
        // first-post-trim diagnostic is specifically a batch-split artifact
        // (different compute path for the identical final KV state).
        if (Environment.GetEnvironmentVariable("DIAG") != "1")
        {
            Assert.Skip("Opt-in diagnostic — run with DIAG=1.");
        }

        var vocab = _fx.Model.Vocab;
        var tokens = vocab.Tokenize(
            "The rain in Spain falls mainly on the plain, and the hills are alive.",
            addSpecial: false, parseSpecial: false);
        int nVocab = vocab.TokenCount;

        _fx.Context.ClearKvCache();
        DecodeRange(_fx.Context, tokens, 0, tokens.Length);
        var logits1 = CopyLastLogits(_fx.Context, nVocab);

        _fx.Context.ClearKvCache();
        DecodeRange(_fx.Context, tokens, 0, tokens.Length);
        var logits2 = CopyLastLogits(_fx.Context, nVocab);

        int mismatches = 0;
        double maxAbs = 0;
        for (int i = 0; i < nVocab; i++)
        {
            var d = Math.Abs((double)(logits1[i] - logits2[i]));
            if (d > 0) mismatches++;
            if (d > maxAbs) maxAbs = d;
        }

        _log.WriteLine($"[Baseline] same-prompt-twice — mismatches: {mismatches}/{nVocab}, max |diff|: {maxAbs:E6}");
    }

    // ----- helpers -----

    private static unsafe void DecodeRange(LlamaContext ctx, int[] tokens, int start, int count)
    {
        if (count <= 0) return;
        int cap = Math.Max(1, ctx.LogicalBatchSize);
        int offset = start;
        int end = start + count;
        while (offset < end)
        {
            int take = Math.Min(cap, end - offset);
            fixed (int* p = &tokens[offset])
            {
                var batch = NativeMethods.llama_batch_get_one(p, take);
                var rc = NativeMethods.llama_decode(ctx.Handle.DangerousHandle, batch);
                if (rc != 0) throw new InvalidOperationException($"llama_decode={rc}");
            }
            offset += take;
        }
    }

    private static unsafe float[] CopyLastLogits(LlamaContext ctx, int nVocab)
    {
        float* ptr = NativeMethods.llama_get_logits_ith(ctx.Handle.DangerousHandle, -1);
        if (ptr is null) throw new InvalidOperationException("llama_get_logits_ith(-1) returned NULL");
        var copy = new float[nVocab];
        new Span<float>(ptr, nVocab).CopyTo(copy);
        return copy;
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        float bestVal = v[0];
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > bestVal) { bestVal = v[i]; best = i; }
        }
        return best;
    }

    private static unsafe List<int> GreedyStream(LlamaContext ctx, int nVocab, int maxSteps)
    {
        // Assumes logits are already available at position -1 from the caller's
        // most recent decode. Greedy-samples maxSteps tokens, decoding each
        // back into the context to advance state.
        var result = new List<int>(maxSteps);
        for (int step = 0; step < maxSteps; step++)
        {
            var logits = CopyLastLogits(ctx, nVocab);
            int tok = ArgMax(logits);
            result.Add(tok);

            var arr = new[] { tok };
            fixed (int* p = arr)
            {
                var batch = NativeMethods.llama_batch_get_one(p, 1);
                var rc = NativeMethods.llama_decode(ctx.Handle.DangerousHandle, batch);
                if (rc != 0) throw new InvalidOperationException($"llama_decode={rc} at step {step}");
            }
        }
        return result;
    }
}
