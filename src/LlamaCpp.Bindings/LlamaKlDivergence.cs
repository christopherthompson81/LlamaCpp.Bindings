using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>Per-chunk progress reported during KL-divergence calculation.</summary>
public readonly record struct LlamaKlDivergenceProgress(
    int ChunkIndex,
    int ChunkCount,
    int TokensScored,
    double RunningMeanKl);

/// <summary>
/// Final summary of a reference-vs-test KL divergence run.
/// All KL values are in nats (natural log); divide by <see cref="Math.Log(2)"/>
/// to convert to bits.
/// </summary>
public sealed record LlamaKlDivergenceResult(
    double MeanKl,
    double MaxKl,
    double MedianKl,
    double P90Kl,
    double P99Kl,
    double Top1AgreementRate,
    double Top5AgreementRate,
    double ReferencePerplexity,
    double TestPerplexity,
    int TokensScored,
    int ChunkCount,
    int ContextSize,
    TimeSpan Elapsed);

/// <summary>Knobs for <see cref="LlamaKlDivergence.ComputeAsync"/>.</summary>
public sealed class LlamaKlDivergenceOptions
{
    /// <summary>Per-chunk context length. <c>0</c> picks <c>min(model.TrainingCtx, 512)</c> — the standard PPL/KL recipe.</summary>
    public int ContextSize { get; set; } = 0;

    /// <summary>
    /// Discard the first half of each chunk as warm-up — same recipe
    /// as <see cref="LlamaPerplexity"/>. Default <c>true</c>; matches
    /// llama-perplexity's wikitext methodology, so KL numbers are
    /// directly comparable to upstream's <c>--kl-divergence</c> output
    /// when the corpus and chunk size match.
    /// </summary>
    public bool ScoreSecondHalfOnly { get; set; } = true;

    /// <summary>CPU thread count. <c>-1</c> = llama.cpp default.</summary>
    public int ThreadCount { get; set; } = -1;

    /// <summary>Prepend BOS to the corpus before tokenizing.</summary>
    public bool AddBeginningOfSequence { get; set; } = true;
}

/// <summary>
/// Computes Kullback–Leibler divergence between a reference model
/// (typically the F16/F32 baseline) and a test model (typically a
/// quantized variant) on a shared text corpus. Reports mean / max /
/// percentile KL plus top-1 and top-5 token-agreement rates and the
/// raw perplexity of each model — the canonical "did this quant
/// preserve quality?" comparison.
/// </summary>
/// <remarks>
/// <para>
/// V1 requires both models to fit in RAM/VRAM simultaneously: we
/// decode each chunk on both models, read each model's full softmax
/// distribution at each scored position, and compute KL per token.
/// Memory budget at peak is roughly the sum of both model sizes plus
/// 4 × n_vocab × n_ctx bytes per chunk for staged logits. For
/// Qwen3-0.6B-F16 (~1.2 GB × 2) plus 152 K vocab × 512 ctx × 4 bytes
/// = ~310 MB chunk staging, the total is well under ~3 GB. Larger
/// models will need a "save reference logits to disk, re-run test"
/// flow that we'll add when the use case arises.
/// </para>
/// <para>
/// The two models must share their tokenizer (same vocab size and
/// token strings). The reference model tokenizes the corpus once;
/// both models decode the same token IDs.
/// </para>
/// </remarks>
public static class LlamaKlDivergence
{
    /// <summary>Compute KL divergence of <paramref name="testModel"/> vs <paramref name="referenceModel"/> over <paramref name="corpus"/>.</summary>
    public static Task<LlamaKlDivergenceResult> ComputeAsync(
        LlamaModel referenceModel,
        LlamaModel testModel,
        string corpus,
        LlamaKlDivergenceOptions? options = null,
        IProgress<LlamaKlDivergenceProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(referenceModel);
        ArgumentNullException.ThrowIfNull(testModel);
        ArgumentException.ThrowIfNullOrEmpty(corpus);
        var opts = options ?? new LlamaKlDivergenceOptions();
        return Task.Run(() => Compute(referenceModel, testModel, corpus, opts, progress, cancellationToken),
            cancellationToken);
    }

    private static LlamaKlDivergenceResult Compute(
        LlamaModel reference, LlamaModel test, string corpus,
        LlamaKlDivergenceOptions opts,
        IProgress<LlamaKlDivergenceProgress>? progress,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        int nVocabRef = reference.Vocab.TokenCount;
        int nVocabTest = test.Vocab.TokenCount;
        if (nVocabRef != nVocabTest)
        {
            throw new InvalidOperationException(
                $"Reference and test models have different vocab sizes ({nVocabRef} vs {nVocabTest}). " +
                "KL divergence comparison requires a shared tokenizer.");
        }

        int[] tokens = reference.Vocab.Tokenize(
            corpus,
            addSpecial: opts.AddBeginningOfSequence,
            parseSpecial: false);
        if (tokens.Length < 2)
        {
            throw new InvalidOperationException(
                $"Corpus tokenized to {tokens.Length} token(s); KL needs at least 2.");
        }

        int trainingCtx = Math.Min(reference.TrainingContextSize, test.TrainingContextSize);
        trainingCtx = Math.Max(8, trainingCtx);
        int chunkSize = opts.ContextSize > 0
            ? Math.Min(opts.ContextSize, trainingCtx)
            : Math.Min(512, trainingCtx);
        if (tokens.Length < chunkSize) chunkSize = Math.Max(8, tokens.Length);
        int chunkCount = tokens.Length / chunkSize;
        if (chunkCount == 0)
        {
            throw new InvalidOperationException(
                $"Corpus has {tokens.Length} tokens but chunk size is {chunkSize}; need at least one full chunk.");
        }

        var ctxParams = new LlamaContextParameters
        {
            ContextSize       = (uint)chunkSize,
            LogicalBatchSize  = (uint)chunkSize,
            PhysicalBatchSize = (uint)chunkSize,
            MaxSequenceCount  = 1,
            ThreadCount       = opts.ThreadCount,
            BatchThreadCount  = opts.ThreadCount,
        };
        using var refCtx  = new LlamaContext(reference, ctxParams);
        using var testCtx = new LlamaContext(test, ctxParams);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var klSamples = new List<double>(chunkCount * chunkSize);
        double sumNllRef = 0, sumNllTest = 0;
        long top1Hits = 0, top5Hits = 0;
        int tokensScored = 0;

        var refBatch  = NativeMethods.llama_batch_init(chunkSize, embd: 0, n_seq_max: 1);
        var testBatch = NativeMethods.llama_batch_init(chunkSize, embd: 0, n_seq_max: 1);
        try
        {
            for (int chunk = 0; chunk < chunkCount; chunk++)
            {
                ct.ThrowIfCancellationRequested();
                int start = chunk * chunkSize;

                refCtx.ClearKvCache();
                testCtx.ClearKvCache();
                PopulateBatchAllLogits(ref refBatch,  tokens, start, chunkSize);
                PopulateBatchAllLogits(ref testBatch, tokens, start, chunkSize);

                unsafe
                {
                    var rc = NativeMethods.llama_decode(refCtx.Handle.DangerousHandle, refBatch);
                    if (rc != 0)
                    {
                        throw new LlamaException("llama_decode (reference)", rc,
                            $"reference llama_decode returned {rc} on KL chunk {chunk}/{chunkCount}.");
                    }
                    rc = NativeMethods.llama_decode(testCtx.Handle.DangerousHandle, testBatch);
                    if (rc != 0)
                    {
                        throw new LlamaException("llama_decode (test)", rc,
                            $"test llama_decode returned {rc} on KL chunk {chunk}/{chunkCount}.");
                    }
                }

                int firstScored = opts.ScoreSecondHalfOnly ? chunkSize / 2 : 1;
                for (int i = firstScored; i < chunkSize; i++)
                {
                    int actualToken = tokens[start + i];
                    var stats = ComputePositionStats(refCtx, testCtx, position: i - 1, actualToken, nVocabRef);
                    klSamples.Add(stats.Kl);
                    sumNllRef  += stats.NllRef;
                    sumNllTest += stats.NllTest;
                    if (stats.Top1Hit) top1Hits++;
                    if (stats.Top5Hit) top5Hits++;
                    tokensScored++;
                }

                if (progress is not null)
                {
                    double meanSoFar = klSamples.Count > 0 ? klSamples.Average() : 0;
                    progress.Report(new LlamaKlDivergenceProgress(
                        ChunkIndex: chunk + 1,
                        ChunkCount: chunkCount,
                        TokensScored: tokensScored,
                        RunningMeanKl: meanSoFar));
                }
            }
        }
        finally
        {
            NativeMethods.llama_batch_free(refBatch);
            NativeMethods.llama_batch_free(testBatch);
        }

        sw.Stop();
        klSamples.Sort();
        double mean   = klSamples.Count > 0 ? klSamples.Average() : 0;
        double max    = klSamples.Count > 0 ? klSamples[^1] : 0;
        double median = Percentile(klSamples, 0.5);
        double p90    = Percentile(klSamples, 0.9);
        double p99    = Percentile(klSamples, 0.99);
        double pplRef  = tokensScored > 0 ? Math.Exp(sumNllRef  / tokensScored) : double.NaN;
        double pplTest = tokensScored > 0 ? Math.Exp(sumNllTest / tokensScored) : double.NaN;

        return new LlamaKlDivergenceResult(
            MeanKl:             mean,
            MaxKl:              max,
            MedianKl:           median,
            P90Kl:              p90,
            P99Kl:              p99,
            Top1AgreementRate:  tokensScored > 0 ? (double)top1Hits / tokensScored : 0,
            Top5AgreementRate:  tokensScored > 0 ? (double)top5Hits / tokensScored : 0,
            ReferencePerplexity: pplRef,
            TestPerplexity:      pplTest,
            TokensScored:        tokensScored,
            ChunkCount:          chunkCount,
            ContextSize:         chunkSize,
            Elapsed:             sw.Elapsed);
    }

    /// <summary>
    /// Stats computed at a single scored position: KL contribution,
    /// per-side NLL, and whether the test model's argmax was within
    /// the reference model's top-1 / top-5.
    /// </summary>
    private readonly record struct PositionStats(double Kl, double NllRef, double NllTest, bool Top1Hit, bool Top5Hit);

    private static unsafe PositionStats ComputePositionStats(
        LlamaContext refCtx, LlamaContext testCtx,
        int position, int actualToken, int nVocab)
    {
        float* refLogits  = NativeMethods.llama_get_logits_ith(refCtx.Handle.DangerousHandle,  position);
        float* testLogits = NativeMethods.llama_get_logits_ith(testCtx.Handle.DangerousHandle, position);
        if (refLogits is null || testLogits is null)
        {
            throw new LlamaException("llama_get_logits_ith",
                $"NULL logits at position {position} (ref={refLogits is null}, test={testLogits is null}).");
        }

        // Stable log-sum-exp on each side. logZ = max + log(Σ exp(x - max)).
        double logZRef  = LogSumExp(refLogits,  nVocab);
        double logZTest = LogSumExp(testLogits, nVocab);

        // Walk the vocab once: KL contribution + argmax tracking + top-5 maintenance.
        double kl = 0;
        int argmaxRef  = 0;     float maxRefLogit  = refLogits[0];
        int argmaxTest = 0;     float maxTestLogit = testLogits[0];

        // Top-5 from reference, by logit (since logZ is constant the
        // logit ordering matches the log-prob ordering).
        Span<int>   top5Ids    = stackalloc int[5];
        Span<float> top5Logits = stackalloc float[5];
        int top5N = 0;

        for (int i = 0; i < nVocab; i++)
        {
            double logP = refLogits[i]  - logZRef;
            double logQ = testLogits[i] - logZTest;
            kl += Math.Exp(logP) * (logP - logQ);

            if (refLogits[i]  > maxRefLogit)  { maxRefLogit  = refLogits[i];  argmaxRef  = i; }
            if (testLogits[i] > maxTestLogit) { maxTestLogit = testLogits[i]; argmaxTest = i; }

            // Maintain top-5 by linear insertion. Cheap because the
            // common case is "this logit is below the 5th-largest" and
            // we exit immediately.
            if (top5N < 5)
            {
                int pos = top5N;
                while (pos > 0 && top5Logits[pos - 1] < refLogits[i])
                {
                    top5Logits[pos] = top5Logits[pos - 1];
                    top5Ids[pos]    = top5Ids[pos - 1];
                    pos--;
                }
                top5Logits[pos] = refLogits[i];
                top5Ids[pos]    = i;
                top5N++;
            }
            else if (refLogits[i] > top5Logits[4])
            {
                int pos = 4;
                while (pos > 0 && top5Logits[pos - 1] < refLogits[i])
                {
                    top5Logits[pos] = top5Logits[pos - 1];
                    top5Ids[pos]    = top5Ids[pos - 1];
                    pos--;
                }
                top5Logits[pos] = refLogits[i];
                top5Ids[pos]    = i;
            }
        }

        // Floor KL at 0 — small negative drift from float noise on
        // identical models would be visually confusing in the UI.
        if (kl < 0) kl = 0;

        bool top1Hit = argmaxTest == argmaxRef;
        bool top5Hit = false;
        for (int i = 0; i < 5 && !top5Hit; i++) top5Hit = top5Ids[i] == argmaxTest;

        double nllRef  = -(refLogits[actualToken]  - logZRef);
        double nllTest = -(testLogits[actualToken] - logZTest);

        return new PositionStats(kl, nllRef, nllTest, top1Hit, top5Hit);
    }

    private static unsafe double LogSumExp(float* logits, int n)
    {
        float max = logits[0];
        for (int i = 1; i < n; i++) if (logits[i] > max) max = logits[i];
        double sum = 0;
        for (int i = 0; i < n; i++) sum += Math.Exp(logits[i] - max);
        return max + Math.Log(sum);
    }

    /// <summary>Linear-interpolated percentile on a sorted list.</summary>
    private static double Percentile(IReadOnlyList<double> sortedAsc, double q)
    {
        if (sortedAsc.Count == 0) return 0;
        if (sortedAsc.Count == 1) return sortedAsc[0];
        double pos = q * (sortedAsc.Count - 1);
        int lo = (int)Math.Floor(pos);
        int hi = (int)Math.Ceiling(pos);
        if (lo == hi) return sortedAsc[lo];
        double frac = pos - lo;
        return sortedAsc[lo] * (1 - frac) + sortedAsc[hi] * frac;
    }

    private static unsafe void PopulateBatchAllLogits(
        ref llama_batch batch, int[] tokens, int offset, int count)
    {
        batch.n_tokens = count;
        var tokPtr   = (int*)batch.token;
        var posPtr   = (int*)batch.pos;
        var nSeqPtr  = (int*)batch.n_seq_id;
        var seqIdArr = (int**)batch.seq_id;
        var logits   = (sbyte*)batch.logits;
        for (int i = 0; i < count; i++)
        {
            tokPtr[i]      = tokens[offset + i];
            posPtr[i]      = i;
            nSeqPtr[i]     = 1;
            seqIdArr[i][0] = 0;
            logits[i]      = 1;
        }
    }
}
