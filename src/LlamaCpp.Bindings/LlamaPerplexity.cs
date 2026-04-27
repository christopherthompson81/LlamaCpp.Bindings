using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// Per-chunk progress reported by <see cref="LlamaPerplexity.ComputeAsync"/>
/// while the calculation is in flight. <c>RunningPerplexity</c> is the
/// perplexity computed over chunks completed so far.
/// </summary>
public readonly record struct LlamaPerplexityProgress(
    int ChunkIndex,
    int ChunkCount,
    int TokensScored,
    double RunningPerplexity);

/// <summary>
/// Final result from <see cref="LlamaPerplexity.ComputeAsync"/>.
/// </summary>
public sealed record LlamaPerplexityResult(
    double Perplexity,
    double NegativeLogLikelihood,
    int TokensScored,
    int ChunkCount,
    int ContextSize,
    TimeSpan Elapsed);

/// <summary>
/// Knobs for <see cref="LlamaPerplexity.ComputeAsync"/>.
/// </summary>
public sealed class LlamaPerplexityOptions
{
    /// <summary>
    /// Per-chunk context length in tokens. <c>0</c> (default) lets the
    /// driver pick: <c>min(model.TrainingContextSize, 2048)</c>, capped at
    /// 512 for tiny models.
    /// </summary>
    public int ContextSize { get; set; } = 0;

    /// <summary>
    /// Discard the first half of each chunk as warm-up — only score tokens
    /// in the second half. This matches <c>llama-perplexity</c>'s wikitext
    /// methodology, so results are comparable against published numbers.
    /// Set <c>false</c> to score every position with a non-empty prefix
    /// (~2× the tokens scored, but with degraded prefix length on early
    /// positions — not directly comparable to llama-perplexity output).
    /// </summary>
    public bool ScoreSecondHalfOnly { get; set; } = true;

    /// <summary>
    /// CPU thread count for context creation. <c>-1</c> (default) inherits
    /// llama.cpp's default. GPU offloading is controlled separately on the
    /// model.
    /// </summary>
    public int ThreadCount { get; set; } = -1;

    /// <summary>
    /// Prepend the BOS token to the first chunk. Default <c>true</c>;
    /// matches <c>llama-perplexity</c>. Has no effect on models whose
    /// vocab has no BOS token.
    /// </summary>
    public bool AddBeginningOfSequence { get; set; } = true;
}

/// <summary>
/// Computes corpus perplexity by sliding a context-sized window across the
/// tokenized text, decoding each chunk with logits at every position, and
/// summing the negative log-likelihood of the actual next token at each
/// position. Returns <c>exp(mean NLL)</c>.
/// </summary>
/// <remarks>
/// <para>
/// The default methodology matches <c>llama-perplexity</c>'s wikitext
/// recipe: chunks of <c>n_ctx</c> tokens with no overlap, score only the
/// second half of each chunk (the first half acts as warm-up context).
/// Results are directly comparable to published perplexity numbers from
/// the upstream tool when the corpus and tokenizer match.
/// </para>
/// <para>
/// All computation is in pure C# on top of the existing decode bindings —
/// no external CLI, no Python.
/// </para>
/// </remarks>
public static class LlamaPerplexity
{
    /// <summary>
    /// Compute perplexity over <paramref name="corpus"/> using
    /// <paramref name="model"/>. The driver creates and disposes its own
    /// <see cref="LlamaContext"/> internally.
    /// </summary>
    /// <param name="model">Loaded model. The caller retains ownership.</param>
    /// <param name="corpus">UTF-8 text to score.</param>
    /// <param name="options">Knobs; pass <c>null</c> for defaults.</param>
    /// <param name="progress">Optional per-chunk progress sink.</param>
    /// <param name="cancellationToken">Honored between chunks.</param>
    public static Task<LlamaPerplexityResult> ComputeAsync(
        LlamaModel model,
        string corpus,
        LlamaPerplexityOptions? options = null,
        IProgress<LlamaPerplexityProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrEmpty(corpus);
        var opts = options ?? new LlamaPerplexityOptions();

        return Task.Run(() => Compute(model, corpus, opts, progress, cancellationToken), cancellationToken);
    }

    private static LlamaPerplexityResult Compute(
        LlamaModel model,
        string corpus,
        LlamaPerplexityOptions options,
        IProgress<LlamaPerplexityProgress>? progress,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // 1. Tokenize the entire corpus. The vocab's Tokenize() handles BOS
        //    when addSpecial=true; that matches the llama-perplexity default.
        int[] tokens = model.Vocab.Tokenize(
            corpus,
            addSpecial: options.AddBeginningOfSequence,
            parseSpecial: false);
        if (tokens.Length < 2)
        {
            throw new InvalidOperationException(
                $"Corpus tokenized to {tokens.Length} token(s); need at least 2 to score.");
        }

        // 2. Decide chunk size. Cap at the model's training context but
        //    don't exceed 2048 by default — perplexity at 4k+ is
        //    proportionally more expensive without changing the metric.
        int trainingCtx = Math.Max(8, model.TrainingContextSize);
        int chunkSize = options.ContextSize > 0
            ? Math.Min(options.ContextSize, trainingCtx)
            : Math.Min(2048, trainingCtx);
        if (chunkSize < 8)
        {
            throw new InvalidOperationException(
                $"Chunk size {chunkSize} is too small (need >= 8); model trainingCtx={trainingCtx}.");
        }
        if (tokens.Length < chunkSize)
        {
            // Round chunk size down to the nearest power-of-two-ish value
            // that fits, with a floor of 8. Keeps the second-half-only
            // recipe meaningful even on short inputs.
            chunkSize = Math.Max(8, tokens.Length);
        }

        int chunkCount = tokens.Length / chunkSize;
        if (chunkCount == 0)
        {
            throw new InvalidOperationException(
                $"Tokenized corpus ({tokens.Length} tokens) is shorter than chunk size {chunkSize}; " +
                "shorten ContextSize or supply more text.");
        }

        // 3. Spin up a context sized to chunkSize. Batch sizes must be
        //    >= chunkSize so the entire chunk decodes in one llama_decode.
        var ctxParams = new LlamaContextParameters
        {
            ContextSize       = (uint)chunkSize,
            LogicalBatchSize  = (uint)chunkSize,
            PhysicalBatchSize = (uint)chunkSize,
            MaxSequenceCount  = 1,
            ThreadCount       = options.ThreadCount,
            BatchThreadCount  = options.ThreadCount,
        };

        using var context = new LlamaContext(model, ctxParams);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        double nllSum = 0;
        int tokensScored = 0;
        int nVocab = model.Vocab.TokenCount;

        // 4. Decode each chunk and accumulate per-position NLLs.
        var batch = NativeMethods.llama_batch_init(chunkSize, embd: 0, n_seq_max: 1);
        try
        {
            for (int chunk = 0; chunk < chunkCount; chunk++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int start = chunk * chunkSize;
                context.ClearKvCache();

                PopulateBatchAllLogits(ref batch, tokens, start, chunkSize);
                unsafe
                {
                    var rc = NativeMethods.llama_decode(context.Handle.DangerousHandle, batch);
                    if (rc != 0)
                    {
                        throw new LlamaException(
                            "llama_decode", rc,
                            $"llama_decode returned {rc} on perplexity chunk {chunk}/{chunkCount} " +
                            $"(chunkSize={chunkSize}).");
                    }
                }

                // Parallelize the per-position softmax loop: each token's
                // log-prob is independent, and the softmax cost
                // (full-vocab log-sum-exp) dominates the per-chunk wall
                // time. Empirically this is the actual bottleneck —
                // GPU forward is fast, the C# softmax is what takes 90%
                // of the chunk's time. Aggregating into thread-local
                // accumulators keeps the hot loop lock-free.
                int firstScored = options.ScoreSecondHalfOnly ? chunkSize / 2 : 1;
                var (chunkNllSum, chunkScored) = ParallelChunkScore(
                    context, tokens, start, firstScored, chunkSize, nVocab,
                    options.ThreadCount);
                nllSum += chunkNllSum;
                tokensScored += chunkScored;

                if (progress is not null)
                {
                    double running = tokensScored > 0 ? Math.Exp(nllSum / tokensScored) : double.NaN;
                    progress.Report(new LlamaPerplexityProgress(
                        ChunkIndex: chunk + 1,
                        ChunkCount: chunkCount,
                        TokensScored: tokensScored,
                        RunningPerplexity: running));
                }
            }
        }
        finally
        {
            NativeMethods.llama_batch_free(batch);
        }

        sw.Stop();
        double meanNll = tokensScored > 0 ? nllSum / tokensScored : double.NaN;
        double ppl = Math.Exp(meanNll);

        return new LlamaPerplexityResult(
            Perplexity: ppl,
            NegativeLogLikelihood: meanNll,
            TokensScored: tokensScored,
            ChunkCount: chunkCount,
            ContextSize: chunkSize,
            Elapsed: sw.Elapsed);
    }

    /// <summary>
    /// Batched variant: process <paramref name="batchSequences"/> chunks
    /// concurrently within a single <c>llama_decode</c> call. Each chunk
    /// gets its own sequence id so KV caches are independent, but they
    /// share the GPU forward pass — giving ~2–3× wall-clock speedup
    /// because the per-chunk batch=1 work was leaving the GPU
    /// underutilized.
    /// </summary>
    /// <remarks>
    /// Numerical equivalence with <see cref="ComputeAsync"/> holds when
    /// the same <c>options</c> are passed and the corpus tokenizes the
    /// same way. The chunk loop is just N at a time instead of one.
    /// </remarks>
    public static Task<LlamaPerplexityResult> ComputeBatchedAsync(
        LlamaModel model,
        string corpus,
        LlamaPerplexityOptions? options = null,
        int batchSequences = 4,
        IProgress<LlamaPerplexityProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrEmpty(corpus);
        if (batchSequences < 1)
            throw new ArgumentOutOfRangeException(nameof(batchSequences), "Must be >= 1.");
        var opts = options ?? new LlamaPerplexityOptions();

        return Task.Run(() => ComputeBatched(model, corpus, opts, batchSequences, progress, cancellationToken), cancellationToken);
    }

    private static LlamaPerplexityResult ComputeBatched(
        LlamaModel model, string corpus, LlamaPerplexityOptions options,
        int batchSequences,
        IProgress<LlamaPerplexityProgress>? progress,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        int[] tokens = model.Vocab.Tokenize(
            corpus,
            addSpecial: options.AddBeginningOfSequence,
            parseSpecial: false);
        if (tokens.Length < 2)
        {
            throw new InvalidOperationException(
                $"Corpus tokenized to {tokens.Length} token(s); need at least 2 to score.");
        }

        int trainingCtx = Math.Max(8, model.TrainingContextSize);
        int chunkSize = options.ContextSize > 0
            ? Math.Min(options.ContextSize, trainingCtx)
            : Math.Min(2048, trainingCtx);
        if (chunkSize < 8)
        {
            throw new InvalidOperationException(
                $"Chunk size {chunkSize} is too small (need >= 8); model trainingCtx={trainingCtx}.");
        }
        if (tokens.Length < chunkSize) chunkSize = Math.Max(8, tokens.Length);

        int chunkCount = tokens.Length / chunkSize;
        if (chunkCount == 0)
        {
            throw new InvalidOperationException(
                $"Tokenized corpus ({tokens.Length} tokens) is shorter than chunk size {chunkSize}; " +
                "shorten ContextSize or supply more text.");
        }

        // Cap parallelism at chunk count and at the model's training
        // context size (a B-sequence batch needs chunkSize * B context
        // positions in total).
        int B = Math.Max(1, Math.Min(batchSequences, chunkCount));
        int totalCtx = chunkSize * B;
        if (totalCtx > trainingCtx)
        {
            // Reduce B until the total fits the training context.
            B = Math.Max(1, trainingCtx / chunkSize);
            totalCtx = chunkSize * B;
        }

        var ctxParams = new LlamaContextParameters
        {
            ContextSize       = (uint)totalCtx,
            LogicalBatchSize  = (uint)totalCtx,
            PhysicalBatchSize = (uint)totalCtx,
            MaxSequenceCount  = (uint)B,
            ThreadCount       = options.ThreadCount,
            BatchThreadCount  = options.ThreadCount,
        };
        using var context = new LlamaContext(model, ctxParams);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        double nllSum = 0;
        int tokensScored = 0;
        int nVocab = model.Vocab.TokenCount;
        int rounds = (chunkCount + B - 1) / B;

        var batch = NativeMethods.llama_batch_init(totalCtx, embd: 0, n_seq_max: B);
        try
        {
            for (int round = 0; round < rounds; round++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int firstChunk = round * B;
                int chunksInRound = Math.Min(B, chunkCount - firstChunk);

                context.ClearKvCache();
                PopulateMultiSeqBatch(ref batch, tokens, firstChunk, chunksInRound, chunkSize);

                unsafe
                {
                    var rc = NativeMethods.llama_decode(context.Handle.DangerousHandle, batch);
                    if (rc != 0)
                    {
                        throw new LlamaException(
                            "llama_decode", rc,
                            $"llama_decode returned {rc} on perplexity round {round}/{rounds} " +
                            $"(B={B}, chunkSize={chunkSize}).");
                    }
                }

                // Same parallel softmax as the sequential path, but the
                // index space is (chunksInRound × scored-positions)
                // flattened into one Parallel.For range.
                int firstScored = options.ScoreSecondHalfOnly ? chunkSize / 2 : 1;
                var (roundNllSum, roundScored) = ParallelRoundScore(
                    context, tokens, firstChunk, chunksInRound, chunkSize,
                    firstScored, nVocab, options.ThreadCount);
                nllSum += roundNllSum;
                tokensScored += roundScored;

                if (progress is not null)
                {
                    int chunksDone = firstChunk + chunksInRound;
                    double running = tokensScored > 0 ? Math.Exp(nllSum / tokensScored) : double.NaN;
                    progress.Report(new LlamaPerplexityProgress(
                        ChunkIndex: chunksDone,
                        ChunkCount: chunkCount,
                        TokensScored: tokensScored,
                        RunningPerplexity: running));
                }
            }
        }
        finally
        {
            NativeMethods.llama_batch_free(batch);
        }

        sw.Stop();
        double meanNll = tokensScored > 0 ? nllSum / tokensScored : double.NaN;
        double ppl = Math.Exp(meanNll);

        return new LlamaPerplexityResult(
            Perplexity: ppl,
            NegativeLogLikelihood: meanNll,
            TokensScored: tokensScored,
            ChunkCount: chunkCount,
            ContextSize: chunkSize,
            Elapsed: sw.Elapsed);
    }

    /// <summary>
    /// Pack <paramref name="chunksInRound"/> consecutive chunks into a
    /// single multi-sequence batch. Sequence i runs over batch positions
    /// [i*chunkSize .. (i+1)*chunkSize), with <c>pos</c> reset to 0 within
    /// each sequence and <c>seq_id</c> = i. Logits enabled at every
    /// position (perplexity scores every token).
    /// </summary>
    private static unsafe void PopulateMultiSeqBatch(
        ref llama_batch batch, int[] tokens,
        int firstChunk, int chunksInRound, int chunkSize)
    {
        int totalTokens = chunksInRound * chunkSize;
        batch.n_tokens = totalTokens;
        var tokPtr   = (int*)batch.token;
        var posPtr   = (int*)batch.pos;
        var nSeqPtr  = (int*)batch.n_seq_id;
        var seqIdArr = (int**)batch.seq_id;
        var logits   = (sbyte*)batch.logits;

        for (int seq = 0; seq < chunksInRound; seq++)
        {
            int chunkStart = (firstChunk + seq) * chunkSize;
            int batchOffset = seq * chunkSize;
            for (int i = 0; i < chunkSize; i++)
            {
                int b = batchOffset + i;
                tokPtr[b]      = tokens[chunkStart + i];
                posPtr[b]      = i;       // position within the sequence
                nSeqPtr[b]     = 1;
                seqIdArr[b][0] = seq;     // sequence index 0..chunksInRound-1
                logits[b]      = 1;
            }
        }
    }

    /// <summary>
    /// One unit of work for <see cref="ParallelPerplexityRunner"/>.
    /// </summary>
    public sealed record PerplexityJob(
        string ModelPath,
        string Corpus,
        LlamaModelParameters? ModelParameters = null,
        LlamaPerplexityOptions? Options = null,
        int Batch = 1,
        object? Tag = null);

    /// <summary>
    /// Result of a single <see cref="PerplexityJob"/>. <see cref="Tag"/>
    /// echoes the job's tag so callers can correlate without keeping an
    /// auxiliary dictionary.
    /// </summary>
    public sealed record PerplexityJobResult(
        string ModelPath,
        LlamaPerplexityResult Result,
        object? Tag);

    /// <summary>
    /// Cross-job parallelism: load up to <c>maxConcurrent</c> models into
    /// VRAM at once and run their PPL passes concurrently. The expensive
    /// softmax loop is already CPU-parallelized inside each
    /// <see cref="ComputeAsync"/>; running multiple jobs at once spreads
    /// model loading + GPU forward across overlap with other jobs'
    /// softmax phases.
    /// </summary>
    /// <remarks>
    /// Each job gets its own <see cref="LlamaModel"/> and
    /// <see cref="LlamaContext"/>; they share the GPU's CUDA context but
    /// have independent KV caches. The semaphore gates concurrent VRAM
    /// occupancy — set it conservatively for big models.
    ///
    /// To prevent over-subscribing the CPU, the per-job
    /// <c>ThreadCount</c> defaults to <c>ProcessorCount / maxConcurrent</c>
    /// when not set explicitly on the job's options. Override by setting
    /// <see cref="LlamaPerplexityOptions.ThreadCount"/> on the job.
    /// </remarks>
    public static async IAsyncEnumerable<PerplexityJobResult> RunParallelAsync(
        IEnumerable<PerplexityJob> jobs,
        int maxConcurrent = 4,
        IProgress<LlamaPerplexityProgress>? sharedProgress = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation]
        CancellationToken cancellationToken = default)
    {
        if (maxConcurrent < 1)
            throw new ArgumentOutOfRangeException(nameof(maxConcurrent), "Must be >= 1.");

        using var slot = new SemaphoreSlim(maxConcurrent, maxConcurrent);
        // Split the full logical-core budget across concurrent jobs.
        // Softmax scales nearly to all 16 logical cores on this rig
        // (see ResolveSoftmaxDop note), so 4 concurrent × 4 threads
        // each gives full CPU utilization without thrashing the
        // ThreadPool. Caller can override via per-job ThreadCount.
        int autoThreadsPerJob = Math.Max(1, Environment.ProcessorCount / maxConcurrent);

        var inFlight = new List<Task<PerplexityJobResult>>();
        foreach (var job in jobs)
        {
            await slot.WaitAsync(cancellationToken).ConfigureAwait(false);
            var captured = job;  // closure capture
            inFlight.Add(Task.Run<PerplexityJobResult>(async () =>
            {
                try
                {
                    var modelOpts = captured.ModelParameters ?? new LlamaModelParameters
                    {
                        UseMmap       = true,
                        GpuLayerCount = -1,
                    };
                    var pplOpts = captured.Options ?? new LlamaPerplexityOptions();
                    if (pplOpts.ThreadCount <= 0)
                    {
                        pplOpts = new LlamaPerplexityOptions
                        {
                            ContextSize         = pplOpts.ContextSize,
                            ScoreSecondHalfOnly = pplOpts.ScoreSecondHalfOnly,
                            AddBeginningOfSequence = pplOpts.AddBeginningOfSequence,
                            ThreadCount         = autoThreadsPerJob,
                        };
                    }

                    using var model = new LlamaModel(captured.ModelPath, modelOpts);
                    var result = captured.Batch > 1
                        ? await ComputeBatchedAsync(model, captured.Corpus, pplOpts, captured.Batch, sharedProgress, cancellationToken).ConfigureAwait(false)
                        : await ComputeAsync       (model, captured.Corpus, pplOpts, sharedProgress, cancellationToken).ConfigureAwait(false);
                    return new PerplexityJobResult(captured.ModelPath, result, captured.Tag);
                }
                finally
                {
                    slot.Release();
                }
            }, cancellationToken));
        }

        while (inFlight.Count > 0)
        {
            var done = await Task.WhenAny(inFlight).ConfigureAwait(false);
            inFlight.Remove(done);
            yield return await done.ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Score one chunk's positions in parallel. Returns the NLL sum and
    /// scored-token count; aggregation across threads is lock-free via
    /// thread-local accumulators.
    /// </summary>
    private static (double sum, int count) ParallelChunkScore(
        LlamaContext context, int[] tokens, int start,
        int firstScored, int chunkSize, int nVocab, int threadCount)
    {
        int dop = ResolveSoftmaxDop(threadCount);
        int totalScored = chunkSize - firstScored;
        if (totalScored <= 0) return (0.0, 0);

        double localSumAccum = 0;
        int localCountAccum = 0;
        var lockObj = new object();

        Parallel.For<(double sum, int count)>(
            firstScored, chunkSize,
            new ParallelOptions { MaxDegreeOfParallelism = dop },
            () => (0.0, 0),
            (i, _, local) =>
            {
                int actual = tokens[start + i];
                double nll = NegativeLogProbability(context, position: i - 1, actualToken: actual, nVocab);
                return (local.sum + nll, local.count + 1);
            },
            local =>
            {
                lock (lockObj)
                {
                    localSumAccum   += local.sum;
                    localCountAccum += local.count;
                }
            });

        return (localSumAccum, localCountAccum);
    }

    /// <summary>
    /// Score a multi-sequence round in parallel. The (sequence, position)
    /// index space is flattened so the same lock-free aggregation works.
    /// </summary>
    private static (double sum, int count) ParallelRoundScore(
        LlamaContext context, int[] tokens,
        int firstChunk, int chunksInRound, int chunkSize,
        int firstScored, int nVocab, int threadCount)
    {
        int dop = ResolveSoftmaxDop(threadCount);
        int scoredPerSeq = chunkSize - firstScored;
        int totalScored = chunksInRound * scoredPerSeq;
        if (totalScored <= 0) return (0.0, 0);

        double localSumAccum = 0;
        int localCountAccum = 0;
        var lockObj = new object();

        Parallel.For<(double sum, int count)>(
            0, totalScored,
            new ParallelOptions { MaxDegreeOfParallelism = dop },
            () => (0.0, 0),
            (idx, _, local) =>
            {
                int seq      = idx / scoredPerSeq;
                int chunkPos = firstScored + (idx % scoredPerSeq);
                int chunkStart  = (firstChunk + seq) * chunkSize;
                int batchOffset = seq * chunkSize;
                int actual = tokens[chunkStart + chunkPos];
                double nll = NegativeLogProbability(
                    context, position: batchOffset + chunkPos - 1, actualToken: actual, nVocab);
                return (local.sum + nll, local.count + 1);
            },
            local =>
            {
                lock (lockObj)
                {
                    localSumAccum   += local.sum;
                    localCountAccum += local.count;
                }
            });

        return (localSumAccum, localCountAccum);
    }

    /// <summary>
    /// Pick the softmax DoP. <c>threadCount</c> ≤ 0 means "auto" — full
    /// logical-core count, which empirically wins on this workload.
    /// Unlike AVX-heavy K-quant kernels (where SMT siblings contend on
    /// the SIMD unit and physical-core count is best), the softmax loop
    /// is Math.Exp + scalar reduction; its hot loop uses different
    /// execution ports than matmul, so HT siblings interleave instead
    /// of contending. Sweep on i7-10700K (8C/16T): 8→16 threads gives
    /// another ~12 % speedup. Positive value is honored verbatim —
    /// useful when the caller is running multiple Tasks concurrently
    /// and wants each Task to stay within a thread budget.
    /// </summary>
    private static int ResolveSoftmaxDop(int threadCount) =>
        threadCount > 0 ? threadCount : Math.Max(1, Environment.ProcessorCount);

    /// <summary>
    /// Compute -log P(actualToken | prefix) using a numerically stable
    /// log-sum-exp on the logits at <paramref name="position"/>.
    /// </summary>
    private static unsafe double NegativeLogProbability(
        LlamaContext context, int position, int actualToken, int nVocab)
    {
        float* logits = NativeMethods.llama_get_logits_ith(
            context.Handle.DangerousHandle, position);
        if (logits is null)
        {
            throw new LlamaException(
                "llama_get_logits_ith",
                $"llama_get_logits_ith returned NULL at position {position}; " +
                $"the batch must enable logits at every position for perplexity.");
        }
        if ((uint)actualToken >= (uint)nVocab)
        {
            // Out-of-vocab next token shouldn't happen on a self-tokenized
            // corpus, but guard so we throw a clear exception rather than
            // index past the logits buffer.
            throw new InvalidOperationException(
                $"Token id {actualToken} out of range for vocab size {nVocab}.");
        }

        // Stable log-sum-exp.
        float maxLogit = logits[0];
        for (int i = 1; i < nVocab; i++)
        {
            if (logits[i] > maxLogit) maxLogit = logits[i];
        }
        double sum = 0;
        for (int i = 0; i < nVocab; i++)
        {
            sum += Math.Exp(logits[i] - maxLogit);
        }
        double logZ = maxLogit + Math.Log(sum);
        double logProb = logits[actualToken] - logZ;
        return -logProb;
    }

    /// <summary>
    /// Fill <paramref name="batch"/> with <paramref name="count"/> tokens
    /// starting at <paramref name="offset"/>, all on sequence 0, with
    /// logits enabled at every position. Mirrors the layout
    /// <see cref="LlamaGenerator"/> uses internally for prompt prefill,
    /// but with <c>logits[i] = 1</c> for all i.
    /// </summary>
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
