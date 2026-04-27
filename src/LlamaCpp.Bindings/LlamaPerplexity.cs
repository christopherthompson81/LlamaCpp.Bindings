using System.Buffers;
using System.Numerics.Tensors;
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

                // Per-chunk wall is dominated by the GPU→CPU logit readback
                // (~65% on Qwen3-1.7B / RTX 3090 — confirmed via two
                // trace iterations + a ComputeBatchedAsync sweep): ~150 MB
                // of logits per chunk arrive lazily as managed-memory
                // page faults during softmax touches. Softmax compute is
                // ~16s genuine compute, already vectorized via
                // TensorPrimitives. ComputeBatchedAsync (n_seq=2,4,8)
                // gives at most 5% wall reduction over the sequential
                // path — chunks share the GPU forward but readback still
                // serializes. The only meaningful win from here is a
                // device-side logsumexp+gather subgraph (return only
                // target_logit + logsumexp per position) — ~3× wall.
                // Tracked but not on the critical path; the parallel
                // runner amortizes the per-job 50s cost when multiple
                // PPL jobs run concurrently.
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
    ///
    /// <para>
    /// <b>Default <c>maxConcurrent = 8</c> rationale</b> — measured on
    /// i7-10700K (8C/16T) + RTX 3090 with Qwen3-0.6B, 8-job ablation
    /// campaign:
    /// <code>
    ///   concurrency  threads/job  wall (s)  speedup  CPU time (s)
    ///       1           16         401.4    1.00x    5902  (softmax-bound)
    ///       2            8         349.8    1.15x    4703
    ///       4            4         283.1    1.42x    3944
    ///       8            2         252.3    1.59x    3777  ← best
    ///      16            1         295.0    1.36x    2335  (cores idle)
    /// </code>
    /// At 16 the per-job softmax falls to single-threaded (235 s/job)
    /// and only N_jobs slots are filled — half the cores idle. At 8 the
    /// softmax stays usefully parallel and the GPU contention amortizes
    /// well across 8 concurrent forwards. Counter-intuitive observation
    /// is that concurrency=8 uses LESS total CPU than concurrency=1
    /// despite finishing in 63 % the wall time — the GPU has become the
    /// new bottleneck and CPU slots free up productively.
    /// </para>
    /// </remarks>
    public static async IAsyncEnumerable<PerplexityJobResult> RunParallelAsync(
        IEnumerable<PerplexityJob> jobs,
        int maxConcurrent = 0,
        IProgress<LlamaPerplexityProgress>? sharedProgress = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation]
        CancellationToken cancellationToken = default)
    {
        // Resolve auto-concurrency from the job sizes. Materialize once so
        // the iteration below sees the same list the heuristic measured.
        var jobsList = jobs as IReadOnlyList<PerplexityJob> ?? jobs.ToList();
        if (maxConcurrent <= 0)
        {
            maxConcurrent = RecommendConcurrency(
                jobsList.Select(j => j.ModelPath),
                expectedJobCount: jobsList.Count);
        }

        using var slot = new SemaphoreSlim(maxConcurrent, maxConcurrent);
        // Split the full logical-core budget across concurrent jobs.
        // Softmax scales nearly to all 16 logical cores on this rig
        // (see ResolveSoftmaxDop note), so 4 concurrent × 4 threads
        // each gives full CPU utilization without thrashing the
        // ThreadPool. Caller can override via per-job ThreadCount.
        int autoThreadsPerJob = Math.Max(1, Environment.ProcessorCount / maxConcurrent);

        var inFlight = new List<Task<PerplexityJobResult>>();
        foreach (var job in jobsList)
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
    /// Pick a sensible <c>maxConcurrent</c> for
    /// <see cref="RunParallelAsync"/> given the actual job sizes. Two
    /// constraints: per-instance VRAM (model weights + KV cache + compute
    /// buffers) and per-instance CPU thread budget (softmax wants &gt;= 2
    /// threads to be useful — see thread sweep in
    /// <c>docs/investigations/qwen3_qk_sensitivity.md</c>).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The hardware (24 GB VRAM, 16 logical cores) stays fixed; what
    /// changes between profile builds is the model size. A 0.6B-class
    /// F16 file is ~1.5 GB / instance → 8 concurrent fits with room to
    /// spare. A 1.7B at F16 is ~3.5 GB → 5 concurrent. A 4B class is
    /// ~8 GB → only 2 concurrent. The runner default of 8 was tuned on
    /// Qwen3-0.6B and silently saturated VRAM (or hung waiting on
    /// allocation) on bigger models; this heuristic auto-scales.
    /// </para>
    /// <para>
    /// Per-instance VRAM estimate is **additive**, not multiplicative:
    /// <c>weights + 700 MiB headroom</c> (compute buffer + KV cache +
    /// allocator slack). Empirically the CUDA compute buffer at
    /// n_ctx=512 is ~300 MiB and is essentially independent of model
    /// size (Qwen3-0.6B logged 298.75 MiB, Qwen3-1.7B logged 300.75
    /// MiB — same order of magnitude as the KV cache). A previous
    /// multiplicative form (<c>weights × 1.85</c>) over-reserved
    /// badly on larger models — the 1.7B build resolved
    /// concurrency=2 while only using ~3.5 GB / instance, leaving
    /// half the GPU idle (Run 13). Additive matches the actual
    /// behavior. Users on tight VRAM can pass an explicit
    /// <paramref name="availableVramBytes"/>.
    /// </para>
    /// <para>
    /// CPU ceiling is half the logical cores (8 on a 16-thread box) —
    /// matches the empirically-best concurrency for softmax-bound PPL
    /// scoring on the reference machine. With CPU/2 jobs, each gets
    /// ~2 threads, which the thread sweep showed retains most of the
    /// softmax-parallelism win.
    /// </para>
    /// </remarks>
    /// <param name="modelPaths">Paths to size representatives. The heuristic uses the *largest* file size as the per-instance VRAM estimate. May be a single representative file (e.g. the source model) when called from a builder before ablation files exist.</param>
    /// <param name="availableVramBytes">Override the assumed available VRAM. Defaults to 24 GB (RTX 3090 / 4090 class). Pass smaller for laptops or shared GPUs.</param>
    /// <param name="cpuConcurrencyCap">Override the CPU ceiling. Defaults to <c>Environment.ProcessorCount / 2</c>.</param>
    /// <param name="expectedJobCount">Optional cap by job count. When unset, the heuristic does not cap by <c>modelPaths.Count</c> — that's important for callers that pass a single representative file but expect to run many jobs. The runner overload calls this with the actual job count to also cap there.</param>
    public static int RecommendConcurrency(
        IEnumerable<string> modelPaths,
        long? availableVramBytes = null,
        int cpuConcurrencyCap = 0,
        int? expectedJobCount = null)
    {
        ArgumentNullException.ThrowIfNull(modelPaths);
        long maxFile = 0;
        int count = 0;
        foreach (var p in modelPaths)
        {
            count++;
            try
            {
                long sz = new FileInfo(p).Length;
                if (sz > maxFile) maxFile = sz;
            }
            catch { /* unreadable file — let the runner surface the error later */ }
        }
        if (count == 0) return 1;
        if (maxFile <= 0) maxFile = 4L * 1024 * 1024 * 1024;  // 4 GB fallback

        // Per-instance VRAM ≈ weights + 700 MiB. The 700 MiB absorbs
        // the CUDA compute buffer (~300 MiB at n_ctx=512, observed
        // identical on Qwen3-0.6B and 1.7B), KV cache (tens to ~150
        // MiB at n_ctx=512), and allocator overhead, with margin.
        // Earlier multiplicative form (weights × 1.85) was tuned on
        // a single 0.6B run; on 1.7B-class models it over-reserved
        // by ~2× and halved usable concurrency (Run 13).
        const long PerInstanceHeadroom = 700L * 1024 * 1024;
        long perInstance = maxFile + PerInstanceHeadroom;
        long vram = availableVramBytes ?? 24L * 1024 * 1024 * 1024;
        long usableVram = (long)(vram * 0.80);       // 20 % left for the OS/DE/other GPU users

        int byVram = Math.Max(1, (int)(usableVram / perInstance));
        int byCpu  = cpuConcurrencyCap > 0
            ? cpuConcurrencyCap
            : Math.Max(1, Environment.ProcessorCount / 2);

        int recommended = Math.Min(byVram, byCpu);
        if (expectedJobCount is int n && n > 0)
        {
            // Don't ask for more concurrency than there are jobs.
            recommended = Math.Min(recommended, n);
        }
        return Math.Max(1, recommended);
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

        // Stable log-sum-exp via vectorized TensorPrimitives. The two
        // hot loops over a ~150k-vocab dominate per-chunk wall time
        // (Run 14: ~14s pure softmax for wikitext-2 at 16-thread
        // parallel). TensorPrimitives.Max / Subtract / Exp / Sum are
        // SIMD-accelerated on AVX2/AVX512, giving roughly 4× over the
        // scalar Math.Exp loop on this hardware.
        var logitSpan = new ReadOnlySpan<float>(logits, nVocab);
        float maxLogit = TensorPrimitives.Max(logitSpan);

        // Rent a per-call buffer for the shifted/exp'd logits. ArrayPool
        // is lock-free at this size class — each parallel scoring task
        // pays a single rent/return per scored position.
        var rented = ArrayPool<float>.Shared.Rent(nVocab);
        try
        {
            var tmp = rented.AsSpan(0, nVocab);
            TensorPrimitives.Subtract(logitSpan, maxLogit, tmp);
            TensorPrimitives.Exp(tmp, tmp);
            float sumExp = TensorPrimitives.Sum((ReadOnlySpan<float>)tmp);
            double logZ = maxLogit + Math.Log(sumExp);
            double logProb = logits[actualToken] - logZ;
            return -logProb;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
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
