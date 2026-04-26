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

                int firstScored = options.ScoreSecondHalfOnly ? chunkSize / 2 : 1;
                for (int i = firstScored; i < chunkSize; i++)
                {
                    // logits[i-1] are the predictive distribution for the
                    // token actually appearing at tokens[start + i].
                    int actual = tokens[start + i];
                    double nll = NegativeLogProbability(context, position: i - 1, actualToken: actual, nVocab);
                    nllSum += nll;
                    tokensScored++;
                }

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
