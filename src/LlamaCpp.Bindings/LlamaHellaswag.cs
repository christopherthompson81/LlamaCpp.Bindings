using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>One HellaSwag task: a context + four candidate endings + the gold ending index.</summary>
public sealed record LlamaHellaswagTask(
    string Context,
    IReadOnlyList<string> Endings,
    int GoldEndingIndex)
{
    public const int EndingCount = 4;
}

/// <summary>Per-task progress reported during HellaSwag scoring.</summary>
public readonly record struct LlamaHellaswagProgress(
    int TaskIndex,
    int TaskCount,
    int RunningCorrect,
    double RunningAccuracy);

/// <summary>Final summary of a HellaSwag run.</summary>
public sealed record LlamaHellaswagResult(
    double AccuracyNorm,        // length-normalized argmax match rate
    double AccuracyRaw,         // raw-sum argmax match rate (no length normalization)
    int TaskCount,
    int CorrectNorm,
    int CorrectRaw,
    int ContextSize,
    TimeSpan Elapsed);

/// <summary>Knobs for <see cref="LlamaHellaswag.ComputeAsync"/>.</summary>
public sealed class LlamaHellaswagOptions
{
    /// <summary>Per-decode context length. <c>0</c> picks <c>min(model.TrainingCtx, 512)</c>.</summary>
    public int ContextSize { get; set; } = 0;

    /// <summary>CPU thread count. <c>-1</c> = llama.cpp default.</summary>
    public int ThreadCount { get; set; } = -1;
}

/// <summary>
/// Computes the HellaSwag <c>acc_norm</c> score — the canonical
/// "is this quant any good at common-sense reasoning?" benchmark.
/// Each task presents a context plus four candidate continuations;
/// the model "answers" by selecting the candidate with highest mean
/// per-token log-likelihood. Accuracy is the fraction of tasks whose
/// argmax matches the labeled gold ending.
/// </summary>
/// <remarks>
/// <para>
/// V1 scores each candidate via a fresh full-sequence decode (clear
/// KV, decode <c>context + ending</c>, read logits at every position).
/// Upstream's perplexity tool batches all 4 endings as separate
/// sequences sharing a common-prefix KV — ~3× faster but
/// significantly more complex to wire from C#. Defer until the
/// performance becomes the binding constraint; for now, default to a
/// reasonable task subset and document the cost.
/// </para>
/// <para>
/// Dataset format expected by <see cref="ParseUpstreamFile"/> is the
/// 6-line-per-task layout from
/// <c>klosax/hellaswag_text_data/hellaswag_val_full.txt</c>:
/// activity-prefixed context, gold index, ending 0, ending 1, ending
/// 2, ending 3.
/// </para>
/// </remarks>
public static class LlamaHellaswag
{
    /// <summary>Score <paramref name="tasks"/> against <paramref name="model"/>.</summary>
    public static Task<LlamaHellaswagResult> ComputeAsync(
        LlamaModel model,
        IReadOnlyList<LlamaHellaswagTask> tasks,
        LlamaHellaswagOptions? options = null,
        IProgress<LlamaHellaswagProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tasks);
        if (tasks.Count == 0)
            throw new ArgumentException("No tasks supplied.", nameof(tasks));
        var opts = options ?? new LlamaHellaswagOptions();
        return Task.Run(() => Compute(model, tasks, opts, progress, cancellationToken), cancellationToken);
    }

    /// <summary>
    /// Parse the 6-line-per-task <c>hellaswag_val_full.txt</c> format
    /// shared by llama.cpp's upstream and the
    /// <c>klosax/hellaswag_text_data</c> mirror.
    /// </summary>
    public static IReadOnlyList<LlamaHellaswagTask> ParseUpstreamFile(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        return ParseUpstreamText(File.ReadAllText(path));
    }

    /// <summary>Parse the same 6-line-per-task format from a string.</summary>
    public static IReadOnlyList<LlamaHellaswagTask> ParseUpstreamText(string content)
    {
        ArgumentNullException.ThrowIfNull(content);
        // Splitting by \n and trimming \r covers both LF and CRLF
        // variants. Upstream expects exactly N×6 lines so we follow
        // suit, but tolerate trailing blank lines (a common artifact
        // of file-tail editors).
        var rawLines = content.Split('\n');
        var lines = new List<string>(rawLines.Length);
        foreach (var raw in rawLines)
        {
            var line = raw.TrimEnd('\r');
            lines.Add(line);
        }
        // Drop trailing blank lines.
        while (lines.Count > 0 && string.IsNullOrEmpty(lines[^1])) lines.RemoveAt(lines.Count - 1);
        if (lines.Count % 6 != 0)
        {
            throw new InvalidDataException(
                $"HellaSwag file has {lines.Count} non-trailing-blank lines, " +
                "but the format expects a multiple of 6 (one task per 6 lines).");
        }
        int taskCount = lines.Count / 6;
        var tasks = new LlamaHellaswagTask[taskCount];
        for (int i = 0; i < taskCount; i++)
        {
            string context = lines[i * 6 + 0];
            if (!int.TryParse(lines[i * 6 + 1].Trim(), out var gold))
            {
                throw new InvalidDataException(
                    $"Task {i}: line {i * 6 + 2} '{lines[i * 6 + 1]}' is not a valid gold-index integer.");
            }
            var endings = new[]
            {
                lines[i * 6 + 2],
                lines[i * 6 + 3],
                lines[i * 6 + 4],
                lines[i * 6 + 5],
            };
            tasks[i] = new LlamaHellaswagTask(context, endings, gold);
        }
        return tasks;
    }

    private static LlamaHellaswagResult Compute(
        LlamaModel model,
        IReadOnlyList<LlamaHellaswagTask> tasks,
        LlamaHellaswagOptions opts,
        IProgress<LlamaHellaswagProgress>? progress,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        int trainingCtx = Math.Max(8, model.TrainingContextSize);
        int chunkSize = opts.ContextSize > 0
            ? Math.Min(opts.ContextSize, trainingCtx)
            : Math.Min(512, trainingCtx);

        var ctxParams = new LlamaContextParameters
        {
            ContextSize       = (uint)chunkSize,
            LogicalBatchSize  = (uint)chunkSize,
            PhysicalBatchSize = (uint)chunkSize,
            MaxSequenceCount  = 1,
            ThreadCount       = opts.ThreadCount,
            BatchThreadCount  = opts.ThreadCount,
        };
        using var context = new LlamaContext(model, ctxParams);

        int nVocab = model.Vocab.TokenCount;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int correctNorm = 0;
        int correctRaw  = 0;

        var batch = NativeMethods.llama_batch_init(chunkSize, embd: 0, n_seq_max: 1);
        // Per-candidate score buffers — allocated outside the task
        // loop and reused per-task to avoid the analyzer's stackalloc-
        // in-a-loop warning (CA2014). Tiny: 4 doubles = 32 bytes each.
        Span<double> rawLogProbs  = stackalloc double[LlamaHellaswagTask.EndingCount];
        Span<double> normLogProbs = stackalloc double[LlamaHellaswagTask.EndingCount];
        try
        {
            for (int t = 0; t < tasks.Count; t++)
            {
                ct.ThrowIfCancellationRequested();
                var task = tasks[t];
                if (task.Endings.Count != LlamaHellaswagTask.EndingCount)
                {
                    throw new InvalidDataException(
                        $"Task {t} has {task.Endings.Count} endings; HellaSwag tasks must have exactly 4.");
                }
                if (task.GoldEndingIndex is < 0 or >= LlamaHellaswagTask.EndingCount)
                {
                    throw new InvalidDataException(
                        $"Task {t} has gold index {task.GoldEndingIndex}; must be 0..3.");
                }

                // Tokenize the four candidate sequences. We tokenize
                // <context + " " + ending> as a single string per
                // candidate — that mirrors upstream's behavior and lets
                // the BPE merge across the boundary the same way it
                // would during normal use of the model.
                var seqs = new int[LlamaHellaswagTask.EndingCount][];
                for (int j = 0; j < LlamaHellaswagTask.EndingCount; j++)
                {
                    seqs[j] = model.Vocab.Tokenize(
                        task.Context + " " + task.Endings[j],
                        addSpecial: true,
                        parseSpecial: false);
                    if (seqs[j].Length > chunkSize)
                    {
                        throw new InvalidOperationException(
                            $"Task {t} ending {j} tokenizes to {seqs[j].Length} tokens, exceeding context " +
                            $"size {chunkSize}. Increase ContextSize or skip overlong tasks.");
                    }
                    if (seqs[j].Length < 2)
                    {
                        throw new InvalidOperationException(
                            $"Task {t} ending {j} tokenized to {seqs[j].Length} tokens; need at least 2 to score.");
                    }
                }

                // Find the longest common token prefix shared across
                // all four sequences. We start scoring from the first
                // diverging position — this matches the standard
                // "score only the ending tokens" semantics, which
                // matters when context tokenization accidentally
                // splits differently across endings.
                int commonPrefix = 0;
                int minLen = seqs.Min(s => s.Length);
                while (commonPrefix < minLen
                    && seqs[0][commonPrefix] == seqs[1][commonPrefix]
                    && seqs[0][commonPrefix] == seqs[2][commonPrefix]
                    && seqs[0][commonPrefix] == seqs[3][commonPrefix])
                {
                    commonPrefix++;
                }

                // Score each candidate: clear KV, decode the full
                // candidate sequence with logits at every position,
                // then sum the log-prob of each post-common-prefix
                // token. The two metrics:
                //   acc_raw  — argmax of raw summed log-prob.
                //   acc_norm — argmax of (log-prob sum) / (token count).
                // The latter is the standard reported HellaSwag metric;
                // it removes the length bias that favours shorter
                // endings on raw sum.
                for (int j = 0; j < LlamaHellaswagTask.EndingCount; j++)
                {
                    var seq = seqs[j];
                    context.ClearKvCache();
                    PopulateBatchAllLogits(ref batch, seq, seq.Length);
                    unsafe
                    {
                        var rc = NativeMethods.llama_decode(context.Handle.DangerousHandle, batch);
                        if (rc != 0)
                        {
                            throw new LlamaException(
                                "llama_decode", rc,
                                $"llama_decode returned {rc} on HellaSwag task {t}/{tasks.Count} ending {j}.");
                        }
                    }

                    double sum = 0;
                    int countedTokens = 0;
                    // For each position p starting at commonPrefix,
                    // logits at (p-1) predict seq[p]. We score from
                    // commonPrefix onward.
                    int firstScored = Math.Max(1, commonPrefix);
                    for (int p = firstScored; p < seq.Length; p++)
                    {
                        sum += LogProbAt(context, p - 1, seq[p], nVocab);
                        countedTokens++;
                    }
                    rawLogProbs[j]  = sum;
                    normLogProbs[j] = countedTokens > 0 ? sum / countedTokens : sum;
                }

                int argmaxRaw  = ArgmaxOf(rawLogProbs);
                int argmaxNorm = ArgmaxOf(normLogProbs);
                if (argmaxRaw  == task.GoldEndingIndex) correctRaw++;
                if (argmaxNorm == task.GoldEndingIndex) correctNorm++;

                progress?.Report(new LlamaHellaswagProgress(
                    TaskIndex: t + 1,
                    TaskCount: tasks.Count,
                    RunningCorrect: correctNorm,
                    RunningAccuracy: (double)correctNorm / (t + 1)));
            }
        }
        finally
        {
            NativeMethods.llama_batch_free(batch);
        }

        sw.Stop();
        return new LlamaHellaswagResult(
            AccuracyNorm: tasks.Count > 0 ? (double)correctNorm / tasks.Count : 0,
            AccuracyRaw:  tasks.Count > 0 ? (double)correctRaw  / tasks.Count : 0,
            TaskCount:    tasks.Count,
            CorrectNorm:  correctNorm,
            CorrectRaw:   correctRaw,
            ContextSize:  chunkSize,
            Elapsed:      sw.Elapsed);
    }

    private static int ArgmaxOf(ReadOnlySpan<double> values)
    {
        int idx = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[idx]) idx = i;
        }
        return idx;
    }

    private static unsafe double LogProbAt(LlamaContext context, int position, int actualToken, int nVocab)
    {
        float* logits = NativeMethods.llama_get_logits_ith(context.Handle.DangerousHandle, position);
        if (logits is null)
        {
            throw new LlamaException("llama_get_logits_ith",
                $"NULL logits at position {position} in HellaSwag scoring.");
        }
        // Stable log-sum-exp.
        float max = logits[0];
        for (int i = 1; i < nVocab; i++) if (logits[i] > max) max = logits[i];
        double sum = 0;
        for (int i = 0; i < nVocab; i++) sum += Math.Exp(logits[i] - max);
        double logZ = max + Math.Log(sum);
        return logits[actualToken] - logZ;
    }

    private static unsafe void PopulateBatchAllLogits(
        ref llama_batch batch, int[] tokens, int count)
    {
        batch.n_tokens = count;
        var tokPtr   = (int*)batch.token;
        var posPtr   = (int*)batch.pos;
        var nSeqPtr  = (int*)batch.n_seq_id;
        var seqIdArr = (int**)batch.seq_id;
        var logits   = (sbyte*)batch.logits;
        for (int i = 0; i < count; i++)
        {
            tokPtr[i]      = tokens[i];
            posPtr[i]      = i;
            nSeqPtr[i]     = 1;
            seqIdArr[i][0] = 0;
            logits[i]      = 1;
        }
    }
}
