namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Smoke tests for <see cref="LlamaPerplexity"/>. Uses the cached default
/// test model (Qwen3-0.6B) on small canned corpora — verifies the metric is
/// finite, monotonic in the obvious way, and that progress + cancellation
/// behave.
/// </summary>
public class PerplexityTests
{
    /// <summary>
    /// A short, fluent passage of English. With a competent base model
    /// the perplexity should be well below 100 — exact value depends on
    /// the model and tokenizer, but anything in the 1-100 range is
    /// "this is working" territory. We assert a generous upper bound.
    /// </summary>
    private const string FluentEnglish =
        "The quick brown fox jumps over the lazy dog. " +
        "Pack my box with five dozen liquor jugs. " +
        "How vexingly quick daft zebras jump! " +
        "The five boxing wizards jump quickly. " +
        "Sphinx of black quartz, judge my vow. " +
        "Amazingly few discotheques provide jukeboxes. " +
        "Jackdaws love my big sphinx of quartz. " +
        "The job requires extra pluck and zeal from every young wage earner.";

    /// <summary>
    /// Random characters — should perplex the model far more than fluent
    /// English. Used to assert the metric increases on noise.
    /// </summary>
    private const string Garbled =
        "qz xj kpvq mwfb zlrk vntx jwqp gxhk lmnv qzpw " +
        "fjkx vbnm lkjh zxcv qwer poiu mnbv lkjh fdsa " +
        "rtyu jklm zxcq pwoe asdf ghjk qwer ytre xnsa " +
        "vbgh kjlp qwed mnbf vcxz lkjr potu nbvc xzas";

    [Fact]
    public async Task Perplexity_On_Fluent_English_Is_Finite_And_Reasonable()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        using var model = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,   // CPU only — keep CI cheap
            UseMmap = true,
        });

        var result = await LlamaPerplexity.ComputeAsync(
            model,
            FluentEnglish,
            new LlamaPerplexityOptions
            {
                ContextSize = 64,           // small for speed; the corpus is tiny anyway
                ScoreSecondHalfOnly = false, // tiny corpus — we want every token scored
            });

        Assert.True(double.IsFinite(result.Perplexity),
            $"Perplexity must be finite, got {result.Perplexity} (NLL={result.NegativeLogLikelihood}).");
        Assert.True(result.Perplexity > 0,
            $"Perplexity must be positive, got {result.Perplexity}.");
        Assert.True(result.Perplexity < 1000,
            $"Perplexity on fluent English with a competent model should be < 1000, got {result.Perplexity}.");
        Assert.True(result.TokensScored > 0);
        Assert.True(result.ChunkCount >= 1);
        Assert.Equal(64, result.ContextSize);
    }

    [Fact]
    public async Task Perplexity_On_Garbled_Text_Exceeds_Fluent_Text()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        using var model = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });

        // Repetitive fluent corpus: the warm-up half teaches the pattern
        // and the scored half should hit ~1.0 PPL.
        // NON-repetitive garbled corpus: deterministic pseudo-random chars
        // so no amount of warm-up can teach it. This gives us a clear gap
        // that the metric should detect.
        var fluentCorpus = string.Concat(Enumerable.Repeat(
            "The cat sat on the mat. The cat sat on the mat. ", 16));
        var garbledCorpus = BuildNonRepetitiveGarbled(charCount: fluentCorpus.Length, seed: 42);

        var opts = new LlamaPerplexityOptions
        {
            ContextSize = 128,
            ScoreSecondHalfOnly = true,
        };

        var fluent = await LlamaPerplexity.ComputeAsync(model, fluentCorpus, opts);
        var garbled = await LlamaPerplexity.ComputeAsync(model, garbledCorpus, opts);

        // With repetitive fluent text and a 64-token warm-up, the model
        // should predict the second half with very low PPL. Garbled text
        // remains hard. Assert a clear gap so the test is robust against
        // tokenizer / quant variation.
        Assert.True(garbled.Perplexity > fluent.Perplexity * 5,
            $"Expected garbled PPL ({garbled.Perplexity:F2}) >> fluent PPL ({fluent.Perplexity:F2}) " +
            $"with repetitive corpora and second-half-only scoring.");
    }

    /// <summary>
    /// Build a deterministic non-repetitive garbled corpus of <c>charCount</c>
    /// characters using a fixed-seed RNG. The output is space-separated
    /// 4-letter "words" of random lowercase letters so the tokenizer
    /// produces many short subword tokens — none of which the model can
    /// learn from the warm-up half because the bytes never repeat in the
    /// same pattern twice.
    /// </summary>
    private static string BuildNonRepetitiveGarbled(int charCount, int seed)
    {
        var rng = new Random(seed);
        var sb = new System.Text.StringBuilder(charCount);
        while (sb.Length < charCount)
        {
            for (int i = 0; i < 4; i++) sb.Append((char)('a' + rng.Next(26)));
            sb.Append(' ');
        }
        return sb.ToString();
    }

    [Fact]
    public async Task Perplexity_Reports_Progress_Per_Chunk()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        using var model = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });

        // Pad the corpus so we get multiple chunks at a small ContextSize.
        var corpus = string.Concat(Enumerable.Repeat(FluentEnglish + " ", 6));
        var reports = new List<LlamaPerplexityProgress>();
        var progress = new Progress<LlamaPerplexityProgress>(p => reports.Add(p));

        var result = await LlamaPerplexity.ComputeAsync(
            model,
            corpus,
            new LlamaPerplexityOptions { ContextSize = 32, ScoreSecondHalfOnly = false },
            progress);

        // Progress<T> dispatches via the captured SyncContext. In an xUnit
        // test there's no UI thread, so reports may arrive on the thread-pool
        // — give the dispatcher a tick to drain.
        for (int waited = 0; reports.Count < result.ChunkCount && waited < 50; waited++)
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        Assert.NotEmpty(reports);
        Assert.Equal(result.ChunkCount, reports[^1].ChunkIndex);
        Assert.Equal(result.ChunkCount, reports[^1].ChunkCount);
        Assert.True(double.IsFinite(reports[^1].RunningPerplexity));
    }

    [Fact]
    public async Task Perplexity_Honors_Cancellation_Between_Chunks()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        using var model = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });

        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            LlamaPerplexity.ComputeAsync(
                model,
                FluentEnglish,
                new LlamaPerplexityOptions { ContextSize = 64, ScoreSecondHalfOnly = false },
                progress: null,
                cancellationToken: cts.Token));
    }
}
