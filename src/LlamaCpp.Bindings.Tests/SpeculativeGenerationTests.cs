using System.Text;
using Xunit;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Speculative-decoding integration tests. Uses its own fixture because the
/// GPU fixture assumes a single loaded model — speculative decoding needs
/// two contexts (main + draft).
///
/// <para>
/// By default both contexts are loaded from the same GGUF (the default test
/// model). That is a degenerate case for speculative decoding — a draft
/// identical to the main will always agree, so the acceptance rate should
/// be 100% and throughput is strictly worse than plain decoding. That is
/// acceptable for exercising the plumbing; real speedup tests need a smaller
/// draft model, pointed at by <c>LLAMACPP_TEST_DRAFT_MODEL</c>.
/// </para>
/// </summary>
public class SpeculativeGenerationTests : IClassFixture<SpeculativeFixture>
{
    private readonly SpeculativeFixture _fx;
    private readonly ITestOutputHelper _log;
    public SpeculativeGenerationTests(SpeculativeFixture fx, ITestOutputHelper log)
    {
        _fx = fx;
        _log = log;
    }

    // ----- Constructor / argument validation (no decode required) -----

    [Fact]
    public void Constructor_Rejects_Null_Arguments()
    {
        using var mainSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        using var draftSampler = new LlamaSamplerBuilder().WithGreedy().Build();

        Assert.Throws<ArgumentNullException>(() => new LlamaSpeculativeGenerator(null!, _fx.DraftContext, mainSampler, draftSampler));
        Assert.Throws<ArgumentNullException>(() => new LlamaSpeculativeGenerator(_fx.MainContext, null!, mainSampler, draftSampler));
        Assert.Throws<ArgumentNullException>(() => new LlamaSpeculativeGenerator(_fx.MainContext, _fx.DraftContext, null!, draftSampler));
        Assert.Throws<ArgumentNullException>(() => new LlamaSpeculativeGenerator(_fx.MainContext, _fx.DraftContext, mainSampler, null!));
    }

    [Fact]
    public void Constructor_Rejects_Same_Context_For_Main_And_Draft()
    {
        using var mainSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        using var draftSampler = new LlamaSamplerBuilder().WithGreedy().Build();

        Assert.Throws<ArgumentException>(() => new LlamaSpeculativeGenerator(_fx.MainContext, _fx.MainContext, mainSampler, draftSampler));
    }

    [Fact]
    public void Constructor_Rejects_Non_Positive_Lookahead()
    {
        using var mainSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        using var draftSampler = new LlamaSamplerBuilder().WithGreedy().Build();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LlamaSpeculativeGenerator(_fx.MainContext, _fx.DraftContext, mainSampler, draftSampler, draftLookahead: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LlamaSpeculativeGenerator(_fx.MainContext, _fx.DraftContext, mainSampler, draftSampler, draftLookahead: -1));
    }

    // ----- End-to-end streaming -----

    [Fact]
    public async Task Produces_Nonempty_Output_With_Same_Model_As_Draft()
    {
        _fx.MainContext.ClearKvCache();
        _fx.DraftContext.ClearKvCache();

        using var mainSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        using var draftSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        using var spec = new LlamaSpeculativeGenerator(
            _fx.MainContext, _fx.DraftContext, mainSampler, draftSampler, draftLookahead: 4);

        var prompt = BuildPrompt(_fx.MainModel, "Say hello.");
        var sb = new StringBuilder();
        await foreach (var piece in spec.GenerateAsync(
            prompt, maxTokens: 32, addSpecial: false, parseSpecial: true,
            cancellationToken: TestContext.Current.CancellationToken))
        {
            sb.Append(piece);
        }

        Assert.False(string.IsNullOrWhiteSpace(sb.ToString()),
            "Speculative generator produced no output. Plumbing likely broken.");
    }

    [Fact]
    public async Task High_Acceptance_When_Draft_Is_Same_Family_As_Main()
    {
        // With a same-family draft (Qwen3-0.6B verifying for Qwen3-1.7B) on
        // a factual prompt, the draft's top-1 pick agrees with the main
        // frequently enough to clear 40% acceptance. Anything dramatically
        // below that on this pair points at a bug in verification, not at
        // the draft being a bad guesser.
        _fx.MainContext.ClearKvCache();
        _fx.DraftContext.ClearKvCache();

        using var mainSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        using var draftSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        using var spec = new LlamaSpeculativeGenerator(
            _fx.MainContext, _fx.DraftContext, mainSampler, draftSampler, draftLookahead: 4);

        var prompt = BuildPrompt(_fx.MainModel, "List three prime numbers.");
        var sb = new StringBuilder();
        await foreach (var piece in spec.GenerateAsync(
            prompt, maxTokens: 24, addSpecial: false, parseSpecial: true,
            cancellationToken: TestContext.Current.CancellationToken))
        {
            sb.Append(piece);
        }

        Assert.False(string.IsNullOrWhiteSpace(sb.ToString()));
        Assert.True(spec.Stats.TotalDrafted > 0, "Draft never proposed any tokens — lookahead bookkeeping is broken.");
        // Floor, not a fixed value: identical-model pairs sit at ~100% and
        // 0.6B→1.7B Qwen3 sits around 85–95%. The floor catches regressions
        // without pinning us to a model-specific ratio.
        Assert.True(spec.Stats.AcceptanceRate >= 0.4,
            $"Acceptance rate {spec.Stats.AcceptanceRate:P1} is implausibly low for a same-family pair " +
            $"({spec.Stats.TotalAccepted}/{spec.Stats.TotalDrafted}).");
    }

    [Fact]
    public async Task Output_Is_Nonempty_And_Stops_Cleanly()
    {
        // Semantic-equivalence between plain greedy and speculative greedy
        // is NOT guaranteed on GPU: batched decode and sequential decode
        // produce subtly different logits (kernel / batch-size dependent
        // floating-point ops), and over enough steps those differences
        // bubble up to different argmax picks. The algorithm is provably
        // correct — we always sample using the main model's sampler at a
        // main-verified position — but byte-equivalence only holds on CPU
        // or for very short runs. This test only asserts that a realistic
        // run produces coherent output and terminates with a recognised
        // stop reason.
        var prompt = BuildPrompt(_fx.MainModel, "Count: 1, 2, 3,");

        _fx.MainContext.ClearKvCache();
        _fx.DraftContext.ClearKvCache();
        using var mainSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        using var draftSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        using var spec = new LlamaSpeculativeGenerator(
            _fx.MainContext, _fx.DraftContext, mainSampler, draftSampler, draftLookahead: 3);
        var speculative = new StringBuilder();
        await foreach (var p in spec.GenerateAsync(
            prompt, maxTokens: 24, addSpecial: false, parseSpecial: true,
            cancellationToken: TestContext.Current.CancellationToken))
        {
            speculative.Append(p);
        }

        Assert.False(string.IsNullOrWhiteSpace(speculative.ToString()));
        Assert.Contains(spec.LastStopReason, new[]
        {
            LlamaStopReason.MaxTokens,
            LlamaStopReason.EndOfGeneration,
        });
    }

    // ----- Distinct main / draft pair -----

    [Fact]
    public async Task Distinct_Pair_Draft_Proposes_And_Main_Accepts_Majority()
    {
        // With Qwen3-1.7B verifying a Qwen3-0.6B draft on generic English,
        // we expect well over half the draft tokens to be accepted. Below
        // that, the draft is effectively dead weight and speculative
        // decoding is losing to plain — worth flagging as a regression.
        if (!_fx.HasDistinctModels)
        {
            Assert.Skip("Speculative main model unavailable — set LLAMACPP_TEST_SPEC_MAIN_MODEL or allow auto-download.");
        }

        var prompt = BuildPrompt(_fx.MainModel, "Name three colours.");

        _fx.MainContext.ClearKvCache();
        _fx.DraftContext.ClearKvCache();
        using var mainSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        using var draftSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        using var spec = new LlamaSpeculativeGenerator(
            _fx.MainContext, _fx.DraftContext, mainSampler, draftSampler, draftLookahead: 4);
        var speculative = new StringBuilder();
        await foreach (var p in spec.GenerateAsync(
            prompt, maxTokens: 32, addSpecial: false, parseSpecial: true,
            cancellationToken: TestContext.Current.CancellationToken))
        {
            speculative.Append(p);
        }

        Assert.False(string.IsNullOrWhiteSpace(speculative.ToString()));
        Assert.True(spec.Stats.TotalDrafted > 0);
        Assert.True(spec.Stats.AcceptanceRate >= 0.5,
            $"Acceptance rate {spec.Stats.AcceptanceRate:P1} on Qwen3 0.6B→1.7B is below the 50% floor " +
            $"({spec.Stats.TotalAccepted}/{spec.Stats.TotalDrafted}). Expect ~85% in healthy state.");
    }

    [Fact]
    public async Task Distinct_Pair_Benchmark_Plain_Vs_Speculative()
    {
        // Pure benchmark: measure wall-clock tokens/sec for plain vs
        // speculative over the same prompt and token budget. Prints the
        // tokens/sec, speedup, and acceptance rate to the test output so
        // regressions and tuning wins are both visible.
        //
        // We do NOT assert a speedup floor: the actual ratio is heavily
        // workload-dependent — with small mains (1.7B) per-token kernel
        // launch overhead dominates real compute, so the speculative path
        // can underperform plain decode even at 85%+ acceptance. Real 2–5×
        // wins only show up once the main is big enough (≥ 7B) that single-
        // token latency is clearly compute-bound. The assertion is on the
        // acceptance-rate floor instead, which catches verification bugs
        // regardless of model size.
        if (!_fx.HasDistinctModels)
        {
            Assert.Skip("Speculative main model unavailable — set LLAMACPP_TEST_SPEC_MAIN_MODEL or allow auto-download.");
        }

        const int MaxTokens = 80;
        var prompt = BuildPrompt(_fx.MainModel,
            "Write a short paragraph explaining what a prime number is.");

        // Plain greedy baseline.
        _fx.MainContext.ClearKvCache();
        using var plainSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        var plainGen = new LlamaGenerator(_fx.MainContext, plainSampler);
        var plainOut = new StringBuilder();
        int plainTokens = 0;
        var plainSw = System.Diagnostics.Stopwatch.StartNew();
        await foreach (var p in plainGen.GenerateAsync(
            prompt, maxTokens: MaxTokens, addSpecial: false, parseSpecial: true,
            cancellationToken: TestContext.Current.CancellationToken))
        {
            plainOut.Append(p);
            plainTokens++;
        }
        plainSw.Stop();

        // Speculative with same greedy samplers.
        _fx.MainContext.ClearKvCache();
        _fx.DraftContext.ClearKvCache();
        using var mainSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        using var draftSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        using var spec = new LlamaSpeculativeGenerator(
            _fx.MainContext, _fx.DraftContext, mainSampler, draftSampler, draftLookahead: 5);
        var specOut = new StringBuilder();
        int specTokens = 0;
        var specSw = System.Diagnostics.Stopwatch.StartNew();
        await foreach (var p in spec.GenerateAsync(
            prompt, maxTokens: MaxTokens, addSpecial: false, parseSpecial: true,
            cancellationToken: TestContext.Current.CancellationToken))
        {
            specOut.Append(p);
            specTokens++;
        }
        specSw.Stop();

        double plainRate = plainTokens / plainSw.Elapsed.TotalSeconds;
        double specRate  = specTokens  / specSw.Elapsed.TotalSeconds;
        double speedup   = specRate / plainRate;
        double accept    = spec.Stats.AcceptanceRate;

        _log.WriteLine(
            $"[SpecBench] plain: {plainTokens} tok in {plainSw.Elapsed.TotalMilliseconds:F0} ms ({plainRate:F1} tok/s) | " +
            $"spec: {specTokens} tok in {specSw.Elapsed.TotalMilliseconds:F0} ms ({specRate:F1} tok/s) | " +
            $"speedup = {speedup:F2}x | accept = {accept:P1} " +
            $"({spec.Stats.TotalAccepted}/{spec.Stats.TotalDrafted})");

        // Both paths should produce non-empty, non-trivial output. Byte
        // equivalence is not asserted: batched vs sequential decode on GPU
        // produce tiny logit differences that compound into different
        // argmax picks over long runs. The algorithm is still correct
        // (every emitted token is sampled by the main's sampler at a
        // main-verified position) — just not bit-reproducible against
        // plain decode.
        Assert.True(plainTokens > 0);
        Assert.True(specTokens > 0);
        _ = speedup; // avoid unused-value warning; the value is in the log line

        // Acceptance-rate floor. A Qwen3-0.6B draft verifying for Qwen3-1.7B
        // on generic English should clear 40% easily. Anything below that
        // means the verification loop is being pessimistic — probably a
        // drift between draft KV state and main KV state.
        Assert.True(accept >= 0.4,
            $"Draft acceptance rate {accept:P1} is implausibly low for a same-family pair. " +
            "Suspect KV-state drift between the two contexts.");
    }

    // ----- helpers -----

    private static string BuildPrompt(LlamaModel model, string userMessage)
    {
        var tmpl = model.GetChatTemplate();
        if (string.IsNullOrEmpty(tmpl)) return userMessage;
        return LlamaChatTemplate.Apply(tmpl, new[]
        {
            new ChatMessage("user", userMessage),
        }, addAssistantPrefix: true);
    }
}

/// <summary>
/// Loads a main + draft context pair for speculative-decoding tests.
///
/// <para>
/// Default main model is Qwen3-1.7B (auto-fetched on first use), default
/// draft is the existing Qwen3-0.6B test model — a realistic ~3× parameter-
/// count gap that exercises the speedup path. Both override via
/// <c>LLAMACPP_TEST_SPEC_MAIN_MODEL</c> / <c>LLAMACPP_TEST_DRAFT_MODEL</c>.
/// </para>
///
/// <para>
/// When the default main fetch fails (no network, user overrode
/// <c>LLAMACPP_TEST_MODEL</c> to a non-Qwen3 family, etc.) the fixture
/// falls back to using the default test model as both sides — plumbing
/// tests still run, speedup/distinctness tests skip themselves.
/// </para>
/// </summary>
public sealed class SpeculativeFixture : IDisposable
{
    public LlamaModel MainModel { get; }
    public LlamaContext MainContext { get; }
    public LlamaModel DraftModel { get; }
    public LlamaContext DraftContext { get; }

    /// <summary>True when the fixture is running with distinct main/draft weights.</summary>
    public bool HasDistinctModels { get; }

    public SpeculativeFixture()
    {
        LlamaBackend.Initialize();

        var draftPath = Environment.GetEnvironmentVariable("LLAMACPP_TEST_DRAFT_MODEL");
        if (string.IsNullOrWhiteSpace(draftPath))
        {
            draftPath = TestModelProvider.EnsureModelPath();
        }
        else if (!File.Exists(draftPath))
        {
            throw new FileNotFoundException(
                $"LLAMACPP_TEST_DRAFT_MODEL='{draftPath}' but the file does not exist.", draftPath);
        }

        var mainPath = TestModelProvider.TryGetSpeculativeMainModelPath() ?? draftPath;
        HasDistinctModels = !string.Equals(
            Path.GetFullPath(mainPath), Path.GetFullPath(draftPath), StringComparison.Ordinal);

        MainModel = new LlamaModel(mainPath, new LlamaModelParameters
        {
            GpuLayerCount = -1,
            UseMmap = true,
        });
        MainContext = new LlamaContext(MainModel, new LlamaContextParameters
        {
            ContextSize = 2048,
            LogicalBatchSize = 512,
            PhysicalBatchSize = 512,
            MaxSequenceCount = 1,
            OffloadKQV = true,
        });

        DraftModel = new LlamaModel(draftPath, new LlamaModelParameters
        {
            GpuLayerCount = -1,
            UseMmap = true,
        });
        DraftContext = new LlamaContext(DraftModel, new LlamaContextParameters
        {
            ContextSize = 2048,
            LogicalBatchSize = 512,
            PhysicalBatchSize = 512,
            MaxSequenceCount = 1,
            OffloadKQV = true,
        });
    }

    public void Dispose()
    {
        DraftContext.Dispose();
        DraftModel.Dispose();
        MainContext.Dispose();
        MainModel.Dispose();
    }
}
