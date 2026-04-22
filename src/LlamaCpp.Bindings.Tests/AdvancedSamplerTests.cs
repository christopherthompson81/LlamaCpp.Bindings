namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Tier-2 T2-1: advanced sampler builder methods. Tests the BUILDER contract
/// (terminal/non-terminal, chain composition) rather than end-to-end output
/// quality — validating that e.g. mirostat produces "less surprising" text
/// would need a calibrated perplexity harness.
/// </summary>
public class AdvancedSamplerTests : IClassFixture<ModelFixture>
{
    private readonly ModelFixture _fx;
    public AdvancedSamplerTests(ModelFixture fx) => _fx = fx;

    [Fact]
    public void Xtc_And_TopNSigma_Are_Non_Terminal()
    {
        LlamaBackend.Initialize();
        // Should be composable with a terminal afterward.
        using var s = new LlamaSamplerBuilder()
            .WithTopNSigma(1.0f)
            .WithXtc(probability: 0.1f, threshold: 0.05f)
            .WithTemperature(0.7f)
            .WithDistribution(seed: 42)
            .Build();
        Assert.Contains(s.ChainStageNames, n => n.Contains("xtc", StringComparison.OrdinalIgnoreCase)
                                             || n.Contains("top", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void ExtendedTemperature_Is_Non_Terminal()
    {
        LlamaBackend.Initialize();
        using var s = new LlamaSamplerBuilder()
            .WithExtendedTemperature(0.7f, delta: 0.3f, exponent: 1.0f)
            .WithDistribution(seed: 1)
            .Build();
        Assert.True(s.ChainLength >= 2);
    }

    [Fact]
    public void MirostatV2_Is_Terminal()
    {
        LlamaBackend.Initialize();
        var builder = new LlamaSamplerBuilder().WithMirostatV2(seed: 1);
        // Adding another stage after a terminal must throw.
        Assert.Throws<InvalidOperationException>(() => builder.WithTopK(40));
        using var s = builder.Build();
        Assert.Equal(1, s.ChainLength);
    }

    [Fact]
    public void Mirostat_V1_Is_Terminal()
    {
        LlamaBackend.Initialize();
        var builder = new LlamaSamplerBuilder()
            .WithTopK(40)
            .WithMirostat(vocabSize: 32000, seed: 1, tau: 5.0f, eta: 0.1f);
        Assert.Throws<InvalidOperationException>(() => builder.WithDistribution(seed: 2));
        using var s = builder.Build();
        Assert.Equal(2, s.ChainLength);
    }

    [Fact]
    public void AdaptiveP_Is_Terminal()
    {
        LlamaBackend.Initialize();
        var builder = new LlamaSamplerBuilder()
            .WithMinP(0.05f)
            .WithAdaptiveP(target: 0.1f, decay: 0.9f, seed: 42);
        Assert.Throws<InvalidOperationException>(() => builder.WithGreedy());
        using var s = builder.Build();
        Assert.Equal(2, s.ChainLength);
    }

    [Fact]
    public void Dry_Requires_Vocab_And_Is_Non_Terminal()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }

        using var s = new LlamaSamplerBuilder()
            .WithDry(
                _fx.Model.Vocab,
                contextTrainSize: _fx.Model.TrainingContextSize,
                multiplier: 0.8f,
                dryBase: 1.75f,
                allowedLength: 2,
                sequenceBreakers: new[] { "\n", ".", "!", "?" })
            .WithTopK(40)
            .WithTemperature(0.7f)
            .WithDistribution(seed: 42)
            .Build();
        Assert.Equal(4, s.ChainLength);

        // Also accepts null seq_breakers.
        using var s2 = new LlamaSamplerBuilder()
            .WithDry(_fx.Model.Vocab, _fx.Model.TrainingContextSize)
            .WithGreedy()
            .Build();
        Assert.Equal(2, s2.ChainLength);
    }

    [Fact]
    public void Infill_Requires_Vocab()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }

        using var s = new LlamaSamplerBuilder()
            .WithTopK(40).WithTopP(0.9f)
            .WithInfill(_fx.Model.Vocab)
            .WithDistribution(seed: 7)
            .Build();
        Assert.Equal(4, s.ChainLength);
    }

    [Fact]
    public void Dry_Rejects_Null_Vocab()
    {
        LlamaBackend.Initialize();
        Assert.Throws<ArgumentNullException>(() =>
            new LlamaSamplerBuilder().WithDry(null!, 2048).WithGreedy().Build());
    }
}

[Collection(GpuCollection.Name)]
public class AdvancedSamplerGenerationTests
{
    private readonly GpuGenerationFixture _fx;
    public AdvancedSamplerGenerationTests(GpuGenerationFixture fx) => _fx = fx;

    [Fact]
    public async Task MirostatV2_Generation_Produces_Output()
    {
        if (_fx.Context is null || _fx.Model is null) { _fx.SkipMessage(); return; }
        _fx.Context.ClearKvCache();

        // Mirostat is a terminal — it stands in for distribution.
        using var sampler = new LlamaSamplerBuilder()
            .WithMirostatV2(seed: 42, tau: 5.0f, eta: 0.1f)
            .Build();
        var gen = new LlamaGenerator(_fx.Context, sampler);

        var sb = new System.Text.StringBuilder();
        await foreach (var p in gen.GenerateAsync(
            "Briefly: say hi.", maxTokens: 20,
            addSpecial: false, parseSpecial: false))
        {
            sb.Append(p);
        }
        Assert.False(string.IsNullOrWhiteSpace(sb.ToString()));
    }

    [Fact]
    public async Task AdaptiveP_Generation_Produces_Output()
    {
        if (_fx.Context is null || _fx.Model is null) { _fx.SkipMessage(); return; }
        _fx.Context.ClearKvCache();

        using var sampler = new LlamaSamplerBuilder()
            .WithMinP(0.05f)
            .WithAdaptiveP(target: 0.3f, decay: 0.9f, seed: 7)
            .Build();
        var gen = new LlamaGenerator(_fx.Context, sampler);

        var sb = new System.Text.StringBuilder();
        await foreach (var p in gen.GenerateAsync(
            "Say yes.", maxTokens: 20,
            addSpecial: false, parseSpecial: false))
        {
            sb.Append(p);
        }
        Assert.False(string.IsNullOrWhiteSpace(sb.ToString()));
    }

    [Fact]
    public async Task Xtc_In_Chain_Produces_Output()
    {
        if (_fx.Context is null || _fx.Model is null) { _fx.SkipMessage(); return; }
        _fx.Context.ClearKvCache();

        using var sampler = new LlamaSamplerBuilder()
            .WithTopK(40)
            .WithXtc(probability: 0.1f, threshold: 0.05f, seed: 42)
            .WithTemperature(0.7f)
            .WithDistribution(seed: 42)
            .Build();
        var gen = new LlamaGenerator(_fx.Context, sampler);

        // XTC randomly removes top-probability tokens — on a small model with a
        // peaked distribution it may end up emitting only whitespace/newlines.
        // The binding-level property the test guards is "the chain wires up
        // and emits tokens without crashing"; what those tokens spell is up to
        // the model. Counting tokens decouples the assertion from model behavior.
        int emitted = 0;
        await foreach (var _ in gen.GenerateAsync(
            "Say ok.", maxTokens: 15,
            addSpecial: false, parseSpecial: false))
        {
            emitted++;
        }
        Assert.True(emitted > 0, "XTC sampler chain should emit at least one token");
    }
}
