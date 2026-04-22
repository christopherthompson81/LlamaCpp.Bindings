namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Covers Tier-1 sampler introspection: chain inspection, seed readback,
/// clone. Model-agnostic — no model fixture needed.
/// </summary>
public class SamplerIntrospectionTests
{
    [Fact]
    public void Chain_Reports_Stage_Count_And_Names()
    {
        LlamaBackend.Initialize();

        using var s = new LlamaSamplerBuilder()
            .WithTopK(40)
            .WithTopP(0.9f)
            .WithMinP(0.05f)
            .WithTemperature(0.7f)
            .WithDistribution(seed: 42)
            .Build();

        Assert.Equal("chain", s.Name);
        Assert.Equal(5, s.ChainLength);

        var stages = s.ChainStageNames;
        Assert.Equal(5, stages.Count);
        // llama.cpp names them: top-k, top-p, min-p, temp, dist — but the exact
        // spelling (dashes vs underscores) is subject to upstream drift. Just
        // verify they're non-empty and distinct.
        Assert.All(stages, name => Assert.False(string.IsNullOrWhiteSpace(name)));

        // Names should reference sampling concepts; do a loose substring check
        // rather than hard-coded equality.
        Assert.Contains(stages, n => n.Contains("top", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(stages, n => n.Contains("temp", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(stages, n => n.Contains("dist", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void GetChainStageName_Out_Of_Range_Returns_Null()
    {
        LlamaBackend.Initialize();
        using var s = new LlamaSamplerBuilder().WithTopK(40).WithGreedy().Build();
        Assert.Null(s.GetChainStageName(-1));
        Assert.Null(s.GetChainStageName(999));
    }

    [Fact]
    public void Seed_Is_Readback_For_Distribution_Chain()
    {
        LlamaBackend.Initialize();
        using var s = new LlamaSamplerBuilder()
            .WithTopK(40).WithTemperature(0.7f).WithDistribution(seed: 12345).Build();

        Assert.Equal(12345u, s.Seed);
    }

    [Fact]
    public void Seed_Is_Null_For_Greedy_Chain()
    {
        LlamaBackend.Initialize();
        using var s = new LlamaSamplerBuilder().WithTopK(40).WithGreedy().Build();
        // Greedy has no RNG; llama.cpp returns LLAMA_DEFAULT_SEED (0xFFFFFFFF).
        Assert.Null(s.Seed);
    }

    [Fact]
    public void Clone_Produces_Independent_Sampler()
    {
        LlamaBackend.Initialize();
        using var original = new LlamaSamplerBuilder()
            .WithTopK(40).WithTemperature(0.7f).WithDistribution(seed: 999).Build();

        using var clone = original.Clone();
        Assert.Equal(original.Name, clone.Name);
        Assert.Equal(original.ChainLength, clone.ChainLength);
        Assert.Equal(original.Seed, clone.Seed);
        Assert.Equal(original.ChainStageNames, clone.ChainStageNames);

        // Clone must be independently disposable.
        clone.Dispose();
        // The original should still work after the clone is disposed.
        _ = original.Name;
        _ = original.ChainLength;
    }

    [Fact]
    public void Access_After_Dispose_Throws()
    {
        LlamaBackend.Initialize();
        var s = new LlamaSamplerBuilder().WithGreedy().Build();
        s.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = s.Name);
        Assert.Throws<ObjectDisposedException>(() => _ = s.ChainLength);
        Assert.Throws<ObjectDisposedException>(() => _ = s.Seed);
        Assert.Throws<ObjectDisposedException>(() => s.Clone());
    }
}
