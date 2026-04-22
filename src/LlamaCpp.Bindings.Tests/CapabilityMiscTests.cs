namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Tier-1 C6: backend capability helpers, context runtime settings, extra
/// vocab + template helpers.
/// </summary>
public class CapabilityMiscTests
{
    [Fact]
    public void Backend_Reports_System_Info_And_Caps()
    {
        LlamaBackend.Initialize();

        // System info is a non-empty CPU-feature / GPU summary string.
        var info = LlamaBackend.SystemInfo();
        Assert.False(string.IsNullOrWhiteSpace(info));

        Assert.True(LlamaBackend.MaxParallelSequences() > 0);
        // SupportsRpc is build-dependent; just check that the call doesn't
        // throw and returns a valid bool.
        _ = LlamaBackend.SupportsRpc();
    }

    [Fact]
    public void BuiltInTemplateNames_Are_Non_Empty()
    {
        LlamaBackend.Initialize();
        var names = LlamaChatTemplate.BuiltInTemplateNames();
        Assert.NotEmpty(names);
        // At least one recognised template should be present (llama2, chatml,
        // mistral, etc.). Don't hard-code the list; just require some length.
        Assert.All(names, n => Assert.False(string.IsNullOrWhiteSpace(n)));
    }

    [Fact]
    public void InitializeNuma_Disabled_Does_Not_Throw()
    {
        LlamaBackend.Initialize();
        // "Disabled" is a safe no-op on every system.
        LlamaBackend.InitializeNuma(LlamaNumaStrategy.Disabled);
    }
}

[Collection(GpuCollection.Name)]
public class ContextRuntimeSettingsTests
{
    private readonly GpuGenerationFixture _fx;
    public ContextRuntimeSettingsTests(GpuGenerationFixture fx) => _fx = fx;

    [Fact]
    public void Thread_Counts_Are_Readable_And_Settable()
    {
        if (_fx.Context is null) { _fx.SkipMessage(); return; }
        var c = _fx.Context;

        int originalGen = c.ThreadCount;
        int originalBatch = c.BatchThreadCount;
        Assert.True(originalGen > 0);
        Assert.True(originalBatch > 0);

        c.SetThreadCounts(generationThreads: 2, batchThreads: 4);
        Assert.Equal(2, c.ThreadCount);
        Assert.Equal(4, c.BatchThreadCount);

        // Restore so this test doesn't bleed into later perf measurements.
        c.SetThreadCounts(originalGen, originalBatch);
    }

    [Fact]
    public void PoolingType_Reports_Default()
    {
        if (_fx.Context is null) { _fx.SkipMessage(); return; }
        var p = _fx.Context.PoolingType;
        // Any valid enum value. Generative models typically report None or Unspecified.
        Assert.InRange((int)p, -1, 4);
    }

    [Fact]
    public void SequenceContextSize_Matches_Or_Below_ContextSize()
    {
        if (_fx.Context is null) { _fx.SkipMessage(); return; }
        var c = _fx.Context;
        Assert.True(c.SequenceContextSize > 0);
        Assert.True(c.SequenceContextSize <= c.ContextSize);
    }

    [Fact]
    public void Synchronize_Does_Not_Throw()
    {
        if (_fx.Context is null) { _fx.SkipMessage(); return; }
        _fx.Context.Synchronize();
    }

    [Fact]
    public void Runtime_Flag_Setters_Run()
    {
        if (_fx.Context is null) { _fx.SkipMessage(); return; }
        var c = _fx.Context;
        // We don't assert behavioural consequences here — each setter just
        // mutates a flag the next decode consults. Verifying the flags'
        // *effects* requires a test for each (embeddings mode needs an
        // embeddings test; causal=false needs an encoder run). Those live
        // under Tier 2. Here we confirm the P/Invoke path doesn't throw.
        c.SetCausalAttention(true);
        c.SetWarmup(false);
        c.SetEmbeddingsMode(false);  // leave the context in generation mode
    }

    [Fact]
    public void DetokenizeNative_Roundtrip()
    {
        if (_fx.Context is null || _fx.Model is null) { _fx.SkipMessage(); return; }
        var v = _fx.Model.Vocab;

        const string text = "Hello, world.";
        var tokens = v.Tokenize(text, addSpecial: false, parseSpecial: false);
        var restored = v.DetokenizeNative(tokens, removeSpecial: false, unparseSpecial: false);
        Assert.Equal(text, restored);
    }
}
