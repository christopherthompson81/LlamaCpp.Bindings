namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Phase 3: end-to-end generation. Builder validation is testable anywhere;
/// actual token production requires a loaded model. Uses a dedicated GPU
/// fixture because running a 35B MoE on CPU for 50 tokens costs minutes —
/// with GPU offload, the same test finishes in 1-2 seconds.
/// </summary>
[Collection(GpuCollection.Name)]
public class GenerationTests
{
    private readonly GpuGenerationFixture _fx;

    public GenerationTests(GpuGenerationFixture fx) => _fx = fx;

    // ----- Builder validation (model-independent) -----

    [Fact]
    public void Builder_Rejects_Empty_Chain()
    {
        LlamaBackend.Initialize();
        Assert.Throws<InvalidOperationException>(() => new LlamaSamplerBuilder().Build());
    }

    [Fact]
    public void Builder_Rejects_Chain_Without_Terminal()
    {
        LlamaBackend.Initialize();
        Assert.Throws<InvalidOperationException>(() =>
            new LlamaSamplerBuilder().WithTopK(40).WithTemperature(0.7f).Build());
    }

    [Fact]
    public void Builder_Rejects_Stage_After_Terminal()
    {
        LlamaBackend.Initialize();
        var b = new LlamaSamplerBuilder().WithDistribution(seed: 1);
        Assert.Throws<InvalidOperationException>(() => b.WithTopK(40));
    }

    [Fact]
    public void Builder_Produces_Disposable_Sampler()
    {
        LlamaBackend.Initialize();
        using var s = new LlamaSamplerBuilder().WithGreedy().Build();
        Assert.NotNull(s);
    }

    // ----- End-to-end generation -----

    [Fact]
    public async Task Greedy_Generation_Produces_Plausible_Output()
    {
        if (_fx.Model is null || _fx.Context is null) { _fx.SkipMessage(); return; }

        // Greedy (argmax) should produce deterministic-ish output — but on CUDA
        // with an MoE model (Qwen3 A3B here) it is NOT bit-reproducible across
        // separate context instances: expert routing + kernel non-determinism
        // diverge. We assert only that greedy produces non-empty output; seeded
        // reproducibility is covered by the companion Distribution_* test below.
        var prompt = BuildPrompt(_fx.Model, "What is 2 + 2? Answer with just a number.");
        _fx.ResetContextFor(prompt);

        var run = await Collect(_fx.Model, _fx.Context, prompt, () =>
            new LlamaSamplerBuilder().WithGreedy().Build(), maxTokens: 80);

        Assert.False(string.IsNullOrWhiteSpace(run));
    }

    [Fact]
    public async Task Distribution_Generation_With_Fixed_Seed_Is_Reproducible()
    {
        if (_fx.Model is null || _fx.Context is null) { _fx.SkipMessage(); return; }

        var prompt = BuildPrompt(_fx.Model, "Write the word 'cat' exactly once.");
        const uint Seed = 42;

        // Both runs must start from identical KV state. The original version
        // only reset *between* runs; under Qwen3 the model converged anyway,
        // but under any other model the leftover state from prior tests
        // diverged the two outputs and the equality assertion fired.
        _fx.ResetContextFor(prompt);
        var run1 = await Collect(_fx.Model, _fx.Context, prompt, () =>
            new LlamaSamplerBuilder()
                .WithTopK(40).WithTopP(0.9f).WithMinP(0.05f).WithTemperature(0.7f)
                .WithDistribution(Seed).Build(),
            maxTokens: 30);

        _fx.ResetContextFor(prompt);
        var run2 = await Collect(_fx.Model, _fx.Context, prompt, () =>
            new LlamaSamplerBuilder()
                .WithTopK(40).WithTopP(0.9f).WithMinP(0.05f).WithTemperature(0.7f)
                .WithDistribution(Seed).Build(),
            maxTokens: 30);

        Assert.Equal(run1, run2);
        Assert.False(string.IsNullOrWhiteSpace(run1));
    }

    [Fact]
    public async Task Generation_Produces_Nonempty_Output_In_Under_Token_Budget()
    {
        if (_fx.Model is null || _fx.Context is null) { _fx.SkipMessage(); return; }

        var prompt = BuildPrompt(_fx.Model, "Say hello.");
        _fx.ResetContextFor(prompt);

        var sampler = new LlamaSamplerBuilder()
            .WithTopK(40).WithTopP(0.9f).WithMinP(0.05f).WithTemperature(0.7f)
            .WithDistribution(seed: 123)
            .Build();

        var generator = new LlamaGenerator(_fx.Context, sampler);

        var pieces = new List<string>();
        await foreach (var p in generator.GenerateAsync(prompt, maxTokens: 50, addSpecial: false, parseSpecial: true))
        {
            pieces.Add(p);
            if (pieces.Count >= 50) break; // belt-and-suspenders
        }

        var output = string.Concat(pieces);
        Assert.False(string.IsNullOrWhiteSpace(output),
            "Generator produced no non-whitespace output. Model/prompt/sampler may be broken.");
        Assert.InRange(pieces.Count, 1, 60);

        sampler.Dispose();
    }

    [Fact]
    public async Task Generation_Respects_Cancellation_Token()
    {
        if (_fx.Model is null || _fx.Context is null) { _fx.SkipMessage(); return; }

        var prompt = BuildPrompt(_fx.Model, "Count to one thousand slowly.");
        _fx.ResetContextFor(prompt);

        using var sampler = new LlamaSamplerBuilder()
            .WithTopK(40).WithTemperature(0.7f).WithDistribution(seed: 7).Build();
        var generator = new LlamaGenerator(_fx.Context, sampler);

        using var cts = new CancellationTokenSource();
        int received = 0;

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var _ in generator.GenerateAsync(prompt, maxTokens: 500, cancellationToken: cts.Token))
            {
                received++;
                if (received == 3) cts.Cancel();
            }
        });

        Assert.True(received >= 3, "Should have streamed at least 3 pieces before cancellation fired.");
    }

    // ----- helpers -----

    private static string BuildPrompt(LlamaModel model, string userMessage)
    {
        var tmpl = model.GetChatTemplate();
        if (string.IsNullOrEmpty(tmpl))
        {
            // Fallback for models lacking a template: naked user text.
            return userMessage;
        }
        return LlamaChatTemplate.Apply(tmpl, new[]
        {
            new ChatMessage("user", userMessage),
        }, addAssistantPrefix: true);
    }

    private static async Task<string> Collect(
        LlamaModel model, LlamaContext ctx, string prompt,
        Func<LlamaSampler> mkSampler, int maxTokens)
    {
        _ = model;
        using var sampler = mkSampler();
        var gen = new LlamaGenerator(ctx, sampler);
        var sb = new System.Text.StringBuilder();
        await foreach (var piece in gen.GenerateAsync(prompt, maxTokens, addSpecial: false, parseSpecial: true))
        {
            sb.Append(piece);
        }
        return sb.ToString();
    }
}

/// <summary>
/// xUnit collection that shares a single <see cref="GpuGenerationFixture"/>
/// across every test class that uses <c>[Collection(GpuCollection.Name)]</c>.
/// Without this, each test class would attempt its own model load and blow
/// past the 3090's VRAM budget.
/// </summary>
[CollectionDefinition(GpuCollection.Name)]
public sealed class GpuCollection : ICollectionFixture<GpuGenerationFixture>
{
    public const string Name = "GpuGenerationFixture";
}

/// <summary>
/// Loads the smoke-test GGUF with full GPU offload and builds a fresh context
/// per test (via <see cref="ResetContextFor"/>) so KV state doesn't leak
/// between tests that share the fixture.
/// </summary>
public sealed class GpuGenerationFixture : IDisposable
{
    private const string DefaultModelPath = "/mnt/data/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf";

    public LlamaModel? Model { get; }
    public LlamaContext? Context { get; private set; }
    public ModelCapabilities Capabilities { get; }

    public GpuGenerationFixture()
    {
        var path = Environment.GetEnvironmentVariable("LLAMACPP_TEST_MODEL");
        if (string.IsNullOrWhiteSpace(path)) path = DefaultModelPath;
        if (!File.Exists(path))
        {
            Capabilities = ModelCapabilities.Empty;
            return;
        }

        LlamaBackend.Initialize();
        Model = new LlamaModel(path, new LlamaModelParameters
        {
            GpuLayerCount = -1, // all layers on GPU for real generation
            UseMmap = true,
        });
        Context = BuildContext();
        Capabilities = ModelCapabilities.Probe(Model);
    }

    public void SkipMessage()
    {
        Console.WriteLine(
            $"SKIP: Generation tests require {DefaultModelPath} (or $LLAMACPP_TEST_MODEL).");
    }

    /// <summary>
    /// Reset KV cache so each test starts fresh. Phase 4 replaced the original
    /// Dispose+recreate hack with a proper ClearKvCache call.
    /// </summary>
    public void ResetContextFor(string prompt)
    {
        _ = prompt;
        Context?.ClearKvCache();
    }

    private LlamaContext BuildContext()
    {
        return new LlamaContext(Model!, new LlamaContextParameters
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
        Context?.Dispose();
        Model?.Dispose();
    }
}
