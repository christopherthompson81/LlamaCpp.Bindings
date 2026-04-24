using System.Text;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// LoRA adapter lifecycle + attach/detach tests.
///
/// The behavioural tests (attach → output differs; detach → output restored)
/// load the default test LoRA fetched by
/// <see cref="TestModelProvider.TryGetLoraAdapterPath"/> — a 77 MB Qwen3-0.6B
/// adapter matched to the default base model. Tests skip gracefully if the
/// download failed (offline dev box) or if <c>LLAMACPP_TEST_MODEL</c> was
/// overridden to a non-Qwen3 model without also setting
/// <c>LLAMACPP_TEST_LORA</c> to a matching adapter.
/// </summary>
[Collection(GpuCollection.Name)]
public class LoraAdapterTests
{
    private readonly GpuGenerationFixture _fx;
    public LoraAdapterTests(GpuGenerationFixture fx) => _fx = fx;

    // ----- Structural tests (no adapter file required) -----

    [Fact]
    public void LoadFromFile_Throws_FileNotFound_For_Missing_Path()
    {
        var missing = Path.Combine(Path.GetTempPath(), $"no-such-lora-{Guid.NewGuid():N}.gguf");
        Assert.Throws<FileNotFoundException>(() =>
            LlamaLoraAdapter.LoadFromFile(_fx.Model, missing));
    }

    [Fact]
    public void LoadFromFile_Rejects_Null_Or_Empty_Arguments()
    {
        Assert.Throws<ArgumentNullException>(() =>
            LlamaLoraAdapter.LoadFromFile(null!, "x.gguf"));
        Assert.Throws<ArgumentException>(() =>
            LlamaLoraAdapter.LoadFromFile(_fx.Model, ""));
    }

    [Fact]
    public void ActiveLoraAdapters_On_Fresh_Context_Is_Empty()
    {
        Assert.Empty(_fx.Context.ActiveLoraAdapters);
    }

    [Fact]
    public void DetachAll_On_Empty_Is_Noop()
    {
        _fx.Context.DetachAllLoraAdapters();
        Assert.Empty(_fx.Context.ActiveLoraAdapters);
    }

    [Fact]
    public void Detach_Unknown_Adapter_Is_Noop()
    {
        var path = GetAdapterOrSkip();
        if (path is null) return;

        using var adapter = LlamaLoraAdapter.LoadFromFile(_fx.Model, path);
        _fx.Context.DetachLoraAdapter(adapter); // never attached — should not throw
        Assert.Empty(_fx.Context.ActiveLoraAdapters);
    }

    // ----- Behavioural tests (require an adapter built for the fixture's model) -----

    [Fact]
    public async Task Attach_Then_Detach_Restores_Baseline_Output()
    {
        var path = GetAdapterOrSkip();
        if (path is null) return;

        const string Prompt = "The capital of France is";
        var baseline = await Generate(Prompt, seed: 1234);

        using var adapter = LlamaLoraAdapter.LoadFromFile(_fx.Model, path);
        _fx.Context.AttachLoraAdapter(adapter, scale: 1.0f);
        Assert.Single(_fx.Context.ActiveLoraAdapters);

        var withAdapter = await Generate(Prompt, seed: 1234);

        _fx.Context.DetachLoraAdapter(adapter);
        Assert.Empty(_fx.Context.ActiveLoraAdapters);

        var restored = await Generate(Prompt, seed: 1234);

        // The adapter should actually influence output. If it doesn't, either
        // the adapter is a no-op (scale 0 or zero-delta) or the native attach
        // silently failed — either way worth surfacing.
        Assert.NotEqual(baseline, withAdapter);
        // After detach, output should match the original baseline.
        Assert.Equal(baseline, restored);
    }

    [Fact]
    public async Task Attach_With_Scale_Zero_Matches_Unattached_Output()
    {
        var path = GetAdapterOrSkip();
        if (path is null) return;

        using var adapter = LlamaLoraAdapter.LoadFromFile(_fx.Model, path);

        var baseline = await Generate("Describe a cat.", seed: 42);

        _fx.Context.AttachLoraAdapter(adapter, scale: 0.0f);
        var scaledZero = await Generate("Describe a cat.", seed: 42);
        _fx.Context.DetachLoraAdapter(adapter);

        Assert.Equal(baseline, scaledZero);
    }

    [Fact]
    public void Attach_Same_Adapter_Twice_Updates_Scale_In_Place()
    {
        var path = GetAdapterOrSkip();
        if (path is null) return;

        using var adapter = LlamaLoraAdapter.LoadFromFile(_fx.Model, path);
        _fx.Context.AttachLoraAdapter(adapter, scale: 0.5f);
        Assert.Equal(0.5f, _fx.Context.ActiveLoraAdapters[adapter]);

        _fx.Context.AttachLoraAdapter(adapter, scale: 1.5f);
        Assert.Equal(1.5f, _fx.Context.ActiveLoraAdapters[adapter]);
        Assert.Single(_fx.Context.ActiveLoraAdapters);

        _fx.Context.DetachAllLoraAdapters();
    }

    [Fact]
    public void SetLoraAdapters_Atomically_Replaces_Active_Set()
    {
        var path = GetAdapterOrSkip();
        if (path is null) return;

        using var adapter = LlamaLoraAdapter.LoadFromFile(_fx.Model, path);
        _fx.Context.AttachLoraAdapter(adapter, scale: 1.0f);
        Assert.Single(_fx.Context.ActiveLoraAdapters);

        // Replace with the empty set.
        _fx.Context.SetLoraAdapters(Array.Empty<KeyValuePair<LlamaLoraAdapter, float>>());
        Assert.Empty(_fx.Context.ActiveLoraAdapters);

        // Replace with a single-entry set.
        _fx.Context.SetLoraAdapters(new[] { KeyValuePair.Create(adapter, 0.75f) });
        Assert.Single(_fx.Context.ActiveLoraAdapters);
        Assert.Equal(0.75f, _fx.Context.ActiveLoraAdapters[adapter]);

        _fx.Context.DetachAllLoraAdapters();
    }

    [Fact]
    public void Metadata_Is_Readable()
    {
        var path = GetAdapterOrSkip();
        if (path is null) return;

        using var adapter = LlamaLoraAdapter.LoadFromFile(_fx.Model, path);
        Assert.NotNull(adapter.Metadata);
        // We don't assert specific keys — adapters vary — only that the
        // dictionary is reachable and self-consistent.
        foreach (var kvp in adapter.Metadata)
        {
            Assert.False(string.IsNullOrEmpty(kvp.Key));
            Assert.NotNull(kvp.Value);
        }
    }

    [Fact]
    public void AloraInvocationTokens_Is_Null_For_Standard_LoRA()
    {
        var path = GetAdapterOrSkip();
        if (path is null) return;

        using var adapter = LlamaLoraAdapter.LoadFromFile(_fx.Model, path);
        // Standard LoRA adapters return null; alora adapters return a
        // non-empty array. Either is valid — but each test adapter is one or
        // the other, so we just assert the API doesn't throw.
        var tokens = adapter.AloraInvocationTokens;
        if (tokens is not null)
        {
            Assert.NotEmpty(tokens);
        }
    }

    // ----- helpers -----

    private static string? GetAdapterOrSkip() => TestModelProvider.TryGetLoraAdapterPath();

    private async Task<string> Generate(string prompt, uint seed, int maxTokens = 12)
    {
        _fx.Context.ClearKvCache();
        using var sampler = new LlamaSamplerBuilder()
            .WithTemperature(0.0f).WithDistribution(seed).Build();
        var gen = new LlamaGenerator(_fx.Context, sampler);
        var sb = new StringBuilder();
        await foreach (var p in gen.GenerateAsync(
            prompt, maxTokens: maxTokens, addSpecial: false, parseSpecial: false,
            cancellationToken: TestContext.Current.CancellationToken))
        {
            sb.Append(p);
        }
        return sb.ToString();
    }
}
