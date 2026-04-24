using LlamaCpp.Bindings.LlamaChat.Services;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// End-to-end smoke test for the auto-configure pipeline. Exercises the
/// real model probe (briefly loads the fixture GGUF via mmap + 0 GPU
/// layers) and the real hardware enumeration, then sanity-checks that the
/// produced <see cref="AutoConfigureResult"/> is internally consistent.
///
/// Lives in the GPU collection to share the fixture's backend init
/// (<see cref="LlamaBackend"/>), not because it needs GPU access itself —
/// the probe runs with <c>GpuLayerCount=0</c>.
/// </summary>
[Collection(GpuCollection.Name)]
public class AutoConfigureServiceTests
{
    private readonly GpuGenerationFixture _fx;
    public AutoConfigureServiceTests(GpuGenerationFixture fx) => _fx = fx;

    /// <summary>
    /// Load the shipped sampling DB from disk so the service can run
    /// without standing up an Avalonia resource system in the test host.
    /// </summary>
    private static SamplingProfileDatabase LoadShippedDb()
    {
        var d = new DirectoryInfo(AppContext.BaseDirectory);
        while (d is not null && !File.Exists(Path.Combine(d.FullName, "LlamaCpp.Bindings.slnx")))
            d = d.Parent;
        Assert.NotNull(d);
        var path = Path.Combine(
            d!.FullName, "src", "LlamaCpp.Bindings.LlamaChat", "Assets", "sampling-profiles.json");
        return SamplingProfileDb.Parse(File.ReadAllText(path));
    }

    [Fact]
    public void Configure_Produces_Usable_Load_Settings_For_Test_Model()
    {
        var path = TestModelProvider.EnsureModelPath();
        var result = AutoConfigureService.Configure(path, LoadShippedDb());

        // Load settings — basic sanity across every field.
        Assert.Equal(path, result.Load.ModelPath);
        Assert.True(result.Load.ContextSize >= 4096, $"context too small: {result.Load.ContextSize}");
        // Ceiling is MaxContextSize (1 M). On a small model with lots of free
        // VRAM (the usual fixture case), the heuristic pushes right up to or
        // near that ceiling — the whole point of "maximize context".
        Assert.True(result.Load.ContextSize <= 1_048_576, $"context too large: {result.Load.ContextSize}");

        // GPU layer count must be either -1 (all) or in [0, n_layers]. The
        // fixture's base model has a known layer count; read it fresh to
        // avoid hard-coding.
        var nLayers = _fx.Model.LayerCount;
        Assert.True(
            result.Load.GpuLayerCount == -1 ||
            (result.Load.GpuLayerCount >= 0 && result.Load.GpuLayerCount <= nLayers),
            $"gpu layers out of range: {result.Load.GpuLayerCount} (n_layers={nLayers})");

        // mmap should default on; mlock default off.
        Assert.True(result.Load.UseMmap);
        Assert.False(result.Load.UseMlock);

        // KV cache K and V always match in the current heuristic (one
        // dropdown in the UI). If that invariant changes, this test should
        // be updated alongside.
        Assert.Equal(result.Load.KvCacheTypeK, result.Load.KvCacheTypeV);

        // Sampler values should fall in plausible ranges.
        var s = result.Sampler;
        Assert.InRange(s.Temperature, 0.01f, 2.0f);
        Assert.True(s.TopP is null or (>= 0.0f and <= 1.0f));
        Assert.True(s.TopK is null or >= 0);
        Assert.True(s.MinP is null or (>= 0.0f and <= 1.0f));

        // Explanation is human-readable — at minimum non-empty and mentions
        // the sampler profile id that was applied.
        Assert.False(string.IsNullOrWhiteSpace(result.Explanation));
        Assert.Contains("Sampler:", result.Explanation);
    }

    [Fact]
    public void Configure_Hits_Qwen3_Sampling_Profile_For_Test_Model()
    {
        // The bundled test model is Qwen3-0.6B — it should land on either
        // the thinking variant (if the GGUF name hints at that) or the
        // generic Qwen3 entry. Either is correct; fallback would be wrong.
        var path = TestModelProvider.EnsureModelPath();
        var result = AutoConfigureService.Configure(path, LoadShippedDb());

        Assert.Contains("qwen3", result.Explanation, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Configure_Throws_FileNotFound_For_Missing_Path()
    {
        var missing = Path.Combine(Path.GetTempPath(), $"no-such-{Guid.NewGuid():N}.gguf");
        Assert.Throws<FileNotFoundException>(() =>
            AutoConfigureService.Configure(missing, LoadShippedDb()));
    }

    [Fact]
    public void Configure_Throws_ArgumentException_For_Empty_Path()
    {
        Assert.Throws<ArgumentException>(() =>
            AutoConfigureService.Configure("", LoadShippedDb()));
    }
}
