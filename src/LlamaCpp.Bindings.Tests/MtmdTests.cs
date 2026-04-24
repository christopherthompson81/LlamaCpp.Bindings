using LlamaCpp.Bindings;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Phase A (multimodal) smoke tests. The vision model + mmproj live under
/// <c>/mnt/data/models/</c> — the fixture gates on availability so the suite
/// stays green without them. Fetch with <c>tools/fetch-test-models.py --only
/// smolvlm-256m-text smolvlm-256m-mmproj</c>.
/// </summary>
public sealed class MtmdFixture : IDisposable
{
    private const string DefaultTextPath   = "/mnt/data/models/SmolVLM-256M-Instruct-Q8_0.gguf";
    private const string DefaultMmprojPath = "/mnt/data/models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf";

    public LlamaModel? Model { get; }
    public MtmdContext? Mtmd { get; }

    public MtmdFixture()
    {
        var textPath   = Environment.GetEnvironmentVariable("LLAMACPP_TEST_VISION_MODEL")  ?? DefaultTextPath;
        var mmprojPath = Environment.GetEnvironmentVariable("LLAMACPP_TEST_VISION_MMPROJ") ?? DefaultMmprojPath;

        if (!File.Exists(textPath) || !File.Exists(mmprojPath)) return;

        LlamaBackend.Initialize();
        Model = new LlamaModel(textPath, new LlamaModelParameters { GpuLayerCount = 0, UseMmap = true });
        // Warmup false keeps cold-start fast for fixture setup; production
        // callers keep the default (true).
        Mtmd = new MtmdContext(Model, mmprojPath, new MtmdContextParameters { UseGpu = false, Warmup = false });
    }

    public void SkipIfMissing()
    {
        Console.WriteLine(
            "SKIP: vision model/mmproj not found at /mnt/data/models/ — set " +
            "LLAMACPP_TEST_VISION_MODEL / _MMPROJ or run " +
            "tools/fetch-test-models.py --only smolvlm-256m-text smolvlm-256m-mmproj.");
    }

    public void Dispose()
    {
        Mtmd?.Dispose();
        Model?.Dispose();
    }
}

public class MtmdTests : IClassFixture<MtmdFixture>
{
    private readonly MtmdFixture _fx;
    public MtmdTests(MtmdFixture fx) => _fx = fx;

    private static string TestImagePath =>
        Path.Combine(AppContext.BaseDirectory, "TestData", "test-1.jpeg");

    [Fact]
    public void Context_Reports_Vision_Support()
    {
        if (_fx.Mtmd is null) { _fx.SkipIfMissing(); return; }

        Assert.True(_fx.Mtmd.SupportsVision, "SmolVLM mmproj should support vision");
        Assert.False(_fx.Mtmd.SupportsAudio, "SmolVLM is image-only");
        Assert.Null(_fx.Mtmd.AudioSampleRate);
        Assert.False(string.IsNullOrEmpty(_fx.Mtmd.DefaultMediaMarker),
            "mtmd_default_marker() should return a non-empty string");
    }

    [Fact]
    public void Bitmap_FromFile_Populates_Dimensions()
    {
        if (_fx.Mtmd is null) { _fx.SkipIfMissing(); return; }
        Assert.True(File.Exists(TestImagePath),
            $"test image missing at {TestImagePath} — check csproj TestData glob");

        using var bmp = MtmdBitmap.FromFile(_fx.Mtmd, TestImagePath);
        Assert.True(bmp.Width  > 0, "decoded bitmap should have a positive width");
        Assert.True(bmp.Height > 0, "decoded bitmap should have a positive height");
        Assert.False(bmp.IsAudio);
        Assert.Equal((long)bmp.Width * bmp.Height * 3, bmp.ByteCount);
    }

    [Fact]
    public void Bitmap_FromBytes_Equivalent_To_FromFile()
    {
        if (_fx.Mtmd is null) { _fx.SkipIfMissing(); return; }

        var bytes = File.ReadAllBytes(TestImagePath);
        using var fromBytes = MtmdBitmap.FromBytes(_fx.Mtmd, bytes);
        using var fromFile  = MtmdBitmap.FromFile (_fx.Mtmd, TestImagePath);

        Assert.Equal(fromFile.Width,  fromBytes.Width);
        Assert.Equal(fromFile.Height, fromBytes.Height);
        Assert.Equal(fromFile.ByteCount, fromBytes.ByteCount);
    }

    [Fact]
    public void Bitmap_FromPixels_Roundtrips_Exact_Size()
    {
        if (_fx.Mtmd is null) { _fx.SkipIfMissing(); return; }

        // 4x4 solid red RGB buffer = 48 bytes.
        var rgb = new byte[4 * 4 * 3];
        for (int i = 0; i < rgb.Length; i += 3) { rgb[i] = 255; rgb[i + 1] = 0; rgb[i + 2] = 0; }

        using var bmp = MtmdBitmap.FromPixels(4, 4, rgb);
        Assert.Equal(4, bmp.Width);
        Assert.Equal(4, bmp.Height);
        Assert.Equal(48, bmp.ByteCount);
    }

    [Fact]
    public void Bitmap_FromPixels_Rejects_Mismatched_Buffer()
    {
        if (_fx.Mtmd is null) { _fx.SkipIfMissing(); return; }
        // 4x4 needs 48 bytes; pass 40 to force the guard.
        var rgb = new byte[40];
        Assert.Throws<ArgumentException>(() => MtmdBitmap.FromPixels(4, 4, rgb));
    }

    [Fact]
    public async Task EvalPrompt_With_Image_Advances_NPast()
    {
        if (_fx.Mtmd is null || _fx.Model is null) { _fx.SkipIfMissing(); return; }

        using var lctx = new LlamaContext(_fx.Model, new LlamaContextParameters
        {
            ContextSize = 2048,
            LogicalBatchSize = 512,
            PhysicalBatchSize = 512,
        });
        using var bmp = MtmdBitmap.FromFile(_fx.Mtmd, TestImagePath);

        var prompt = $"{_fx.Mtmd.DefaultMediaMarker} describe.";
        var newNPast = await _fx.Mtmd.EvalPromptAsync(
            lctx, prompt, [bmp], nPast: 0,
            addSpecial: true, parseSpecial: true,
            cancellationToken: TestContext.Current.CancellationToken);

        // A 512x512-ish test image plus the surrounding text should consume
        // well over a handful of positions. The exact count depends on the
        // mmproj's token budget, so we only assert a lower bound.
        Assert.True(newNPast > 10,
            $"eval_chunks should advance n_past substantially; got {newNPast}");
    }

    [Fact]
    public async Task EvalPrompt_Rejects_Marker_Bitmap_Mismatch()
    {
        if (_fx.Mtmd is null || _fx.Model is null) { _fx.SkipIfMissing(); return; }

        using var lctx = new LlamaContext(_fx.Model, new LlamaContextParameters
        {
            ContextSize = 1024,
            LogicalBatchSize = 256,
            PhysicalBatchSize = 256,
        });
        using var bmp = MtmdBitmap.FromFile(_fx.Mtmd, TestImagePath);

        // Two markers, one bitmap — tokenize should return status 1.
        var prompt = $"{_fx.Mtmd.DefaultMediaMarker} and {_fx.Mtmd.DefaultMediaMarker}";
        var ex = await Assert.ThrowsAsync<LlamaException>(() =>
            _fx.Mtmd.EvalPromptAsync(lctx, prompt, [bmp], nPast: 0,
                cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal("mtmd_tokenize", ex.FunctionName);
        Assert.Equal(1, ex.StatusCode);
    }
}
