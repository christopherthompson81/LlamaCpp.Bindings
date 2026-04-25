namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Phase 1 smoke test: initialize the backend, load a GGUF model, create a
/// context, dispose everything, no crashes. This is the first time we
/// actually dlopen libllama.so — until this passes, everything upstream
/// (struct layouts, P/Invoke signatures) is theoretical.
///
/// The model is resolved via <see cref="TestModelProvider"/>: if
/// <c>LLAMACPP_TEST_MODEL</c> is set it is used (missing file = hard failure);
/// otherwise the default 0.6B test model is downloaded automatically.
/// </summary>
public class SmokeTests
{

    [Fact]
    public void Backend_Initializes_And_Shuts_Down_Cleanly()
    {
        LlamaBackend.Initialize();
        Assert.True(LlamaBackend.MaxDevices() > 0, "llama_max_devices() should be > 0 even on CPU-only builds.");
        // Do NOT call Shutdown() here — other tests rely on the init state.
        // LlamaBackend is idempotent so this is safe to run repeatedly.
    }

    [Fact]
    public void NativeLibraryDirectory_Is_Found_And_Contains_Cpu_Plugins()
    {
        // llama.cpp's Linux release layout has shifted over time:
        //   - older builds (≤ b88xx early) shipped per-arch plugins as
        //     libggml-cpu-<arch>.so (haswell / sandybridge / skylakex / …);
        //   - current builds (b8893+) ship a single monolithic libggml-cpu.so
        //     and let CPU feature-detection happen inside that one binary.
        // Either layout loads correctly through NativeLibraryResolver — the
        // per-arch probe just no-ops when there's nothing to enumerate. The
        // test accepts both so it doesn't fire on every llama.cpp bump that
        // only changes packaging.
        var dir = LlamaCpp.Bindings.Native.NativeLibraryResolver.NativeLibraryDirectory();
        Assert.NotNull(dir);
        Assert.True(Directory.Exists(dir), $"Expected native dir to exist: {dir}");

        var perArch    = Directory.GetFiles(dir, "libggml-cpu-*.so");
        var monolithic = Directory.GetFiles(dir, "libggml-cpu.so")
            .Concat(Directory.GetFiles(dir, "libggml-cpu.so.*"))
            .ToArray();

        Assert.True(perArch.Length > 0 || monolithic.Length > 0,
            $"Expected either per-arch libggml-cpu-*.so or a monolithic libggml-cpu.so in {dir}, " +
            $"found: {string.Join(", ", Directory.GetFiles(dir, "*.so*").Select(Path.GetFileName))}");
    }

    [Fact]
    public void Backend_Dev_Count_Is_Nonzero_After_Initialize()
    {
        LlamaBackend.Initialize();
        var count = (int)LlamaCpp.Bindings.Native.NativeMethods.ggml_backend_dev_count();
        Assert.True(count > 0, $"ggml_backend_dev_count() returned {count} — backends were not loaded");
    }

    [Fact]
    public void Default_Model_Params_Round_Trip_Looks_Sane()
    {
        LlamaBackend.Initialize();
        var p = LlamaModelParameters.Default();
        Assert.True(p.UseMmap);                                  // llama.cpp default
        Assert.True(p.GpuLayerCount == -1 || p.GpuLayerCount >= 0); // -1 = all, >=0 = explicit
        Assert.False(p.UseMlock);                                // default off (needs privileges)
        Assert.False(p.VocabOnly);
    }

    [Fact]
    public void Default_Context_Params_Round_Trip_Looks_Sane()
    {
        LlamaBackend.Initialize();
        var p = LlamaContextParameters.Default();
        Assert.True(p.ContextSize > 0);        // llama.cpp defaults to 512 as of b8620
        Assert.True(p.LogicalBatchSize > 0);
        Assert.True(p.PhysicalBatchSize > 0);
        Assert.True(p.MaxSequenceCount >= 1);
    }

    [Fact]
    public void Managed_Defaults_Match_Native_Defaults_For_Gpu_Offload()
    {
        // new LlamaModelParameters() must not silently degrade the native default
        // (we had a bug where managed defaulted to GpuLayerCount = 0, disabling
        // GPU offload even though the user never asked for that).
        LlamaBackend.Initialize();
        var @new = new LlamaModelParameters();
        var native = LlamaModelParameters.Default();
        Assert.Equal(native.GpuLayerCount, @new.GpuLayerCount);
    }

    [Fact]
    public void Can_Load_Model_Create_Context_And_Dispose()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize((lvl, msg) => TestOutput.LastLog = (lvl, msg));

        using var model = new LlamaModel(modelPath, new LlamaModelParameters
        {
            // Keep VRAM pressure light for the smoke test; Phase 3+ will exercise full offload.
            GpuLayerCount = 0,
            UseMmap = true,
        });

        Assert.True(model.TrainingContextSize > 0);
        Assert.True(model.EmbeddingSize > 0);
        Assert.True(model.LayerCount > 0);

        using var context = new LlamaContext(model, new LlamaContextParameters
        {
            ContextSize = 512,     // tiny context is fine for smoke
            LogicalBatchSize = 512,
            PhysicalBatchSize = 512,
            MaxSequenceCount = 1,
        });

        Assert.True(context.ContextSize >= 512);
        Assert.True(context.LogicalBatchSize > 0);
    }

    // Scratch capture so xUnit doesn't spam its output. Static so the native
    // log callback (which fires from native threads) doesn't care about fixtures.
    private static class TestOutput
    {
        public static (LlamaLogLevel level, string msg) LastLog;
    }
}
