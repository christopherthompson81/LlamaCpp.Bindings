namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// End-to-end tests for <see cref="LlamaQuantizer"/>. The default test model
/// is already quantized (Q6_K_XL) so we exercise the binding via the
/// "only-copy" path — no actual quantization, just a tensor copy. That's
/// enough to validate struct marshalling + return-code handling without
/// requiring a separate F16 download.
/// </summary>
public class QuantizerTests
{
    [Fact]
    public void Default_Quantization_Params_Round_Trip_Looks_Sane()
    {
        LlamaBackend.Initialize();
        var p = LlamaQuantizationParameters.Default();
        // llama_model_quantize_default_params reports ftype = Q5_1.
        Assert.Equal(LlamaFileType.Q5_1, p.FileType);
        Assert.Equal(0, p.ThreadCount);                 // 0 = hardware_concurrency
        Assert.True(p.QuantizeOutputTensor);            // native default is true
        Assert.False(p.AllowRequantize);
        Assert.False(p.OnlyCopy);
        Assert.False(p.Pure);
        Assert.False(p.KeepSplit);
        Assert.False(p.DryRun);
        // The two tensor-type overrides default to GGML_TYPE_COUNT (out of
        // our public enum range), which the binding surfaces as null.
        Assert.Null(p.OutputTensorType);
        Assert.Null(p.TokenEmbeddingType);
    }

    [Fact]
    public void Managed_Defaults_Match_Native_Defaults()
    {
        LlamaBackend.Initialize();
        var @new = new LlamaQuantizationParameters();
        var native = LlamaQuantizationParameters.Default();
        Assert.Equal(native.FileType, @new.FileType);
        Assert.Equal(native.QuantizeOutputTensor, @new.QuantizeOutputTensor);
        Assert.Equal(native.AllowRequantize, @new.AllowRequantize);
    }

    [Fact]
    public void Quantize_With_Missing_Input_Throws_FileNotFound()
    {
        LlamaBackend.Initialize();
        var dir = Path.Combine(Path.GetTempPath(), "llama-quantize-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var bogus = Path.Combine(dir, "does-not-exist.gguf");
            var output = Path.Combine(dir, "out.gguf");
            Assert.Throws<FileNotFoundException>(() =>
                LlamaQuantizer.Quantize(bogus, output));
        }
        finally
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public async Task QuantizeAsync_OnlyCopy_Produces_A_Loadable_Output()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        var dir = Path.Combine(Path.GetTempPath(), "llama-quantize-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var output = Path.Combine(dir, "copy.gguf");

            await LlamaQuantizer.QuantizeAsync(
                modelPath,
                output,
                new LlamaQuantizationParameters { OnlyCopy = true });

            Assert.True(File.Exists(output), $"Expected output GGUF at {output}");
            var inputSize = new FileInfo(modelPath).Length;
            var outputSize = new FileInfo(output).Length;
            // Only-copy should yield approximately the same byte count.
            // Allow a few KB of metadata jitter — the GGUF header rewriter
            // may re-emit padding differently between versions.
            var delta = Math.Abs(inputSize - outputSize);
            Assert.True(delta < 64 * 1024,
                $"Only-copy output size ({outputSize}) drifted from input ({inputSize}) by {delta} bytes — unexpected.");

            // Round-trip: load the output via the existing model loader.
            // If the file is malformed this throws.
            using var model = new LlamaModel(output, new LlamaModelParameters
            {
                GpuLayerCount = 0,
                UseMmap = true,
            });
            Assert.True(model.LayerCount > 0);
        }
        finally
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public async Task QuantizeAsync_Cancelled_Before_Start_Throws_Cancelled()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        var dir = Path.Combine(Path.GetTempPath(), "llama-quantize-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            using var cts = new CancellationTokenSource();
            cts.Cancel();
            var output = Path.Combine(dir, "wont-exist.gguf");
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
                LlamaQuantizer.QuantizeAsync(
                    modelPath, output,
                    new LlamaQuantizationParameters { OnlyCopy = true },
                    cts.Token));
            Assert.False(File.Exists(output), "Output file should not have been created on pre-flight cancellation.");
        }
        finally
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }
}
