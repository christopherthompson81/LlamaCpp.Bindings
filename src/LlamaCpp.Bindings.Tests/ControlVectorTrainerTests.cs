namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Smoke tests for <see cref="LlamaControlVectorTrainer"/>. We train a
/// small control vector against the cached test model with a handful of
/// short prompt pairs, then load it back through the existing
/// <see cref="LlamaControlVector"/> reader to verify the GGUF format is
/// what consumers expect.
/// </summary>
public class ControlVectorTrainerTests
{
    // Two short, contrasting prompt pairs. The Qwen3 tokenizer doesn't
    // need any special chat-template framing for the trainer to work —
    // we're just looking at residual-stream activation differences,
    // which exist for any pair of distinct inputs. Real-world cvectors
    // usually use chat-formatted prompts; keeping these plain keeps the
    // test fast and tokenizer-independent.
    private static readonly string[] PositivePrompts =
    {
        "I am extremely happy and full of joy today.",
        "What a wonderful, delightful, sunny morning!",
    };

    private static readonly string[] NegativePrompts =
    {
        "I am deeply sad and full of sorrow today.",
        "What a miserable, dreary, gloomy morning.",
    };

    [Fact]
    public async Task Trainer_Mean_Method_Writes_Loadable_ControlVector()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        using var model = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });

        var dir = MakeTempDir();
        try
        {
            var outPath = Path.Combine(dir, "happy.cvector.gguf");

            var result = await LlamaControlVectorTrainer.ComputeAsync(
                model,
                PositivePrompts,
                NegativePrompts,
                outPath,
                new LlamaControlVectorOptions
                {
                    Method = LlamaControlVectorMethod.Mean,
                });

            Assert.True(File.Exists(outPath));
            Assert.Equal(2, result.PromptPairCount);
            Assert.Equal(model.LayerCount - 1, result.LayerCount);
            Assert.Equal(model.EmbeddingSize, result.EmbeddingSize);
            Assert.Equal(LlamaControlVectorMethod.Mean, result.Method);

            // Round-trip through the existing reader.
            var cv = LlamaControlVector.LoadFromFile(outPath);
            Assert.Equal(model.EmbeddingSize, cv.NEmbd);
            // Reader pads up to the highest layer index found in the
            // file; for upstream-style output that's exactly n_layers - 1.
            Assert.Equal(model.LayerCount - 1, cv.LayerCount);
            Assert.Equal(cv.NEmbd * cv.LayerCount, cv.Data.Length);

            // Every direction should be (approximately) unit-normalized.
            // Layers absent from the file would be all-zero; we assert at
            // least one direction has unit length and none has length > 1+eps.
            int unitLayerCount = 0;
            for (int li = 0; li < cv.LayerCount; li++)
            {
                double sumSq = 0;
                for (int j = 0; j < cv.NEmbd; j++)
                {
                    var v = cv.Data[li * cv.NEmbd + j];
                    sumSq += (double)v * v;
                }
                double norm = Math.Sqrt(sumSq);
                Assert.True(norm <= 1.0 + 1e-3,
                    $"Layer {li + 1} has norm {norm:F4} > 1 — directions should be unit length.");
                if (norm > 0.99) unitLayerCount++;
            }
            Assert.True(unitLayerCount > 0,
                "No layer produced a unit-norm direction — collector likely failed to capture activations.");
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Trainer_Pca_Method_Writes_Loadable_ControlVector()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        using var model = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });

        var dir = MakeTempDir();
        try
        {
            var outPath = Path.Combine(dir, "happy.pca.cvector.gguf");

            var result = await LlamaControlVectorTrainer.ComputeAsync(
                model,
                PositivePrompts,
                NegativePrompts,
                outPath,
                new LlamaControlVectorOptions
                {
                    Method = LlamaControlVectorMethod.Pca,
                    // Keep iteration count modest for CI speed; PCA on
                    // ~2-4 prompt pairs converges quickly.
                    PcaIterations = 64,
                });

            Assert.True(File.Exists(outPath));
            Assert.Equal(LlamaControlVectorMethod.Pca, result.Method);

            var cv = LlamaControlVector.LoadFromFile(outPath);
            Assert.Equal(model.EmbeddingSize, cv.NEmbd);
            Assert.Equal(model.LayerCount - 1, cv.LayerCount);

            // PCA direction vs Mean direction should differ (otherwise
            // PCA isn't actually doing anything). Train both and compare.
            var meanPath = Path.Combine(dir, "happy.mean.cvector.gguf");
            await LlamaControlVectorTrainer.ComputeAsync(
                model, PositivePrompts, NegativePrompts, meanPath,
                new LlamaControlVectorOptions { Method = LlamaControlVectorMethod.Mean });
            var meanCv = LlamaControlVector.LoadFromFile(meanPath);

            // Cosine similarity per layer should be < 0.999 (i.e. not
            // identical) on at least some of them.
            int diverged = 0;
            for (int li = 0; li < cv.LayerCount; li++)
            {
                double dot = 0;
                for (int j = 0; j < cv.NEmbd; j++)
                {
                    dot += (double)cv.Data[li * cv.NEmbd + j] * meanCv.Data[li * cv.NEmbd + j];
                }
                if (Math.Abs(dot) < 0.999) diverged++;
            }
            Assert.True(diverged > 0, "PCA and Mean methods produced identical directions on every layer — PCA path likely no-op.");
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Trainer_Honors_Cancellation()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        using var model = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });

        var dir = MakeTempDir();
        try
        {
            using var cts = new CancellationTokenSource();
            cts.Cancel();
            var outPath = Path.Combine(dir, "cancelled.cvector.gguf");

            await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
                LlamaControlVectorTrainer.ComputeAsync(
                    model,
                    PositivePrompts,
                    NegativePrompts,
                    outPath,
                    options: null,
                    progress: null,
                    cancellationToken: cts.Token));

            Assert.False(File.Exists(outPath));
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Trainer_Rejects_Mismatched_Prompt_Counts()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        using var model = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });

        var dir = MakeTempDir();
        try
        {
            var outPath = Path.Combine(dir, "wont-write.cvector.gguf");
            await Assert.ThrowsAsync<ArgumentException>(() =>
                LlamaControlVectorTrainer.ComputeAsync(
                    model,
                    new[] { "happy" },
                    new[] { "sad", "extra" },
                    outPath));
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    private static string MakeTempDir()
    {
        var d = Path.Combine(Path.GetTempPath(), "llama-cvector-train-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    private static void DeleteDir(string d)
    {
        try { if (Directory.Exists(d)) Directory.Delete(d, recursive: true); } catch { /* best-effort */ }
    }
}
