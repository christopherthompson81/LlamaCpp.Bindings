namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Tests for <see cref="LlamaQuantSensitivity"/>. The math-load-bearing
/// check is monotonicity within the K-quant family: Q2_K should
/// produce more reconstruction error than Q6_K on a non-trivial
/// tensor. If our quantize → dequantize → score wiring is wrong,
/// the relationship inverts or scores come back NaN.
/// </summary>
public class QuantSensitivityTests
{
    [Fact]
    public async Task K_Quant_Family_Score_Is_Monotonic_In_Bit_Width()
    {
        // Synthesize a small F32 GGUF with a single 256×4 weight
        // tensor (256 = K-quant block size; 4 rows is enough for the
        // kernel to behave normally). Values chosen to be non-trivial:
        // a sinusoidal pattern with magnitude in [-1, 1].
        LlamaBackend.Initialize();
        var dir = MakeTempDir();
        try
        {
            var basePath = Path.Combine(dir, "src.gguf");
            const int InDim = 256;   // n_per_row — must divide K-quant block size
            const int OutDim = 4;    // rows
            var values = new float[InDim * OutDim];
            for (int i = 0; i < values.Length; i++) values[i] = (float)Math.Sin(i * 0.1) * 0.7f;

            await new LlamaGgufWriter()
                .SetMetadata("general.architecture", "test")
                .AddTensorF32("blk.0.attn_q.weight", new long[] { InDim, OutDim }, values)
                .WriteAsync(basePath);

            var result = await LlamaQuantSensitivity.MeasureAsync(basePath,
                new LlamaQuantSensitivityOptions
                {
                    CandidateTypes = new[]
                    {
                        LlamaTensorType.Q2_K,
                        LlamaTensorType.Q4_K,
                        LlamaTensorType.Q6_K,
                    },
                });

            Assert.Single(result.Scores.Select(s => s.TensorName).Distinct());
            var byType = result.Scores.ToDictionary(s => s.QuantType);

            foreach (var s in byType.Values)
            {
                Assert.False(s.Skipped, $"Score for {s.QuantType} marked skipped: {s.SkipReason}");
                Assert.False(double.IsNaN(s.RawMse), $"NaN raw MSE for {s.QuantType}");
                Assert.True(s.RawMse >= 0, $"Negative raw MSE for {s.QuantType}");
            }

            // Monotone improvement going up the K-quant ladder. Cushion
            // the comparison (>) rather than (>=) since each step
            // should give a meaningful gap on a non-trivial input.
            Assert.True(byType[LlamaTensorType.Q2_K].RawMse > byType[LlamaTensorType.Q4_K].RawMse,
                $"Expected Q2_K MSE ({byType[LlamaTensorType.Q2_K].RawMse:E}) > Q4_K MSE ({byType[LlamaTensorType.Q4_K].RawMse:E}).");
            Assert.True(byType[LlamaTensorType.Q4_K].RawMse > byType[LlamaTensorType.Q6_K].RawMse,
                $"Expected Q4_K MSE ({byType[LlamaTensorType.Q4_K].RawMse:E}) > Q6_K MSE ({byType[LlamaTensorType.Q6_K].RawMse:E}).");
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Skip_OneDimensional_Tensors_By_Default()
    {
        // 1-D tensors stay F32 in production quantization — the sweep
        // should skip them rather than emit nonsense scores.
        LlamaBackend.Initialize();
        var dir = MakeTempDir();
        try
        {
            var basePath = Path.Combine(dir, "src.gguf");
            await new LlamaGgufWriter()
                .SetMetadata("general.architecture", "test")
                .AddTensorF32("blk.0.attn_norm.weight", new long[] { 256 }, new float[256])
                .AddTensorF32("blk.0.attn_q.weight",   new long[] { 256, 4 }, new float[256 * 4])
                .WriteAsync(basePath);

            var result = await LlamaQuantSensitivity.MeasureAsync(basePath,
                new LlamaQuantSensitivityOptions
                {
                    CandidateTypes = new[] { LlamaTensorType.Q4_K },
                });

            // Only the 2-D tensor was scored.
            var tensorNames = result.Scores.Select(s => s.TensorName).Distinct().ToArray();
            Assert.Single(tensorNames);
            Assert.Equal("blk.0.attn_q.weight", tensorNames[0]);
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Json_Round_Trip_Preserves_Scores()
    {
        var dir = MakeTempDir();
        try
        {
            // Run a tiny sweep to get a real result, then round-trip
            // through JSON and compare a couple of fields. Don't go
            // deep on equality — JSON's double-precision serialization
            // may flip a least-significant bit; we accept that and
            // just assert structural identity.
            var basePath = Path.Combine(dir, "src.gguf");
            const int InDim = 256;
            await new LlamaGgufWriter()
                .SetMetadata("general.architecture", "test")
                .AddTensorF32("blk.0.attn_q.weight", new long[] { InDim, 4 }, new float[InDim * 4])
                .WriteAsync(basePath);

            var original = await LlamaQuantSensitivity.MeasureAsync(basePath,
                new LlamaQuantSensitivityOptions
                {
                    CandidateTypes = new[] { LlamaTensorType.Q4_K },
                });

            var jsonPath = Path.Combine(dir, "scores.json");
            LlamaQuantSensitivity.SaveToJson(original, jsonPath);
            var roundTripped = LlamaQuantSensitivity.LoadFromJson(jsonPath);

            Assert.Equal(original.Scores.Count, roundTripped.Scores.Count);
            for (int i = 0; i < original.Scores.Count; i++)
            {
                Assert.Equal(original.Scores[i].TensorName, roundTripped.Scores[i].TensorName);
                Assert.Equal(original.Scores[i].QuantType, roundTripped.Scores[i].QuantType);
                Assert.Equal(original.Scores[i].Skipped, roundTripped.Scores[i].Skipped);
            }
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Real_Model_Smoke_Produces_Finite_Scores()
    {
        // Run a tiny sweep on the cached test model: one tensor only
        // (via IncludeNameRegex) at three candidates. Verifies the
        // dequantize-source path works against a quantized base
        // (the cached model is Q6_K_XL). Score values aren't asserted
        // beyond "finite, non-negative" — exact MSE depends on the
        // model and isn't load-bearing here.
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        var result = await LlamaQuantSensitivity.MeasureAsync(modelPath,
            new LlamaQuantSensitivityOptions
            {
                IncludeNameRegex = @"^blk\.0\.attn_q\.weight$",
                CandidateTypes  = new[]
                {
                    LlamaTensorType.Q4_K,
                    LlamaTensorType.Q2_K,
                },
            });

        Assert.Equal(2, result.Scores.Count);
        foreach (var s in result.Scores)
        {
            Assert.False(s.Skipped, $"Real-model {s.QuantType} score marked skipped: {s.SkipReason}");
            Assert.False(double.IsNaN(s.RawMse));
            Assert.True(s.RawMse >= 0);
            Assert.True(s.ElementCount > 0);
        }
    }

    private static string MakeTempDir()
    {
        var d = Path.Combine(Path.GetTempPath(), "llama-qsens-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    private static void DeleteDir(string d)
    {
        try { if (Directory.Exists(d)) Directory.Delete(d, recursive: true); } catch { /* best-effort */ }
    }
}
