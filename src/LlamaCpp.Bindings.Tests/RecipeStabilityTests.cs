namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// End-to-end stability tests for the sensitivity-sweep + recipe-build
/// pipeline. Existing tests cover the math (monotonicity, threshold
/// logic) and JSON round-trip; this one's job is to catch silent
/// regressions where a future optimization, an upstream kernel
/// improvement, or a parallel-ordering change perturbs the per-tensor
/// rel-MSE enough to flip recipe contents — same input, same options,
/// different recipe.
/// </summary>
public class RecipeStabilityTests
{
    /// <summary>
    /// Two runs of the same sweep + recipe-build on the same input
    /// must produce identical recipes. Catches any future change that
    /// introduces non-determinism: parallel reduction reorder, RNG in
    /// a sub-sampling path, etc.
    /// </summary>
    [Fact]
    public async Task Sweep_Plus_Recipe_Build_Is_Deterministic()
    {
        LlamaBackend.Initialize();
        var dir = MakeTempDir();
        try
        {
            var basePath = Path.Combine(dir, "src.gguf");
            await WriteFixtureGgufAsync(basePath);

            var options = new LlamaQuantSensitivityOptions
            {
                CandidateTypes = new[]
                {
                    LlamaTensorType.Q2_K,
                    LlamaTensorType.Q4_K,
                    LlamaTensorType.Q6_K,
                },
            };

            var run1 = await LlamaQuantSensitivity.MeasureAsync(basePath, options);
            var run2 = await LlamaQuantSensitivity.MeasureAsync(basePath, options);

            const double Tau = 0.05;
            var recipe1 = LlamaQuantRecipe.Build(run1, threshold: Tau);
            var recipe2 = LlamaQuantRecipe.Build(run2, threshold: Tau);

            Assert.Equal(recipe1.Entries.Count, recipe2.Entries.Count);
            for (int i = 0; i < recipe1.Entries.Count; i++)
            {
                var a = recipe1.Entries[i];
                var b = recipe2.Entries[i];
                Assert.Equal(a.TensorName,        b.TensorName);
                Assert.Equal(a.ChosenType,        b.ChosenType);
                Assert.Equal(a.ExceededThreshold, b.ExceededThreshold);
            }
        }
        finally { DeleteDir(dir); }
    }

    /// <summary>
    /// Pin the recipe contents at a chosen τ on a fixture GGUF. The
    /// fixture has two tensors whose K-quant MSE values are 2+ orders
    /// of magnitude apart, well clear of any single-τ boundary, so a
    /// kernel improvement that shifts MSE by &lt;10× will not flip the
    /// recipe. If this assertion ever fails, *investigate before
    /// updating the snapshot* — it could be a real regression in the
    /// sweep logic, the recipe build, or a meaningful kernel change.
    /// </summary>
    [Fact]
    public async Task Recipe_Snapshot_Matches_For_Fixture()
    {
        LlamaBackend.Initialize();
        var dir = MakeTempDir();
        try
        {
            var basePath = Path.Combine(dir, "src.gguf");
            await WriteFixtureGgufAsync(basePath);

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

            // τ = 0.05 is comfortably between the two tensors' difficulty
            // bands by construction. The easy tensor is well below τ at
            // every K-quant; the hard tensor's Q2_K rel-MSE is above τ,
            // forcing the recipe to step up to Q4_K or Q6_K. Both edges
            // sit ~5–10× away from τ so a kernel improvement that
            // shifts MSE by a small factor won't tip the picks.
            const double Tau = 0.05;
            var recipe = LlamaQuantRecipe.Build(result, threshold: Tau);

            var picks = recipe.Entries.ToDictionary(e => e.TensorName, e => e.ChosenType);

            Assert.Equal(2, picks.Count);

            // The easy tensor is reconstructed cleanly even by Q2_K.
            // The hard tensor needs more bits — anything *larger* than
            // Q2_K is acceptable; pinning Q4_K specifically would be
            // brittle to any kernel-side gain that shifts hard@Q2_K
            // back under τ.
            Assert.Equal(LlamaTensorType.Q2_K, picks["blk.0.easy.weight"]);
            Assert.True(picks["blk.0.hard.weight"] is LlamaTensorType.Q4_K
                                                   or LlamaTensorType.Q6_K,
                $"hard tensor picked {picks["blk.0.hard.weight"]}; expected " +
                "Q4_K or Q6_K (anything but Q2_K — the recipe should see " +
                "the hard tensor as harder than the easy one).");

            // Belt-and-braces: the chosen MSE for each tensor is
            // inside τ (so the picks aren't fallback-from-failure).
            foreach (var e in recipe.Entries)
            {
                Assert.False(e.ExceededThreshold,
                    $"{e.TensorName} fell back to lowest-MSE; expected a passing pick.");
                Assert.True(e.RelativeMse <= Tau,
                    $"{e.TensorName}: rel-MSE {e.RelativeMse:E} exceeds τ={Tau}.");
            }
        }
        finally { DeleteDir(dir); }
    }

    /// <summary>
    /// Build the two-tensor stability fixture. Both tensors are 256-
    /// wide so they fit every K-quant block-size constraint; the
    /// per-tensor row counts (4 and 8) keep the file small enough for
    /// CI but big enough for the kernels to behave normally.
    /// <list type="bullet">
    ///   <item><c>blk.0.easy.weight</c> — small-amplitude sinusoid. Dynamic
    ///         range is narrow, so even Q2_K reconstructs it cleanly.</item>
    ///   <item><c>blk.0.hard.weight</c> — wide-band sinusoid plus outlier
    ///         spikes. Forces Q2_K into trouble and lets Q4_K shine.</item>
    /// </list>
    /// </summary>
    private static async Task WriteFixtureGgufAsync(string path)
    {
        const int InDim = 256;

        const int EasyRows = 4;
        var easy = new float[InDim * EasyRows];
        for (int i = 0; i < easy.Length; i++) easy[i] = (float)Math.Sin(i * 0.1) * 0.1f;

        const int HardRows = 8;
        var hard = new float[InDim * HardRows];
        // Deterministic high-entropy fill. Smooth signals (sinusoids,
        // ramps) compress well even at Q2_K because the per-block scale
        // covers the range and the values cluster. Uniform random over
        // [-1, 1] has no structure to exploit — every value lands at a
        // different point in the per-block quant grid, so the residual
        // is bounded below by the grid spacing. Q2_K's 4 levels per
        // block leave a relative MSE that exceeds any reasonable τ.
        var rng = new Random(0xC0FFEE);
        for (int i = 0; i < hard.Length; i++)
        {
            hard[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        await new LlamaGgufWriter()
            .SetMetadata("general.architecture", "test")
            .AddTensorF32("blk.0.easy.weight", new long[] { InDim, EasyRows }, easy)
            .AddTensorF32("blk.0.hard.weight", new long[] { InDim, HardRows }, hard)
            .WriteAsync(path);
    }

    private static string MakeTempDir()
    {
        var d = Path.Combine(Path.GetTempPath(), "llama-recipe-stability-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    private static void DeleteDir(string d)
    {
        try { if (Directory.Exists(d)) Directory.Delete(d, recursive: true); } catch { /* best-effort */ }
    }
}
