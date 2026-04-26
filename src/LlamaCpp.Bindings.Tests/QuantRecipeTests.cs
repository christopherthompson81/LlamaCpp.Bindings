namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Tests for <see cref="LlamaQuantRecipe"/>. We build synthetic
/// <see cref="LlamaQuantSensitivityResult"/> tables so the recipe logic
/// can be exercised deterministically without running a real sweep.
/// </summary>
public class QuantRecipeTests
{
    private static LlamaQuantSensitivityResult MakeResult(
        params (string tensor, LlamaTensorType type, double relMse)[] rows)
    {
        var scores = rows.Select(r => new LlamaQuantSensitivityScore(
            TensorName:      r.tensor,
            QuantType:       r.type,
            RawMse:          r.relMse,           // doesn't matter for recipe build
            RelativeMse:     r.relMse,
            ElementCount:    1024,
            ImatrixWeighted: false,
            Skipped:         false,
            SkipReason:      null)).ToList();
        var candidates = rows.Select(r => r.type).Distinct().ToList();
        return new LlamaQuantSensitivityResult(
            ModelPath:      "synthetic.gguf",
            ImatrixPath:    null,
            CandidateTypes: candidates,
            Scores:         scores,
            ComputedAtUtc:  DateTime.UtcNow,
            Elapsed:        TimeSpan.Zero);
    }

    [Fact]
    public void Build_Picks_Smallest_Type_That_Meets_Threshold()
    {
        LlamaBackend.Initialize();
        // Q4_K is well under threshold; Q2_K is too noisy. The recipe
        // should land on Q4_K — smaller than Q6_K, accurate enough.
        var scores = MakeResult(
            ("blk.0.attn_q.weight", LlamaTensorType.Q2_K, 0.10),
            ("blk.0.attn_q.weight", LlamaTensorType.Q4_K, 0.02),
            ("blk.0.attn_q.weight", LlamaTensorType.Q6_K, 0.005));

        var recipe = LlamaQuantRecipe.Build(scores, threshold: 0.05);

        var entry = Assert.Single(recipe.Entries);
        Assert.Equal(LlamaTensorType.Q4_K, entry.ChosenType);
        Assert.False(entry.ExceededThreshold);
        Assert.Equal(0.02, entry.RelativeMse);
    }

    [Fact]
    public void Build_Falls_Back_To_Lowest_Mse_When_Threshold_Unmet()
    {
        LlamaBackend.Initialize();
        // Threshold is tighter than every candidate's MSE — fall back
        // to the safest available (lowest MSE) and flag the entry.
        var scores = MakeResult(
            ("blk.0.attn_q.weight", LlamaTensorType.Q2_K, 0.20),
            ("blk.0.attn_q.weight", LlamaTensorType.Q4_K, 0.10),
            ("blk.0.attn_q.weight", LlamaTensorType.Q6_K, 0.05));

        var recipe = LlamaQuantRecipe.Build(scores, threshold: 0.001);

        var entry = Assert.Single(recipe.Entries);
        Assert.Equal(LlamaTensorType.Q6_K, entry.ChosenType);
        Assert.True(entry.ExceededThreshold);
    }

    [Fact]
    public void Build_Handles_Multiple_Tensors_Independently()
    {
        LlamaBackend.Initialize();
        // Easy tensor lands at Q2_K; hard one needs Q6_K.
        var scores = MakeResult(
            ("blk.0.attn_q.weight",  LlamaTensorType.Q2_K, 0.01),
            ("blk.0.attn_q.weight",  LlamaTensorType.Q4_K, 0.005),
            ("blk.0.attn_q.weight",  LlamaTensorType.Q6_K, 0.002),
            ("blk.0.ffn_down.weight", LlamaTensorType.Q2_K, 0.20),
            ("blk.0.ffn_down.weight", LlamaTensorType.Q4_K, 0.08),
            ("blk.0.ffn_down.weight", LlamaTensorType.Q6_K, 0.03));

        var recipe = LlamaQuantRecipe.Build(scores, threshold: 0.05);

        Assert.Equal(2, recipe.Entries.Count);
        var byTensor = recipe.Entries.ToDictionary(e => e.TensorName);
        Assert.Equal(LlamaTensorType.Q2_K, byTensor["blk.0.attn_q.weight"].ChosenType);
        Assert.Equal(LlamaTensorType.Q6_K, byTensor["blk.0.ffn_down.weight"].ChosenType);
    }

    [Fact]
    public void ToTtOverrides_Emits_Anchored_Escaped_Regex()
    {
        LlamaBackend.Initialize();
        var scores = MakeResult(
            ("blk.0.attn_q.weight", LlamaTensorType.Q4_K, 0.01));
        var recipe = LlamaQuantRecipe.Build(scores, threshold: 0.05);

        var overrides = recipe.ToTtOverrides();
        var pair = Assert.Single(overrides);
        // Literal dots must be escaped so the pattern doesn't widen.
        Assert.Equal(@"^blk\.0\.attn_q\.weight$", pair.Key);
        Assert.Equal(LlamaTensorType.Q4_K, pair.Value);
    }

    [Fact]
    public void GetBitsPerElement_Orders_K_Quants_Correctly()
    {
        // Sanity: ggml_type_traits gives sensible bits-per-element
        // values, and they form the expected order.
        LlamaBackend.Initialize();
        var f16 = LlamaQuantRecipe.GetBitsPerElement(LlamaTensorType.F16);
        var q8  = LlamaQuantRecipe.GetBitsPerElement(LlamaTensorType.Q8_0);
        var q6  = LlamaQuantRecipe.GetBitsPerElement(LlamaTensorType.Q6_K);
        var q4  = LlamaQuantRecipe.GetBitsPerElement(LlamaTensorType.Q4_K);
        var q2  = LlamaQuantRecipe.GetBitsPerElement(LlamaTensorType.Q2_K);

        Assert.Equal(16.0, f16);
        Assert.True(q8 > q6 && q6 > q4 && q4 > q2,
            $"Expected F16 > Q8_0 > Q6_K > Q4_K > Q2_K, got " +
            $"F16={f16}, Q8_0={q8}, Q6_K={q6}, Q4_K={q4}, Q2_K={q2}");
    }

    [Fact]
    public void Json_Round_Trip_Preserves_Entries()
    {
        LlamaBackend.Initialize();
        var scores = MakeResult(
            ("blk.0.attn_q.weight",  LlamaTensorType.Q4_K, 0.02),
            ("blk.0.ffn_down.weight", LlamaTensorType.Q4_K, 0.10));
        var recipe = LlamaQuantRecipe.Build(scores, threshold: 0.05);

        var dir = Path.Combine(Path.GetTempPath(), "llama-recipe-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var path = Path.Combine(dir, "recipe.json");
            LlamaQuantRecipe.SaveToJson(recipe, path);
            var loaded = LlamaQuantRecipe.LoadFromJson(path);

            Assert.Equal(recipe.Threshold, loaded.Threshold);
            Assert.Equal(recipe.Entries.Count, loaded.Entries.Count);
            for (int i = 0; i < recipe.Entries.Count; i++)
            {
                Assert.Equal(recipe.Entries[i].TensorName,        loaded.Entries[i].TensorName);
                Assert.Equal(recipe.Entries[i].ChosenType,        loaded.Entries[i].ChosenType);
                Assert.Equal(recipe.Entries[i].ExceededThreshold, loaded.Entries[i].ExceededThreshold);
            }
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { /* best-effort */ }
        }
    }

    [Fact]
    public void Build_Skips_Tensor_When_All_Rows_Are_Skipped()
    {
        LlamaBackend.Initialize();
        // 1-D tensors come back marked Skipped; recipe should drop them
        // rather than emit nonsense entries.
        var scores = new LlamaQuantSensitivityResult(
            ModelPath:      "synthetic.gguf",
            ImatrixPath:    null,
            CandidateTypes: new[] { LlamaTensorType.Q4_K },
            Scores: new[]
            {
                new LlamaQuantSensitivityScore(
                    TensorName: "blk.0.attn_norm.weight",
                    QuantType:  LlamaTensorType.Q4_K,
                    RawMse:     double.NaN,
                    RelativeMse: double.NaN,
                    ElementCount: 256,
                    ImatrixWeighted: false,
                    Skipped:    true,
                    SkipReason: "1-D"),
            },
            ComputedAtUtc: DateTime.UtcNow,
            Elapsed:       TimeSpan.Zero);

        var recipe = LlamaQuantRecipe.Build(scores, threshold: 0.05);
        Assert.Empty(recipe.Entries);
    }
}
