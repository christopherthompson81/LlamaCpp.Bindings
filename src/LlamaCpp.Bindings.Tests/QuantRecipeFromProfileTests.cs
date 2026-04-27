namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Algorithm-level tests for <see cref="LlamaQuantRecipeFromProfile"/>.
/// These exercise the greedy bpw walk + size scaling + floor
/// enforcement against synthetic profiles and synthetic tensor
/// layouts; no real GGUF required. The Qwen3 reference profiles are
/// loaded from <c>data/profiles/</c> for end-to-end sanity checks.
/// </summary>
public class QuantRecipeFromProfileTests
{
    static QuantRecipeFromProfileTests()
    {
        // GetBitsPerElement reads ggml_type_traits via P/Invoke and needs
        // the backend up. Idempotent.
        LlamaBackend.Initialize();
    }

    private static string RepoRoot()
    {
        var dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "LlamaCpp.Bindings.slnx")))
            dir = Path.GetDirectoryName(dir);
        return dir ?? throw new InvalidOperationException("Could not locate repo root.");
    }

    /// <summary>
    /// Two-category synthetic profile: ffn_up is highly sensitive,
    /// ffn_gate is barely sensitive. Same shape as the real Qwen3
    /// data, just compressed.
    /// </summary>
    private static LlamaSensitivityProfile SyntheticProfile(long sourceParams = 100_000_000) =>
        new(
            SchemaVersion:         LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId:        "synthetic",
            LayerCount:            4,
            FamilyNotes:           null,
            Provenance:            new LlamaSensitivityProvenance(
                                       Method:               "ablation",
                                       SourceModel:          "synthetic.gguf",
                                       SourceParameterCount: sourceParams,
                                       Corpus:               "synthetic",
                                       BuiltAtUtc:           DateTime.UtcNow,
                                       BuilderVersion:       "test"),
            F16BaselinePerplexity: 10.0,
            BaselineContextSize:   512,
            Categories: new Dictionary<string, LlamaSensitivityCategoryCoefficient>
            {
                ["ffn_up"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q2_K] = 4.0,    // catastrophic at Q2_K
                        [LlamaTensorType.Q4_K] = 0.5,    // moderate at Q4_K
                        [LlamaTensorType.Q6_K] = 0.05,   // near-zero at Q6_K
                    },
                    RecommendedFloor: LlamaTensorType.Q4_K,    // floor blocks Q2_K
                    Notes: null),
                ["ffn_gate"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q2_K] = 0.5,
                        [LlamaTensorType.Q4_K] = 0.05,
                        [LlamaTensorType.Q6_K] = 0.005,
                    },
                    RecommendedFloor: LlamaTensorType.Q2_K,
                    Notes: null),
            });

    /// <summary>Synthetic tensor layout: equal-sized tensors per category.</summary>
    private static IReadOnlyList<(string Name, long Elements)> SyntheticLayout(int tensorsPerCategory, long elementsEach) =>
        Enumerable.Range(0, tensorsPerCategory)
            .SelectMany(i => new[]
            {
                ($"blk.{i}.ffn_up.weight", elementsEach),
                ($"blk.{i}.ffn_gate.weight", elementsEach),
            })
            .ToList();

    [Fact]
    public void TargetBpw_Q4K_AllCategoriesLandAtQ4K()
    {
        var profile = SyntheticProfile();
        var layout = SyntheticLayout(4, 1_000_000);
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5);
        Assert.All(recipe.Entries, e => Assert.Equal(LlamaTensorType.Q4_K, e.ChosenType));
        Assert.InRange(recipe.AverageBitsPerElement, 4.45, 4.55);
    }

    [Fact]
    public void TargetBpw_AboveQ4K_PromotesMostSensitiveFirst()
    {
        // Halfway between Q4_K (4.5) and Q6_K (6.5625) is ~5.5. The greedy
        // walk should promote ffn_up (huge ΔPPL win) before ffn_gate.
        var profile = SyntheticProfile();
        var layout = SyntheticLayout(4, 1_000_000);
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 5.5);

        var byCat = recipe.Entries.GroupBy(e => e.TensorName.Contains("ffn_up") ? "ffn_up" : "ffn_gate")
                                   .ToDictionary(g => g.Key, g => g.First().ChosenType);
        Assert.Equal(LlamaTensorType.Q6_K, byCat["ffn_up"]);     // promoted
        Assert.Equal(LlamaTensorType.Q4_K, byCat["ffn_gate"]);   // not promoted
    }

    [Fact]
    public void UncategorizedProtections_OutputWeightDefaultsToQ6K()
    {
        // The default protection table puts output.weight at Q6_K — even
        // when the profile knows nothing about it. Run 15 showed that
        // demoting output.weight to Q4_K (the old UncategorizedType
        // default) costs ~3 PPL on Qwen3-1.7B. The protection should
        // hold regardless of the bpw budget — we'd rather pay a bpw
        // penalty than ship output.weight at Q4_K.
        var profile = SyntheticProfile();
        var layout = new List<(string Name, long Elements)>
        {
            ("blk.0.ffn_up.weight",   1_000_000L),
            ("blk.0.ffn_gate.weight", 1_000_000L),
            ("output.weight",         1_000_000L),    // uncategorized in synthetic profile
            ("token_embd.weight",     1_000_000L),    // uncategorized in synthetic profile
        };
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5);

        var output = recipe.Entries.First(e => e.TensorName == "output.weight");
        Assert.Equal(LlamaTensorType.Q6_K, output.ChosenType);

        var embd = recipe.Entries.First(e => e.TensorName == "token_embd.weight");
        Assert.Equal(LlamaTensorType.Q4_K, embd.ChosenType);
    }

    [Fact]
    public void UncategorizedProtections_CallerCanOverride()
    {
        // The protection table is just a default — callers can replace
        // it. Pass an empty map and observe everything fall to UncategorizedDefault.
        var profile = SyntheticProfile();
        var layout = new List<(string Name, long Elements)>
        {
            ("blk.0.ffn_up.weight",   1_000_000L),
            ("blk.0.ffn_gate.weight", 1_000_000L),
            ("output.weight",         1_000_000L),
        };
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                UncategorizedProtections = new Dictionary<string, LlamaTensorType>(),
                UncategorizedDefault = LlamaTensorType.Q4_K,
            });

        var output = recipe.Entries.First(e => e.TensorName == "output.weight");
        Assert.Equal(LlamaTensorType.Q4_K, output.ChosenType);
    }

    [Fact]
    public void Floor_NeverChosenBelowFloorEvenAtAggressiveBudget()
    {
        // ffn_up has a Q4_K floor in the synthetic profile. Even when
        // the budget is so tight that pure-Q2_K would otherwise be the
        // PPL-optimal recipe, ffn_up must stay at Q4_K or above.
        var profile = SyntheticProfile();
        var layout = SyntheticLayout(4, 1_000_000);
        foreach (var bpw in new[] { 2.625, 3.0, 3.5, 4.0 })
        {
            var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
                profile, layout, targetParameterCount: 100_000_000,
                targetBitsPerElement: bpw);
            var ffnUp = recipe.Entries.First(e => e.TensorName.Contains("ffn_up"));
            Assert.True(
                LlamaQuantRecipe.GetBitsPerElement(ffnUp.ChosenType) >=
                LlamaQuantRecipe.GetBitsPerElement(LlamaTensorType.Q4_K),
                $"ffn_up dropped below floor at target={bpw}: chose {ffnUp.ChosenType}");
        }
    }

    [Fact]
    public void SizeScaling_AppliesToCoefficients()
    {
        // Profile sourced on a 100M model, applied to a 300M target with
        // exponent 1.0 → coefficients triple. Verify that the entries'
        // RelativeMse field (which we repurpose as predicted-ΔPPL ×
        // size-scale) reflects the scaled value, not the raw profile value.
        var profile = SyntheticProfile(sourceParams: 100_000_000);
        var layout = SyntheticLayout(4, 1_000_000);
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 300_000_000,
            targetBitsPerElement: 4.5);

        // ffn_up at Q4_K: profile says 0.5, scaling 3× → expect ~1.5.
        var ffnUp = recipe.Entries.First(e => e.TensorName.Contains("ffn_up"));
        Assert.Equal(LlamaTensorType.Q4_K, ffnUp.ChosenType);
        Assert.InRange(ffnUp.RelativeMse, 1.45, 1.55);
    }

    [Fact]
    public void SizeScaling_ExponentZero_DisablesScaling()
    {
        var profile = SyntheticProfile(sourceParams: 100_000_000);
        var layout = SyntheticLayout(4, 1_000_000);
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 5_000_000_000,    // 50× larger
            targetBitsPerElement: 4.5,
            options: new LlamaQuantRecipeFromProfileOptions { SizeScalingExponent = 0.0 });
        var ffnUp = recipe.Entries.First(e => e.TensorName.Contains("ffn_up"));
        Assert.Equal(0.5, ffnUp.RelativeMse, precision: 4);    // unscaled
    }

    [Fact]
    public void EndToEnd_FromQwen3Profile_BuildsWithoutTargetGguf()
    {
        // Real reference profile + synthetic target the same size as the
        // profile's source. The greedy walk should land at ~Q4_K_M-ish
        // (roughly 4.6 bpw) and respect the per-category floors.
        var path = Path.Combine(RepoRoot(), "data", "profiles", "qwen3-0.6B.profile.json");
        var profile = LlamaSensitivityProfile.LoadFromJson(path);
        // Synthetic 7-category layout (one tensor per category, equal size)
        // so the test is deterministic regardless of GGUF availability.
        var layout = profile.Categories.Keys.Select(c =>
        {
            var tensorName = c.Contains('.') ? $"blk.0.{c}" : $"blk.0.{c}.weight";
            return (tensorName, 1_000_000L);
        }).ToList();
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: profile.Provenance.SourceParameterCount ?? 100_000_000,
            targetBitsPerElement: 4.7);
        Assert.Equal(profile.Categories.Count, recipe.Entries.Count);
        Assert.InRange(recipe.AverageBitsPerElement, 4.5, 5.0);
        // Every entry should match a profile category (no uncategorized).
        Assert.All(recipe.Entries, e =>
            Assert.Contains(profile.Categories.Keys, c =>
                c.Contains('.') ? e.TensorName.EndsWith(c) : e.TensorName.Contains(c)));
    }
}
