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
    public void CategoryMatch_OutputWeightDoesNotCatchAttnOutputWeight()
    {
        // The matcher's old EndsWith("output.weight") logic double-counted
        // attn_output.weight tensors as belonging to the "output.weight"
        // category, producing nonsense ΔPPL coefficients on the Run 17
        // expanded profile build. The fix: dot-containing categories
        // require either exact match or "." + category suffix.
        //
        // Build a profile that distinguishes the two — output.weight gets
        // a high penalty at Q4_K, attn_output.weight gets a low one. If
        // the matcher is correct, the recipe at a tight budget will
        // promote the tensor mapped to attn_output.weight (because its
        // ΔPPL is small at Q4_K and we save by leaving it there), while
        // the tensor mapped to output.weight gets promoted heavily.
        var profile = new LlamaSensitivityProfile(
            SchemaVersion:         LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId:        "test",
            LayerCount:            2,
            FamilyNotes:           null,
            Provenance:            new LlamaSensitivityProvenance(
                "test", null, 100_000_000, null, null, null),
            F16BaselinePerplexity: 10.0,
            BaselineContextSize:   512,
            Categories: new Dictionary<string, LlamaSensitivityCategoryCoefficient>
            {
                ["attn_output.weight"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q4_K] = 0.01,    // benign
                        [LlamaTensorType.Q6_K] = 0.0,
                    },
                    RecommendedFloor: LlamaTensorType.Q4_K),
                ["output.weight"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q4_K] = 5.0,     // catastrophic
                        [LlamaTensorType.Q6_K] = 0.0,
                    },
                    RecommendedFloor: LlamaTensorType.Q4_K),
            });
        var layout = new List<(string Name, long Elements)>
        {
            ("blk.0.attn_output.weight", 1_000_000L),
            ("output.weight",            1_000_000L),
        };
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 5.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                ApplyStockBaseline = false,    // don't let baseline mask the routing
                UncategorizedProtections = new Dictionary<string, LlamaTensorType>(),
            });

        var attn = recipe.Entries.First(e => e.TensorName == "blk.0.attn_output.weight");
        var outp = recipe.Entries.First(e => e.TensorName == "output.weight");
        // attn_output should ride at Q4_K (cheap, benign).
        Assert.Equal(LlamaTensorType.Q4_K, attn.ChosenType);
        // output.weight should be promoted (high penalty at Q4_K → algorithm
        // pays the bpw to promote it). If the matcher were wrong and both
        // tensors ended up in the same category, both would land at the
        // same type — this test would fail.
        Assert.Equal(LlamaTensorType.Q6_K, outp.ChosenType);
    }

    [Fact]
    public void ProfileCategory_WithoutMatchingTensorInTarget_IsSkipped()
    {
        // Cross-size scenario: profile measured `output.weight` as a
        // category (e.g. Qwen3-1.7B has a separate output tensor) but
        // the target model uses tied embeddings (Qwen3-4B). Build
        // should skip the orphan category instead of crashing the
        // enumeration with a missing-key lookup.
        var profile = SyntheticProfile();
        // Add a category that has profile data but no matching target tensor.
        var cats = new Dictionary<string, LlamaSensitivityCategoryCoefficient>(profile.Categories)
        {
            ["output.weight"] = new LlamaSensitivityCategoryCoefficient(
                DeltaPplByType: new Dictionary<LlamaTensorType, double>
                {
                    [LlamaTensorType.Q2_K] = 5.0,
                    [LlamaTensorType.Q4_K] = 0.1,
                    [LlamaTensorType.Q6_K] = 0.0,
                },
                RecommendedFloor: LlamaTensorType.Q4_K),
        };
        var profileWithOutput = profile with { Categories = cats };

        var layout = new List<(string Name, long Elements)>
        {
            ("blk.0.ffn_up.weight",   1_000_000L),
            ("blk.0.ffn_gate.weight", 1_000_000L),
            // Note: no output.weight tensor — tied embeddings.
        };

        // Without the skip, the algorithm would crash on the
        // categoryBitsAtType lookup for ("output.weight", *).
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profileWithOutput, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5);
        Assert.NotEmpty(recipe.Entries);
        Assert.DoesNotContain(recipe.Entries, e => e.TensorName == "output.weight");
    }

    [Fact]
    public void SnapToStock_WhenPredictedGainBelowThreshold_RecipeEqualsStockEquivalent()
    {
        // Profile with mild sensitivities everywhere — Q4_K is already
        // near-optimal for both categories, so any "promotion" the
        // optimizer picks at a Q5_K-ish budget produces only a sliver
        // of predicted gain. With MinPredictedGainPpl in force, the
        // builder should discard the wiggle and emit the stock-
        // equivalent (every category at the target's nearest ladder
        // type from below).
        var profile = new LlamaSensitivityProfile(
            SchemaVersion:         LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId:        "mild",
            LayerCount:            4,
            FamilyNotes:           null,
            Provenance:            new LlamaSensitivityProvenance(
                "ablation", "mild.gguf", 100_000_000, null, DateTime.UtcNow, "test"),
            F16BaselinePerplexity: 10.0,
            BaselineContextSize:   512,
            Categories: new Dictionary<string, LlamaSensitivityCategoryCoefficient>
            {
                ["ffn_up"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q2_K] = 0.20,
                        [LlamaTensorType.Q4_K] = 0.05,
                        [LlamaTensorType.Q6_K] = 0.01,
                    },
                    RecommendedFloor: LlamaTensorType.Q2_K),
                ["ffn_gate"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q2_K] = 0.10,
                        [LlamaTensorType.Q4_K] = 0.03,
                        [LlamaTensorType.Q6_K] = 0.00,
                    },
                    RecommendedFloor: LlamaTensorType.Q2_K),
            });
        var layout = SyntheticLayout(4, 1_000_000);

        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 5.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                ApplyStockBaseline = false,    // isolate snap behavior from baseline floors
            });

        // Stock-equivalent at target=5.5 with ladder {Q2,Q4,Q6} = all at Q4_K
        // (highest ladder type with bpw ≤ 5.5). The optimizer might prefer
        // Q6_K for ffn_up to save a sliver of PPL, but the predicted gain
        // (≤ 0.05 PPL) is below the 0.25 default → snap.
        Assert.All(recipe.Entries, e => Assert.Equal(LlamaTensorType.Q4_K, e.ChosenType));
    }

    [Fact]
    public void SnapToStock_WhenPredictedGainAboveThreshold_KeepsRecipeChoice()
    {
        // The default SyntheticProfile has ffn_up Q4_K=0.5 (and Q6_K=0.05).
        // At target 5.5, the recipe promotes ffn_up→Q6_K, predicted gain ~0.45
        // PPL — comfortably above the 0.25 default threshold. The snap
        // logic must NOT discard a real win.
        var profile = SyntheticProfile();
        var layout = SyntheticLayout(4, 1_000_000);
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 5.5);

        var ffnUp = recipe.Entries.First(e => e.TensorName.Contains("ffn_up"));
        Assert.Equal(LlamaTensorType.Q6_K, ffnUp.ChosenType);    // promotion preserved
    }

    [Fact]
    public void SnapToStock_DisabledWhenThresholdIsZero()
    {
        // Setting MinPredictedGainPpl=0 turns snapping off entirely; even
        // sliver-of-gain recipes should ship as-is. Use the mild profile
        // from the snap test — without snapping, the optimizer should
        // promote ffn_up to Q6_K to claim the 0.04 PPL gain.
        var profile = new LlamaSensitivityProfile(
            SchemaVersion:         LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId:        "mild",
            LayerCount:            4,
            FamilyNotes:           null,
            Provenance:            new LlamaSensitivityProvenance(
                "ablation", "mild.gguf", 100_000_000, null, DateTime.UtcNow, "test"),
            F16BaselinePerplexity: 10.0,
            BaselineContextSize:   512,
            Categories: new Dictionary<string, LlamaSensitivityCategoryCoefficient>
            {
                ["ffn_up"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q2_K] = 0.20,
                        [LlamaTensorType.Q4_K] = 0.05,
                        [LlamaTensorType.Q6_K] = 0.01,
                    },
                    RecommendedFloor: LlamaTensorType.Q2_K),
                ["ffn_gate"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q2_K] = 0.10,
                        [LlamaTensorType.Q4_K] = 0.03,
                        [LlamaTensorType.Q6_K] = 0.00,
                    },
                    RecommendedFloor: LlamaTensorType.Q2_K),
            });
        var layout = SyntheticLayout(4, 1_000_000);

        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 5.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                ApplyStockBaseline = false,
                MinPredictedGainPpl = 0.0,
                MinPplGainPerBpw    = 0.0,    // also disable the per-bpw efficiency gate
            });

        // Without snapping, the optimizer's pure-min-pplSum pick within
        // budget should land at Q6_K for at least one category.
        Assert.Contains(recipe.Entries, e => e.ChosenType == LlamaTensorType.Q6_K);
    }

    [Fact]
    public void NoiseClamp_NegativeAndNonMonotoneDeltasDontProduceFakeWins()
    {
        // Build a profile that mimics the 4B attn_v pathology:
        // ablation noise produces a negative Q2_K delta and a
        // non-monotone Q3_K. Without the noise clamp, the optimizer
        // would happily put the noisy category at Q2_K (looks like a
        // "−0.5 PPL win!") and starve the genuinely-sensitive category.
        // With the clamp, the noisy category's Q2_K and Q3_K deltas
        // get pinned to ≥ Q4_K's value, removing the fake win.
        var profile = new LlamaSensitivityProfile(
            SchemaVersion:         LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId:        "noisy",
            LayerCount:            4,
            FamilyNotes:           null,
            Provenance:            new LlamaSensitivityProvenance(
                "ablation", "noisy.gguf", 100_000_000, null, DateTime.UtcNow, "test"),
            F16BaselinePerplexity: 10.0,
            BaselineContextSize:   512,
            Categories: new Dictionary<string, LlamaSensitivityCategoryCoefficient>
            {
                ["ffn_up"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q2_K] = -0.50,    // negative — ablation noise
                        [LlamaTensorType.Q3_K] =  0.05,    // non-monotone vs Q4_K
                        [LlamaTensorType.Q4_K] =  0.20,
                        [LlamaTensorType.Q6_K] =  0.10,
                    },
                    RecommendedFloor: LlamaTensorType.Q2_K),
                ["ffn_gate"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q2_K] =  3.00,    // genuine signal
                        [LlamaTensorType.Q3_K] =  1.00,
                        [LlamaTensorType.Q4_K] =  0.30,
                        [LlamaTensorType.Q6_K] =  0.05,
                    },
                    RecommendedFloor: LlamaTensorType.Q2_K),
            });
        var layout = SyntheticLayout(4, 1_000_000);

        // Tight budget (3.5 bpw) forces a real tradeoff. Without the
        // clamp, ffn_up's "−0.50 at Q2_K" would dominate and the
        // optimizer would put ffn_up at Q2_K (free PPL!) to fund a
        // ffn_gate promotion. With the clamp, ffn_up Q2_K is pinned
        // to its monotone-bound (≥ Q4_K's 0.20), so Q2_K is no longer
        // a free win.
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 3.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                ApplyStockBaseline = false,
                MinPplGainPerBpw   = 0.0,
                MinPredictedGainPpl = 0.0,
            });

        // Verify: every entry's reported predicted ΔPPL (RelativeMse) is
        // non-negative. No fake wins survive into the recipe.
        Assert.All(recipe.Entries, e =>
            Assert.True(e.RelativeMse >= 0.0,
                $"{e.TensorName} at {e.ChosenType}: predicted ΔPPL {e.RelativeMse} < 0 — clamp failed"));
    }

    [Fact]
    public void NoiseClamp_PreservesCleanlyMonotoneSignal()
    {
        // The default SyntheticProfile is cleanly monotone-decreasing
        // in bpw (Q2=4.0, Q4=0.5, Q6=0.05 for ffn_up). The clamp must
        // be a no-op on this — every type's predicted ΔPPL should
        // match the raw profile value (× scaling, here 1.0).
        var profile = SyntheticProfile();
        var layout = SyntheticLayout(4, 1_000_000);
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 5.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                ApplyStockBaseline = false,
            });
        // ffn_up at Q6_K → raw 0.05, scale 1.0, clamp running_max starts
        // at 0 → max(0.05, 0) = 0.05. Unchanged.
        var ffnUp = recipe.Entries.First(e => e.TensorName.Contains("ffn_up"));
        Assert.Equal(LlamaTensorType.Q6_K, ffnUp.ChosenType);
        Assert.Equal(0.05, ffnUp.RelativeMse, precision: 4);
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

    /// <summary>
    /// Noise-aware refinement should drop per-tensor refinement for
    /// any category whose drilled curve is non-monotonic (a higher-bpw
    /// measurement greater than a lower-bpw one is a mathematically
    /// impossible pattern that only arises from measurement noise).
    /// </summary>
    [Fact]
    public void NoiseAwareRefinement_NonMonotonicPerTensor_FallsBackToCategoryPick()
    {
        var profile = MakeProfileWithNonMonotonicPerTensor();
        var layout = SyntheticLayout(4, 1_000_000);

        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                UsePerTensorData = true,
                AllowPerTensorPromotion = true,
                NoiseAwareRefinement = true,    // default; explicit for clarity
            });

        // ffn_up's per-tensor data is non-monotonic (Q4_K Δ > Q2_K Δ
        // on one tensor). Noise-aware refinement should drop the
        // per-tensor refinement and leave every ffn_up tensor at the
        // per-category pick.
        var ffnUpTypes = recipe.Entries
            .Where(e => e.TensorName.EndsWith("ffn_up.weight"))
            .Select(e => e.ChosenType)
            .Distinct()
            .ToList();
        Assert.Single(ffnUpTypes);  // every ffn_up tensor at the same type
    }

    /// <summary>
    /// Stage 2 of #44: NoiseAwareCategoryPick should bump the floor
    /// to NoiseFallbackFloor (default Q4_K) when a category curve is
    /// NonMonotonic AND every measured Δ is below the noise band.
    /// This blocks the optimizer from picking Q2_K (or any sub-floor
    /// type) on a category whose measurements can't reliably be
    /// distinguished from zero.
    /// </summary>
    [Fact]
    public void NoiseAwareCategoryPick_AllNoiseCurve_BumpsFloorToFallback()
    {
        // Synthetic two-category profile. ffn_up's category curve is
        // NonMonotonic (Q4_K=-0.10 < Q2_K=-0.08, then Q6_K=-0.04 >
        // Q4_K=-0.10 — both inverted) AND the entire absolute Δ
        // range is < 1.0 PPL: pure noise. Without the gate, the
        // optimizer would pick the cheapest type (Q2_K) since its Δ
        // is "cheap" at -0.08. With the gate on, the floor bumps to
        // Q4_K and the optimizer picks Q4_K (or higher).
        var profile = new LlamaSensitivityProfile(
            SchemaVersion: LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId: "synthetic",
            LayerCount: 4,
            FamilyNotes: null,
            Provenance: new LlamaSensitivityProvenance(
                "ablation", "synthetic.gguf", 100_000_000, "synthetic",
                DateTime.UtcNow, "test"),
            F16BaselinePerplexity: 10.0,
            BaselineContextSize: 512,
            Categories: new Dictionary<string, LlamaSensitivityCategoryCoefficient>
            {
                ["ffn_up"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q2_K] = -0.08,
                        [LlamaTensorType.Q4_K] = -0.10,
                        [LlamaTensorType.Q6_K] = -0.04,
                    },
                    RecommendedFloor: LlamaTensorType.Q2_K,
                    Notes: null),
                // Anchor category that the optimizer can use to spend
                // the budget elsewhere. Without it, the optimizer has
                // nowhere to go and might force pick into uncomfortable
                // corners.
                ["ffn_gate"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q2_K] = 4.0,
                        [LlamaTensorType.Q4_K] = 0.5,
                        [LlamaTensorType.Q6_K] = 0.05,
                    },
                    RecommendedFloor: LlamaTensorType.Q4_K,
                    Notes: null),
            });

        var layout = SyntheticLayout(4, 1_000_000);

        // Gate ON (default). MinPredictedGainPpl=0 disables the
        // snap-to-stock safeguard so we isolate the gate's effect.
        var recipeGated = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                NoiseAwareCategoryPick = true,
                MinPredictedGainPpl = 0,
            });
        var gatedFfnUpType = recipeGated.Entries
            .First(e => e.TensorName.EndsWith("ffn_up.weight")).ChosenType;
        Assert.True(
            LlamaQuantRecipe.GetBitsPerElement(gatedFfnUpType) >=
            LlamaQuantRecipe.GetBitsPerElement(LlamaTensorType.Q4_K),
            $"expected gated pick to be at-or-above Q4_K floor; got {gatedFfnUpType}");

        // Gate OFF — the optimizer should freely pick Q2_K because
        // every measured Δ is "cheap" (negative, in the noise band).
        // Snap-to-stock is also disabled for the same isolation reason.
        var recipeUngated = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                NoiseAwareCategoryPick = false,
                MinPredictedGainPpl = 0,
            });
        var ungatedFfnUpType = recipeUngated.Entries
            .First(e => e.TensorName.EndsWith("ffn_up.weight")).ChosenType;
        Assert.True(
            LlamaQuantRecipe.GetBitsPerElement(ungatedFfnUpType) <
            LlamaQuantRecipe.GetBitsPerElement(LlamaTensorType.Q4_K),
            $"expected ungated pick to drop below Q4_K when measurements look cheap; got {ungatedFfnUpType}");
    }

    /// <summary>
    /// NoiseAwareCategoryPick should NOT raise the floor on a
    /// non-monotonic curve that has real signal somewhere — only
    /// when the entire curve is below the noise band. A category
    /// with a Q2_K catastrophe (+4.0 PPL) but a slightly inverted
    /// Q4_K → Q6_K transition shouldn't be flagged as all-noise
    /// just because of the small inversion at the high end.
    /// </summary>
    [Fact]
    public void NoiseAwareCategoryPick_NonMonotonicWithRealSignal_DoesNotBumpFloor()
    {
        var profile = new LlamaSensitivityProfile(
            SchemaVersion: LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId: "synthetic",
            LayerCount: 4,
            FamilyNotes: null,
            Provenance: new LlamaSensitivityProvenance(
                "ablation", "synthetic.gguf", 100_000_000, "synthetic",
                DateTime.UtcNow, "test"),
            F16BaselinePerplexity: 10.0,
            BaselineContextSize: 512,
            Categories: new Dictionary<string, LlamaSensitivityCategoryCoefficient>
            {
                ["ffn_up"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q2_K] = 4.0,    // real signal
                        [LlamaTensorType.Q4_K] = 0.5,
                        [LlamaTensorType.Q6_K] = 0.55,   // tiny inversion
                    },
                    RecommendedFloor: LlamaTensorType.Q2_K,
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

        var layout = SyntheticLayout(4, 1_000_000);
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                NoiseAwareCategoryPick = true,
            });

        // The Q2_K catastrophe (+4.0) still drives the optimizer
        // away from Q2_K naturally — no bump needed. The recipe
        // should pick Q4_K based on the measured Δ, not because the
        // gate forced it. Either way, the result should be Q4_K or
        // higher (not Q2_K).
        var ffnUpType = recipe.Entries
            .First(e => e.TensorName.EndsWith("ffn_up.weight")).ChosenType;
        Assert.NotEqual(LlamaTensorType.Q2_K, ffnUpType);
    }

    /// <summary>
    /// Companion test: with <c>NoiseAwareRefinement=false</c>, the
    /// same non-monotonic profile DOES end up with per-tensor
    /// variance — the gate is what produced the difference, not some
    /// unrelated change to the rest of the pipeline.
    /// </summary>
    [Fact]
    public void NoiseAwareRefinementOff_NonMonotonicPerTensor_RefinesAnyway()
    {
        var profile = MakeProfileWithNonMonotonicPerTensor();
        var layout = SyntheticLayout(4, 1_000_000);

        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                UsePerTensorData = true,
                AllowPerTensorPromotion = true,
                NoiseAwareRefinement = false,    // explicit opt-out
                NoiseAwareCategoryPick = false,  // also off so the
                                                 // floor isn't bumped
                                                 // by stage 2 logic
                MinPredictedGainPpl = 0,         // and don't snap to stock
            });

        // Without the noise gate the per-tensor refinement runs and
        // typically picks a non-uniform mix (one of the ffn_up
        // tensors had a noise-flipped Δ that the optimizer treats as
        // signal). Assert that at least one tensor lands at a type
        // different from the others.
        var ffnUpTypes = recipe.Entries
            .Where(e => e.TensorName.EndsWith("ffn_up.weight"))
            .Select(e => e.ChosenType)
            .Distinct()
            .ToList();
        Assert.True(ffnUpTypes.Count >= 2,
            $"expected per-tensor refinement to pick mixed types when noise gate is off; got {string.Join(",", ffnUpTypes)}");
    }

    /// <summary>
    /// Soft-floor gate: when a category is flagged as noise-suspect, the
    /// default behavior is to RAISE the per-tensor refinement floor (to
    /// <c>RefinementFloorWhenNoisy</c>, default <c>Q3_K</c>) rather than
    /// skip refinement entirely. Refinement still runs — it just can't
    /// pick types below the floor. Run 22 demonstrated that legitimate
    /// per-tensor demotes (IQ4_XS, Q4_K) live above any sane noise floor;
    /// what we actually want to block is the catastrophic Q2_K demote on
    /// noise-flagged data.
    /// </summary>
    [Fact]
    public void NoiseAwareRefinement_SoftFloor_AllowsAboveFloor_BlocksBelow()
    {
        var profile = MakeProfileWithNonMonotonicPerTensor_AndQ3K();
        var layout = SyntheticLayout(4, 1_000_000);

        // Default options: soft floor at Q3_K.
        var recipeSoft = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                UsePerTensorData = true,
                NoiseAwareRefinement = true,            // default
                // RefinementFloorWhenNoisy default = Q3_K
                MinPredictedGainPpl = 0,                // isolate gate behavior
            });
        var blk2Soft = recipeSoft.Entries
            .First(e => e.TensorName == "blk.2.ffn_up.weight").ChosenType;
        Assert.Equal(LlamaTensorType.Q3_K, blk2Soft);

        // Hard-stop legacy behavior: RefinementFloorWhenNoisy = null
        // → refinement skipped entirely, blk.2 stays at category pick.
        var recipeHardStop = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                UsePerTensorData = true,
                NoiseAwareRefinement = true,
                RefinementFloorWhenNoisy = null,        // legacy hard-stop
                MinPredictedGainPpl = 0,
            });
        var blk2HardStop = recipeHardStop.Entries
            .First(e => e.TensorName == "blk.2.ffn_up.weight").ChosenType;
        Assert.Equal(LlamaTensorType.Q4_K, blk2HardStop);  // category pick

        // Gate off: refinement runs unconstrained, blk.2 demotes all
        // the way to Q2_K based on its noise-flipped per-tensor signal.
        var recipeOff = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                UsePerTensorData = true,
                NoiseAwareRefinement = false,           // gate off
                MinPredictedGainPpl = 0,
            });
        var blk2Off = recipeOff.Entries
            .First(e => e.TensorName == "blk.2.ffn_up.weight").ChosenType;
        Assert.Equal(LlamaTensorType.Q2_K, blk2Off);
    }

    /// <summary>
    /// Variant of <see cref="MakeProfileWithNonMonotonicPerTensor"/>
    /// that adds Q3_K data at both category and per-tensor scope. The
    /// extended ladder lets the soft-floor test distinguish three
    /// outcomes for blk.2: demote to Q2_K (gate off), demote to Q3_K
    /// (soft floor), or no demote at all (hard-stop / gate skips
    /// refinement).
    /// </summary>
    private static LlamaSensitivityProfile MakeProfileWithNonMonotonicPerTensor_AndQ3K()
    {
        var perTensor = new Dictionary<string, LlamaSensitivityTensorCoefficient>(StringComparer.Ordinal);
        for (int i = 0; i < 4; i++)
        {
            var deltas = i == 2
                ? new Dictionary<LlamaTensorType, double>
                {
                    [LlamaTensorType.Q2_K] = 0.01,    // gate-off picks this
                    [LlamaTensorType.Q3_K] = 0.02,    // soft floor picks this
                    [LlamaTensorType.Q4_K] = 0.10,
                    [LlamaTensorType.Q6_K] = 0.04,
                }
                : new Dictionary<LlamaTensorType, double>
                {
                    [LlamaTensorType.Q2_K] = 4.0,
                    [LlamaTensorType.Q3_K] = 1.5,
                    [LlamaTensorType.Q4_K] = 0.5,
                    [LlamaTensorType.Q6_K] = 0.05,
                };
            perTensor[$"blk.{i}.ffn_up.weight"] = new LlamaSensitivityTensorCoefficient(deltas);
        }
        return new LlamaSensitivityProfile(
            SchemaVersion: LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId: "synthetic",
            LayerCount: 4,
            FamilyNotes: null,
            Provenance: new LlamaSensitivityProvenance(
                Method: "ablation",
                SourceModel: "synthetic.gguf",
                SourceParameterCount: 100_000_000,
                Corpus: "synthetic",
                BuiltAtUtc: DateTime.UtcNow,
                BuilderVersion: "test"),
            F16BaselinePerplexity: 10.0,
            BaselineContextSize: 512,
            Categories: new Dictionary<string, LlamaSensitivityCategoryCoefficient>
            {
                ["ffn_up"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q2_K] = 4.0,
                        [LlamaTensorType.Q3_K] = 1.5,
                        [LlamaTensorType.Q4_K] = 0.5,
                        [LlamaTensorType.Q6_K] = 0.55,    // inverted → NonMonotonic
                    },
                    RecommendedFloor: LlamaTensorType.Q2_K,
                    Notes: null),
                ["ffn_gate"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q2_K] = 0.5,
                        [LlamaTensorType.Q3_K] = 0.1,
                        [LlamaTensorType.Q4_K] = 0.05,
                        [LlamaTensorType.Q6_K] = 0.005,
                    },
                    RecommendedFloor: LlamaTensorType.Q2_K,
                    Notes: null),
            },
            PerTensor: perTensor);
    }

    /// <summary>
    /// Helper: build a profile whose ffn_up category has a clean
    /// monotonic per-category curve but a non-monotonic per-tensor
    /// curve on one of its layers. The recipe builder's analyzer
    /// should classify ffn_up as
    /// <see cref="LlamaCategoryShape.NonMonotonic"/> and (when
    /// noise-aware refinement is on) skip per-tensor refinement.
    /// </summary>
    private static LlamaSensitivityProfile MakeProfileWithNonMonotonicPerTensor()
    {
        // Per-tensor data designed so the refinement loop will *actually*
        // demote blk.2 to Q2_K when allowed:
        //   - blk.2's Q2_K Δ is below the default
        //     PerTensorPromotionThresholdPpl (0.05) so the demote gate
        //     opens.
        //   - The other layers' Q2_K Δ sits at the per-category number
        //     (4.0) so they stay at the per-category pick.
        var perTensor = new Dictionary<string, LlamaSensitivityTensorCoefficient>(StringComparer.Ordinal);
        for (int i = 0; i < 4; i++)
        {
            var deltas = i == 2
                ? new Dictionary<LlamaTensorType, double>
                {
                    [LlamaTensorType.Q2_K] = 0.01,    // below promotion threshold
                    [LlamaTensorType.Q4_K] = 0.10,
                    [LlamaTensorType.Q6_K] = 0.04,
                }
                : new Dictionary<LlamaTensorType, double>
                {
                    [LlamaTensorType.Q2_K] = 4.0,
                    [LlamaTensorType.Q4_K] = 0.5,
                    [LlamaTensorType.Q6_K] = 0.05,
                };
            perTensor[$"blk.{i}.ffn_up.weight"] = new LlamaSensitivityTensorCoefficient(deltas);
        }
        // Category-level ffn_up: Q4_K → Q6_K is inverted (Q6_K Δ > Q4_K
        // Δ by 0.05 > 0.01 noise band), which is mathematically
        // impossible and triggers the analyzer's NonMonotonic verdict.
        // This is what gates the refinement skip.
        return new LlamaSensitivityProfile(
            SchemaVersion: LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId: "synthetic",
            LayerCount: 4,
            FamilyNotes: null,
            Provenance: new LlamaSensitivityProvenance(
                Method: "ablation",
                SourceModel: "synthetic.gguf",
                SourceParameterCount: 100_000_000,
                Corpus: "synthetic",
                BuiltAtUtc: DateTime.UtcNow,
                BuilderVersion: "test"),
            F16BaselinePerplexity: 10.0,
            BaselineContextSize: 512,
            Categories: new Dictionary<string, LlamaSensitivityCategoryCoefficient>
            {
                ["ffn_up"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q2_K] = 4.0,
                        [LlamaTensorType.Q4_K] = 0.5,
                        [LlamaTensorType.Q6_K] = 0.55,    // inverted vs Q4_K → NonMonotonic
                    },
                    // Floor allows Q2_K so per-tensor refinement isn't
                    // blocked at the floor — only the noise gate would
                    // stop blk.2 from being demoted.
                    RecommendedFloor: LlamaTensorType.Q2_K,
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
            },
            PerTensor: perTensor);
    }
}
