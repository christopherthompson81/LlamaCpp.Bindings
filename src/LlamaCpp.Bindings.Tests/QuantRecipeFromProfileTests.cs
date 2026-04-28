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
    public void NoiseClamp_PerFamilyMonotone_PreservesIQuantSignalAgainstNoisyKQuant()
    {
        // Run-21 case: a noisy K-family Q5_K reading was inflating the
        // clamped delta of every K-family type below it (Q4_K, Q3_K…)
        // AND of the I-family IQ4_XS sitting between Q5_K and Q4_K in
        // bpw. The cross-family monotone propagation suppressed IQ4_XS's
        // genuine signal; per-family monotone preserves it.
        //
        // Profile: Q5_K is noisy (Δ = 0.30), Q4_K is moderate (0.10),
        // IQ4_XS measures very low (0.01) — its true signal. Under
        // per-family clamping, IQ4_XS should land at 0.01 (not pulled
        // up by Q5_K), and the optimizer at a 4.5 bpw target should
        // prefer IQ4_XS (lower bpw, better delta).
        var profile = new LlamaSensitivityProfile(
            SchemaVersion:         LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId:        "test",
            LayerCount:            1,
            FamilyNotes:           null,
            Provenance:            new LlamaSensitivityProvenance(
                "ablation", "test.gguf", 100_000_000, null, DateTime.UtcNow, "test"),
            F16BaselinePerplexity: 10.0,
            BaselineContextSize:   512,
            Categories: new Dictionary<string, LlamaSensitivityCategoryCoefficient>
            {
                ["ffn_up"] = new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q6_K]   = 0.001,    // K-family, near-zero
                        [LlamaTensorType.Q5_K]   = 0.30,     // K-family, NOISY (worse than Q4_K)
                        [LlamaTensorType.Q4_K]   = 0.10,     // K-family, moderate
                        [LlamaTensorType.IQ4_XS] = 0.01,     // I-family, low — preserved by per-family clamp
                    },
                    // Floor at Q3_K (3.4375 bpw) so IQ4_XS (4.25) is
                    // above floor. Floor semantics for cross-family
                    // ladders is its own design question; this test
                    // isolates the clamp behaviour.
                    RecommendedFloor: LlamaTensorType.Q3_K),
            });
        var layout = SyntheticLayout(1, 1_000_000);
        // Reuse one ffn_up tensor name pattern; only ffn_up matters.
        var ffnUpLayout = layout.Where(t => t.Name.Contains("ffn_up")).ToList();

        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, ffnUpLayout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5,    // Q4_K-class budget
            options: new LlamaQuantRecipeFromProfileOptions
            {
                ApplyStockBaseline = false,
                MinPredictedGainPpl = 0.0,    // disable snap so the optimizer freely picks
                MinPplGainPerBpw   = 0.05,
            });

        // Optimizer should pick IQ4_XS (lower bpw 4.25, lower clamped Δ 0.01)
        // over Q4_K (4.5 bpw, clamped 0.30 from Q5_K propagation)
        // and over Q5_K (5.5 bpw, 0.30) and Q6_K (6.5625 bpw, 0).
        // The score function prefers IQ4_XS:
        //   IQ4_XS: 0.01 + 0.05 × 4.25 = 0.2225
        //   Q6_K:   0    + 0.05 × 6.56 = 0.328
        var ffnUp = recipe.Entries.First(e => e.TensorName.Contains("ffn_up"));
        Assert.Equal(LlamaTensorType.IQ4_XS, ffnUp.ChosenType);
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
