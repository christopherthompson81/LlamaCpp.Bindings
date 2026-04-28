namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// DB → profile derivation, completeness reporting, and per-tensor
/// refinement in the recipe builder. Covers the Phase 4 wiring that
/// turns accumulated measurement rows into actionable recipes.
/// </summary>
public class SensitivityProfileDeriveTests
{
    static SensitivityProfileDeriveTests()
    {
        // Required by LlamaQuantRecipe.GetBitsPerElement (P/Invoke into ggml).
        LlamaBackend.Initialize();
    }

    private static string TempDbPath() =>
        Path.Combine(Path.GetTempPath(), $"llama-derive-{Guid.NewGuid():N}.sqlite");

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch { /* best-effort */ }
        try { if (File.Exists(path + "-wal")) File.Delete(path + "-wal"); }
        catch { /* best-effort */ }
        try { if (File.Exists(path + "-shm")) File.Delete(path + "-shm"); }
        catch { /* best-effort */ }
    }

    private static LlamaMeasurementRecord MakeRecord(
        string target, LlamaTensorType type, double ablationPpl,
        double baselinePpl = 14.0, string modelSha = "model_a") =>
        new(
            ModelSha:        modelSha,
            ArchId:          "qwen3",
            ParamCount:      1_000_000_000,
            CorpusSha:       "corpus_a",
            CorpusName:      "wiki.test.raw",
            ImatrixSha:      LlamaInvestigationDb.NoImatrixSha,
            ContextSize:     512,
            AblationTarget:  target,
            AblationType:    type,
            BaselineType:    LlamaTensorType.F16,
            BaselinePpl:     baselinePpl,
            AblationPpl:     ablationPpl,
            DeltaPpl:        ablationPpl - baselinePpl,
            MeasuredAtUtc:   DateTime.UtcNow,
            BuilderVersion:  "test/1.0",
            LlamaCppVersion: "b8893",
            GpuModel:        null,
            Notes:           null);

    [Fact]
    public void DeriveFromDb_NoBaseline_ReturnsNull()
    {
        var path = TempDbPath();
        try
        {
            using var db = LlamaInvestigationDb.Open(path);
            // Insert ablation rows but no baseline — derivation has nothing to anchor against.
            db.RecordMeasurement(MakeRecord("category:ffn_up", LlamaTensorType.Q4_K, 14.5));

            var profile = LlamaSensitivityProfile.DeriveFromDb(
                db, modelSha: "model_a", corpusSha: "corpus_a",
                imatrixSha: LlamaInvestigationDb.NoImatrixSha,
                contextSize: 512, archId: "qwen3", paramCount: 1_000_000_000, layerCount: 4);
            Assert.Null(profile);
        }
        finally { TryDelete(path); }
    }

    [Fact]
    public void DeriveFromDb_BaselinePlusAblations_ProducesProfile()
    {
        var path = TempDbPath();
        try
        {
            using var db = LlamaInvestigationDb.Open(path);
            db.RecordMeasurement(MakeRecord("baseline", LlamaTensorType.F16, 14.0));
            db.RecordMeasurement(MakeRecord("category:ffn_up", LlamaTensorType.Q4_K, 14.5));
            db.RecordMeasurement(MakeRecord("category:ffn_up", LlamaTensorType.Q2_K, 18.0));
            db.RecordMeasurement(MakeRecord("category:ffn_down", LlamaTensorType.Q4_K, 14.1));

            var profile = LlamaSensitivityProfile.DeriveFromDb(
                db, modelSha: "model_a", corpusSha: "corpus_a",
                imatrixSha: LlamaInvestigationDb.NoImatrixSha,
                contextSize: 512, archId: "qwen3", paramCount: 1_000_000_000, layerCount: 4);

            Assert.NotNull(profile);
            Assert.Equal(14.0, profile.F16BaselinePerplexity, precision: 4);

            Assert.True(profile.Categories.ContainsKey("ffn_up"));
            Assert.Equal(0.5, profile.Categories["ffn_up"].DeltaPplByType[LlamaTensorType.Q4_K], precision: 4);
            Assert.Equal(4.0, profile.Categories["ffn_up"].DeltaPplByType[LlamaTensorType.Q2_K], precision: 4);

            Assert.True(profile.Categories.ContainsKey("ffn_down"));
            Assert.Equal(0.1, profile.Categories["ffn_down"].DeltaPplByType[LlamaTensorType.Q4_K], precision: 4);
        }
        finally { TryDelete(path); }
    }

    [Fact]
    public void DeriveFromDb_PerTensorRows_PopulatePerTensorField()
    {
        var path = TempDbPath();
        try
        {
            using var db = LlamaInvestigationDb.Open(path);
            db.RecordMeasurement(MakeRecord("baseline", LlamaTensorType.F16, 14.0));
            db.RecordMeasurement(MakeRecord("category:ffn_up", LlamaTensorType.Q4_K, 14.5));
            db.RecordMeasurement(MakeRecord("tensor:blk.0.ffn_up.weight", LlamaTensorType.Q4_K, 14.05));
            db.RecordMeasurement(MakeRecord("tensor:blk.1.ffn_up.weight", LlamaTensorType.Q4_K, 14.20));

            var profile = LlamaSensitivityProfile.DeriveFromDb(
                db, modelSha: "model_a", corpusSha: "corpus_a",
                imatrixSha: LlamaInvestigationDb.NoImatrixSha,
                contextSize: 512, archId: "qwen3", paramCount: 1_000_000_000, layerCount: 4);

            Assert.NotNull(profile);
            Assert.NotNull(profile.PerTensor);
            Assert.Equal(2, profile.PerTensor!.Count);
            Assert.Equal(0.05, profile.PerTensor["blk.0.ffn_up.weight"].DeltaPplByType[LlamaTensorType.Q4_K], precision: 4);
            Assert.Equal(0.20, profile.PerTensor["blk.1.ffn_up.weight"].DeltaPplByType[LlamaTensorType.Q4_K], precision: 4);
        }
        finally { TryDelete(path); }
    }

    [Fact]
    public void DeriveFromDb_MultipleSamples_KeepsLatest()
    {
        // Re-running a measurement appends a new row; derivation should
        // pick the most recent one (Query returns DESC by date).
        var path = TempDbPath();
        try
        {
            using var db = LlamaInvestigationDb.Open(path);
            db.RecordMeasurement(MakeRecord("baseline", LlamaTensorType.F16, 14.0));
            db.RecordMeasurement(MakeRecord("category:ffn_up", LlamaTensorType.Q4_K, 14.5));
            // Slight delay so timestamps differ deterministically.
            System.Threading.Thread.Sleep(10);
            db.RecordMeasurement(MakeRecord("category:ffn_up", LlamaTensorType.Q4_K, 14.6));

            var profile = LlamaSensitivityProfile.DeriveFromDb(
                db, modelSha: "model_a", corpusSha: "corpus_a",
                imatrixSha: LlamaInvestigationDb.NoImatrixSha,
                contextSize: 512, archId: "qwen3", paramCount: 1_000_000_000, layerCount: 4);

            Assert.NotNull(profile);
            // Latest sample (14.6 - 14.0 = 0.6) wins.
            Assert.Equal(0.6, profile.Categories["ffn_up"].DeltaPplByType[LlamaTensorType.Q4_K], precision: 4);
        }
        finally { TryDelete(path); }
    }

    [Fact]
    public void Completeness_PartialProfile_ReportsMissingCells()
    {
        // Profile measured only Q4_K for two of three expected categories.
        var profile = MakeMinimalProfile(measuredCategories: new[] { "ffn_up", "ffn_down" });
        var c = profile.ComputeCompleteness(
            expectedCategories: new[] { "ffn_up", "ffn_down", "attn_q.weight" },
            expectedTypes:      new[] { LlamaTensorType.Q4_K });

        Assert.Equal(3, c.TotalCategoryCells);
        Assert.Equal(2, c.MeasuredCategoryCells);
        Assert.Single(c.MissingCategoryCells);
        Assert.Equal(("attn_q.weight", LlamaTensorType.Q4_K), c.MissingCategoryCells[0]);
        Assert.False(c.IsComplete);
    }

    [Fact]
    public void Completeness_FullyMeasured_ReportsComplete()
    {
        var profile = MakeMinimalProfile(measuredCategories: new[] { "ffn_up", "ffn_down" });
        var c = profile.ComputeCompleteness(
            expectedCategories: new[] { "ffn_up", "ffn_down" },
            expectedTypes:      new[] { LlamaTensorType.Q4_K });

        Assert.Equal(2, c.TotalCategoryCells);
        Assert.Equal(2, c.MeasuredCategoryCells);
        Assert.Empty(c.MissingCategoryCells);
        Assert.True(c.IsComplete);
    }

    [Fact]
    public void RecipeBuilder_PerTensorDataDemotesIndividualTensor()
    {
        // Profile says category ffn_up should be at Q6_K (catastrophic at Q2_K).
        // Per-tensor data shows blk.0.ffn_up.weight is fine at Q4_K.
        // With UsePerTensorData on, just that tensor demotes; others stay at Q6_K.
        var profile = new LlamaSensitivityProfile(
            SchemaVersion:         LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId:        "test",
            LayerCount:            4,
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
                        [LlamaTensorType.Q4_K] = 1.50,    // bad
                        [LlamaTensorType.Q6_K] = 0.05,    // good
                    },
                    RecommendedFloor: LlamaTensorType.Q4_K),
            },
            PerTensor: new Dictionary<string, LlamaSensitivityTensorCoefficient>
            {
                ["blk.0.ffn_up.weight"] = new LlamaSensitivityTensorCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q4_K] = 0.02,    // this specific tensor is fine at Q4_K
                        [LlamaTensorType.Q6_K] = 0.01,
                    }),
            });

        var layout = new List<(string Name, long Elements)>
        {
            ("blk.0.ffn_up.weight", 1_000_000L),    // has per-tensor data
            ("blk.1.ffn_up.weight", 1_000_000L),    // no per-tensor data
            ("blk.2.ffn_up.weight", 1_000_000L),
        };
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 7.0,    // budget high enough that Q6_K (6.5625) fits
            options: new LlamaQuantRecipeFromProfileOptions
            {
                ApplyStockBaseline = false,
                UsePerTensorData   = true,
                MinPredictedGainPpl = 0.0,    // disable snap so we see the refinement
            });

        var byName = recipe.Entries.ToDictionary(e => e.TensorName);
        // blk.0 demoted to Q4_K (per-tensor data permits)
        Assert.Equal(LlamaTensorType.Q4_K, byName["blk.0.ffn_up.weight"].ChosenType);
        // blk.1 and blk.2 stay at the category pick (Q6_K)
        Assert.Equal(LlamaTensorType.Q6_K, byName["blk.1.ffn_up.weight"].ChosenType);
        Assert.Equal(LlamaTensorType.Q6_K, byName["blk.2.ffn_up.weight"].ChosenType);
    }

    [Fact]
    public void RecipeBuilder_PerTensorPromotion_PromotesSensitiveTensorFundedByDemoteSavings()
    {
        // Per-category data says ffn_up is fine at Q4_K (small Δ), so
        // the optimizer picks Q4_K. Per-tensor data says blk.0 is
        // unusually sensitive at Q4_K (above the promotion threshold)
        // while blk.1, blk.2, blk.3 are unusually robust (Q3_K data
        // shows they tolerate it). The refinement should:
        //   - demote blk.1, blk.2, blk.3 to Q3_K (free bpw)
        //   - promote blk.0 to Q6_K (paid for by demote savings)
        // Bpw stays within the per-category budget.
        var profile = new LlamaSensitivityProfile(
            SchemaVersion:         LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId:        "test",
            LayerCount:            4,
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
                        [LlamaTensorType.Q2_K] = 4.0,    // catastrophic at Q2_K (knee)
                        [LlamaTensorType.Q3_K] = 1.0,    // measured so it appears in the ladder
                        [LlamaTensorType.Q4_K] = 0.50,   // category average
                        [LlamaTensorType.Q6_K] = 0.05,
                    },
                    RecommendedFloor: LlamaTensorType.Q3_K),
            },
            PerTensor: new Dictionary<string, LlamaSensitivityTensorCoefficient>
            {
                // Sensitive at Q4_K — wants promotion
                ["blk.0.ffn_up.weight"] = new LlamaSensitivityTensorCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q3_K] = 1.0,     // far above category 0.50 → demote blocked
                        [LlamaTensorType.Q4_K] = 0.20,    // > 0.05 promotion threshold
                        [LlamaTensorType.Q6_K] = 0.01,    // gain ≈ 0.19, ΔBpw ≈ 2.06 → 0.092 PPL/bpw
                    }),
                // Robust at Q3_K — safe to demote
                ["blk.1.ffn_up.weight"] = new LlamaSensitivityTensorCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q3_K] = 0.01,    // ≤ category 0.50 → demote OK
                        [LlamaTensorType.Q4_K] = 0.01,
                        [LlamaTensorType.Q6_K] = 0.005,
                    }),
                ["blk.2.ffn_up.weight"] = new LlamaSensitivityTensorCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q3_K] = 0.01,
                        [LlamaTensorType.Q4_K] = 0.01,
                        [LlamaTensorType.Q6_K] = 0.005,
                    }),
                ["blk.3.ffn_up.weight"] = new LlamaSensitivityTensorCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q3_K] = 0.01,
                        [LlamaTensorType.Q4_K] = 0.01,
                        [LlamaTensorType.Q6_K] = 0.005,
                    }),
            });

        var layout = new List<(string Name, long Elements)>
        {
            ("blk.0.ffn_up.weight", 1_000_000L),
            ("blk.1.ffn_up.weight", 1_000_000L),
            ("blk.2.ffn_up.weight", 1_000_000L),
            ("blk.3.ffn_up.weight", 1_000_000L),
        };
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5,    // Q4_K-class budget
            options: new LlamaQuantRecipeFromProfileOptions
            {
                ApplyStockBaseline       = false,
                UsePerTensorData         = true,
                AllowPerTensorPromotion  = true,
                PerTensorPromotionThresholdPpl = 0.05,
                MinPredictedGainPpl      = 0.0,    // disable snap-to-stock so the recipe diverges
            });

        var byName = recipe.Entries.ToDictionary(e => e.TensorName);
        // blk.0 promoted (sensitive at Q4_K, refinement found Q6_K worth the bpw)
        Assert.Equal(LlamaTensorType.Q6_K, byName["blk.0.ffn_up.weight"].ChosenType);
        // blk.1/2/3 stay at Q4_K (or below — IQ4_XS is also accepted as
        // "no worse than category" demote target; at minimum they don't promote)
        var bpw1 = byName["blk.1.ffn_up.weight"].BitsPerElement;
        var bpw2 = byName["blk.2.ffn_up.weight"].BitsPerElement;
        var bpw3 = byName["blk.3.ffn_up.weight"].BitsPerElement;
        Assert.True(bpw1 <= 4.5, $"blk.1 unexpectedly promoted to {byName["blk.1.ffn_up.weight"].ChosenType}");
        Assert.True(bpw2 <= 4.5, $"blk.2 unexpectedly promoted to {byName["blk.2.ffn_up.weight"].ChosenType}");
        Assert.True(bpw3 <= 4.5, $"blk.3 unexpectedly promoted to {byName["blk.3.ffn_up.weight"].ChosenType}");
    }

    [Fact]
    public void RecipeBuilder_PromotionDisabled_StillDemotesBugDoesNotPromote()
    {
        // Same shape as the previous test, but with AllowPerTensorPromotion=false.
        // blk.0 should NOT promote (preserves the existing demote-only contract).
        var profile = new LlamaSensitivityProfile(
            SchemaVersion:         LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId:        "test",
            LayerCount:            2,
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
                        [LlamaTensorType.Q2_K] = 4.0,
                        [LlamaTensorType.Q4_K] = 0.50,
                        [LlamaTensorType.Q6_K] = 0.05,
                    },
                    RecommendedFloor: LlamaTensorType.Q4_K),
            },
            PerTensor: new Dictionary<string, LlamaSensitivityTensorCoefficient>
            {
                ["blk.0.ffn_up.weight"] = new LlamaSensitivityTensorCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q4_K] = 0.20,
                        [LlamaTensorType.Q6_K] = 0.01,
                    }),
            });
        var layout = new List<(string Name, long Elements)>
        {
            ("blk.0.ffn_up.weight", 1_000_000L),
            ("blk.1.ffn_up.weight", 1_000_000L),
        };
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                ApplyStockBaseline      = false,
                UsePerTensorData        = true,
                AllowPerTensorPromotion = false,    // <-- promotion off
                MinPredictedGainPpl     = 0.0,
            });

        var byName = recipe.Entries.ToDictionary(e => e.TensorName);
        // blk.0 stays at category pick (Q4_K), no promotion despite high per-tensor delta
        Assert.Equal(LlamaTensorType.Q4_K, byName["blk.0.ffn_up.weight"].ChosenType);
    }

    [Fact]
    public void RecipeBuilder_PromotionBudgetExhausted_StopsBeforeBlowingBpw()
    {
        // Two sensitive tensors competing for promotion, only one cheap
        // demote available to fund it. The greedy promotion should pick
        // the higher gain-per-bit, leave the other at the category pick.
        // Sized so blk.0 (small) fits within blk.2's demote savings while
        // blk.1 (full size) does not.
        var profile = new LlamaSensitivityProfile(
            SchemaVersion:         LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId:        "test",
            LayerCount:            3,
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
                        [LlamaTensorType.Q2_K] = 4.0,
                        [LlamaTensorType.Q3_K] = 1.0,
                        [LlamaTensorType.Q4_K] = 0.50,
                        [LlamaTensorType.Q6_K] = 0.05,
                    },
                    RecommendedFloor: LlamaTensorType.Q3_K),
            },
            PerTensor: new Dictionary<string, LlamaSensitivityTensorCoefficient>
            {
                // High gain-per-bit promotion target (very sensitive at Q4_K, small tensor)
                ["blk.0.ffn_up.weight"] = new LlamaSensitivityTensorCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q3_K] = 1.0,     // demote blocked
                        [LlamaTensorType.Q4_K] = 0.40,
                        [LlamaTensorType.Q6_K] = 0.01,
                    }),
                // Lower gain-per-bit promotion target (sensitive but less so, full size)
                ["blk.1.ffn_up.weight"] = new LlamaSensitivityTensorCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q3_K] = 1.0,
                        [LlamaTensorType.Q4_K] = 0.15,
                        [LlamaTensorType.Q6_K] = 0.01,
                    }),
                // Lone demote candidate — savings only fund the cheaper promotion
                ["blk.2.ffn_up.weight"] = new LlamaSensitivityTensorCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q3_K] = 0.01,
                        [LlamaTensorType.Q4_K] = 0.01,
                        [LlamaTensorType.Q6_K] = 0.005,
                    }),
            });
        var layout = new List<(string Name, long Elements)>
        {
            ("blk.0.ffn_up.weight",   400_000L),    // small — promote cost ~825K bits
            ("blk.1.ffn_up.weight", 1_000_000L),    // full — promote cost ~2.06M bits
            ("blk.2.ffn_up.weight", 1_000_000L),    // demote saves ~1.06M bits
        };
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 4.5,
            options: new LlamaQuantRecipeFromProfileOptions
            {
                ApplyStockBaseline      = false,
                UsePerTensorData        = true,
                AllowPerTensorPromotion = true,
                MinPredictedGainPpl     = 0.0,
            });

        var byName = recipe.Entries.ToDictionary(e => e.TensorName);
        // blk.0 wins promotion (highest gain per bit)
        Assert.Equal(LlamaTensorType.Q6_K, byName["blk.0.ffn_up.weight"].ChosenType);
        // blk.1 doesn't fit within remaining savings → stays at category pick
        Assert.Equal(LlamaTensorType.Q4_K, byName["blk.1.ffn_up.weight"].ChosenType);
        // Total category bpw must not exceed the all-Q4_K naive budget
        // (sum of elements × 4.5 bpw).
        long totalElements = layout.Sum(l => l.Elements);
        var totalBits = recipe.Entries
            .Where(e => e.TensorName.Contains("ffn_up"))
            .Sum(e => e.BitsPerElement * e.ElementCount);
        Assert.True(totalBits <= totalElements * 4.5,
            $"Category bpw blew budget: {totalBits} > {totalElements * 4.5}");
    }

    [Fact]
    public void RecipeBuilder_UsePerTensorDataDisabled_IgnoresPerTensorField()
    {
        // Same profile as above but with UsePerTensorData = false.
        // Recipe should keep all tensors at the category pick.
        var profile = new LlamaSensitivityProfile(
            SchemaVersion:         LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId:        "test",
            LayerCount:            4,
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
                        [LlamaTensorType.Q4_K] = 1.50,
                        [LlamaTensorType.Q6_K] = 0.05,
                    },
                    RecommendedFloor: LlamaTensorType.Q4_K),
            },
            PerTensor: new Dictionary<string, LlamaSensitivityTensorCoefficient>
            {
                ["blk.0.ffn_up.weight"] = new LlamaSensitivityTensorCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q4_K] = 0.02,
                        [LlamaTensorType.Q6_K] = 0.01,
                    }),
            });

        var layout = new List<(string Name, long Elements)>
        {
            ("blk.0.ffn_up.weight", 1_000_000L),
        };
        var recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
            profile, layout, targetParameterCount: 100_000_000,
            targetBitsPerElement: 7.0,    // budget high enough that Q6_K (6.5625) fits
            options: new LlamaQuantRecipeFromProfileOptions
            {
                ApplyStockBaseline = false,
                UsePerTensorData   = false,
                MinPredictedGainPpl = 0.0,
            });

        // Without per-tensor refinement, blk.0 stays at category pick (Q6_K).
        Assert.Equal(LlamaTensorType.Q6_K, recipe.Entries[0].ChosenType);
    }

    private static LlamaSensitivityProfile MakeMinimalProfile(string[] measuredCategories) =>
        new(
            SchemaVersion:         LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId:        "test",
            LayerCount:            4,
            FamilyNotes:           null,
            Provenance:            new LlamaSensitivityProvenance(
                "ablation", null, 100_000_000, null, null, null),
            F16BaselinePerplexity: 10.0,
            BaselineContextSize:   512,
            Categories: measuredCategories.ToDictionary(
                c => c,
                c => new LlamaSensitivityCategoryCoefficient(
                    DeltaPplByType: new Dictionary<LlamaTensorType, double>
                    {
                        [LlamaTensorType.Q4_K] = 0.1,
                    },
                    RecommendedFloor: LlamaTensorType.Q4_K)));
}
