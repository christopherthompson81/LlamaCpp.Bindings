using LlamaCpp.Bindings;

namespace LlamaCpp.Bindings.Tests;

public class SensitivityDrillCandidatesTests
{
    private static LlamaSensitivityProfile MakeProfile(
        params (string Name, (LlamaTensorType Type, double Delta)[] Curve)[] categories)
    {
        var dict = new Dictionary<string, LlamaSensitivityCategoryCoefficient>();
        foreach (var (name, curve) in categories)
        {
            var deltas = new Dictionary<LlamaTensorType, double>();
            foreach (var (t, d) in curve) deltas[t] = d;
            dict[name] = new LlamaSensitivityCategoryCoefficient(deltas, RecommendedFloor: null);
        }
        return new LlamaSensitivityProfile(
            SchemaVersion: 1,
            ArchitectureId: "test",
            LayerCount: 28,
            FamilyNotes: null,
            Provenance: new LlamaSensitivityProvenance(
                "test", "test", 0, "test", DateTime.UtcNow, "test"),
            F16BaselinePerplexity: 16.9,
            BaselineContextSize: 512,
            Categories: dict);
    }

    [Fact]
    public void SmoothCurve_IsClassifiedSmooth()
    {
        var profile = MakeProfile((
            Name: "ffn_up",
            Curve: new[]
            {
                (LlamaTensorType.Q2_K, 1.5),
                (LlamaTensorType.Q4_K, 0.4),
                (LlamaTensorType.Q6_K, 0.05),
            }));

        var candidates = LlamaSensitivityDrillCandidates.Analyze(profile);
        Assert.Single(candidates);
        Assert.Equal(LlamaCategoryShape.Smooth, candidates[0].Shape);
        Assert.Contains("smooth", candidates[0].Recommendation);
    }

    [Fact]
    public void CliffCurve_IsClassifiedCliff_AndFlaggedForDrilling()
    {
        // Big jump from Q2_K (catastrophe) to Q4_K (small), then small
        // jump Q4_K → Q6_K. Run-22 ffn_down × Q2_K shape.
        var profile = MakeProfile((
            Name: "ffn_down",
            Curve: new[]
            {
                (LlamaTensorType.Q2_K, 100.0),
                (LlamaTensorType.Q4_K, 0.5),
                (LlamaTensorType.Q6_K, 0.05),
            }));

        var candidates = LlamaSensitivityDrillCandidates.Analyze(profile);
        Assert.Equal(LlamaCategoryShape.Cliff, candidates[0].Shape);
        Assert.Contains("cliff", candidates[0].Recommendation);
        Assert.Contains("drill recommended", candidates[0].Recommendation);
        Assert.True(candidates[0].DrillPriority > 100, "cliff drill priority should be high");
    }

    [Fact]
    public void NonMonotonicCurve_FlaggedForClarification()
    {
        // Q4_K is "worse" than Q2_K (inverted) — likely measurement noise
        // or layer heterogeneity averaging out.
        var profile = MakeProfile((
            Name: "attn_v.weight",
            Curve: new[]
            {
                (LlamaTensorType.Q2_K, 0.2),
                (LlamaTensorType.Q4_K, 0.8),  // higher delta despite higher bpw → inverted
                (LlamaTensorType.Q6_K, 0.05),
            }));

        var candidates = LlamaSensitivityDrillCandidates.Analyze(profile);
        Assert.Equal(LlamaCategoryShape.NonMonotonic, candidates[0].Shape);
        Assert.Contains("non-monotonic", candidates[0].Recommendation);
    }

    [Fact]
    public void AnalyzeOrdersByDrillPriority()
    {
        var profile = MakeProfile(
            (Name: "smooth_lowdelta",
             Curve: new[]
             {
                 (LlamaTensorType.Q2_K, 0.3),
                 (LlamaTensorType.Q4_K, 0.1),
                 (LlamaTensorType.Q6_K, 0.01),
             }),
            (Name: "cliff_highdelta",
             Curve: new[]
             {
                 (LlamaTensorType.Q2_K, 50.0),
                 (LlamaTensorType.Q4_K, 0.2),
                 (LlamaTensorType.Q6_K, 0.05),
             }));

        var candidates = LlamaSensitivityDrillCandidates.Analyze(profile);
        Assert.Equal(2, candidates.Count);
        // Cliff with higher worst Δ should rank ahead of smooth with low Δ.
        Assert.Equal("cliff_highdelta", candidates[0].CategoryName);
        Assert.Equal("smooth_lowdelta", candidates[1].CategoryName);
    }

    [Fact]
    public void AlreadyDrilledCategory_DropsInPriority_AndShowsVariance()
    {
        // Cliff in category aggregate, but per-tensor data exists.
        // High variance per-tensor at the worst type → "high variance"
        // hint; priority drops because the recipe builder is already
        // using the per-tensor data.
        var perTensor = new Dictionary<string, LlamaSensitivityTensorCoefficient>
        {
            ["blk.0.ffn_down.weight"] = new(
                new() { [LlamaTensorType.Q2_K] = 5.0 }),
            ["blk.13.ffn_down.weight"] = new(
                new() { [LlamaTensorType.Q2_K] = 200.0 }),
            ["blk.27.ffn_down.weight"] = new(
                new() { [LlamaTensorType.Q2_K] = 10.0 }),
        };
        var profile = new LlamaSensitivityProfile(
            SchemaVersion: 1,
            ArchitectureId: "test",
            LayerCount: 28,
            FamilyNotes: null,
            Provenance: new LlamaSensitivityProvenance(
                "test", "test", 0, "test", DateTime.UtcNow, "test"),
            F16BaselinePerplexity: 16.9,
            BaselineContextSize: 512,
            Categories: new()
            {
                ["ffn_down"] = new(new()
                {
                    [LlamaTensorType.Q2_K] = 70.0,
                    [LlamaTensorType.Q4_K] = 0.3,
                    [LlamaTensorType.Q6_K] = 0.05,
                }, RecommendedFloor: null),
            },
            PerTensor: perTensor);

        var candidates = LlamaSensitivityDrillCandidates.Analyze(profile);
        Assert.Single(candidates);
        Assert.True(candidates[0].AlreadyDrilled);
        Assert.NotNull(candidates[0].PerTensorRelativeStdAtWorst);
        Assert.True(candidates[0].PerTensorRelativeStdAtWorst!.Value > 1.0,
            "expected high relative std for skewed deltas (5/200/10)");
        Assert.Contains("already drilled", candidates[0].Recommendation);
    }

    [Fact]
    public void NotEnoughData_GetsLowPriority()
    {
        // Single-type profile — nothing to characterize.
        var profile = MakeProfile((
            Name: "thin",
            Curve: new[] { (LlamaTensorType.Q2_K, 5.0) }));

        var candidates = LlamaSensitivityDrillCandidates.Analyze(profile);
        Assert.Equal(LlamaCategoryShape.NotEnoughData, candidates[0].Shape);
        Assert.Equal(0.0, candidates[0].DrillPriority);
    }
}
