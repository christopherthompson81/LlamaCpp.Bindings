namespace LlamaCpp.Bindings;

/// <summary>
/// Shape of a category's bpw → ΔPPL curve, derived from its
/// <see cref="LlamaSensitivityCategoryCoefficient.DeltaPplByType"/>.
/// Used by the drill-candidates analyzer to flag categories whose
/// per-tensor story is likely to differ from their averaged
/// per-category story.
/// </summary>
public enum LlamaCategoryShape
{
    /// <summary>Not enough types in the ladder to characterize.</summary>
    NotEnoughData,
    /// <summary>
    /// Smooth, monotonic decrease as bpw increases — Q2_K worst,
    /// Q6_K best, no inversions, no large jumps. Drilling probably
    /// won't reveal heterogeneity.
    /// </summary>
    Smooth,
    /// <summary>
    /// One large jump on a ladder dense enough that intermediate
    /// rungs would have caught it if it weren't real. Drilling per
    /// layer is the recommended action — the catastrophe is likely
    /// concentrated in a few specific layers.
    /// </summary>
    Cliff,
    /// <summary>
    /// One large jump but the ladder is sparse around it: there are
    /// canonical-ladder types (Q3_K / IQ3_S / IQ4_XS / Q5_K) that
    /// would fit between the two cliff endpoints but haven't been
    /// measured. Adding intermediate rungs first is cheaper than
    /// drilling per-layer and will tell us whether the cliff is real
    /// (Q3_K is also catastrophic → drill) or a granularity artifact
    /// (Q3_K is mild → no drill needed).
    /// </summary>
    SparseCliff,
    /// <summary>
    /// Non-monotonic curve (e.g. Q6_K worse than Q4_K). Indicates
    /// per-tensor measurement noise or non-uniform layer behavior;
    /// drilling clarifies which.
    /// </summary>
    NonMonotonic,
}

/// <summary>
/// One row in the drill-candidates table — a category's analyzed
/// curve plus a recommendation hint for whether per-tensor drilling
/// is likely to be worthwhile.
/// </summary>
public sealed record LlamaDrillCandidate(
    string CategoryName,
    /// <summary>Largest <see cref="LlamaSensitivityCategoryCoefficient.DeltaPplByType"/> entry across the measured ladder.</summary>
    double WorstDeltaPpl,
    /// <summary>Type that hit <see cref="WorstDeltaPpl"/>; usually the lowest-bpw type measured.</summary>
    LlamaTensorType WorstType,
    /// <summary>Pre-existing floor from the profile — first type whose ΔPPL stays under the knee threshold.</summary>
    LlamaTensorType? RecommendedFloor,
    /// <summary>Curve shape — drives the recommendation.</summary>
    LlamaCategoryShape Shape,
    /// <summary>Human-readable hint, e.g. <c>"cliff at Q2_K — drill recommended"</c>.</summary>
    string Recommendation,
    /// <summary>Higher = more likely to benefit from drilling. Sort the table by this descending.</summary>
    double DrillPriority,
    /// <summary>True if the profile already contains per-tensor data covering this category.</summary>
    bool AlreadyDrilled,
    /// <summary>When <see cref="AlreadyDrilled"/>, std/mean of per-tensor ΔPPL at the worst type — high = heterogeneous.</summary>
    double? PerTensorRelativeStdAtWorst,
    /// <summary>Curve points sorted by ascending bpw (Q2_K first, Q6_K last). Useful for spark-line rendering.</summary>
    IReadOnlyList<(LlamaTensorType Type, double DeltaPpl, double BitsPerElement)> Curve);

/// <summary>
/// Analyzes a <see cref="LlamaSensitivityProfile"/> and returns one
/// <see cref="LlamaDrillCandidate"/> per category, sorted by drill
/// priority. Pure function — no I/O, no DB access — suitable for
/// running inside a UI binding to refresh the panel as the profile
/// is rebuilt.
/// </summary>
public static class LlamaSensitivityDrillCandidates
{
    /// <summary>
    /// Threshold for the cliff/smooth distinction. A "cliff" is when
    /// the largest jump between adjacent ladder entries is
    /// <see cref="CliffJumpRatio"/>× the second-largest. Tuned to
    /// catch Run 22's classic Q2_K-only catastrophes (e.g. ffn_down
    /// Q2_K × 28 layers = +3709 PPL) while ignoring mild slopes.
    /// </summary>
    public const double CliffJumpRatio = 4.0;

    /// <summary>
    /// Canonical practical-quant ladder, ordered by ascending bpw.
    /// Used to detect "sparse" cliffs — categories whose curve has
    /// gaps in this canonical sequence get the
    /// <see cref="LlamaCategoryShape.SparseCliff"/> recommendation
    /// (add the missing rungs before drilling per-layer). Excludes
    /// F16/BF16/F32 (lossless, no point in adding) and the ternary /
    /// FP4 families (rarely useful for AQ work).
    /// </summary>
    private static readonly LlamaTensorType[] CanonicalLadder = new[]
    {
        LlamaTensorType.IQ2_S,    // 2.5
        LlamaTensorType.Q2_K,     // 2.625
        LlamaTensorType.IQ3_S,    // 3.4375
        LlamaTensorType.Q3_K,     // 3.4375
        LlamaTensorType.IQ4_XS,   // 4.25
        LlamaTensorType.Q4_K,     // 4.5
        LlamaTensorType.Q5_K,     // 5.5
        LlamaTensorType.Q6_K,     // 6.5625
        LlamaTensorType.Q8_0,     // 8.5
    };

    /// <summary>
    /// Build the drill-candidates table from a profile. Categories
    /// without enough measured types (<2) get
    /// <see cref="LlamaCategoryShape.NotEnoughData"/> and a low
    /// priority — the user can address them by extending the ladder
    /// rather than drilling.
    /// </summary>
    public static IReadOnlyList<LlamaDrillCandidate> Analyze(LlamaSensitivityProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);
        var results = new List<LlamaDrillCandidate>(profile.Categories.Count);

        foreach (var (catName, coef) in profile.Categories)
        {
            var curve = coef.DeltaPplByType
                .Select(kv => (Type: kv.Key, DeltaPpl: kv.Value, Bpw: BitsPerElement(kv.Key)))
                .OrderBy(x => x.Bpw)
                .ToList();

            // Drilled if PerTensor exists with at least one tensor that
            // matches this category's name pattern.
            bool drilled = false;
            double? perTensorRelStdAtWorst = null;

            // Find worst (highest ΔPPL).
            double worstDelta = curve.Count > 0 ? curve.Max(x => x.DeltaPpl) : 0;
            var worstType = curve.Count > 0
                ? curve.OrderByDescending(x => x.DeltaPpl).First().Type
                : LlamaTensorType.F16;

            (LlamaCategoryShape shape, string hint, double cliffStrength) = ClassifyCurve(curve);

            // If we have per-tensor data, compute its variance at the
            // worst type — high relative std signals heterogeneity that
            // category-level numbers averaged away.
            if (profile.PerTensor is not null)
            {
                var matching = profile.PerTensor
                    .Where(kv => MatchesCategory(kv.Key, catName))
                    .Select(kv => kv.Value)
                    .ToList();
                if (matching.Count > 1)
                {
                    drilled = true;
                    var deltas = matching
                        .Where(t => t.DeltaPplByType.ContainsKey(worstType))
                        .Select(t => t.DeltaPplByType[worstType])
                        .ToList();
                    if (deltas.Count > 1)
                    {
                        var mean = deltas.Average();
                        var variance = deltas.Select(d => (d - mean) * (d - mean)).Average();
                        var std = Math.Sqrt(variance);
                        perTensorRelStdAtWorst = Math.Abs(mean) > 1e-9 ? std / Math.Abs(mean) : 0;
                    }
                }
            }

            // Drill priority = worst-ΔPPL × shape factor × (1 − already-drilled).
            // Already-drilled categories drop in priority because the
            // recipe builder is already using their per-tensor data.
            // Priority weighting: cliff > sparse-cliff > non-monotonic
            // > smooth. SparseCliff lands above NonMonotonic because
            // the recommended action (add rungs) is cheap and
            // diagnostic; the user should do it before drilling other
            // categories. But it's below proper Cliff because real
            // cliffs are the highest-yield drill targets.
            double shapeFactor = shape switch
            {
                LlamaCategoryShape.Cliff => 1.0 + cliffStrength,
                LlamaCategoryShape.SparseCliff => 0.8 + 0.5 * cliffStrength,
                LlamaCategoryShape.NonMonotonic => 0.7,
                LlamaCategoryShape.Smooth => 0.3,
                _ => 0.0,
            };
            double drilledFactor = drilled ? 0.2 : 1.0;
            double priority = Math.Max(0, worstDelta) * shapeFactor * drilledFactor;

            // Override hint when already drilled.
            if (drilled)
            {
                hint = perTensorRelStdAtWorst is double rs && rs > 0.5
                    ? $"already drilled — high per-layer variance ({rs:F2}× rel-std at {worstType})"
                    : "already drilled — uniform across layers";
            }

            results.Add(new LlamaDrillCandidate(
                CategoryName: catName,
                WorstDeltaPpl: worstDelta,
                WorstType: worstType,
                RecommendedFloor: coef.RecommendedFloor,
                Shape: shape,
                Recommendation: hint,
                DrillPriority: priority,
                AlreadyDrilled: drilled,
                PerTensorRelativeStdAtWorst: perTensorRelStdAtWorst,
                Curve: curve.Select(x => (x.Type, x.DeltaPpl, x.Bpw)).ToList()));
        }

        return results.OrderByDescending(c => c.DrillPriority).ToList();
    }

    /// <summary>
    /// Decide curve shape + hint string + cliff-strength scalar (used
    /// to weight the drill-priority score).
    /// </summary>
    private static (LlamaCategoryShape, string Hint, double CliffStrength)
        ClassifyCurve(List<(LlamaTensorType Type, double DeltaPpl, double Bpw)> sortedAscBpw)
    {
        if (sortedAscBpw.Count < 2)
            return (LlamaCategoryShape.NotEnoughData, "needs more types in the ladder", 0);

        // Walk by ascending bpw; deltas should generally decrease
        // (lower bpw = higher PPL = higher delta).
        // Inversion: any place where higher-bpw ΔPPL > lower-bpw ΔPPL.
        bool inverted = false;
        for (int i = 1; i < sortedAscBpw.Count; i++)
        {
            if (sortedAscBpw[i].DeltaPpl > sortedAscBpw[i - 1].DeltaPpl + 0.01)
            {
                inverted = true;
                break;
            }
        }
        if (inverted)
            return (LlamaCategoryShape.NonMonotonic,
                    "non-monotonic — drilling clarifies whether it's noise or layer heterogeneity",
                    0);

        // Compute jumps between adjacent ladder points (lower bpw vs higher).
        // jump[i] = delta[i] - delta[i+1]  (always positive in monotonic case).
        var jumps = new List<double>();
        for (int i = 0; i < sortedAscBpw.Count - 1; i++)
            jumps.Add(sortedAscBpw[i].DeltaPpl - sortedAscBpw[i + 1].DeltaPpl);
        if (jumps.Count == 0)
            return (LlamaCategoryShape.NotEnoughData, "needs more types in the ladder", 0);

        var sortedJumps = jumps.OrderByDescending(x => x).ToList();
        double largest = sortedJumps[0];
        double secondLargest = sortedJumps.Count > 1 ? sortedJumps[1] : 0;
        if (secondLargest < 0.001) secondLargest = 0.001;  // avoid div-by-zero
        double jumpRatio = largest / secondLargest;

        if (jumpRatio >= CliffJumpRatio)
        {
            // Find the type at the cliff's lower-bpw end and the next
            // measured rung above it.
            int cliffIdx = jumps.IndexOf(largest);
            var cliffLowType = sortedAscBpw[cliffIdx].Type;
            var cliffHighType = sortedAscBpw[cliffIdx + 1].Type;

            // Decide sparse vs dense: count canonical-ladder types
            // whose bpw falls strictly between the two cliff endpoints
            // and which haven't been measured. Any such type is a
            // candidate intermediate rung that would localize the
            // cliff before we commit to drilling.
            var measuredTypes = sortedAscBpw.Select(x => x.Type).ToHashSet();
            double lowBpw = sortedAscBpw[cliffIdx].Bpw;
            double highBpw = sortedAscBpw[cliffIdx + 1].Bpw;
            var unmeasuredInSpan = CanonicalLadder
                .Where(t =>
                {
                    double bpw = BitsPerElement(t);
                    return bpw > lowBpw && bpw < highBpw && !measuredTypes.Contains(t);
                })
                .ToList();

            double cliffStrength = Math.Min(2.0, jumpRatio / CliffJumpRatio);
            if (unmeasuredInSpan.Count > 0)
            {
                var suggested = string.Join(", ", unmeasuredInSpan);
                return (LlamaCategoryShape.SparseCliff,
                        $"cliff between {cliffLowType} and {cliffHighType} on a sparse ladder — " +
                        $"add intermediate rung(s) ({suggested}) before drilling to tell whether " +
                        $"the cliff is real or a granularity artifact",
                        cliffStrength);
            }
            return (LlamaCategoryShape.Cliff,
                    $"cliff between {cliffLowType} and {cliffHighType} on a dense ladder — " +
                    $"drill per-layer to find the responsible layers",
                    cliffStrength);
        }
        return (LlamaCategoryShape.Smooth, "smooth degradation — drilling unlikely to help much", 0);
    }

    /// <summary>
    /// Match the per-tensor key (e.g. <c>blk.0.attn_q.weight</c>) against
    /// a category name (e.g. <c>attn_q.weight</c> or <c>ffn_down</c>).
    /// Mirrors the <see cref="LlamaSensitivityProfileBuilder.CategoryMatch"/>
    /// rule but works at analyze time without a builder reference.
    /// </summary>
    private static bool MatchesCategory(string tensorName, string category) =>
        category.Contains('.')
            ? tensorName == category ||
              tensorName.EndsWith("." + category, StringComparison.Ordinal)
            : tensorName.Contains(category, StringComparison.Ordinal);

    private static double BitsPerElement(LlamaTensorType t) => t switch
    {
        LlamaTensorType.F32     => 32.0,
        LlamaTensorType.F16     => 16.0,
        LlamaTensorType.BF16    => 16.0,
        LlamaTensorType.Q8_0    => 8.5,
        LlamaTensorType.Q6_K    => 6.5625,
        LlamaTensorType.Q5_K    => 5.5,
        LlamaTensorType.Q4_K    => 4.5,
        LlamaTensorType.IQ4_XS  => 4.25,
        LlamaTensorType.Q3_K    => 3.4375,
        LlamaTensorType.IQ3_S   => 3.4375,
        LlamaTensorType.Q2_K    => 2.625,
        LlamaTensorType.IQ2_S   => 2.5,
        _                       => 8.0,
    };
}
