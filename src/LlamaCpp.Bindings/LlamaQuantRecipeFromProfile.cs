namespace LlamaCpp.Bindings;

/// <summary>
/// Tunable knobs for <see cref="LlamaQuantRecipeFromProfile.Build"/>.
/// </summary>
public sealed class LlamaQuantRecipeFromProfileOptions
{
    /// <summary>
    /// Pattern-based protection for uncategorized weight tensors.
    /// Maps a tensor-name pattern (suffix-match if the pattern contains
    /// a dot, contains-match otherwise — same matcher as profile
    /// categories) to a minimum type to use when the tensor doesn't
    /// match any profile category. First match wins.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Why this exists: Run 15 showed that a profile without
    /// <c>output.weight</c> as a category shipped a 4.7 bpw recipe
    /// that put <c>output.weight</c> at Q4_K (the
    /// <see cref="UncategorizedDefault"/>) and lost 3 PPL to stock
    /// Q4_K_M, because llama-quant's heuristic protects
    /// <c>output.weight</c> at Q6_K. Hand-rolled per-architecture
    /// protection lists are brittle; this generic pattern map preserves
    /// the spirit of llama-quant's <c>use_more_bits</c> table for the
    /// few tensors known to be high-leverage across all architectures.
    /// </para>
    /// <para>
    /// Defaults mirror what stock <c>llama_tensor_get_type</c> does on
    /// the highest-impact non-categorical tensors. Override or extend
    /// to cover architecture-specific cases (e.g. expert routers in
    /// MoE architectures).
    /// </para>
    /// </remarks>
    public IReadOnlyDictionary<string, LlamaTensorType> UncategorizedProtections { get; set; } =
        new Dictionary<string, LlamaTensorType>(StringComparer.Ordinal)
        {
            // Stock Q4_K_M assigns output.weight → Q6_K via use_more_bits.
            // The lm_head projection is the highest-leverage single tensor
            // in most decoder-only models; demoting it to Q4_K loses
            // ~3 PPL on Qwen3-1.7B (Run 15). Q6_K gives the protection
            // back without completely fixing what a proper profile
            // category would do.
            ["output.weight"] = LlamaTensorType.Q6_K,
            // Stock Q4_K_M leaves token_embd.weight at Q4_K. Keeping it
            // there matches stock; uncategorized handling will respect
            // this floor.
            ["token_embd.weight"] = LlamaTensorType.Q4_K,
        };

    /// <summary>
    /// Default type for uncategorized weight tensors that don't match
    /// any pattern in <see cref="UncategorizedProtections"/>. Default
    /// Q4_K — picks up the long tail of tensors the profile didn't
    /// characterize and there's no "high leverage" prior on.
    /// </summary>
    public LlamaTensorType UncategorizedDefault { get; set; } = LlamaTensorType.Q4_K;

    /// <summary>
    /// Exponent applied to the parameter-count ratio when projecting
    /// profile coefficients to a different-sized target:
    /// <c>scaled_coeff = source_coeff × (target_params / source_params)^exponent</c>.
    /// Default <c>1.0</c> (linear). Run 13's empirical Qwen3 0.6B → 1.7B
    /// data showed Q4_K coefficients scaled at ~0.85 (sub-linear), so
    /// linear is a conservative over-estimate of sensitivity. Override
    /// once cross-size validation runs prove a tighter exponent.
    /// </summary>
    public double SizeScalingExponent { get; set; } = 1.0;

    /// <summary>
    /// Override the target's parameter count. When null (default), the
    /// builder reads it from the GGUF. Useful for synthetic targets
    /// in tests or when callers already know the count.
    /// </summary>
    public long? TargetParameterCountOverride { get; set; }

    /// <summary>
    /// Tolerance band around <c>targetBitsPerElement</c>; the greedy
    /// walk stops when current bpw lands within this many bits.
    /// Default <c>0.05</c> bpw — tighter than that produces no
    /// observable file-size difference.
    /// </summary>
    public double BitsPerElementTolerance { get; set; } = 0.05;

    /// <summary>
    /// Optional override for the candidate type ladder. When null
    /// (default) the builder uses the union of types measured in the
    /// profile (typically Q2_K / Q4_K / Q6_K). To use intermediate
    /// types like Q5_K, the profile must have measured them — the
    /// builder refuses to interpolate, since the per-category curves
    /// have non-monotone knees (Run 11). Pass an explicit list to
    /// restrict further (e.g. forbid Q2_K family-wide).
    /// </summary>
    public IReadOnlyList<LlamaTensorType>? CandidateTypes { get; set; }

    /// <summary>
    /// Apply <see cref="LlamaStockBaseline"/> as a per-tensor floor:
    /// every tensor's effective type is <c>max(baseline, recipe-pick)</c>,
    /// so the recipe can promote above stock's per-layer protection
    /// but never demote below it. Default <c>true</c>. Run 15 showed
    /// that without this, profile recipes lose ~3 PPL to stock at
    /// Q4_K_M-class budgets because single-tensor profile ablations
    /// can't capture the per-layer interaction effects stock's
    /// <c>use_more_bits</c> alternation pattern protects against.
    /// </summary>
    public bool ApplyStockBaseline { get; set; } = true;

    /// <summary>
    /// Pre-built stock baseline map. When null and <see cref="ApplyStockBaseline"/>
    /// is true, the builder computes one from the target GGUF using
    /// <see cref="LlamaStockBaseline.Build"/>. Tests can supply an
    /// explicit map to exercise specific baseline scenarios.
    /// </summary>
    public IReadOnlyDictionary<string, LlamaTensorType>? StockBaselineMap { get; set; }

    /// <summary>
    /// Efficiency threshold in PPL-per-bpw units. The optimizer
    /// minimizes <c>pplSum + MinPplGainPerBpw × bpw</c> within the
    /// budget cap, so a promotion is only taken when its predicted
    /// PPL gain divided by added bpw exceeds this threshold.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Without this, strict <c>min(pplSum within budget)</c> happily
    /// trades 0.06 bpw for 0.001 PPL gain — visible on Run 17 as
    /// the algorithm picking <c>attn_v Q8_0×28</c> over
    /// <c>attn_v Q6_K×28</c> for a noise-level predicted improvement.
    /// Run 17b's variant B verified that switching attn_v back to
    /// Q6_K reclaimed 0.056 bpw AND slightly improved PPL — the
    /// strict-min-pplSum was actively harmful.
    /// </para>
    /// <para>
    /// Default <c>0.05</c> rejects trades worse than 0.05 PPL per
    /// added bpw. Set to <c>0</c> for the original strict-min
    /// behavior.
    /// </para>
    /// </remarks>
    public double MinPplGainPerBpw { get; set; } = 0.05;

    /// <summary>
    /// Snap-to-stock threshold in absolute PPL units. If the optimal
    /// recipe's predicted ΔPPL gain over the stock-equivalent
    /// assignment (every category at the highest ladder type whose
    /// bpw is ≤ <c>targetBitsPerElement</c>, baseline floors applied)
    /// falls below this value, the builder discards the recipe and
    /// emits the stock-equivalent assignment instead.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Why this exists: Run 20 (Qwen3-4B) showed the recipe builder
    /// will happily emit a recipe that differs from stock and predict
    /// a sub-0.2 PPL improvement when the profile has no real headroom
    /// to exploit (all Q4_K deltas &lt; 0.4, several already optimal
    /// or negative). A profile that says "stock is near-optimal"
    /// should produce a recipe that says the same — not a noise-level
    /// wiggle on top of stock that ships with implied confidence.
    /// </para>
    /// <para>
    /// Default <c>0.25</c> PPL — comfortably above measured noise on
    /// our wiki.test corpus while still letting through the 1.7B-class
    /// wins (those predict multi-PPL gains). Set to <c>0</c> to
    /// disable snapping.
    /// </para>
    /// </remarks>
    public double MinPredictedGainPpl { get; set; } = 0.25;

    /// <summary>
    /// When the profile has <see cref="LlamaSensitivityProfile.PerTensor"/>
    /// data, use it to refine per-tensor type choices after the
    /// per-category optimizer has run. Refinement is conservative:
    /// it can only <em>demote</em> a tensor to a lower-bpw type where
    /// the per-tensor data shows the demotion is essentially free —
    /// never promote (which would violate the budget the per-category
    /// pass committed to). Default <c>true</c>: when measured data
    /// disagrees with the category average, prefer the measurement.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Why demote-only by default: an ill-paired promotion violates the
    /// budget the per-category optimizer chose. Set
    /// <see cref="AllowPerTensorPromotion"/> to enable a per-category
    /// rebalance pass that spends demote savings on promotions where the
    /// per-tensor signal warrants it.
    /// </para>
    /// </remarks>
    public bool UsePerTensorData { get; set; } = true;

    /// <summary>
    /// Enable per-tensor <em>promotion</em> alongside the existing
    /// demote-only refinement. When on, the per-tensor refinement
    /// works per-category: demotes and promotes share a common budget
    /// pool such that the category's total bpw never exceeds what the
    /// category-level optimizer's pick would have allocated. Useful
    /// when individual tensors disagree sharply with the category
    /// average — Qwen3-1.7B ffn_down has 3 layers with Q4_K Δ &gt;
    /// 0.05 PPL while the category average is ~0.004; promoting just
    /// those layers is funded by demoting the much larger set of
    /// near-zero-Δ layers.
    /// </summary>
    /// <remarks>
    /// Default <c>false</c> preserves the demote-only behavior. Turn
    /// on after measuring per-tensor data and confirming individual
    /// layers materially deviate from the category average. Bpw is
    /// strictly conserved within the category — the recipe builder
    /// won't blow its budget.
    /// </remarks>
    public bool AllowPerTensorPromotion { get; set; } = false;

    /// <summary>
    /// Absolute per-tensor ΔPPL threshold (in PPL units) above which
    /// a tensor at the category-chosen type is considered "materially
    /// sensitive" and becomes a promotion candidate. Below this
    /// threshold the per-tensor signal is treated as noise-level and
    /// not worth committing budget to.
    /// </summary>
    /// <remarks>
    /// Default <c>0.05</c> PPL — comfortably above measured noise on
    /// our wiki.test corpus, captures genuinely-sensitive layers like
    /// Qwen3-1.7B blk.3/7 ffn_down (Δ ≈ 0.06–0.07 at Q4_K) without
    /// over-promoting on noise. Lower it to be more aggressive with
    /// promotions; raise it to be more conservative.
    /// </remarks>
    public double PerTensorPromotionThresholdPpl { get; set; } = 0.05;
}

/// <summary>
/// Build a <see cref="LlamaQuantRecipe"/> from a
/// <see cref="LlamaSensitivityProfile"/> and a target model. The
/// profile supplies per-category ΔPPL coefficients; this picks a
/// type per category to hit a bpw budget while honoring each
/// category's recommended floor.
/// </summary>
/// <remarks>
/// <para>
/// Algorithm — exhaustive enumeration over per-category type choices,
/// filtered by floors. With 7 categories × 3 measured types that's
/// ≤ 2187 combinations, trivially enumerable. Selection objective:
/// minimize total predicted ΔPPL subject to weighted bpw ≤ target +
/// tolerance. Falls back to lowest-PPL recipe overall when the budget
/// is infeasible (collective floors exceed the budget).
/// </para>
/// <para>
/// Coefficients are size-scaled — a profile built on a 0.6B model
/// applied to a 1.7B target multiplies its coefficients by
/// <c>(1.7/0.6)^exponent</c>. Default exponent <c>1.0</c> is the
/// safe (over-estimating) choice; cross-size validation refines it.
/// </para>
/// <para>
/// The output recipe is meant to be consumed by
/// <see cref="LlamaCustomQuantizer.QuantizeWithRecipeAsync"/>, which
/// realizes the per-tensor type choices verbatim. The legacy path
/// through <see cref="LlamaQuantizer.QuantizeAsync"/> with
/// <see cref="LlamaQuantizationParameters.TensorTypeOverrides"/>
/// will silently drop demotions, so recipe-built recipes ship larger
/// than predicted on that path (Run 14).
/// </para>
/// </remarks>
public static class LlamaQuantRecipeFromProfile
{
    public static LlamaQuantRecipe Build(
        LlamaSensitivityProfile profile,
        string targetModelPath,
        double targetBitsPerElement,
        LlamaQuantRecipeFromProfileOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentException.ThrowIfNullOrEmpty(targetModelPath);
        if (!File.Exists(targetModelPath))
            throw new FileNotFoundException($"Target model not found: {targetModelPath}", targetModelPath);

        var opts = options ?? new LlamaQuantRecipeFromProfileOptions();
        LlamaBackend.EnsureInitialized();

        // Read target tensor layout, then defer to the core algorithm.
        var ggufFile = LlamaGgufFile.Open(targetModelPath);
        var weightTensors = ggufFile.Tensors
            .Where(t => t.Dimensions.Length > 1 && t.Name.EndsWith(".weight", StringComparison.Ordinal))
            .Select(t => (Name: t.Name, Elements: t.Dimensions.Aggregate(1L, (a, b) => a * (long)b)))
            .ToList();
        long targetParamCount = opts.TargetParameterCountOverride ??
            ggufFile.Tensors.Sum(t => t.Dimensions.Aggregate(1L, (a, b) => a * (long)b));

        // Auto-compute the stock baseline if enabled and not already
        // supplied. Layer count comes from the architecture's block_count
        // metadata; falls back to the profile's LayerCount.
        if (opts.ApplyStockBaseline && opts.StockBaselineMap is null)
        {
            int layerCount = ResolveLayerCount(ggufFile, profile.LayerCount);
            var baselineInput = ggufFile.Tensors
                .Where(t => t.Dimensions.Length > 1 && t.Name.EndsWith(".weight", StringComparison.Ordinal))
                .Select(t => (Name: t.Name, Dimensions: t.Dimensions))
                .ToList();
            opts.StockBaselineMap = LlamaStockBaseline.Build(baselineInput, layerCount);
        }

        return BuildFromTensorLayout(profile, weightTensors, targetParamCount, targetBitsPerElement, opts);
    }

    private static int ResolveLayerCount(LlamaGgufFile ggufFile, int profileFallback)
    {
        var arch = ggufFile.Metadata.FirstOrDefault(m => m.Key == "general.architecture")
            ?.Value.AsString();
        if (string.IsNullOrEmpty(arch)) return profileFallback > 0 ? profileFallback : 1;
        var entry = ggufFile.Metadata.FirstOrDefault(m => m.Key == $"{arch}.block_count");
        if (entry is null) return profileFallback > 0 ? profileFallback : 1;
        return entry.Value.Type switch
        {
            LlamaGgufType.Uint32 => (int)entry.Value.AsUInt32(),
            LlamaGgufType.Int32  => entry.Value.AsInt32(),
            LlamaGgufType.Uint64 => (int)entry.Value.AsUInt64(),
            _                    => profileFallback > 0 ? profileFallback : 1,
        };
    }

    /// <summary>
    /// Algorithm core: takes a pre-resolved tensor layout instead of a
    /// GGUF path. Public so tests (and synthetic targets) can exercise
    /// the greedy walk without a real model file. Production callers
    /// should use <see cref="Build"/>.
    /// </summary>
    public static LlamaQuantRecipe BuildFromTensorLayout(
        LlamaSensitivityProfile profile,
        IReadOnlyList<(string Name, long Elements)> weightTensors,
        long targetParameterCount,
        double targetBitsPerElement,
        LlamaQuantRecipeFromProfileOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(weightTensors);
        if (!(targetBitsPerElement > 0))
            throw new ArgumentOutOfRangeException(nameof(targetBitsPerElement), "Must be > 0.");
        var opts = options ?? new LlamaQuantRecipeFromProfileOptions();
        LlamaBackend.EnsureInitialized();
        long targetParamCount = opts.TargetParameterCountOverride ?? targetParameterCount;

        // ---- 2. Size-scaling factor ----
        double sizeScale = 1.0;
        if (profile.Provenance.SourceParameterCount is long srcParams && srcParams > 0 && targetParamCount > 0)
        {
            var ratio = (double)targetParamCount / srcParams;
            sizeScale = Math.Pow(ratio, opts.SizeScalingExponent);
        }

        // ---- 3. Candidate ladder = profile-measured types (intersected with opts) ----
        var measuredTypes = profile.Categories.Values
            .SelectMany(c => c.DeltaPplByType.Keys)
            .Distinct()
            .ToHashSet();
        var ladder = (opts.CandidateTypes ?? measuredTypes.ToList())
            .Where(measuredTypes.Contains)
            .Distinct()
            .OrderBy(LlamaQuantRecipe.GetBitsPerElement)
            .ToList();
        if (ladder.Count == 0)
            throw new InvalidOperationException(
                "Empty candidate ladder. The profile measured no types matching options.CandidateTypes.");

        // ---- 4. Map each weight tensor to its category (or "uncategorized") ----
        // A tensor that matches multiple categories takes the first match in
        // declaration order — same matcher as LlamaSensitivityProfileBuilder.
        var tensorCategory = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (name, _) in weightTensors)
        {
            foreach (var cat in profile.Categories.Keys)
            {
                if (CategoryMatch(name, cat))
                {
                    tensorCategory[name] = cat;
                    break;
                }
            }
        }

        // ---- 5. Element-count totals per category, with stock-baseline floor ----
        // For uncategorized tensors we resolve their effective type up front
        // (max of stock baseline and the user's protection table). Their bit
        // contribution is constant across recipe enumeration.
        //
        // For categorized tensors we precompute, *per (category, candidate type T)*:
        //   - bits if cat is at T: sum over t in cat of bpw(max(baseline(t), T)) × elements(t)
        //   - elements at T (vs riding baseline): used for ΔPPL pro-rating
        // Stock baseline acts as a per-tensor floor: the recipe can promote
        // above it but never demote below. This guarantees recipes ship at
        // least as good as stock at the protected layers.
        var baseline = opts.StockBaselineMap ?? new Dictionary<string, LlamaTensorType>(StringComparer.Ordinal);

        LlamaTensorType MaxType(LlamaTensorType a, LlamaTensorType b) =>
            LlamaQuantRecipe.GetBitsPerElement(a) >= LlamaQuantRecipe.GetBitsPerElement(b) ? a : b;

        var categoryElements = new Dictionary<string, long>(StringComparer.Ordinal);
        var uncategorizedTypeByName = new Dictionary<string, LlamaTensorType>(StringComparer.Ordinal);
        long uncategorizedElements = 0;
        double uncategorizedBitsTotal = 0;
        // Per (cat, type) → (bits, elements_at_or_below_baseline-overridden, elements_at_T).
        var categoryBitsAtType = new Dictionary<(string Cat, LlamaTensorType T), double>();
        var categoryAtTElements = new Dictionary<(string Cat, LlamaTensorType T), long>();

        foreach (var (name, elements) in weightTensors)
        {
            if (tensorCategory.TryGetValue(name, out var cat))
            {
                categoryElements[cat] = categoryElements.GetValueOrDefault(cat) + elements;

                // Precompute (bits, atTElements) for every candidate type for this tensor.
                baseline.TryGetValue(name, out var b);  // default = LlamaTensorType.None ≈ no opinion
                bool hasBaseline = baseline.ContainsKey(name);
                foreach (var T in ladder)
                {
                    LlamaTensorType effective = hasBaseline ? MaxType(b, T) : T;
                    var bits = LlamaQuantRecipe.GetBitsPerElement(effective) * elements;
                    var key = (cat, T);
                    categoryBitsAtType[key] = categoryBitsAtType.GetValueOrDefault(key) + bits;
                    if (effective == T)  // tensor uses category type, not baseline
                        categoryAtTElements[key] = categoryAtTElements.GetValueOrDefault(key) + elements;
                }
            }
            else
            {
                // Uncategorized: max(stock baseline, protection-table type).
                var protection = ResolveUncategorizedType(name, opts);
                var effective = baseline.TryGetValue(name, out var b)
                    ? MaxType(b, protection)
                    : protection;
                uncategorizedTypeByName[name] = effective;
                uncategorizedElements += elements;
                uncategorizedBitsTotal += LlamaQuantRecipe.GetBitsPerElement(effective) * elements;
            }
        }
        long totalElements = categoryElements.Values.Sum() + uncategorizedElements;
        if (totalElements <= 0)
            throw new InvalidOperationException("Target model has no weight tensors.");

        // ---- 6. Pick a type per category ----
        // Exhaustive enumeration over the candidate ladder, filtered by
        // each category's floor. With 7 categories × 3 types that's
        // ≤ 2187 combinations — trivial. Greedy with sparse ladders
        // (Q2_K → Q4_K → Q6_K, +2 bpw gaps) overshoots wildly because
        // the smallest move per-category is +2 bpw / category-fraction,
        // which is bigger than typical tolerance bands. Exhaustive is
        // optimal, deterministic, and cheap at this scale. Revisit if
        // candidate sets grow past ~10 types × 8+ categories.
        //
        // Clamp the per-category deltas before scoring:
        //   1. Zero-floor: a quantized type can never strictly improve
        //      PPL over F16, so any δ < 0 is measurement noise (small
        //      effect size + finite-corpus PPL variance). Floor at 0.
        //   2. Monotone-from-above-bpw: a lower-bpw type can't have
        //      smaller PPL impact than a higher-bpw type from the same
        //      category. Walk the ladder by bpw descending; each lower
        //      type's δ is at least its higher neighbor's δ.
        // The clamp is uniform across categories — no special-casing
        // attn_v or other "noisy" tensors. The combined clamp
        // automatically degrades to a no-op for cleanly monotone
        // categories (1.7B's ffn_down Q2_K=3709 is unaffected) and
        // suppresses noise where it dominates (4B's attn_v Q2_K=−0.45
        // becomes 0.93 from monotone-clamping against Q3_K).
        var clampedDelta = BuildClampedScaledDeltas(profile, ladder, sizeScale);
        double ScaledDelta(string cat, LlamaTensorType type) =>
            clampedDelta.GetValueOrDefault((cat, type), 0.0);

        // Only enumerate over categories that have at least one matching
        // tensor in the target model. This skips e.g. <c>output.weight</c>
        // when the profile has data for it but the target uses tied
        // embeddings (Qwen3-4B, Llama-3.2-1B): the profile category is
        // present but no target tensor matches, so it contributes nothing
        // and would otherwise crash the per-(cat, T) lookup below.
        var perCategoryChoices = new Dictionary<string, IReadOnlyList<LlamaTensorType>>();
        foreach (var (cat, coef) in profile.Categories)
        {
            if (!categoryElements.ContainsKey(cat)) continue;  // no matching target tensors

            var allowed = ladder
                .Where(t => coef.RecommendedFloor is not LlamaTensorType f ||
                            LlamaQuantRecipe.GetBitsPerElement(t) >= LlamaQuantRecipe.GetBitsPerElement(f))
                .ToList();
            if (allowed.Count == 0)
                allowed = new List<LlamaTensorType> { ladder[^1] };  // floor outside ladder — pin to top
            perCategoryChoices[cat] = allowed;
        }

        // Selection objective: target bpw is a *budget*. Minimize predicted
        // total ΔPPL subject to bpw ≤ target + tolerance. If no in-budget
        // assignment exists (all category floors collectively force higher
        // bpw than the budget), fall back to the in-budget-or-over recipe
        // with the lowest predicted ΔPPL — better to ship a good recipe
        // slightly over budget than refuse to build one.
        double budgetCap = targetBitsPerElement + opts.BitsPerElementTolerance;
        // Composite score: pplSum + MinPplGainPerBpw × bpw. With λ in
        // PPL-per-bpw units, an assignment that costs +Δbpw and saves
        // -Δppl is preferred only when Δppl/Δbpw > λ. Threshold defaults
        // to 0.05 so we don't burn ~0.06 bpw chasing 0.001 PPL gains
        // (Run 17b variant B finding).
        double Score(double pplSum, double bpw) =>
            pplSum + opts.MinPplGainPerBpw * bpw;

        Dictionary<string, LlamaTensorType>? bestInBudget = null;
        double bestInBudgetScore = double.PositiveInfinity;
        Dictionary<string, LlamaTensorType>? bestOverall = null;
        double bestOverallScore = double.PositiveInfinity;

        var catList = perCategoryChoices.Keys.ToList();
        var current = new Dictionary<string, LlamaTensorType>(StringComparer.Ordinal);
        EnumerateAssignments(catList, 0, perCategoryChoices, current, assignment =>
        {
            double bpwSum = 0;
            double pplSum = 0;
            foreach (var (cat, type) in assignment)
            {
                // Per-(cat, T) bit contribution accounts for stock baseline:
                // tensors whose baseline > T stay at baseline, others use T.
                bpwSum += categoryBitsAtType[(cat, type)];
                // Pro-rate the predicted ΔPPL by the fraction of category
                // elements actually riding at the recipe type (not the
                // baseline override). Baseline-overridden tensors are
                // assumed "good enough" — they don't count against this
                // category's predicted cost.
                long catTotal = categoryElements[cat];
                long atT = categoryAtTElements.GetValueOrDefault((cat, type));
                if (catTotal > 0)
                    pplSum += ScaledDelta(cat, type) * ((double)atT / catTotal);
            }
            bpwSum += uncategorizedBitsTotal;
            var bpw = bpwSum / totalElements;
            var score = Score(pplSum, bpw);

            if (score < bestOverallScore)
            {
                bestOverallScore = score;
                bestOverall = new Dictionary<string, LlamaTensorType>(assignment, StringComparer.Ordinal);
            }
            if (bpw <= budgetCap && score < bestInBudgetScore)
            {
                bestInBudgetScore = score;
                bestInBudget = new Dictionary<string, LlamaTensorType>(assignment, StringComparer.Ordinal);
            }
        });

        var pick = bestInBudget ?? bestOverall
            ?? throw new InvalidOperationException(
                "Recipe enumeration produced no assignment — empty category space.");

        // ---- 6b. Snap to stock when predicted gain is below threshold ----
        // Build a "stock-equivalent" assignment: every category at the
        // highest ladder type whose bpw is ≤ targetBitsPerElement (with
        // each category's floor honored). Compare its predicted pplSum
        // to the picked recipe's pplSum; if the gain is below the
        // configured threshold, use the stock-equivalent instead.
        // Rationale: when the profile says "no real headroom"
        // (Run 20 case), shipping a noise-level wiggle off stock is
        // strictly worse than just declaring stock optimal.
        if (opts.MinPredictedGainPpl > 0 && perCategoryChoices.Count > 0)
        {
            var stockType = ladder
                .Where(t => LlamaQuantRecipe.GetBitsPerElement(t) <= targetBitsPerElement)
                .DefaultIfEmpty(ladder[0])
                .Last();

            var stockEquivalent = new Dictionary<string, LlamaTensorType>(StringComparer.Ordinal);
            foreach (var (cat, allowed) in perCategoryChoices)
            {
                // Pick the allowed type closest to stockType from below;
                // floor handling means some categories may be pinned above.
                var chosen = allowed
                    .Where(t => LlamaQuantRecipe.GetBitsPerElement(t) <=
                                LlamaQuantRecipe.GetBitsPerElement(stockType))
                    .DefaultIfEmpty(allowed[0])
                    .Last();
                stockEquivalent[cat] = chosen;
            }

            double pickPplSum = 0;
            double stockPplSum = 0;
            foreach (var (cat, type) in pick)
            {
                long catTotal = categoryElements[cat];
                long atT = categoryAtTElements.GetValueOrDefault((cat, type));
                if (catTotal > 0)
                    pickPplSum += ScaledDelta(cat, type) * ((double)atT / catTotal);
            }
            foreach (var (cat, type) in stockEquivalent)
            {
                long catTotal = categoryElements[cat];
                long atT = categoryAtTElements.GetValueOrDefault((cat, type));
                if (catTotal > 0)
                    stockPplSum += ScaledDelta(cat, type) * ((double)atT / catTotal);
            }

            double predictedGain = stockPplSum - pickPplSum;
            if (predictedGain < opts.MinPredictedGainPpl)
            {
                pick = stockEquivalent;
            }
        }

        // ---- 7. Materialize per-tensor entries ----
        // ---- 7. Per-category per-tensor refinement (optional) ----
        // Run RefineCategoryPerTensor once per category before the entry
        // materialization loop. The category function does demote-only
        // by default; with AllowPerTensorPromotion on, it also runs a
        // promotion pass funded by demote savings within the same
        // category. Bpw is strictly conserved at the category level —
        // the recipe builder never blows the budget the per-category
        // optimizer chose.
        var perTensorRefined = new Dictionary<string, (LlamaTensorType Type, double Delta)>(StringComparer.Ordinal);
        if (opts.UsePerTensorData && profile.PerTensor is { } perTensorMap)
        {
            // Group weight tensors by category so we can refine each
            // category independently.
            var byCategory = new Dictionary<string, List<(string Name, long Elements)>>(StringComparer.Ordinal);
            foreach (var (name, elements) in weightTensors)
            {
                if (!tensorCategory.TryGetValue(name, out var cat)) continue;
                if (!byCategory.TryGetValue(cat, out var list))
                    byCategory[cat] = list = new List<(string, long)>();
                list.Add((name, elements));
            }

            foreach (var (cat, tensorsInCat) in byCategory)
            {
                if (!pick.TryGetValue(cat, out var picked)) continue;
                // Skip refinement when no tensor in this category has per-tensor data.
                var perTensorInCat = tensorsInCat
                    .Where(t => perTensorMap.ContainsKey(t.Name))
                    .ToDictionary(t => t.Name, t => perTensorMap[t.Name], StringComparer.Ordinal);
                if (perTensorInCat.Count == 0) continue;

                var refined = RefineCategoryPerTensor(
                    cat, picked, ScaledDelta(cat, picked),
                    tensorsInCat,
                    perTensorInCat,
                    ladder, sizeScale,
                    floor: profile.Categories[cat].RecommendedFloor,
                    baseline: baseline,
                    minPplGainPerBpw: opts.MinPplGainPerBpw,
                    allowPromotion: opts.AllowPerTensorPromotion,
                    promotionThreshold: opts.PerTensorPromotionThresholdPpl);
                foreach (var kv in refined)
                    perTensorRefined[kv.Key] = kv.Value;
            }
        }

        // Each tensor's effective type = the per-category-refined pick
        // when available, otherwise max(baseline, category pick).
        var entries = new List<LlamaQuantRecipeEntry>(weightTensors.Count);
        foreach (var (name, elements) in weightTensors)
        {
            LlamaTensorType chosen;
            double estDelta;
            if (perTensorRefined.TryGetValue(name, out var refined))
            {
                chosen = refined.Type;
                estDelta = refined.Delta;
            }
            else if (tensorCategory.TryGetValue(name, out var cat))
            {
                var picked = pick[cat];
                chosen = baseline.TryGetValue(name, out var b) ? MaxType(b, picked) : picked;
                estDelta = chosen == picked ? ScaledDelta(cat, picked) : 0.0;
            }
            else
            {
                chosen = uncategorizedTypeByName[name];
                estDelta = 0.0;  // no profile signal — baseline + protection table picked it
            }
            entries.Add(new LlamaQuantRecipeEntry(
                TensorName:        name,
                ChosenType:        chosen,
                BitsPerElement:    LlamaQuantRecipe.GetBitsPerElement(chosen),
                RelativeMse:       estDelta,    // repurposed: predicted ΔPPL × sizeScale
                ExceededThreshold: false,
                ElementCount:      elements));
        }

        return new LlamaQuantRecipe(
            Threshold:            targetBitsPerElement,    // repurposed: target bpw
            SourceScoreTablePath: $"profile:{profile.ArchitectureId}@{profile.Provenance.SourceModel}",
            Entries:              entries,
            BuiltAtUtc:           DateTime.UtcNow);
    }

    /// <summary>
    /// Recursive depth-first walk over the per-category choice space.
    /// At a leaf (all categories assigned) the visitor sees the
    /// completed assignment via <paramref name="visit"/>; the visitor
    /// must not retain a reference to the dictionary, since the
    /// recursion mutates it as it backtracks.
    /// </summary>
    private static void EnumerateAssignments(
        IReadOnlyList<string> categories,
        int index,
        IReadOnlyDictionary<string, IReadOnlyList<LlamaTensorType>> choices,
        Dictionary<string, LlamaTensorType> current,
        Action<Dictionary<string, LlamaTensorType>> visit)
    {
        if (index == categories.Count)
        {
            visit(current);
            return;
        }
        var cat = categories[index];
        foreach (var t in choices[cat])
        {
            current[cat] = t;
            EnumerateAssignments(categories, index + 1, choices, current, visit);
        }
        current.Remove(cat);
    }

    /// <summary>
    /// Per-category refinement of per-tensor type choices given the
    /// category-level optimizer's pick. Runs two passes:
    /// </summary>
    /// <remarks>
    /// <list type="number">
    ///   <item><b>Demote.</b> For each tensor with per-tensor data,
    ///     walk the ladder from lowest bpw to highest. The first type
    ///     whose clamped per-tensor delta is ≤ the category-level
    ///     predicted delta is the cheapest "no worse than the category
    ///     average" pick. The category-level delta is the right yardstick
    ///     because it's what the optimizer used to commit to this
    ///     tensor's bpw; a per-tensor coefficient below it means this
    ///     specific tensor is less sensitive than its peers.</item>
    ///   <item><b>Promote</b> (only when <paramref name="allowPromotion"/>).
    ///     Identify tensors whose per-tensor delta at the
    ///     currently-assigned type exceeds <paramref name="promotionThreshold"/>
    ///     — these are individually sensitive in a way the category
    ///     average smooths over. Find each such tensor's best higher-bpw
    ///     target (highest gain-per-added-bpw), then apply promotions
    ///     greedily until the savings pool from pass 1's demotions is
    ///     exhausted. Bpw is strictly conserved at the category level.</item>
    /// </list>
    /// <para>
    /// Per-tensor data not in the profile for a tensor leaves that tensor
    /// at its category-level pick (with the baseline floor honored). The
    /// returned dictionary contains an entry for every tensor in
    /// <paramref name="tensorsInCategory"/> regardless of whether it
    /// had per-tensor data, so callers can use it as a complete
    /// per-tensor assignment for the category.
    /// </para>
    /// </remarks>
    private static IReadOnlyDictionary<string, (LlamaTensorType Type, double Delta)>
        RefineCategoryPerTensor(
            string categoryName,
            LlamaTensorType categoryChosen,
            double categoryDelta,
            IReadOnlyList<(string Name, long Elements)> tensorsInCategory,
            IReadOnlyDictionary<string, LlamaSensitivityTensorCoefficient> perTensor,
            IReadOnlyList<LlamaTensorType> ladder,
            double sizeScale,
            LlamaTensorType? floor,
            IReadOnlyDictionary<string, LlamaTensorType> baseline,
            double minPplGainPerBpw,
            bool allowPromotion,
            double promotionThreshold)
    {
        var assignment = new Dictionary<string, (LlamaTensorType Type, double Delta)>(StringComparer.Ordinal);
        var floorBpw = floor is { } f ? LlamaQuantRecipe.GetBitsPerElement(f) : 0.0;

        // Build per-tensor clamped delta tables (zero-floor + monotone-from-above-bpw),
        // and track which types each tensor actually has MEASURED data for.
        // The clamp fills missing types via monotone-from-above so they
        // appear "no worse than the next-higher measured type" — fine for
        // noise smoothing on measured negatives, but treating monotone-
        // filled types as valid demote/promote targets would extrapolate
        // beyond the data. Restrict refinement decisions to measured types
        // only.
        var clampedByTensor = new Dictionary<string, Dictionary<LlamaTensorType, double>>(StringComparer.Ordinal);
        var measuredByTensor = new Dictionary<string, HashSet<LlamaTensorType>>(StringComparer.Ordinal);
        foreach (var (name, _) in tensorsInCategory)
        {
            if (!perTensor.TryGetValue(name, out var coef)) continue;
            var clamped = new Dictionary<LlamaTensorType, double>();
            // Per-family running max at per-tensor scope. Within a
            // family the monotone rule still suppresses ablation noise
            // (a higher-bpw K-quant should be at least as good as a
            // lower-bpw K-quant); across families it doesn't hold (an
            // IQ-quant can genuinely beat a higher-bpw K-quant on the
            // same tensor). Per-tensor refinement only swaps a small
            // number of tensors at a time, so the multi-tensor IQ
            // compounding cost we observed at category scope (Run 21
            // follow-up) is much smaller here. Variant A in those
            // experiments — entire attn_k category swapped K→IQ4_XS —
            // cost only +0.04 PPL beyond the all-K Run 21 recipe; an
            // individual-tensor swap is far below noise.
            var runningMaxByFamily = new Dictionary<LlamaQuantFamily, double>();
            foreach (var T in ladder.OrderByDescending(LlamaQuantRecipe.GetBitsPerElement))
            {
                var family = LlamaQuantRecipe.GetFamily(T);
                var familyMax = runningMaxByFamily.GetValueOrDefault(family, 0.0);
                var raw = coef.DeltaPplByType.GetValueOrDefault(T, 0.0) * sizeScale;
                var c = Math.Max(raw, familyMax);
                clamped[T] = c;
                runningMaxByFamily[family] = c;
            }
            clampedByTensor[name] = clamped;
            measuredByTensor[name] = new HashSet<LlamaTensorType>(coef.DeltaPplByType.Keys);
        }

        // Initial assignment: max(baseline, categoryChosen) for every
        // tensor. Tensors without per-tensor data stay here unchanged.
        foreach (var (name, _) in tensorsInCategory)
        {
            var initial = baseline.TryGetValue(name, out var b)
                ? (LlamaQuantRecipe.GetBitsPerElement(b) >= LlamaQuantRecipe.GetBitsPerElement(categoryChosen) ? b : categoryChosen)
                : categoryChosen;
            var initialDelta = clampedByTensor.TryGetValue(name, out var c)
                ? c.GetValueOrDefault(initial, categoryDelta)
                : (initial == categoryChosen ? categoryDelta : 0.0);
            assignment[name] = (initial, initialDelta);
        }

        // ---- Pass 1: demote ----
        foreach (var (name, _) in tensorsInCategory)
        {
            if (!clampedByTensor.TryGetValue(name, out var clamped)) continue;
            var measured = measuredByTensor[name];
            var current = assignment[name].Type;
            var currentBpw = LlamaQuantRecipe.GetBitsPerElement(current);
            var baselineFloor = baseline.TryGetValue(name, out var bb) ? LlamaQuantRecipe.GetBitsPerElement(bb) : 0.0;
            var minBpw = Math.Max(floorBpw, baselineFloor);

            foreach (var T in ladder)
            {
                var bpw = LlamaQuantRecipe.GetBitsPerElement(T);
                if (bpw > currentBpw) break;
                if (bpw < minBpw) continue;
                if (!measured.Contains(T)) continue;    // need actual data, not a monotone-fill
                if (!clamped.TryGetValue(T, out var d)) continue;
                if (d <= categoryDelta)
                {
                    assignment[name] = (T, d);
                    break;
                }
            }
        }

        if (!allowPromotion) return assignment;

        // ---- Pass 2: promote ----
        // Compute the bpw savings produced by pass 1 vs the naive
        // every-tensor-at-categoryChosen baseline. Promotions can spend
        // up to that pool but no more.
        long totalElements = tensorsInCategory.Sum(t => t.Elements);
        double naiveBits = LlamaQuantRecipe.GetBitsPerElement(categoryChosen) * totalElements;
        double currentBits = tensorsInCategory.Sum(t =>
            LlamaQuantRecipe.GetBitsPerElement(assignment[t.Name].Type) * t.Elements);
        double savings = naiveBits - currentBits;
        if (savings <= 0) return assignment;

        // Identify promotion candidates and their best higher-bpw target.
        var candidates = new List<(string Name, LlamaTensorType Target, double NewDelta, double GainPerBpw, double Gain, double CostBits)>();
        foreach (var (name, elements) in tensorsInCategory)
        {
            if (!clampedByTensor.TryGetValue(name, out var clamped)) continue;
            var current = assignment[name].Type;
            var currentBpw = LlamaQuantRecipe.GetBitsPerElement(current);
            var currentDelta = clamped.GetValueOrDefault(current, 0.0);
            // Threshold: tensor must be materially sensitive at the
            // current type (above measurement noise) to justify spending
            // budget on a promotion.
            if (currentDelta <= promotionThreshold) continue;

            LlamaTensorType? bestTarget = null;
            double bestNewDelta = currentDelta;
            double bestGainPerBpw = minPplGainPerBpw;
            double bestGain = 0;
            double bestCostBits = 0;
            var measured = measuredByTensor[name];
            foreach (var T in ladder)
            {
                var bpw = LlamaQuantRecipe.GetBitsPerElement(T);
                if (bpw <= currentBpw) continue;
                if (!measured.Contains(T)) continue;    // need actual data, not a monotone-fill
                if (!clamped.TryGetValue(T, out var d)) continue;
                if (d >= currentDelta) continue;    // promotion must measurably reduce delta

                var deltaBpw = bpw - currentBpw;
                var gain = currentDelta - d;
                var gainPerBpw = gain / deltaBpw;
                // Pick the highest gain-per-bpw above the efficiency
                // floor — same metric the per-category optimizer uses
                // for promotion decisions.
                if (gainPerBpw > bestGainPerBpw)
                {
                    bestTarget = T;
                    bestNewDelta = d;
                    bestGainPerBpw = gainPerBpw;
                    bestGain = gain;
                    bestCostBits = deltaBpw * elements;
                }
            }

            if (bestTarget is { } target)
                candidates.Add((name, target, bestNewDelta, bestGainPerBpw, bestGain, bestCostBits));
        }

        // Apply highest-impact promotions first (most absolute gain per
        // bit spent), stopping when the savings pool is exhausted.
        foreach (var c in candidates.OrderByDescending(x => x.Gain / Math.Max(1.0, x.CostBits)))
        {
            if (c.CostBits > savings) continue;
            assignment[c.Name] = (c.Target, c.NewDelta);
            savings -= c.CostBits;
        }

        return assignment;
    }

    /// <summary>
    /// Build a per-(category, ladder type) table of size-scaled
    /// per-category ΔPPL coefficients with zero-floor + cross-family
    /// monotone-from-above-bpw clamping applied.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Two clamps applied jointly via a single descending-bpw pass:
    /// </para>
    /// <list type="number">
    ///   <item><b>Zero-floor.</b> Quantization is a strict information
    ///     loss vs F16. A measured δ &lt; 0 means the F16-vs-quantized
    ///     PPL difference is below the corpus's PPL measurement
    ///     variance — noise, not signal. Floor at 0 to refuse phantom
    ///     "free wins" (Run 20 case: 4B attn_v Q2_K = −0.45).</item>
    ///   <item><b>Cross-family monotone-from-above-bpw.</b> Each lower-bpw
    ///     type's clamped δ is at least the running max of every higher-bpw
    ///     type (regardless of family). Theoretically this is over-conservative
    ///     across families — IQ-quants use different machinery than K-quants
    ///     and can genuinely outperform higher-bpw K-quants on the same tensor
    ///     — but empirically (Run 21 follow-up isolation experiments,
    ///     2026-04-28) per-category single-category ablation
    ///     <em>under-predicts</em> the category-in-mix cost when the
    ///     optimizer picks IQ-types for whole categories. ffn_down at IQ4_XS
    ///     measured +0.007 PPL in single-category ablation but cost +0.31 PPL
    ///     when applied as the recipe pick across all 28 layers. The
    ///     cross-family clamp's accidental side-effect of biasing the
    ///     optimizer away from "all IQ4_XS for category X" turned out to
    ///     be the empirically correct behavior. Per-family clamping at this
    ///     scope was tried and produced a measurably worse recipe.</item>
    /// </list>
    /// <para>
    /// Per-family clamping <em>is</em> the right rule at per-tensor
    /// refinement scope, where individual tensors can be moved to IQ
    /// types without the multi-tensor compounding regression. See
    /// <see cref="RefineCategoryPerTensor"/>.
    /// </para>
    /// </remarks>
    private static Dictionary<(string Cat, LlamaTensorType T), double>
        BuildClampedScaledDeltas(
            LlamaSensitivityProfile profile,
            IReadOnlyList<LlamaTensorType> ladder,
            double sizeScale)
    {
        var result = new Dictionary<(string, LlamaTensorType), double>();
        var byBpwDescending = ladder
            .OrderByDescending(LlamaQuantRecipe.GetBitsPerElement)
            .ToList();
        foreach (var (cat, coef) in profile.Categories)
        {
            double runningMax = 0.0;    // zero-floor seed
            foreach (var T in byBpwDescending)
            {
                var raw = coef.DeltaPplByType.GetValueOrDefault(T, 0.0) * sizeScale;
                var clamped = Math.Max(raw, runningMax);
                result[(cat, T)] = clamped;
                runningMax = clamped;
            }
        }
        return result;
    }

    /// <summary>
    /// Resolve an uncategorized weight tensor's type by walking the
    /// protection patterns. Returns the matched protection type, or
    /// <see cref="LlamaQuantRecipeFromProfileOptions.UncategorizedDefault"/>
    /// when nothing matches. Pattern-match semantics mirror
    /// <see cref="CategoryMatch"/>: dot in pattern → suffix; no dot →
    /// contains.
    /// </summary>
    private static LlamaTensorType ResolveUncategorizedType(
        string tensorName, LlamaQuantRecipeFromProfileOptions opts)
    {
        foreach (var (pattern, type) in opts.UncategorizedProtections)
        {
            if (CategoryMatch(tensorName, pattern)) return type;
        }
        return opts.UncategorizedDefault;
    }

    /// <summary>
    /// Same matcher as <see cref="LlamaSensitivityProfileBuilder"/>:
    /// exact-or-period-prefixed-suffix for dot-containing category
    /// names so <c>output.weight</c> doesn't catch
    /// <c>blk.N.attn_output.weight</c>.
    /// </summary>
    private static bool CategoryMatch(string tensorName, string category) =>
        category.Contains('.')
            ? tensorName == category ||
              tensorName.EndsWith("." + category, StringComparison.Ordinal)
            : tensorName.Contains(category, StringComparison.Ordinal);
}
