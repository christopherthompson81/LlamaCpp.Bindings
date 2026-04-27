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
        double ScaledDelta(string cat, LlamaTensorType type) =>
            profile.Categories[cat].DeltaPplByType.GetValueOrDefault(type, 0.0) * sizeScale;

        var perCategoryChoices = new Dictionary<string, IReadOnlyList<LlamaTensorType>>();
        foreach (var (cat, coef) in profile.Categories)
        {
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
        Dictionary<string, LlamaTensorType>? bestInBudget = null;
        double bestInBudgetPpl = double.PositiveInfinity;
        Dictionary<string, LlamaTensorType>? bestOverall = null;
        double bestOverallPpl = double.PositiveInfinity;

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

            if (pplSum < bestOverallPpl)
            {
                bestOverallPpl = pplSum;
                bestOverall = new Dictionary<string, LlamaTensorType>(assignment, StringComparer.Ordinal);
            }
            if (bpw <= budgetCap && pplSum < bestInBudgetPpl)
            {
                bestInBudgetPpl = pplSum;
                bestInBudget = new Dictionary<string, LlamaTensorType>(assignment, StringComparer.Ordinal);
            }
        });

        var pick = bestInBudget ?? bestOverall
            ?? throw new InvalidOperationException(
                "Recipe enumeration produced no assignment — empty category space.");

        // ---- 7. Materialize per-tensor entries ----
        // Each tensor's effective type = max(baseline, recipe choice).
        var entries = new List<LlamaQuantRecipeEntry>(weightTensors.Count);
        foreach (var (name, elements) in weightTensors)
        {
            LlamaTensorType chosen;
            double estDelta;
            if (tensorCategory.TryGetValue(name, out var cat))
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
