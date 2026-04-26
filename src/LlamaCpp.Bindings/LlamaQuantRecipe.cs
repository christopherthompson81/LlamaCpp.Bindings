using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>One per-tensor decision in a recipe.</summary>
public sealed record LlamaQuantRecipeEntry(
    /// <summary>Exact GGUF tensor name (anchored regex emitted at apply time).</summary>
    string TensorName,
    /// <summary>The chosen ftype for this tensor.</summary>
    LlamaTensorType ChosenType,
    /// <summary>Bits-per-element of the chosen type — the "size" half of the tradeoff.</summary>
    double BitsPerElement,
    /// <summary>The relative MSE that drove the choice (or that of the safest available type when the threshold couldn't be met).</summary>
    double RelativeMse,
    /// <summary>
    /// True when no candidate type satisfied the recipe's threshold and
    /// we fell back to the lowest-MSE candidate available. Treat these
    /// as a flag in the GUI — they're either the threshold being too
    /// tight or a tensor that's fundamentally hard to quantize.
    /// </summary>
    bool ExceededThreshold);

/// <summary>
/// Per-tensor ftype assignment derived from a sensitivity-sweep score
/// table. Maps cleanly to <see cref="LlamaQuantizationParameters.TensorTypeOverrides"/>
/// via <see cref="ToTtOverrides"/> so phase-4's "Apply" path is a one-liner.
/// </summary>
public sealed record LlamaQuantRecipe(
    /// <summary>The relative-MSE threshold used to pick types.</summary>
    double Threshold,
    /// <summary>Path of the sensitivity table this recipe was built from. Informational; not load-bearing.</summary>
    string? SourceScoreTablePath,
    /// <summary>One entry per scored tensor. Order matches the score table's tensor order.</summary>
    IReadOnlyList<LlamaQuantRecipeEntry> Entries,
    DateTime BuiltAtUtc)
{
    /// <summary>
    /// Build a recipe from a sensitivity-sweep result by choosing, per
    /// tensor, the lowest-bits-per-element candidate whose
    /// <see cref="LlamaQuantSensitivityScore.RelativeMse"/> stays at
    /// or below <paramref name="threshold"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Bits-per-element is computed once from
    /// <c>ggml_type_traits</c> per candidate type (e.g. F16=16,
    /// Q8_0=8.5, Q6_K≈6.56, Q4_K=4.5, IQ2_S≈2.56). The "lowest bits"
    /// candidate is therefore the smallest representation that meets
    /// the quality threshold.
    /// </para>
    /// <para>
    /// When no candidate satisfies <paramref name="threshold"/> for a
    /// given tensor, we fall back to the candidate with the smallest
    /// observed MSE — the safest available choice — and flag the
    /// entry's <see cref="LlamaQuantRecipeEntry.ExceededThreshold"/>
    /// so callers can surface it to the user. We never throw; an
    /// always-finishes recipe is more useful than a partial failure.
    /// </para>
    /// </remarks>
    public static LlamaQuantRecipe Build(
        LlamaQuantSensitivityResult scores,
        double threshold,
        string? sourceScoreTablePath = null)
    {
        ArgumentNullException.ThrowIfNull(scores);
        if (!(threshold > 0))
        {
            throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be > 0.");
        }
        LlamaBackend.EnsureInitialized();

        // Cache bits-per-element per candidate type. Same ordering is
        // shared across every tensor's selection.
        var bitsPerType = scores.CandidateTypes.ToDictionary(
            t => t, t => GetBitsPerElement(t));

        // Group score rows by tensor name, preserving the order the
        // tensors appeared in the sweep.
        var byTensor = new Dictionary<string, List<LlamaQuantSensitivityScore>>(StringComparer.Ordinal);
        var tensorOrder = new List<string>();
        foreach (var s in scores.Scores)
        {
            if (!byTensor.TryGetValue(s.TensorName, out var list))
            {
                list = new List<LlamaQuantSensitivityScore>();
                byTensor[s.TensorName] = list;
                tensorOrder.Add(s.TensorName);
            }
            list.Add(s);
        }

        var entries = new List<LlamaQuantRecipeEntry>(tensorOrder.Count);
        foreach (var name in tensorOrder)
        {
            var rows = byTensor[name];
            // Drop skipped rows — they have no meaningful score.
            var usable = rows.Where(r => !r.Skipped).ToList();
            if (usable.Count == 0)
            {
                // Nothing to recommend; the production quantizer will
                // fall through to its ftype default for this tensor.
                continue;
            }

            // Smallest-bits-first ordering of usable types. Ties
            // broken by lower MSE (rarely matters in practice).
            var ordered = usable
                .OrderBy(r => bitsPerType.TryGetValue(r.QuantType, out var bpe) ? bpe : double.MaxValue)
                .ThenBy(r => r.RelativeMse)
                .ToList();

            // Walk small→large; first one under threshold wins.
            LlamaQuantSensitivityScore? chosen = null;
            foreach (var r in ordered)
            {
                if (!double.IsNaN(r.RelativeMse) && r.RelativeMse <= threshold)
                {
                    chosen = r;
                    break;
                }
            }

            bool exceeded = chosen is null;
            if (chosen is null)
            {
                // No candidate met the threshold — fall back to the
                // candidate with the smallest observed MSE (the safest
                // available choice).
                chosen = usable
                    .Where(r => !double.IsNaN(r.RelativeMse))
                    .OrderBy(r => r.RelativeMse)
                    .FirstOrDefault();
                if (chosen is null) continue;  // every score is NaN — give up on this tensor
            }

            entries.Add(new LlamaQuantRecipeEntry(
                TensorName:        name,
                ChosenType:        chosen.QuantType,
                BitsPerElement:    bitsPerType.TryGetValue(chosen.QuantType, out var bp) ? bp : double.NaN,
                RelativeMse:       chosen.RelativeMse,
                ExceededThreshold: exceeded));
        }

        return new LlamaQuantRecipe(
            Threshold: threshold,
            SourceScoreTablePath: sourceScoreTablePath,
            Entries: entries,
            BuiltAtUtc: DateTime.UtcNow);
    }

    /// <summary>
    /// Convert this recipe to the format
    /// <see cref="LlamaQuantizationParameters.TensorTypeOverrides"/>
    /// accepts. Each tensor name becomes an anchored regex with literal
    /// dots escaped (<c>"^&lt;name&gt;$"</c>) so the pattern matches
    /// the tensor exactly without widening to siblings.
    /// </summary>
    /// <remarks>
    /// Reminder from the
    /// <see cref="LlamaQuantizationParameters.TensorTypeOverrides"/>
    /// docs: the consumer must run with <c>Pure=false</c> for the
    /// override list to take effect. If you're using a recipe, you've
    /// implicitly opted in to non-pure.
    /// </remarks>
    public IReadOnlyList<KeyValuePair<string, LlamaTensorType>> ToTtOverrides()
    {
        var list = new List<KeyValuePair<string, LlamaTensorType>>(Entries.Count);
        foreach (var e in Entries)
        {
            var pattern = "^" + Regex.Escape(e.TensorName) + "$";
            list.Add(new KeyValuePair<string, LlamaTensorType>(pattern, e.ChosenType));
        }
        return list;
    }

    /// <summary>
    /// Estimate the average bits-per-element across this recipe's
    /// covered tensors. Useful for "this recipe lands at ~5.2 bpw"
    /// summaries in the GUI without re-quantizing.
    /// </summary>
    public double AverageBitsPerElement
    {
        get
        {
            if (Entries.Count == 0) return double.NaN;
            // Unweighted mean — every tensor counts equally. A weighted
            // version (by element count) is a more honest "what's the
            // file size going to be" estimate, but we don't carry
            // element counts here yet; callers with the score table
            // can compute it themselves.
            double sum = 0;
            int n = 0;
            foreach (var e in Entries)
            {
                if (!double.IsNaN(e.BitsPerElement))
                {
                    sum += e.BitsPerElement;
                    n++;
                }
            }
            return n > 0 ? sum / n : double.NaN;
        }
    }

    // ----- bits-per-element helper -----

    /// <summary>Read <c>ggml_type_traits</c> and compute bits-per-element for a candidate type.</summary>
    public static double GetBitsPerElement(LlamaTensorType type)
    {
        LlamaBackend.EnsureInitialized();
        var ptr = NativeMethods.ggml_get_type_traits((ggml_type)(int)type);
        if (ptr == IntPtr.Zero)
        {
            throw new ArgumentException(
                $"ggml_get_type_traits returned null for {type} — type is not registered.",
                nameof(type));
        }
        var traits = Marshal.PtrToStructure<ggml_type_traits>(ptr);
        long blockSize = traits.blck_size <= 0 ? 1 : traits.blck_size;
        return ((double)traits.type_size * 8) / blockSize;
    }

    // ----- JSON save/load -----

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter() },
    };

    public static void SaveToJson(LlamaQuantRecipe recipe, string path)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        ArgumentException.ThrowIfNullOrEmpty(path);
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(path, JsonSerializer.Serialize(recipe, JsonOpts));
    }

    public static LlamaQuantRecipe LoadFromJson(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<LlamaQuantRecipe>(json, JsonOpts)
            ?? throw new InvalidDataException($"Failed to deserialize recipe from {path}.");
    }
}
