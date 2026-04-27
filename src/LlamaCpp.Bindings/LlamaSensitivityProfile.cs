using System.Text.Json;
using System.Text.Json.Serialization;

namespace LlamaCpp.Bindings;

/// <summary>
/// Per-category quantization sensitivity coefficient measured by ablation:
/// for each candidate <see cref="LlamaTensorType"/>, how much wikitext-PPL
/// goes up when only this category is quantized to that type and the rest
/// stays at F16.
/// </summary>
/// <remarks>
/// Built by <see cref="LlamaSensitivityProfileBuilder"/>. The recipe builder
/// uses <see cref="DeltaPplByType"/> to score candidate recipes (sum the
/// per-(category, type) ΔPPL across all categories) and to honor the
/// <see cref="RecommendedFloor"/> — a "do not go below this type" guardrail
/// derived from the knee in the curve.
/// </remarks>
public sealed record LlamaSensitivityCategoryCoefficient(
    string Category,
    Dictionary<LlamaTensorType, double> DeltaPplByType,
    LlamaTensorType? RecommendedFloor);

/// <summary>
/// Per-architecture quantization sensitivity profile. Captures how each
/// tensor category responds to being quantized at various bit-widths,
/// measured against an F16 baseline on a calibration corpus.
/// </summary>
/// <remarks>
/// <para>
/// Profiles are slow to build (~6 min on Qwen3-0.6B with the parallel PPL
/// runner; longer on bigger models) but cheap to apply. Build once per
/// (architecture, size-class), then the recipe builder solves a
/// constrained optimization over the profile to produce a target-bpw
/// recipe.
/// </para>
/// <para>
/// The profile is *empirical* — it doesn't try to model architecture-
/// specific structure. Two Qwen3 sizes can have meaningfully different
/// profiles; one Qwen3 vs one Llama definitely will (see the
/// QK-norm investigation in <c>docs/investigations/qwen3_qk_sensitivity.md</c>).
/// </para>
/// </remarks>
public sealed record LlamaSensitivityProfile(
    string ArchitectureId,
    int LayerCount,
    string SourceModelPath,
    string CalibrationCorpusName,
    double F16BaselinePerplexity,
    Dictionary<string, LlamaSensitivityCategoryCoefficient> Categories,
    DateTime BuiltAtUtc)
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter() },
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    public static LlamaSensitivityProfile LoadFromJson(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<LlamaSensitivityProfile>(json, JsonOpts)
            ?? throw new InvalidDataException($"Failed to deserialize profile from {path}.");
    }

    public void SaveToJson(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(path, JsonSerializer.Serialize(this, JsonOpts));
    }

    /// <summary>
    /// The categories this profile knows about, in stable display order
    /// (most-sensitive first per the F16-relative ΔPPL at Q4_K).
    /// </summary>
    public IEnumerable<string> CategoriesByDescendingSensitivityAtQ4K
    {
        get
        {
            return Categories
                .OrderByDescending(kv => kv.Value.DeltaPplByType.GetValueOrDefault(LlamaTensorType.Q4_K, 0.0))
                .Select(kv => kv.Key);
        }
    }
}
