using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace LlamaCpp.Bindings;

/// <summary>
/// In-process registry of supported model architectures. Reads the
/// embedded <c>HfConvert/architectures/*.json</c> resources to derive
/// per-architecture metadata: HF↔GGUF tensor maps, category lists for
/// the sensitivity profile builder, per-layer tensor templates for
/// per-layer ablation mode.
/// </summary>
/// <remarks>
/// This is the canonical source for "what tensors does an architecture
/// have" — both the HF converter and the profile builder consume it.
/// Adding support for a new architecture means dropping a JSON file in
/// <c>HfConvert/architectures/</c>; the registry picks it up at startup.
/// </remarks>
public static class LlamaArchitectureRegistry
{
    private const string ResourcePrefix = "LlamaCpp.Bindings.HfConvert.architectures.";

    private static readonly Lazy<IReadOnlyDictionary<string, LlamaArchitectureSpec>> _cache = new(LoadAll);

    /// <summary>All architectures shipped in the binding, keyed by GGUF arch id (e.g. <c>qwen3</c>).</summary>
    public static IReadOnlyDictionary<string, LlamaArchitectureSpec> All => _cache.Value;

    /// <summary>Look up by GGUF arch id; returns null if unknown.</summary>
    public static LlamaArchitectureSpec? Lookup(string ggufArchId) =>
        _cache.Value.TryGetValue(ggufArchId, out var s) ? s : null;

    /// <summary>
    /// Fallback for unknown architectures: the seven canonical decoder-only
    /// transformer categories. Profiles built with this fallback are still
    /// usable, just without architecture-specific extras (like Qwen3's
    /// QK-norms or MoE expert routers).
    /// </summary>
    public static LlamaArchitectureSpec StandardTransformer { get; } = new(
        GgufArch:           "unknown",
        HfArchitecture:     "Unknown",
        Description:        "Standard decoder-only transformer fallback (used when an architecture isn't in the registry).",
        Categories:         new[]
        {
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
            "ffn_up", "ffn_gate", "ffn_down",
            "token_embd.weight", "output.weight",
        },
        PerLayerTensorTemplates: new[]
        {
            "blk.{i}.attn_q.weight",
            "blk.{i}.attn_k.weight",
            "blk.{i}.attn_v.weight",
            "blk.{i}.attn_output.weight",
            "blk.{i}.ffn_up.weight",
            "blk.{i}.ffn_gate.weight",
            "blk.{i}.ffn_down.weight",
        },
        TopLevelTensors:    new[] { "token_embd.weight", "output.weight" });

    private static IReadOnlyDictionary<string, LlamaArchitectureSpec> LoadAll()
    {
        var asm = typeof(LlamaArchitectureRegistry).Assembly;
        var result = new Dictionary<string, LlamaArchitectureSpec>(StringComparer.Ordinal);

        foreach (var resName in asm.GetManifestResourceNames())
        {
            if (!resName.StartsWith(ResourcePrefix, StringComparison.Ordinal)) continue;
            if (!resName.EndsWith(".json", StringComparison.Ordinal)) continue;

            using var s = asm.GetManifestResourceStream(resName);
            if (s is null) continue;
            using var reader = new StreamReader(s);
            var text = reader.ReadToEnd();

            ArchJson? raw;
            try { raw = JsonSerializer.Deserialize<ArchJson>(text, JsonOpts); }
            catch { continue; }    // skip malformed
            if (raw is null || string.IsNullOrEmpty(raw.GgufArch)) continue;

            var spec = BuildSpec(raw);
            result[spec.GgufArch] = spec;
        }
        return result;
    }

    private static LlamaArchitectureSpec BuildSpec(ArchJson raw)
    {
        var perLayerTemplates = new List<string>();
        var topLevel = new List<string>();
        foreach (var t in raw.TensorMap ?? new List<TensorMapEntry>())
        {
            if (string.IsNullOrEmpty(t.Gguf)) continue;
            if (!IsQuantizableWeight(t.Gguf)) continue;

            if (t.Gguf.Contains("{i}", StringComparison.Ordinal))
                perLayerTemplates.Add(t.Gguf);
            else
                topLevel.Add(t.Gguf);
        }

        // Categories: derive from per-layer templates by stripping the
        // blk.{i}. prefix. Top-level tensors become their own categories.
        // Note: the FFN categories use the bare convention (ffn_up,
        // ffn_gate, ffn_down — no .weight) to match the existing
        // matcher's contains-form lookup; the four attention categories
        // and top-level tensors keep their .weight suffix and use the
        // dot-anchored suffix form. This asymmetry is the convention
        // the existing reference profiles encode; the registry preserves
        // it for backwards compatibility.
        var categories = new List<string>();
        foreach (var t in perLayerTemplates)
        {
            var stripped = StripLayerPrefix(t);                 // "blk.{i}.ffn_up.weight" → "ffn_up.weight"
            categories.Add(NormalizeCategoryName(stripped));    // "ffn_up.weight" → "ffn_up"
        }
        categories.AddRange(topLevel);

        return new LlamaArchitectureSpec(
            GgufArch:                raw.GgufArch ?? "unknown",
            HfArchitecture:          raw.HfArchitecture ?? "Unknown",
            Description:             raw.Description ?? string.Empty,
            Categories:              Dedupe(categories),
            PerLayerTensorTemplates: perLayerTemplates,
            TopLevelTensors:         topLevel);
    }

    /// <summary>
    /// Match the rules in llama-quant.cpp's <c>tensor_allows_quantization</c>:
    /// only <c>.weight</c> tensors, no norms, no positional embeddings.
    /// </summary>
    private static bool IsQuantizableWeight(string ggufName)
    {
        if (!ggufName.EndsWith(".weight", StringComparison.Ordinal)) return false;
        if (ggufName.Contains("_norm", StringComparison.Ordinal)) return false;
        if (ggufName.Contains("rope_freqs", StringComparison.Ordinal)) return false;
        if (ggufName.Contains("position_embd", StringComparison.Ordinal)) return false;
        return true;
    }

    /// <summary>"blk.{i}.attn_q.weight" → "attn_q.weight".</summary>
    private static string StripLayerPrefix(string template)
    {
        // Match either "blk.{i}." (template) or "blk.<digits>." (resolved name).
        return Regex.Replace(template, @"^blk\.(\{i\}|\d+)\.", "");
    }

    /// <summary>
    /// Map a stripped tensor name to the category-string convention used
    /// by <see cref="LlamaSensitivityProfileBuilder"/> and the recipe
    /// matcher. FFN tensors use bare names (matcher's contains-form);
    /// everything else keeps <c>.weight</c> (matcher's suffix-form).
    /// </summary>
    private static string NormalizeCategoryName(string strippedName)
    {
        if (strippedName.StartsWith("ffn_", StringComparison.Ordinal) &&
            strippedName.EndsWith(".weight", StringComparison.Ordinal))
        {
            return strippedName[..^".weight".Length];
        }
        return strippedName;
    }

    private static IReadOnlyList<string> Dedupe(IEnumerable<string> seq) =>
        seq.Distinct(StringComparer.Ordinal).ToList();

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        Converters = { new JsonStringEnumConverter() },
        AllowTrailingCommas = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
    };

    // Minimal JSON shape for deserialization. We only read the bits the
    // registry needs; the HF converter parses more fields from the same
    // files using its own DTOs.
    private sealed class ArchJson
    {
        public string? HfArchitecture { get; set; }
        public string? GgufArch { get; set; }
        public string? Description { get; set; }
        public List<TensorMapEntry>? TensorMap { get; set; }
    }

    private sealed class TensorMapEntry
    {
        public string? Gguf { get; set; }
        public string? Hf { get; set; }
    }
}

/// <summary>
/// Architecture metadata derived from <c>HfConvert/architectures/*.json</c>.
/// Used by the profile builder to enumerate categories and per-layer
/// tensor names without hard-coding them per architecture.
/// </summary>
public sealed record LlamaArchitectureSpec(
    string GgufArch,
    string HfArchitecture,
    string Description,
    /// <summary>Category strings for the (existing) per-category sensitivity profile.</summary>
    IReadOnlyList<string> Categories,
    /// <summary>Per-layer tensor templates with <c>{i}</c> placeholders, e.g. <c>blk.{i}.attn_q.weight</c>.</summary>
    IReadOnlyList<string> PerLayerTensorTemplates,
    /// <summary>Top-level tensors (no <c>blk.{i}</c> prefix), e.g. <c>output.weight</c>.</summary>
    IReadOnlyList<string> TopLevelTensors)
{
    /// <summary>Expand per-layer templates to concrete tensor names for a given layer count.</summary>
    public IEnumerable<string> ExpandPerLayerTensors(int layerCount)
    {
        for (int i = 0; i < layerCount; i++)
            foreach (var t in PerLayerTensorTemplates)
                yield return t.Replace("{i}", i.ToString());
    }

    /// <summary>
    /// Targeted-drill helper: enumerate the concrete tensor names that
    /// match a category filter and a layer filter. Drives the per-layer
    /// builder mode when the user only wants to drill into specific
    /// categories on specific layers (e.g. attn_v on all 36 layers of
    /// Qwen3-4B without touching ffn_down at all). Returns top-level
    /// tensors (no layer index) only when their full name appears in
    /// <paramref name="categoryFilter"/>.
    /// </summary>
    /// <param name="totalLayerCount">
    /// The model's actual block count (architecture metadata's
    /// <c>{arch}.block_count</c>). Required because the spec doesn't
    /// know the count.
    /// </param>
    /// <param name="categoryFilter">
    /// Categories to include (matched against the same names that appear
    /// in <see cref="Categories"/>). <c>null</c> = all categories.
    /// </param>
    /// <param name="layerFilter">
    /// Layer indices to include for per-layer templates. <c>null</c> =
    /// every layer in <c>0..totalLayerCount-1</c>. Out-of-range indices
    /// are silently dropped (lets callers pass user input without
    /// pre-validation).
    /// </param>
    public IEnumerable<string> ResolveTensors(
        int totalLayerCount,
        IReadOnlyList<string>? categoryFilter = null,
        IReadOnlyList<int>? layerFilter = null)
    {
        bool catAllowed(string c) =>
            categoryFilter is null || categoryFilter.Contains(c);

        var effectiveLayers = (layerFilter ?? Enumerable.Range(0, Math.Max(0, totalLayerCount)).ToList())
            .Where(i => i >= 0 && i < totalLayerCount)
            .Distinct()
            .OrderBy(i => i)
            .ToList();

        foreach (var template in PerLayerTensorTemplates)
        {
            var cat = StripAndNormalizeForCategory(template);
            if (!catAllowed(cat)) continue;
            foreach (var i in effectiveLayers)
                yield return template.Replace("{i}", i.ToString());
        }

        foreach (var top in TopLevelTensors)
        {
            if (catAllowed(top))
                yield return top;
        }
    }

    /// <summary>
    /// Inverse of the registry's category-naming convention: strip a
    /// per-layer template's <c>blk.{i}.</c> prefix, then apply the same
    /// FFN-bare normalization the registry uses when building
    /// <see cref="Categories"/>. Public so the UI can predict what
    /// category name a template will produce.
    /// </summary>
    public static string StripAndNormalizeForCategory(string template)
    {
        // Match either "blk.{i}." or a numeric "blk.13." (already-resolved).
        var stripped = System.Text.RegularExpressions.Regex.Replace(
            template, @"^blk\.(\{i\}|\d+)\.", "");
        if (stripped.StartsWith("ffn_", StringComparison.Ordinal) &&
            stripped.EndsWith(".weight", StringComparison.Ordinal))
        {
            return stripped[..^".weight".Length];
        }
        return stripped;
    }
}
