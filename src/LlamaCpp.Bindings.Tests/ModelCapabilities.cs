namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Probed (not declared) capabilities of the loaded test model. Lets tests
/// guard model-specific assertions while keeping the harness usable across
/// any GGUF — frontier model, small dev model, embedding model.
///
/// Tests that depend on something model-specific (a particular family,
/// a parameter floor, a vocab shape) call one of the <c>SkipUnless*</c>
/// helpers, which write a SKIP line to stdout and return <c>true</c> if the
/// model doesn't qualify; the caller does <c>return</c>. Tests that want to
/// verify universal binding behavior should NOT use these helpers and
/// instead phrase their assertions as properties (e.g. roundtrip stability)
/// rather than literal-content matches.
/// </summary>
public sealed class ModelCapabilities
{
    /// <summary>True when a model was loaded; false in CI/dev environments without the GGUF.</summary>
    public bool ModelLoaded { get; }

    /// <summary>
    /// Architecture family from <c>general.architecture</c> in GGUF metadata,
    /// lowercased. Empty when no model is loaded. Examples: "qwen3", "llama",
    /// "mistral", "gemma2", "bert", "qwen2".
    /// </summary>
    public string Family { get; } = string.Empty;

    /// <summary><c>general.name</c> from GGUF metadata, when present.</summary>
    public string? Name { get; }

    public long ParameterCount { get; }
    public int VocabSize { get; }
    public bool AddsBosAutomatically { get; }
    public bool HasChatTemplate { get; }
    public bool IsEncoderOnly { get; }
    public bool HasDecoder { get; }
    public LlamaRopeType RopeType { get; }

    private ModelCapabilities() { }

    private ModelCapabilities(LlamaModel model)
    {
        ModelLoaded = true;
        model.Metadata.TryGetValue("general.architecture", out var arch);
        Family = (arch ?? string.Empty).ToLowerInvariant();
        model.Metadata.TryGetValue("general.name", out var name);
        Name = string.IsNullOrWhiteSpace(name) ? null : name;
        ParameterCount = model.ParameterCount;
        VocabSize = model.Vocab.TokenCount;
        AddsBosAutomatically = model.Vocab.AddsBosAutomatically;
        HasChatTemplate = !string.IsNullOrWhiteSpace(model.GetChatTemplate());
        HasDecoder = model.HasDecoder;
        IsEncoderOnly = model.HasEncoder && !model.HasDecoder;
        RopeType = model.RopeType;
    }

    public static ModelCapabilities Empty { get; } = new();

    public static ModelCapabilities Probe(LlamaModel? model)
        => model is null ? Empty : new ModelCapabilities(model);

    public string DisplayLabel
    {
        get
        {
            if (!ModelLoaded) return "(no model)";
            var sizeB = ParameterCount / 1e9;
            var fam = string.IsNullOrEmpty(Family) ? "?" : Family;
            return $"{fam} {sizeB:0.#}B" + (Name is null ? string.Empty : $" — {Name}");
        }
    }

    // ---------------- skip helpers ----------------
    // Return true if the test should bail (caller does `return;`).
    // All write a SKIP line to stdout in the established codebase pattern.

    public bool SkipUnlessLoaded()
    {
        if (ModelLoaded) return false;
        Console.WriteLine("SKIP: no test model loaded (set LLAMACPP_TEST_MODEL).");
        return true;
    }

    public bool SkipUnlessFamily(params string[] families)
    {
        if (SkipUnlessLoaded()) return true;
        foreach (var f in families)
        {
            if (string.Equals(Family, f, StringComparison.OrdinalIgnoreCase)) return false;
        }
        Console.WriteLine(
            $"SKIP: needs model family in [{string.Join(", ", families)}]; got '{Family}' ({DisplayLabel}).");
        return true;
    }

    public bool SkipUnlessMinParameters(long minParams)
    {
        if (SkipUnlessLoaded()) return true;
        if (ParameterCount >= minParams) return false;
        Console.WriteLine(
            $"SKIP: needs >= {minParams / 1e9:0.#}B params; got {ParameterCount / 1e9:0.#}B ({DisplayLabel}).");
        return true;
    }

    public bool SkipUnlessMinVocab(int minVocab)
    {
        if (SkipUnlessLoaded()) return true;
        if (VocabSize >= minVocab) return false;
        Console.WriteLine(
            $"SKIP: needs vocab size >= {minVocab}; got {VocabSize} ({DisplayLabel}).");
        return true;
    }
}
