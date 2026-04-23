namespace LlamaCpp.Bindings.LlamaChat.Models;

/// <summary>
/// Mirrors the knobs in the llama-server webui "Sampling" panel.
/// A null value on a truncation/penalty stage means "don't include that sampler in the chain".
/// Order of stages when building matches llama.cpp's default: penalties → DRY → top-n-sigma →
/// top-k → typical → top-p → min-p → XTC → temperature → (mirostat|distribution).
/// </summary>
public sealed record SamplerSettings
{
    // --- Core randomness ---
    public float Temperature { get; init; } = 0.7f;

    /// <summary>If &gt; 0, use dynamic temperature with (delta, exponent).</summary>
    public float DynaTempRange { get; init; } = 0f;
    public float DynaTempExponent { get; init; } = 1f;

    public uint Seed { get; init; } = 0xDEADBEEF;

    // --- Truncation stages (null = off) ---
    public int? TopK { get; init; } = 40;
    public float? TopP { get; init; } = 0.95f;
    public float? MinP { get; init; } = 0.05f;
    public float? Typical { get; init; } = null;
    public float? TopNSigma { get; init; } = null;

    // --- XTC ---
    public float? XtcProbability { get; init; } = null;
    public float XtcThreshold { get; init; } = 0.1f;

    // --- DRY (enabled when Multiplier > 0) ---
    public float DryMultiplier { get; init; } = 0f;
    public float DryBase { get; init; } = 1.75f;
    public int DryAllowedLength { get; init; } = 2;
    public int DryPenaltyLastN { get; init; } = -1;

    // --- Repetition penalties ---
    // PenaltyRepeat defaults to 1.1 (matching samples/LlamaChat). 1.0 is the
    // identity and disables the stage; a value that low lets the model fall
    // into trivial loops ("1. [1]", "2. [2]", ...) once it finds a repeatable
    // pattern, with nothing to push it back out. 1.1 is a mild nudge that
    // costs nothing on well-behaved generations.
    public int PenaltyLastN { get; init; } = 64;
    public float PenaltyRepeat { get; init; } = 1.1f;
    public float PenaltyFrequency { get; init; } = 0.0f;
    public float PenaltyPresence { get; init; } = 0.0f;

    // --- Terminal sampler (mutually exclusive) ---
    public MirostatMode Mirostat { get; init; } = MirostatMode.Off;
    public float MirostatTau { get; init; } = 5f;
    public float MirostatEta { get; init; } = 0.1f;

    // --- Response format constraints ---
    /// <summary>
    /// How <see cref="ResponseFormatText"/> gets interpreted. Drives which
    /// path <see cref="Services.SamplerFactory"/> takes to compile a grammar.
    /// Off means no constraint at all.
    /// </summary>
    public ResponseFormatMode ResponseFormat { get; init; } = ResponseFormatMode.Off;

    /// <summary>
    /// Free-form text whose meaning depends on <see cref="ResponseFormat"/>:
    /// empty when Off / Json; a JSON Schema document when JsonSchema; a raw
    /// GBNF grammar when Gbnf. Lives here rather than on the profile so
    /// it's snapshot/round-tripped with the rest of the sampler state.
    /// </summary>
    public string ResponseFormatText { get; init; } = string.Empty;

    public string GrammarStartRule { get; init; } = "root";

    /// <summary>
    /// Legacy slot — raw GBNF. Retained for back-compat with profiles
    /// saved before <see cref="ResponseFormat"/> existed; loaders migrate
    /// a non-empty value here into <see cref="ResponseFormatText"/> with
    /// <see cref="ResponseFormatMode.Gbnf"/>. Don't write through this
    /// in new code.
    /// </summary>
    public string? GbnfGrammar { get; init; } = null;

    public static SamplerSettings Default { get; } = new();
}

public enum MirostatMode { Off, V1, V2 }

/// <summary>
/// How the response-format free-text field should be interpreted.
/// </summary>
public enum ResponseFormatMode
{
    /// <summary>No grammar constraint. Sampler runs free.</summary>
    Off,
    /// <summary>Any valid JSON value. Uses <c>LlamaGrammar.Json</c> — no schema needed.</summary>
    Json,
    /// <summary>Free-text holds a JSON Schema; compile via <see cref="JsonSchemaToGbnf"/>.</summary>
    JsonSchema,
    /// <summary>Free-text holds a hand-written GBNF grammar. Passed through as-is.</summary>
    Gbnf,
}
