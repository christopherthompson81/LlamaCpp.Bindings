using System.Text.Json;
using System.Text.Json.Serialization;

namespace LlamaCpp.Bindings.HfConvert;

/// <summary>
/// Declarative definition of one HuggingFace → GGUF model conversion.
/// The architecture engine consumes a definition + the model's
/// <c>config.json</c> + a safetensors file + a tokenizer reader, and
/// emits a GGUF.
/// </summary>
/// <remarks>
/// <para>
/// Definitions ship as embedded JSON resources under
/// <c>HfConvert/architectures/</c>. Adding a new architecture is
/// usually just a new JSON file — only architectures that need new
/// reshape/permute logic require code (a named transform in
/// <see cref="LlamaHfTensorTransforms"/>).
/// </para>
/// <para>
/// Schema is intentionally small: tensor name templates with
/// <c>{i}</c> for block-index substitution, named transforms that
/// reference algorithmic ops by name, optional/tied flags, and a
/// tokenizer-family hint. Per-arch quirks that resist this
/// (e.g. Qwen3-reranker's yes/no row extraction) get a named
/// transform; truly bespoke logic gets a code subclass that overrides
/// <see cref="LlamaHfConverter.PostProcessAsync"/>.
/// </para>
/// </remarks>
public sealed class LlamaHfArchitectureDefinition
{
    /// <summary>
    /// Primary entry in the model's <c>config.json</c> "architectures"
    /// array that this definition handles. Singular by convention —
    /// derivative models that piggy-back on the same architecture (e.g.
    /// <c>InternS1ForConditionalGeneration</c> riding on Qwen3,
    /// <c>Qwen3ForSequenceClassification</c> as a reranker head) go in
    /// <see cref="HfArchitectureAliases"/>.
    /// </summary>
    [JsonPropertyName("hf_architecture")]
    public string HfArchitecture { get; init; } = string.Empty;

    /// <summary>
    /// Additional HF architecture names this definition also matches.
    /// Subordinate to <see cref="HfArchitecture"/> for display
    /// purposes; the engine treats them equally for arch detection.
    /// Empty for most definitions.
    /// </summary>
    [JsonPropertyName("hf_architecture_aliases")]
    public IReadOnlyList<string> HfArchitectureAliases { get; init; } = Array.Empty<string>();

    /// <summary>All HF arch names the definition matches against — primary first, then aliases.</summary>
    public IEnumerable<string> AllHfArchitectures
    {
        get
        {
            if (!string.IsNullOrEmpty(HfArchitecture)) yield return HfArchitecture;
            foreach (var a in HfArchitectureAliases) yield return a;
        }
    }

    /// <summary>GGUF <c>general.architecture</c> value (e.g. "qwen3", "llama").</summary>
    [JsonPropertyName("gguf_arch")]
    public string GgufArchitecture { get; init; } = string.Empty;

    /// <summary>Human description for diagnostics. Optional.</summary>
    [JsonPropertyName("description")]
    public string? Description { get; init; }

    /// <summary>HF config field → GGUF metadata key mappings.</summary>
    [JsonPropertyName("metadata_map")]
    public IReadOnlyList<MetadataMappingEntry> MetadataMap { get; init; } = Array.Empty<MetadataMappingEntry>();

    /// <summary>HF tensor name templates → GGUF tensor name templates with <c>{i}</c> for block index.</summary>
    [JsonPropertyName("tensor_map")]
    public IReadOnlyList<TensorMappingEntry> TensorMap { get; init; } = Array.Empty<TensorMappingEntry>();

    /// <summary>
    /// Tokenizer-family hint. Recognized values: <c>"bpe-gpt2"</c>
    /// (HF tokenizer.json with BPE model + GPT2-style pre-tokenizer —
    /// covers Llama-3, Mistral, Qwen2/3). Future entries can map to
    /// SentencePiece etc.
    /// </summary>
    [JsonPropertyName("tokenizer_family")]
    public string TokenizerFamily { get; init; } = "bpe-gpt2";

    /// <summary>
    /// Optional <c>tokenizer.ggml.pre</c> hint written into the output
    /// GGUF; this is what llama.cpp uses to pick the right
    /// pre-tokenizer regex at load time. If null, the engine derives
    /// from the model name or falls back to "default".
    /// </summary>
    [JsonPropertyName("tokenizer_pre")]
    public string? TokenizerPre { get; init; }

    /// <summary>
    /// How the engine determines the row count to pad
    /// <c>tokenizer.ggml.tokens</c> to (HF embeddings are commonly
    /// padded past the filled vocab for GPU alignment, and llama.cpp
    /// checks <c>tokens.Count == embed_tokens.shape[vocab]</c>).
    /// Format: <c>"config:&lt;dotted_path&gt;"</c> reads a numeric field
    /// from <c>config.json</c>; <c>"tensor:&lt;name&gt;:dim&lt;N&gt;"</c>
    /// reads a tensor's shape entry. Default <c>"config:vocab_size"</c>.
    /// </summary>
    [JsonPropertyName("vocab_anchor")]
    public string VocabAnchor { get; init; } = "config:vocab_size";

    /// <summary>
    /// Glob patterns. Tokens in <c>tokenizer.json</c>'s
    /// <c>added_tokens</c> whose <c>content</c> matches any pattern get
    /// classified as <see cref="LlamaTokenTypeId.Control"/> regardless
    /// of their declared <c>special</c> flag. Silences llama.cpp's
    /// <c>"control-looking token was not control-type"</c> warnings
    /// without the converter having to re-derive HF's added-token
    /// semantics. <c>null</c> falls back to a conventional default
    /// covering chat-template (<c>&lt;|...|&gt;</c>) and
    /// SentencePiece-style (<c>&lt;s&gt;, &lt;/s&gt;, &lt;unk&gt;,
    /// &lt;pad&gt;, &lt;bos&gt;, &lt;eos&gt;</c>) tokens.
    /// </summary>
    [JsonPropertyName("force_control_token_patterns")]
    public IReadOnlyList<string>? ForceControlTokenPatterns { get; init; }

    /// <summary>Default control-token patterns when the definition leaves <see cref="ForceControlTokenPatterns"/> null.</summary>
    public static readonly IReadOnlyList<string> DefaultControlTokenPatterns = new[]
    {
        "<|*|>",
        "<s>",
        "</s>",
        "<unk>",
        "<pad>",
        "<bos>",
        "<eos>",
    };

    /// <summary>
    /// Optional end-of-turn token string (e.g. <c>"&lt;|im_end|&gt;"</c>
    /// for Qwen3 chat models). Resolved against the BPE vocab and
    /// emitted as <c>tokenizer.ggml.eot_token_id</c>; llama.cpp uses
    /// that to extend its end-of-generation set so chat models stop at
    /// turn boundaries instead of running through to <c>eos</c>.
    /// </summary>
    [JsonPropertyName("eot_token")]
    public string? EotToken { get; init; }

    public static LlamaHfArchitectureDefinition FromJson(string json)
    {
        var def = JsonSerializer.Deserialize<LlamaHfArchitectureDefinition>(json, JsonOptions)
            ?? throw new InvalidDataException("Architecture definition JSON deserialized to null.");
        if (string.IsNullOrEmpty(def.HfArchitecture))
            throw new InvalidDataException("Architecture definition is missing hf_architecture.");
        if (string.IsNullOrEmpty(def.GgufArchitecture))
            throw new InvalidDataException("Architecture definition is missing gguf_arch.");
        return def;
    }

    /// <summary>
    /// Deserialization options shared by this type and its child entries.
    /// We use reflection-based deserialization rather than a source-gen
    /// context: source-gen skips C# field initializers when the JSON
    /// omits a property, so init defaults like
    /// <c>= "config:vocab_size"</c> were silently dropped — biting us
    /// with subtle "string came back empty" bugs that needed null-checks
    /// at every read site. Reflection runs the parameterless ctor, the
    /// initializers fire, then JSON values overwrite — so defaults stick.
    /// </summary>
    internal static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };
}

/// <summary>
/// One entry in <see cref="LlamaHfArchitectureDefinition.MetadataMap"/>.
/// </summary>
public sealed class MetadataMappingEntry
{
    /// <summary>
    /// GGUF key suffix (the engine prepends <c>"&lt;arch&gt;."</c>).
    /// E.g. <c>"context_length"</c> becomes <c>"qwen3.context_length"</c>.
    /// </summary>
    [JsonPropertyName("gguf")]
    public string Gguf { get; init; } = string.Empty;

    /// <summary>
    /// HF config.json field path. Dotted paths supported for nested
    /// objects (e.g. <c>"rope_scaling.factor"</c>).
    /// </summary>
    [JsonPropertyName("hf")]
    public string Hf { get; init; } = string.Empty;

    /// <summary>
    /// Output type. One of <c>"uint32" | "int32" | "uint64" | "int64"
    /// | "float32" | "float64" | "string" | "bool"</c>.
    /// </summary>
    [JsonPropertyName("type")]
    public string Type { get; init; } = "uint32";

    /// <summary>
    /// Whether absence in <c>config.json</c> is allowed. When optional
    /// and absent, the engine simply skips the entry.
    /// </summary>
    [JsonPropertyName("optional")]
    public bool Optional { get; init; }
}

/// <summary>
/// One entry in <see cref="LlamaHfArchitectureDefinition.TensorMap"/>.
/// </summary>
public sealed class TensorMappingEntry
{
    /// <summary>GGUF tensor name template, with <c>{i}</c> for block index when applicable.</summary>
    [JsonPropertyName("gguf")]
    public string Gguf { get; init; } = string.Empty;

    /// <summary>HF tensor name template, with <c>{i}</c> for block index when applicable.</summary>
    [JsonPropertyName("hf")]
    public string Hf { get; init; } = string.Empty;

    /// <summary>
    /// Named transform applied to the tensor bytes during conversion.
    /// V1 ships <c>"passthrough"</c> only — bytes copy through with
    /// dtype conversion handled by the output type setting. New
    /// transforms (e.g. <c>"permute_qk"</c>, <c>"pack_qkv"</c>) get
    /// added to <see cref="LlamaHfTensorTransforms"/> by name.
    /// </summary>
    [JsonPropertyName("transform")]
    public string Transform { get; init; } = "passthrough";

    /// <summary>If true, the converter doesn't error when this tensor is missing in HF.</summary>
    [JsonPropertyName("optional")]
    public bool Optional { get; init; }

    /// <summary>
    /// If true and HF lacks this tensor, AND <c>config.tie_word_embeddings == true</c>,
    /// the engine quietly omits it from the output GGUF (llama.cpp ties
    /// embeddings on its side at load time). Used for <c>output.weight</c>
    /// on tied-embedding models.
    /// </summary>
    [JsonPropertyName("optional_when_tied")]
    public bool OptionalWhenTied { get; init; }
}

