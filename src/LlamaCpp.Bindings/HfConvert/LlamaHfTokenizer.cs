using System.Text.Json;

namespace LlamaCpp.Bindings.HfConvert;

/// <summary>
/// GGUF token-type integer values. Match llama.cpp's <c>llama_token_attr</c>
/// "type" subset — we don't import the enum to avoid pulling internal
/// types into the converter. Numeric values are stable; if upstream
/// renumbers, this will need to track.
/// </summary>
public static class LlamaTokenTypeId
{
    public const int Normal       = 1;
    public const int Unknown      = 2;
    public const int Control      = 3;
    public const int UserDefined  = 4;
    public const int Unused       = 5;
    public const int Byte         = 6;
}

/// <summary>
/// Parsed HuggingFace BPE tokenizer (the one used by Llama-3 / Qwen2/3 /
/// Mistral and any other model using the <c>tokenizers</c>-library
/// JSON format). Produces the GGUF metadata needed for
/// <see cref="LlamaHfConverter"/> to emit a working
/// <c>tokenizer.ggml.*</c> block.
/// </summary>
/// <remarks>
/// V1 supports the <c>"BPE"</c> model type only with a flat vocab map
/// (string → id) and a list of merges. SentencePiece (Llama-1/2,
/// Gemma) and Unigram tokenizers are deferred — they need separate
/// readers.
/// </remarks>
public sealed class LlamaHfTokenizer
{
    /// <summary>Token strings indexed by id. Unfilled ids become "[UNUSED_n]".</summary>
    public IReadOnlyList<string> Tokens { get; }

    /// <summary>Per-token type tag (parallel to <see cref="Tokens"/>).</summary>
    public IReadOnlyList<int> TokenTypes { get; }

    /// <summary>BPE merges in order (each entry is two tokens separated by a space).</summary>
    public IReadOnlyList<string> Merges { get; }

    /// <summary>Optional chat template string from <c>tokenizer_config.json</c> or <c>chat_template.json</c>.</summary>
    public string? ChatTemplate { get; }

    /// <summary>Resolved special token IDs (bos/eos/pad/...) keyed by GGUF metadata-suffix name.</summary>
    public IReadOnlyDictionary<string, uint> SpecialTokenIds { get; }

    private LlamaHfTokenizer(IReadOnlyList<string> tokens, IReadOnlyList<int> types, IReadOnlyList<string> merges,
        string? chatTemplate, IReadOnlyDictionary<string, uint> specials)
    {
        Tokens = tokens;
        TokenTypes = types;
        Merges = merges;
        ChatTemplate = chatTemplate;
        SpecialTokenIds = specials;
    }

    /// <summary>Load <c>tokenizer.json</c> (+ <c>tokenizer_config.json</c> if present) from <paramref name="dir"/>.</summary>
    public static LlamaHfTokenizer FromDirectory(string dir)
    {
        ArgumentException.ThrowIfNullOrEmpty(dir);
        var tokenizerPath = Path.Combine(dir, "tokenizer.json");
        if (!File.Exists(tokenizerPath))
            throw new FileNotFoundException($"HF model directory '{dir}' is missing tokenizer.json.", tokenizerPath);

        using var doc = JsonDocument.Parse(File.ReadAllText(tokenizerPath), new JsonDocumentOptions
        {
            CommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
        });
        var root = doc.RootElement;

        var model = root.GetProperty("model");
        var modelType = model.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String
            ? t.GetString() ?? "" : "";
        if (!string.Equals(modelType, "BPE", StringComparison.Ordinal))
        {
            throw new NotSupportedException(
                $"tokenizer.json model.type is '{modelType}'; V1 supports BPE only.");
        }

        // Vocab: { "tokenstr": id, ... }. Build the inverse: id → tokenstr.
        // BPE vocabs have ~50K-200K entries; we preallocate the array
        // sized to the highest id we see.
        var vocab = model.GetProperty("vocab");
        int maxId = -1;
        foreach (var p in vocab.EnumerateObject())
        {
            int id = p.Value.GetInt32();
            if (id > maxId) maxId = id;
        }

        // Pull in added_tokens too so their ids are reachable. Most
        // chat models keep specials (e.g. "<|endoftext|>") in
        // added_tokens with ids past the BPE vocab range.
        var addedTokens = new List<(int id, string content, bool special)>();
        if (root.TryGetProperty("added_tokens", out var added) && added.ValueKind == JsonValueKind.Array)
        {
            foreach (var a in added.EnumerateArray())
            {
                int id = a.GetProperty("id").GetInt32();
                string content = a.GetProperty("content").GetString() ?? "";
                bool special = a.TryGetProperty("special", out var sp)
                    && sp.ValueKind is JsonValueKind.True or JsonValueKind.False
                    && sp.GetBoolean();
                addedTokens.Add((id, content, special));
                if (id > maxId) maxId = id;
            }
        }
        if (maxId < 0)
            throw new InvalidDataException("tokenizer.json vocab + added_tokens are empty — nothing to convert.");

        int vocabSize = maxId + 1;
        var tokens = new string[vocabSize];
        var types = new int[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            tokens[i] = $"[UNUSED_{i}]";
            types[i] = LlamaTokenTypeId.Unused;
        }

        foreach (var p in vocab.EnumerateObject())
        {
            int id = p.Value.GetInt32();
            tokens[id] = p.Name;
            types[id] = LlamaTokenTypeId.Normal;
        }
        // added_tokens override vocab entries when ids overlap.
        foreach (var (id, content, special) in addedTokens)
        {
            tokens[id] = content;
            types[id] = special ? LlamaTokenTypeId.Control : LlamaTokenTypeId.UserDefined;
        }

        // Merges: an array of strings or an array of [a, b] pairs (newer
        // tokenizer.json versions use the latter). Normalize to "a b".
        var mergesNode = model.GetProperty("merges");
        var merges = new List<string>(mergesNode.GetArrayLength());
        foreach (var m in mergesNode.EnumerateArray())
        {
            if (m.ValueKind == JsonValueKind.String)
            {
                merges.Add(m.GetString() ?? string.Empty);
            }
            else if (m.ValueKind == JsonValueKind.Array && m.GetArrayLength() == 2)
            {
                merges.Add($"{m[0].GetString()} {m[1].GetString()}");
            }
            else
            {
                throw new InvalidDataException(
                    "tokenizer.json model.merges entry is neither a string nor a two-element array.");
            }
        }

        // Chat template + special token defaults from sibling files.
        // The resolver needs the full assembled tokens[] (BPE + added)
        // because tokenizer_config.json's bos/eos/pad fields commonly
        // refer to specials in added_tokens (e.g. Qwen3's
        // eos_token = "<|im_end|>") that aren't in the BPE vocab proper.
        string? chatTemplate = TryReadChatTemplate(dir);
        var specials = ResolveSpecialTokenIds(dir, tokens);

        return new LlamaHfTokenizer(tokens, types, merges, chatTemplate, specials);
    }

    private static string? TryReadChatTemplate(string dir)
    {
        // chat_template.json takes precedence (newer convention).
        var ct = Path.Combine(dir, "chat_template.json");
        if (File.Exists(ct))
        {
            using var d = JsonDocument.Parse(File.ReadAllText(ct));
            if (d.RootElement.TryGetProperty("chat_template", out var node)
                && node.ValueKind == JsonValueKind.String)
            {
                return node.GetString();
            }
        }
        // tokenizer_config.json is the older convention.
        var tc = Path.Combine(dir, "tokenizer_config.json");
        if (File.Exists(tc))
        {
            using var d = JsonDocument.Parse(File.ReadAllText(tc));
            if (d.RootElement.TryGetProperty("chat_template", out var node)
                && node.ValueKind == JsonValueKind.String)
            {
                return node.GetString();
            }
        }
        return null;
    }

    private static IReadOnlyDictionary<string, uint> ResolveSpecialTokenIds(string dir, IReadOnlyList<string> tokens)
    {
        // tokenizer_config.json holds bos_token / eos_token / pad_token
        // / unk_token as either a string (the raw token text) or an
        // object {"content": "<|...|>", ...}. Resolve each by walking
        // the assembled tokens[] (BPE + added) for an exact match — the
        // vocab-only path missed chat models whose bos/eos live in
        // added_tokens (Qwen3's "<|im_end|>" is the classic case).
        var result = new Dictionary<string, uint>(StringComparer.Ordinal);
        var tc = Path.Combine(dir, "tokenizer_config.json");
        if (!File.Exists(tc)) return result;
        using var d = JsonDocument.Parse(File.ReadAllText(tc));
        var r = d.RootElement;

        // Build a one-pass content → id reverse map. Most BPE vocabs
        // are 50K-200K entries; building once and reusing for the four
        // lookups beats a per-lookup linear scan.
        var byContent = new Dictionary<string, int>(tokens.Count, StringComparer.Ordinal);
        for (int i = 0; i < tokens.Count; i++)
        {
            // First-write-wins: BPE vocab entries are written before
            // added_tokens overrides at the same id, but the override's
            // string is now the slot's content — so we still get the
            // right id for either form.
            byContent.TryAdd(tokens[i], i);
        }

        Resolve("bos_token", "bos_token_id");
        Resolve("eos_token", "eos_token_id");
        Resolve("pad_token", "padding_token_id");
        Resolve("unk_token", "unknown_token_id");
        return result;

        void Resolve(string field, string ggufSuffix)
        {
            if (!r.TryGetProperty(field, out var node)) return;
            string? content = node.ValueKind switch
            {
                JsonValueKind.String => node.GetString(),
                JsonValueKind.Object => node.TryGetProperty("content", out var c)
                    && c.ValueKind == JsonValueKind.String ? c.GetString() : null,
                _ => null,
            };
            if (string.IsNullOrEmpty(content)) return;
            if (byContent.TryGetValue(content, out var id) && id >= 0)
            {
                result[ggufSuffix] = (uint)id;
            }
        }
    }
}
