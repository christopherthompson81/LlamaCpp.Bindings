using System.Text.Json;
using System.Text.Json.Serialization;

namespace LlamaCpp.Bindings.Server.Models;

/// <summary>
/// OpenAI-style embedding request. The <c>input</c> field can be either a
/// single string or an array of strings — OpenAI's schema accepts both and
/// client libraries depend on that. We parse the raw <see cref="JsonElement"/>
/// and hand the handler a normalised <c>string[]</c>.
/// </summary>
public sealed class EmbeddingsRequest
{
    [JsonPropertyName("input")]
    public JsonElement Input { get; set; }

    [JsonPropertyName("model")]
    public string? Model { get; set; }

    /// <summary>
    /// OpenAI accepts <c>"float"</c> or <c>"base64"</c>. V1 supports only
    /// <c>"float"</c> (the default); <c>"base64"</c> returns 400.
    /// </summary>
    [JsonPropertyName("encoding_format")]
    public string? EncodingFormat { get; set; }

    /// <summary>
    /// Resolve <see cref="Input"/> to an array of strings. Throws
    /// <see cref="ArgumentException"/> for malformed input.
    /// </summary>
    public string[] NormalizeInput()
    {
        if (Input.ValueKind == JsonValueKind.String)
        {
            var s = Input.GetString();
            if (s is null) throw new ArgumentException("input must not be null");
            return new[] { s };
        }
        if (Input.ValueKind == JsonValueKind.Array)
        {
            var list = new List<string>();
            foreach (var el in Input.EnumerateArray())
            {
                if (el.ValueKind != JsonValueKind.String)
                {
                    throw new ArgumentException("input array entries must be strings");
                }
                var s = el.GetString() ?? throw new ArgumentException("input array entries must not be null");
                list.Add(s);
            }
            if (list.Count == 0) throw new ArgumentException("input array must not be empty");
            return list.ToArray();
        }
        throw new ArgumentException($"input must be a string or an array of strings (got {Input.ValueKind})");
    }
}

public sealed class EmbeddingsResponse
{
    [JsonPropertyName("object")]
    public string Object { get; set; } = "list";

    [JsonPropertyName("data")]
    public List<EmbeddingEntry> Data { get; set; } = new();

    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("usage")]
    public EmbeddingUsage Usage { get; set; } = new();
}

public sealed class EmbeddingEntry
{
    [JsonPropertyName("object")]
    public string Object { get; set; } = "embedding";

    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("embedding")]
    public float[] Embedding { get; set; } = Array.Empty<float>();
}

public sealed class EmbeddingUsage
{
    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; set; }

    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; set; }
}
