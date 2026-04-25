using System.Text.Json.Serialization;

namespace LlamaCpp.Bindings.Server.Models;

/// <summary>
/// <c>POST /v1/rerank</c> request. Matches the de-facto rerank
/// schema shared by Cohere, Jina, and llama-server: a single query
/// scored against an array of documents, optionally truncated to the
/// top N most relevant.
/// </summary>
public sealed class RerankRequest
{
    [JsonPropertyName("model")]
    public string? Model { get; set; }

    [JsonPropertyName("query")]
    public string Query { get; set; } = "";

    [JsonPropertyName("documents")]
    public List<string> Documents { get; set; } = new();

    /// <summary>
    /// When set, trim the response to this many highest-scoring entries
    /// (after sorting by descending relevance). When unset, every input
    /// document is returned.
    /// </summary>
    [JsonPropertyName("top_n")]
    public int? TopN { get; set; }

    /// <summary>
    /// When true, mirror Cohere's optional <c>document</c> echo on each
    /// result entry (handy for clients that want one self-contained
    /// payload). Default false to keep responses small.
    /// </summary>
    [JsonPropertyName("return_documents")]
    public bool? ReturnDocuments { get; set; }
}

public sealed class RerankResponse
{
    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("results")]
    public List<RerankResult> Results { get; set; } = new();

    [JsonPropertyName("usage")]
    public RerankUsage Usage { get; set; } = new();
}

public sealed class RerankResult
{
    /// <summary>The document's index in the request's <c>documents</c> array.</summary>
    [JsonPropertyName("index")]
    public int Index { get; set; }

    /// <summary>Raw score from the rank head. Higher = more relevant. Range is model-specific.</summary>
    [JsonPropertyName("relevance_score")]
    public float RelevanceScore { get; set; }

    /// <summary>Optional document echo, populated when <c>return_documents=true</c>.</summary>
    [JsonPropertyName("document")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Document { get; set; }
}

public sealed class RerankUsage
{
    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; set; }
}
