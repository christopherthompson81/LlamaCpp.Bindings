namespace LlamaCpp.Bindings.Server.Configuration;

/// <summary>
/// Configuration bound from <c>appsettings.json</c> and CLI overrides via the
/// standard <c>Microsoft.Extensions.Configuration</c> provider chain.
/// </summary>
/// <remarks>
/// V1 is intentionally small. Anything not here (API keys, SSL, model URL
/// fetch, embeddings/rerank endpoints, per-slot snapshots, metrics, web UI)
/// lives in follow-up issues rather than this struct.
/// </remarks>
public sealed class ServerOptions
{
    public const string Section = "LlamaServer";

    // ----- Model sources -----

    /// <summary>Path to the GGUF file to load on startup. Required.</summary>
    public string ModelPath { get; set; } = "";

    /// <summary>Optional alias reported via <c>/v1/models</c>. Defaults to the file stem.</summary>
    public string? ModelAlias { get; set; }

    // ----- Context sizing -----

    public int ContextSize { get; set; } = 2048;
    public int LogicalBatchSize { get; set; } = 512;
    public int PhysicalBatchSize { get; set; } = 512;

    /// <summary>
    /// Number of parallel conversation slots. Each in-flight request
    /// occupies one slot; beyond this, requests queue.
    /// </summary>
    public int MaxSequenceCount { get; set; } = 4;

    // ----- GPU offload -----

    /// <summary><c>-1</c> = offload every layer to the best available backend. <c>0</c> = CPU only.</summary>
    public int GpuLayerCount { get; set; } = -1;

    public bool OffloadKqv { get; set; } = true;

    // ----- File I/O -----

    public bool UseMmap { get; set; } = true;
    public bool UseMlock { get; set; } = false;

    // ----- HTTP -----

    /// <summary>Serialized ASP.NET Kestrel URLs, e.g. <c>http://127.0.0.1:8080</c>.</summary>
    public string Urls { get; set; } = "http://127.0.0.1:8080";

    /// <summary>
    /// Maximum tokens any single request may generate, regardless of what
    /// the caller asks for. Guards against runaway generations on a shared
    /// server; individual requests may still specify a smaller
    /// <c>max_tokens</c>.
    /// </summary>
    public int MaxOutputTokens { get; set; } = 2048;

    // ----- Authentication -----

    /// <summary>
    /// API keys accepted on all endpoints except <c>/health</c>. Empty =
    /// auth disabled (localhost dev default). Clients present keys as
    /// <c>Authorization: Bearer &lt;key&gt;</c> (OpenAI-style) or
    /// <c>X-Api-Key: &lt;key&gt;</c>.
    /// </summary>
    public List<string> ApiKeys { get; set; } = new();

    /// <summary>
    /// Optional path to a text file containing one API key per line.
    /// Lines may be blank or start with <c>#</c> (comments). Keys loaded
    /// from this file are merged with <see cref="ApiKeys"/>; either
    /// source is sufficient. Matches llama-server's <c>--api-key-file</c>.
    /// </summary>
    public string? ApiKeyFile { get; set; }
}
