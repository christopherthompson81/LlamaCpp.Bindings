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

    /// <summary>
    /// Serialized ASP.NET Kestrel URLs, e.g. <c>http://127.0.0.1:8080</c>.
    /// To enable TLS, prefix with <c>https://</c> and set
    /// <see cref="HttpsCertificatePath"/>.
    /// </summary>
    public string Urls { get; set; } = "http://127.0.0.1:8080";

    /// <summary>
    /// Path to a PKCS#12 (<c>.pfx</c>) certificate file. When set, Kestrel's
    /// HTTPS default is configured to serve this certificate, and any
    /// <c>https://</c> URL in <see cref="Urls"/> becomes reachable.
    /// </summary>
    public string? HttpsCertificatePath { get; set; }

    /// <summary>Password protecting the PKCS#12 file, if any.</summary>
    public string? HttpsCertificatePassword { get; set; }

    // ----- CORS (for browser clients) -----

    /// <summary>
    /// Origins to allow on cross-origin requests. <c>null</c> or empty =
    /// CORS middleware disabled (default — localhost clients don't need
    /// it). <c>["*"]</c> = any origin (matches llama-server's default).
    /// Otherwise an exact-match allow-list of origin strings (e.g.
    /// <c>"https://chat.example.com"</c>).
    /// </summary>
    public List<string>? CorsAllowedOrigins { get; set; }

    /// <summary>
    /// When true, responses carry <c>Access-Control-Allow-Credentials: true</c>
    /// so browsers attach cookies / auth headers to cross-origin requests.
    /// Incompatible with <c>CorsAllowedOrigins = ["*"]</c> per the CORS spec
    /// — when both are set, the wildcard is silently downgraded to mirror
    /// the incoming <c>Origin</c> instead.
    /// </summary>
    public bool CorsAllowCredentials { get; set; } = false;

    /// <summary>
    /// Maximum tokens any single request may generate, regardless of what
    /// the caller asks for. Guards against runaway generations on a shared
    /// server; individual requests may still specify a smaller
    /// <c>max_tokens</c>.
    /// </summary>
    public int MaxOutputTokens { get; set; } = 2048;

    // ----- Multimodal (optional mmproj) -----

    /// <summary>
    /// Path to a multimodal projector GGUF (<c>mmproj-*.gguf</c>) that
    /// pairs with the main model. When set, <c>/v1/chat/completions</c>
    /// accepts OpenAI multi-part content with <c>image_url</c> entries;
    /// when unset, image parts in a request return HTTP 400.
    /// </summary>
    public string? MmprojPath { get; set; }

    /// <summary>
    /// Run mmproj encoder on CPU even when the main model is GPU-offloaded.
    /// Some vision encoders (Gemma-3, Qwen2.5-VL) only have CPU paths in
    /// llama.cpp at the moment; others benefit from shared GPU memory.
    /// Null = llama.cpp picks the default for the model.
    /// </summary>
    public bool? MmprojOnCpu { get; set; }

    /// <summary>
    /// Floor / ceiling on how many tokens a single image occupies in
    /// context. Different VLMs accept different ranges; leave null to
    /// inherit the mmproj file's defaults.
    /// </summary>
    public int? MmprojImageMinTokens { get; set; }

    /// <summary>Ceiling on how many tokens a single image occupies. See <see cref="MmprojImageMinTokens"/>.</summary>
    public int? MmprojImageMaxTokens { get; set; }

    // ----- Embeddings (optional second model) -----

    /// <summary>
    /// Path to a GGUF for the <c>/v1/embeddings</c> endpoint. A dedicated
    /// embedding model (BGE, nomic-embed, etc.) is required — a chat model
    /// loaded in embeddings mode will produce vectors but not meaningful
    /// ones, since it wasn't trained with a pooling head.
    /// </summary>
    /// <remarks>
    /// When null or empty, the <c>/v1/embeddings</c> endpoint is still
    /// registered but returns HTTP 501 with an explanatory body. The chat
    /// endpoints remain unaffected.
    /// </remarks>
    public string? EmbeddingModelPath { get; set; }

    /// <summary>Context length for the embedding model. Most embedding models cap at 512–8192.</summary>
    public int EmbeddingContextSize { get; set; } = 2048;

    /// <summary>Logical batch size for the embedding model.</summary>
    public int EmbeddingBatchSize { get; set; } = 512;

    /// <summary>GPU offload for the embedding model. <c>-1</c> = all, <c>0</c> = CPU.</summary>
    public int EmbeddingGpuLayerCount { get; set; } = -1;

    /// <summary>Alias reported via <c>/v1/embeddings</c> responses. Falls back to the file stem.</summary>
    public string? EmbeddingModelAlias { get; set; }

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
