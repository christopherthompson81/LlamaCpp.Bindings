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

    /// <summary>
    /// Index of the GPU to use when <see cref="SplitMode"/> is
    /// <c>None</c>; the device that holds the model when the model
    /// fits on a single GPU. Default <c>0</c> = first GPU.
    /// </summary>
    public int MainGpu { get; set; } = 0;

    /// <summary>
    /// How to split a model that doesn't fit on one GPU.
    /// <c>None</c> = single-GPU (uses <see cref="MainGpu"/>);
    /// <c>Layer</c> = layer-wise split (default);
    /// <c>Row</c> = row-wise split (rare; needed only for some
    /// models / hardware combinations).
    /// </summary>
    public LlamaSplitMode SplitMode { get; set; } = LlamaSplitMode.Layer;

    /// <summary>
    /// Verify tensor data during model load. Slow but catches
    /// corrupted GGUFs early instead of producing garbage at decode
    /// time. Off by default — operators can flip this on for ops
    /// that ship third-party model files.
    /// </summary>
    public bool CheckTensors { get; set; } = false;

    /// <summary>
    /// Threads used for single-token decode (per-token work). <c>-1</c>
    /// = let llama.cpp pick (default). Most chat workloads with a
    /// GPU model don't benefit from raising this; CPU-only deployments
    /// usually want it pinned to physical-core count.
    /// </summary>
    public int ThreadCount { get; set; } = -1;

    /// <summary>
    /// Threads used for batched (prompt-processing) decode. <c>-1</c> =
    /// match <see cref="ThreadCount"/>. Useful when the prompt eval is
    /// the bottleneck and decode benefits from a different thread budget.
    /// </summary>
    public int BatchThreadCount { get; set; } = -1;

    /// <summary>
    /// Flash Attention selection. <c>Auto</c> (default) lets llama.cpp
    /// pick per-model based on its FA support; <c>Enabled</c> forces it
    /// on (required for any quantised KV cache); <c>Disabled</c> turns
    /// it off (debug only).
    /// </summary>
    public LlamaFlashAttention FlashAttention { get; set; } = LlamaFlashAttention.Auto;

    /// <summary>
    /// Use a full-size SWA cache rather than the compact one. The
    /// binding defaults this to <c>true</c> because it allows the KV
    /// cache to be edited (multi-turn chat with retried turns); set
    /// <c>false</c> for memory-constrained workloads where the KV is
    /// append-only.
    /// </summary>
    public bool UseFullSwaCache { get; set; } = true;

    /// <summary>
    /// Element type for the K component of the KV cache. <c>F16</c>
    /// = llama.cpp default. Quantised options (<c>Q8_0</c>, <c>Q4_0</c>,
    /// etc.) shrink the cache footprint but require Flash Attention —
    /// see <see cref="FlashAttention"/>. The llama.cpp header flags
    /// these as EXPERIMENTAL; behaviour can change across version bumps.
    /// </summary>
    public LlamaKvCacheType KvCacheTypeK { get; set; } = LlamaKvCacheType.F16;

    /// <summary>Element type for the V component of the KV cache. See <see cref="KvCacheTypeK"/>.</summary>
    public LlamaKvCacheType KvCacheTypeV { get; set; } = LlamaKvCacheType.F16;

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

    /// <summary>
    /// Hard cap on tokenised prompt length. Requests with longer prompts
    /// reject with HTTP 413 before any pool slot is taken — fails fast
    /// instead of letting the model decode hit a "no KV slot" error
    /// halfway through. <c>0</c> or negative = derive at request time
    /// from <c>ContextSize - MaxOutputTokens</c> (leaves room for the
    /// reply); <c>&gt; 0</c> = use this value directly.
    /// </summary>
    public int MaxPromptTokens { get; set; } = 0;

    /// <summary>
    /// Server-side wall-clock cap on a single request's generation
    /// budget. <c>0</c> = disabled (rely on client cancellation only);
    /// <c>&gt; 0</c> = cancel the generator after this many seconds and
    /// return HTTP 504 from non-streaming requests. Streaming requests
    /// just close the connection — clients observe end-of-stream early.
    /// Default 300 (5 minutes).
    /// </summary>
    public int RequestTimeoutSeconds { get; set; } = 300;

    /// <summary>
    /// Maximum seconds the host waits for in-flight requests to drain
    /// after SIGTERM before forcibly closing connections. Default 30
    /// matches ASP.NET Core's own host default; raise for long-running
    /// generation jobs that can't tolerate truncation, lower for
    /// faster restarts in CI / development.
    /// </summary>
    public int ShutdownDrainSeconds { get; set; } = 30;

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

    // ----- Rerank (optional third model) -----

    /// <summary>
    /// Path to a reranker GGUF (bge-reranker, jina-reranker, etc.) for
    /// the <c>/v1/rerank</c> endpoint. Reranker models output a single
    /// relevance score per (query, document) pair via a rank pooling
    /// head — not interchangeable with an embedding model. When unset,
    /// <c>/v1/rerank</c> returns 501.
    /// </summary>
    public string? RerankModelPath { get; set; }

    public int RerankContextSize { get; set; } = 2048;
    public int RerankBatchSize { get; set; } = 512;
    public int RerankGpuLayerCount { get; set; } = -1;
    public string? RerankModelAlias { get; set; }

    // ----- Speculative decoding (optional draft model) -----

    /// <summary>
    /// Path to a small "draft" GGUF whose tokens propose ahead for the main
    /// model. When set, the server loads the draft model + a dedicated
    /// speculative main context and exposes per-request opt-in via the
    /// chat-completions <c>speculative</c> field. Unset = endpoint ignores
    /// any <c>speculative</c> flag and runs through the normal generator.
    /// </summary>
    /// <remarks>
    /// <para>The draft must share the main model's tokenizer family (same
    /// SPM/BPE, same special-token ids) — the binding's
    /// <c>LlamaSpeculativeGenerator</c> rejects mismatched pairs at
    /// construction time. Pair a large main with a small fast draft
    /// (e.g. Qwen3-14B + Qwen3-0.6B); a mismatched draft is slower than
    /// plain decoding because every rejection still costs one draft
    /// decode plus a wasted main batch slot.</para>
    /// <para>Speculative requests serialize through a dedicated semaphore —
    /// concurrency is capped at 1 — and don't share state with the
    /// non-speculative session pool, so they don't get prefix caching.</para>
    /// </remarks>
    public string? DraftModelPath { get; set; }

    /// <summary>Context length for the draft model. Larger eats VRAM.</summary>
    public int DraftContextSize { get; set; } = 2048;

    /// <summary>Logical batch size for the draft model.</summary>
    public int DraftLogicalBatchSize { get; set; } = 512;

    /// <summary>Physical batch size for the draft model.</summary>
    public int DraftPhysicalBatchSize { get; set; } = 512;

    /// <summary>GPU offload for the draft model. <c>-1</c> = all, <c>0</c> = CPU.</summary>
    public int DraftGpuLayerCount { get; set; } = -1;

    /// <summary>
    /// Number of tokens the draft proposes per speculation round. 5 is
    /// llama.cpp's default and tends to be a sweet spot. Higher values pay
    /// more on rejections; lower values leave throughput on the table.
    /// </summary>
    public int DraftLookahead { get; set; } = 5;

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
