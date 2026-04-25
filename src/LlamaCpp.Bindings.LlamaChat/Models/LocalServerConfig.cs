using System.Collections.Generic;

namespace LlamaCpp.Bindings.LlamaChat.Models;

/// <summary>
/// Persisted configuration for the in-process "Server" tab — drives a child
/// <c>LlamaCpp.Bindings.Server</c> process spawned from inside LlamaChat.
/// Stored at <c>~/.config/LlamaChat/local-server-config.json</c>.
///
/// Mirrors most fields of <c>LlamaCpp.Bindings.Server.Configuration.ServerOptions</c>
/// directly; what isn't here is the list-of-records surface
/// (<c>TensorBuftOverrides</c>, <c>LoraAdapters</c>, <c>ControlVectors</c>) —
/// those need bespoke editors and live in <see cref="ExtraArgs"/> for now.
/// </summary>
public sealed record LocalServerConfig
{
    // ----- LlamaChat-side knobs (not in ServerOptions) -----

    /// <summary>
    /// Explicit path to the server executable (or .dll). Null = auto-discover
    /// alongside LlamaChat's output (<c>./server/</c>) with a dev-mode
    /// fallback to the sibling project's <c>bin/</c>.
    /// </summary>
    public string? ServerExecutablePath { get; init; }

    public bool LaunchOnAppStart { get; init; }
    public bool AutoCreateRemoteProfile { get; init; } = true;
    public bool AutoSelectProfileOnLaunch { get; init; }

    /// <summary>One arg per entry, passed verbatim after the generated args.</summary>
    public List<string> ExtraArgs { get; init; } = new();

    /// <summary>How long to wait for <c>GET /health</c> to start returning 200.</summary>
    public int StartupTimeoutSeconds { get; init; } = 30;

    // ----- Model -----

    public string ModelPath { get; init; } = "";

    /// <summary>Reported via <c>/v1/models</c>. Empty = file stem.</summary>
    public string ModelAlias { get; init; } = "";

    // ----- Context / batching -----

    public int ContextSize { get; init; } = 4096;
    public int LogicalBatchSize { get; init; } = 512;
    public int PhysicalBatchSize { get; init; } = 512;
    public int MaxSequenceCount { get; init; } = 4;

    // ----- GPU offload -----

    /// <summary><c>-1</c> = all layers; <c>0</c> = CPU.</summary>
    public int GpuLayerCount { get; init; } = -1;
    public bool OffloadKqv { get; init; } = true;
    public int MainGpu { get; init; } = 0;
    public LlamaSplitMode SplitMode { get; init; } = LlamaSplitMode.Layer;
    public bool NoHost { get; init; } = false;
    public bool UseExtraBufts { get; init; } = true;
    public bool CpuMoe { get; init; } = false;
    public bool CheckTensors { get; init; } = false;
    public bool UseDirectIo { get; init; } = false;

    /// <summary>One device name per entry (e.g. <c>CUDA0</c>). Empty = all devices.</summary>
    public List<string> Devices { get; init; } = new();

    /// <summary>Per-device offload proportions. Empty = even split.</summary>
    public List<float> TensorSplit { get; init; } = new();

    // ----- CPU / threading -----

    /// <summary><c>-1</c> = let llama.cpp pick.</summary>
    public int ThreadCount { get; init; } = -1;
    /// <summary><c>-1</c> = match <see cref="ThreadCount"/>.</summary>
    public int BatchThreadCount { get; init; } = -1;
    public LlamaNumaStrategy NumaStrategy { get; init; } = LlamaNumaStrategy.Disabled;

    // ----- KV cache -----

    public LlamaFlashAttention FlashAttention { get; init; } = LlamaFlashAttention.Auto;
    public LlamaKvCacheType KvCacheTypeK { get; init; } = LlamaKvCacheType.F16;
    public LlamaKvCacheType KvCacheTypeV { get; init; } = LlamaKvCacheType.F16;
    public bool UseFullSwaCache { get; init; } = true;

    // ----- File I/O -----

    public bool UseMmap { get; init; } = true;
    public bool UseMlock { get; init; } = false;

    // ----- HTTP -----

    public string BindAddress { get; init; } = "127.0.0.1";
    public int Port { get; init; } = 8080;

    /// <summary>PKCS#12 certificate path. Empty = HTTP only.</summary>
    public string HttpsCertificatePath { get; init; } = "";
    public string HttpsCertificatePassword { get; init; } = "";

    // ----- CORS -----

    /// <summary>Empty = CORS disabled. <c>*</c> = any origin.</summary>
    public List<string> CorsAllowedOrigins { get; init; } = new();
    public bool CorsAllowCredentials { get; init; } = false;

    // ----- Auth -----

    /// <summary>Bearer token. Empty = no auth (localhost dev default).</summary>
    public string ApiKey { get; init; } = "";

    /// <summary>Path to a text file with one API key per line. Merged with <see cref="ApiKey"/>.</summary>
    public string ApiKeyFile { get; init; } = "";

    // ----- Limits -----

    public int MaxOutputTokens { get; init; } = 2048;
    /// <summary><c>0</c> = derive from <see cref="ContextSize"/> at request time.</summary>
    public int MaxPromptTokens { get; init; } = 0;
    /// <summary><c>0</c> = no server-side wall-clock cap.</summary>
    public int RequestTimeoutSeconds { get; init; } = 300;
    public int ShutdownDrainSeconds { get; init; } = 30;

    // ----- Endpoints -----

    public bool ExposeMetricsEndpoint { get; init; } = true;
    public bool ExposeSlotsEndpoint { get; init; } = true;

    // ----- RoPE / YARN -----

    public LlamaRopeScalingType RopeScalingType { get; init; } = LlamaRopeScalingType.Unspecified;
    public float RopeFreqBase { get; init; } = 0f;
    public float RopeFreqScale { get; init; } = 0f;
    public float YarnExtFactor { get; init; } = -1f;
    public float YarnAttnFactor { get; init; } = 1f;
    public float YarnBetaFast { get; init; } = 32f;
    public float YarnBetaSlow { get; init; } = 1f;
    public uint YarnOriginalContext { get; init; } = 0;

    // ----- Multimodal (mmproj) -----

    public string MmprojPath { get; init; } = "";
    public bool MmprojAuto { get; init; } = false;

    /// <summary>Force mmproj encoder on CPU even when GPU-offloaded. Default = let llama.cpp pick.</summary>
    public bool MmprojOnCpu { get; init; } = false;

    /// <summary><c>0</c> = inherit from mmproj defaults.</summary>
    public int MmprojImageMinTokens { get; init; } = 0;
    /// <summary><c>0</c> = inherit from mmproj defaults.</summary>
    public int MmprojImageMaxTokens { get; init; } = 0;

    // ----- Embeddings model -----

    public string EmbeddingModelPath { get; init; } = "";
    public int EmbeddingContextSize { get; init; } = 2048;
    public int EmbeddingBatchSize { get; init; } = 512;
    public int EmbeddingGpuLayerCount { get; init; } = -1;
    public string EmbeddingModelAlias { get; init; } = "";

    // ----- Rerank model -----

    public string RerankModelPath { get; init; } = "";
    public int RerankContextSize { get; init; } = 2048;
    public int RerankBatchSize { get; init; } = 512;
    public int RerankGpuLayerCount { get; init; } = -1;
    public string RerankModelAlias { get; init; } = "";

    // ----- Speculative decoding (draft model) -----

    public string DraftModelPath { get; init; } = "";
    public int DraftContextSize { get; init; } = 2048;
    public int DraftLogicalBatchSize { get; init; } = 512;
    public int DraftPhysicalBatchSize { get; init; } = 512;
    public int DraftGpuLayerCount { get; init; } = -1;
    public int DraftLookahead { get; init; } = 5;

    public string BaseUrl => HttpsCertificatePath.Length > 0
        ? $"https://{BindAddress}:{Port}"
        : $"http://{BindAddress}:{Port}";
}
