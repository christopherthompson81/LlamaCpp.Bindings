using System.Collections.Generic;

namespace LlamaCpp.Bindings.LlamaChat.Models;

/// <summary>
/// Persisted configuration for the in-process "Server" tab — drives a child
/// <c>LlamaCpp.Bindings.Server</c> process spawned from inside LlamaChat.
/// Stored at <c>~/.config/LlamaChat/local-server-config.json</c>. Anything
/// not exposed here can be reached through <see cref="ExtraArgs"/> or by
/// editing the JSON directly.
/// </summary>
public sealed record LocalServerConfig
{
    /// <summary>
    /// Explicit path to the server executable (or .dll). Null = auto-discover
    /// alongside LlamaChat's output (<c>./server/</c>) with a dev-mode
    /// fallback to the sibling project's <c>bin/</c>.
    /// </summary>
    public string? ServerExecutablePath { get; init; }

    /// <summary>Path to the GGUF model file the server should load on startup.</summary>
    public string ModelPath { get; init; } = "";

    /// <summary>Address Kestrel binds to. Default loopback only.</summary>
    public string BindAddress { get; init; } = "127.0.0.1";

    public int Port { get; init; } = 8080;

    public int ContextSize { get; init; } = 4096;

    /// <summary><c>-1</c> = offload all layers; <c>0</c> = CPU.</summary>
    public int GpuLayerCount { get; init; } = -1;

    public int MaxSequenceCount { get; init; } = 4;

    public LlamaFlashAttention FlashAttention { get; init; } = LlamaFlashAttention.Auto;

    /// <summary>Bearer token. Null/empty = no auth (localhost dev default).</summary>
    public string? ApiKey { get; init; }

    public bool LaunchOnAppStart { get; init; }

    public bool AutoCreateRemoteProfile { get; init; } = true;

    public bool AutoSelectProfileOnLaunch { get; init; }

    /// <summary>One arg per entry, passed verbatim after the generated args.</summary>
    public List<string> ExtraArgs { get; init; } = new();

    /// <summary>How long to wait for <c>GET /health</c> to start returning 200.</summary>
    public int StartupTimeoutSeconds { get; init; } = 30;

    public string BaseUrl => $"http://{BindAddress}:{Port}";
}
