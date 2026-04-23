using System;
using System.Collections.Generic;

namespace LlamaCpp.Bindings.LlamaChat.Models;

/// <summary>
/// A user-declared MCP server the client will connect to. Persisted to
/// <c>mcp-servers.json</c>; instances are mutable records so the settings UI
/// can edit them in place before saving.
/// </summary>
public sealed record McpServerConfig
{
    public Guid Id { get; init; } = Guid.NewGuid();

    /// <summary>Human-readable label shown in lists and avatars.</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Base URL of the MCP endpoint — Streamable HTTP transport. e.g.
    /// <c>https://example.com/mcp</c>. The client POSTs JSON-RPC messages
    /// here; the server may respond with a JSON body or upgrade to an SSE
    /// stream for streaming responses.
    /// </summary>
    public string Url { get; set; } = string.Empty;

    /// <summary>
    /// Extra HTTP headers to send on every request (e.g. bearer tokens).
    /// One entry per key, case-insensitive by standard HTTP convention.
    /// </summary>
    public Dictionary<string, string> Headers { get; set; } = new();

    /// <summary>Per-server enable toggle. Disabled servers skip health-check + tool exposure.</summary>
    public bool Enabled { get; set; } = true;
}
