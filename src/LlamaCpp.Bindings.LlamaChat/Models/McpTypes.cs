using System;
using System.Collections.Generic;
using System.Text.Json;

namespace LlamaCpp.Bindings.LlamaChat.Models;

/// <summary>
/// Health-check state for a single server. <see cref="Idle"/> is the initial
/// state before any connection attempt.
/// </summary>
public enum McpConnectionState
{
    Idle,
    Connecting,
    Ready,
    Error,
    Disabled,
}

/// <summary>
/// A tool advertised by an MCP server, as returned by <c>tools/list</c>.
/// The input schema is kept as raw JSON so we can forward it verbatim into
/// the chat-template Jinja context without a round-trip through a typed
/// model.
/// </summary>
public sealed record McpToolInfo(
    string Name,
    string? Description,
    JsonElement InputSchema);

/// <summary>
/// One named prompt exposed by an MCP server. <see cref="Arguments"/> is the
/// argument schema returned by <c>prompts/list</c>.
/// </summary>
public sealed record McpPromptInfo(
    string Name,
    string? Description,
    IReadOnlyList<McpPromptArgument> Arguments);

public sealed record McpPromptArgument(
    string Name,
    string? Description,
    bool Required);

/// <summary>
/// A resource advertised by <c>resources/list</c>. The MCP spec defines
/// URIs as opaque strings with server-defined semantics; we don't attempt
/// hierarchical parsing beyond what the server explicitly returns.
/// </summary>
public sealed record McpResourceInfo(
    string Uri,
    string? Name,
    string? Description,
    string? MimeType);

/// <summary>
/// Advertised capability flags — matches the MCP initialize response. Each
/// flag mirrors whether the server implements the corresponding feature
/// cluster. Used to render the per-server capability badge row.
/// </summary>
public sealed record McpCapabilities(
    bool Tools,
    bool Prompts,
    bool Resources);

/// <summary>
/// One entry in the rolling execution log shown in the "MCP logs" dialog.
/// Captures the JSON-RPC request/response round-trip on the wire plus any
/// local parse/exec errors. Entries are ephemeral — not persisted.
/// </summary>
public sealed record McpExecutionLogEntry(
    DateTimeOffset Timestamp,
    Guid ServerId,
    string ServerName,
    string Direction,
    string Summary,
    string? Payload);
