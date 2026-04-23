using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Minimum-viable MCP client over the Streamable HTTP transport (MCP spec
/// 2025-06). We POST JSON-RPC messages to the server URL; the server either
/// replies with a JSON body or upgrades to an SSE stream. For the surface the
/// UI needs (initialize, tools/list, tools/call, prompts/*, resources/*), the
/// happy path terminates in a single JSON-RPC response, which is all this
/// client has to understand. Streaming intermediate results are surfaced but
/// we wait for the final response before returning.
///
/// Not thread-safe. One instance per server connection; <see cref="McpClientService"/>
/// owns the lifecycle.
/// </summary>
internal sealed class McpClient : IDisposable
{
    public Guid ServerId { get; }
    public string ServerName { get; }
    public string Url { get; }
    public McpCapabilities? Capabilities { get; private set; }

    private readonly HttpClient _http;
    private readonly Action<McpExecutionLogEntry>? _log;
    private string? _sessionId;
    private long _nextRequestId = 1;
    private bool _disposed;

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
    };

    public McpClient(
        McpServerConfig cfg,
        HttpClient http,
        Action<McpExecutionLogEntry>? log = null)
    {
        ServerId = cfg.Id;
        ServerName = cfg.Name;
        Url = cfg.Url;
        _http = http;
        _log = log;

        foreach (var (k, v) in cfg.Headers)
        {
            if (string.IsNullOrWhiteSpace(k)) continue;
            if (!_http.DefaultRequestHeaders.TryAddWithoutValidation(k, v))
            {
                // Some headers (Content-Type etc.) belong on the request body,
                // not defaults. Best-effort.
            }
        }
    }

    /// <summary>
    /// Run the MCP <c>initialize</c> handshake, then the <c>initialized</c>
    /// notification. Populates <see cref="Capabilities"/> on success.
    /// Throws on transport/JSON-RPC error.
    /// </summary>
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        var initParams = new Dictionary<string, object?>
        {
            ["protocolVersion"] = "2025-06-18",
            ["capabilities"] = new Dictionary<string, object?>(),
            ["clientInfo"] = new Dictionary<string, object?>
            {
                ["name"] = "LlamaChat",
                ["version"] = "0.1",
            },
        };
        var res = await RequestAsync("initialize", initParams, ct).ConfigureAwait(false);

        bool hasTools = false, hasPrompts = false, hasResources = false;
        if (res.TryGetProperty("capabilities", out var caps) && caps.ValueKind == JsonValueKind.Object)
        {
            hasTools = caps.TryGetProperty("tools", out _);
            hasPrompts = caps.TryGetProperty("prompts", out _);
            hasResources = caps.TryGetProperty("resources", out _);
        }
        Capabilities = new McpCapabilities(hasTools, hasPrompts, hasResources);

        // initialized notification — fire and forget.
        await NotifyAsync("notifications/initialized", null, ct).ConfigureAwait(false);
    }

    public async Task<IReadOnlyList<McpToolInfo>> ListToolsAsync(CancellationToken ct = default)
    {
        if (Capabilities?.Tools != true) return Array.Empty<McpToolInfo>();
        var res = await RequestAsync("tools/list", null, ct).ConfigureAwait(false);
        var list = new List<McpToolInfo>();
        if (res.TryGetProperty("tools", out var tools) && tools.ValueKind == JsonValueKind.Array)
        {
            foreach (var t in tools.EnumerateArray())
            {
                var name = t.TryGetProperty("name", out var n) ? n.GetString() ?? string.Empty : string.Empty;
                if (string.IsNullOrEmpty(name)) continue;
                var desc = t.TryGetProperty("description", out var d) ? d.GetString() : null;
                var schema = t.TryGetProperty("inputSchema", out var s)
                    ? s.Clone()
                    : JsonDocument.Parse("{}").RootElement.Clone();
                list.Add(new McpToolInfo(name, desc, schema));
            }
        }
        return list;
    }

    public async Task<JsonElement> CallToolAsync(
        string name,
        JsonElement arguments,
        CancellationToken ct = default)
    {
        var paramsObj = new Dictionary<string, object?>
        {
            ["name"] = name,
            ["arguments"] = JsonDocument.Parse(arguments.GetRawText()).RootElement,
        };
        return await RequestAsync("tools/call", paramsObj, ct).ConfigureAwait(false);
    }

    public async Task<IReadOnlyList<McpPromptInfo>> ListPromptsAsync(CancellationToken ct = default)
    {
        if (Capabilities?.Prompts != true) return Array.Empty<McpPromptInfo>();
        var res = await RequestAsync("prompts/list", null, ct).ConfigureAwait(false);
        var list = new List<McpPromptInfo>();
        if (res.TryGetProperty("prompts", out var prompts) && prompts.ValueKind == JsonValueKind.Array)
        {
            foreach (var p in prompts.EnumerateArray())
            {
                var name = p.TryGetProperty("name", out var n) ? n.GetString() ?? string.Empty : string.Empty;
                if (string.IsNullOrEmpty(name)) continue;
                var desc = p.TryGetProperty("description", out var d) ? d.GetString() : null;
                var args = new List<McpPromptArgument>();
                if (p.TryGetProperty("arguments", out var ar) && ar.ValueKind == JsonValueKind.Array)
                {
                    foreach (var a in ar.EnumerateArray())
                    {
                        var an = a.TryGetProperty("name", out var x) ? x.GetString() : null;
                        if (string.IsNullOrEmpty(an)) continue;
                        var ad = a.TryGetProperty("description", out var y) ? y.GetString() : null;
                        var req = a.TryGetProperty("required", out var r)
                            && r.ValueKind == JsonValueKind.True;
                        args.Add(new McpPromptArgument(an, ad, req));
                    }
                }
                list.Add(new McpPromptInfo(name, desc, args));
            }
        }
        return list;
    }

    public async Task<string> GetPromptAsync(
        string name, IReadOnlyDictionary<string, string>? arguments,
        CancellationToken ct = default)
    {
        var paramsObj = new Dictionary<string, object?> { ["name"] = name };
        if (arguments is { Count: > 0 })
        {
            paramsObj["arguments"] = arguments;
        }
        var res = await RequestAsync("prompts/get", paramsObj, ct).ConfigureAwait(false);

        // MCP prompts/get returns { description, messages: [ { role, content: { type: "text", text } } ] }
        // Concatenate all text content for a simple insert. Complex messages
        // (images, multi-part) are beyond v1 picker scope — we fall back to
        // the raw JSON in that case.
        if (res.TryGetProperty("messages", out var msgs) && msgs.ValueKind == JsonValueKind.Array)
        {
            var sb = new StringBuilder();
            foreach (var m in msgs.EnumerateArray())
            {
                if (!m.TryGetProperty("content", out var c)) continue;
                if (c.ValueKind == JsonValueKind.Object
                    && c.TryGetProperty("type", out var ct2) && ct2.GetString() == "text"
                    && c.TryGetProperty("text", out var tx))
                {
                    if (sb.Length > 0) sb.Append("\n\n");
                    sb.Append(tx.GetString() ?? string.Empty);
                }
            }
            if (sb.Length > 0) return sb.ToString();
        }
        return res.GetRawText();
    }

    public async Task<IReadOnlyList<McpResourceInfo>> ListResourcesAsync(CancellationToken ct = default)
    {
        if (Capabilities?.Resources != true) return Array.Empty<McpResourceInfo>();
        var res = await RequestAsync("resources/list", null, ct).ConfigureAwait(false);
        var list = new List<McpResourceInfo>();
        if (res.TryGetProperty("resources", out var resources) && resources.ValueKind == JsonValueKind.Array)
        {
            foreach (var r in resources.EnumerateArray())
            {
                var uri = r.TryGetProperty("uri", out var u) ? u.GetString() ?? string.Empty : string.Empty;
                if (string.IsNullOrEmpty(uri)) continue;
                var name = r.TryGetProperty("name", out var n) ? n.GetString() : null;
                var desc = r.TryGetProperty("description", out var d) ? d.GetString() : null;
                var mime = r.TryGetProperty("mimeType", out var m) ? m.GetString() : null;
                list.Add(new McpResourceInfo(uri, name, desc, mime));
            }
        }
        return list;
    }

    /// <summary>
    /// Fetch a resource's textual contents. Returns the concatenated text of
    /// every <c>text</c>-typed content block; non-text (binary/blob) contents
    /// return a placeholder string since the preview UI only renders text.
    /// </summary>
    public async Task<string> ReadResourceAsync(string uri, CancellationToken ct = default)
    {
        var res = await RequestAsync("resources/read",
            new Dictionary<string, object?> { ["uri"] = uri }, ct).ConfigureAwait(false);
        var sb = new StringBuilder();
        if (res.TryGetProperty("contents", out var contents) && contents.ValueKind == JsonValueKind.Array)
        {
            foreach (var c in contents.EnumerateArray())
            {
                if (c.TryGetProperty("text", out var t) && t.ValueKind == JsonValueKind.String)
                {
                    if (sb.Length > 0) sb.Append("\n\n");
                    sb.Append(t.GetString() ?? string.Empty);
                }
                else if (c.TryGetProperty("blob", out _))
                {
                    sb.Append("(binary content, not shown)");
                }
            }
        }
        return sb.Length > 0 ? sb.ToString() : res.GetRawText();
    }

    // ============================================================
    // JSON-RPC transport
    // ============================================================

    private async Task<JsonElement> RequestAsync(
        string method, object? paramsObj, CancellationToken ct)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(McpClient));

        var id = Interlocked.Increment(ref _nextRequestId);
        var envelope = new Dictionary<string, object?>
        {
            ["jsonrpc"] = "2.0",
            ["id"] = id,
            ["method"] = method,
        };
        if (paramsObj is not null) envelope["params"] = paramsObj;

        var body = JsonSerializer.Serialize(envelope, JsonOpts);
        _log?.Invoke(new McpExecutionLogEntry(
            DateTimeOffset.UtcNow, ServerId, ServerName, "→", method, body));

        using var req = new HttpRequestMessage(HttpMethod.Post, Url)
        {
            Content = new StringContent(body, Encoding.UTF8, "application/json"),
        };
        req.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        req.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));
        if (_sessionId is not null)
        {
            req.Headers.TryAddWithoutValidation("Mcp-Session-Id", _sessionId);
        }

        using var res = await _http.SendAsync(
            req, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);

        // Pick up the session id the server assigns on the initialize reply.
        if (res.Headers.TryGetValues("Mcp-Session-Id", out var sidVals))
        {
            foreach (var v in sidVals) { _sessionId = v; break; }
        }

        if (!res.IsSuccessStatusCode)
        {
            var err = await res.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            _log?.Invoke(new McpExecutionLogEntry(
                DateTimeOffset.UtcNow, ServerId, ServerName,
                "✕", $"HTTP {(int)res.StatusCode}: {res.ReasonPhrase}", err));
            throw new InvalidOperationException(
                $"MCP {method} failed: HTTP {(int)res.StatusCode} {res.ReasonPhrase}");
        }

        var contentType = res.Content.Headers.ContentType?.MediaType ?? "application/json";
        JsonDocument doc;
        if (contentType.Contains("event-stream", StringComparison.OrdinalIgnoreCase))
        {
            doc = await ReadSseFinalResponseAsync(res, id, ct).ConfigureAwait(false);
        }
        else
        {
            var json = await res.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            _log?.Invoke(new McpExecutionLogEntry(
                DateTimeOffset.UtcNow, ServerId, ServerName, "←", method, json));
            doc = JsonDocument.Parse(json);
        }

        try
        {
            var root = doc.RootElement;
            if (root.TryGetProperty("error", out var errEl))
            {
                var msg = errEl.TryGetProperty("message", out var m) ? m.GetString() : "(no message)";
                var code = errEl.TryGetProperty("code", out var c) && c.TryGetInt32(out var ci) ? ci : 0;
                throw new InvalidOperationException($"MCP {method} error {code}: {msg}");
            }
            if (root.TryGetProperty("result", out var result))
            {
                return result.Clone();
            }
            return root.Clone();
        }
        finally
        {
            doc.Dispose();
        }
    }

    private async Task NotifyAsync(string method, object? paramsObj, CancellationToken ct)
    {
        if (_disposed) return;

        var envelope = new Dictionary<string, object?>
        {
            ["jsonrpc"] = "2.0",
            ["method"] = method,
        };
        if (paramsObj is not null) envelope["params"] = paramsObj;

        var body = JsonSerializer.Serialize(envelope, JsonOpts);
        _log?.Invoke(new McpExecutionLogEntry(
            DateTimeOffset.UtcNow, ServerId, ServerName, "→", method + " (notification)", body));

        using var req = new HttpRequestMessage(HttpMethod.Post, Url)
        {
            Content = new StringContent(body, Encoding.UTF8, "application/json"),
        };
        req.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        req.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));
        if (_sessionId is not null)
        {
            req.Headers.TryAddWithoutValidation("Mcp-Session-Id", _sessionId);
        }
        try
        {
            using var res = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct)
                .ConfigureAwait(false);
            // Spec: notifications typically return 202 Accepted with empty body.
            // Discard whatever we get; we don't care for notifications.
        }
        catch
        {
            // Notifications are fire-and-forget.
        }
    }

    /// <summary>
    /// Drain the SSE stream until we see a <c>message</c> event whose JSON-RPC
    /// payload id matches <paramref name="requestId"/>. Intermediate
    /// notifications from the server (progress / log / request) are logged but
    /// otherwise ignored — we only care about the final response.
    /// </summary>
    private async Task<JsonDocument> ReadSseFinalResponseAsync(
        HttpResponseMessage res, long requestId, CancellationToken ct)
    {
        using var stream = await res.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var reader = new StreamReader(stream, Encoding.UTF8);

        var dataBuf = new StringBuilder();
        while (!ct.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync(ct).ConfigureAwait(false);
            if (line is null) break; // end of stream

            if (line.Length == 0)
            {
                // Dispatch accumulated event.
                if (dataBuf.Length == 0) continue;
                var payload = dataBuf.ToString();
                dataBuf.Clear();
                _log?.Invoke(new McpExecutionLogEntry(
                    DateTimeOffset.UtcNow, ServerId, ServerName, "←", "sse", payload));

                try
                {
                    var doc = JsonDocument.Parse(payload);
                    var root = doc.RootElement;
                    if (root.TryGetProperty("id", out var idEl) && MatchesRequestId(idEl, requestId))
                    {
                        return doc;
                    }
                    doc.Dispose();
                }
                catch (JsonException)
                {
                    // Malformed SSE payload — skip, keep draining.
                }
                continue;
            }
            if (line.StartsWith("data:", StringComparison.Ordinal))
            {
                var body = line.Length > 5 && line[5] == ' ' ? line[6..] : line[5..];
                if (dataBuf.Length > 0) dataBuf.Append('\n');
                dataBuf.Append(body);
            }
            // Ignore `event:`, `id:`, `retry:` for now — only `data:` is
            // required to reconstruct the JSON-RPC message.
        }

        throw new InvalidOperationException("MCP SSE stream ended without a matching response.");
    }

    private static bool MatchesRequestId(JsonElement idEl, long expected)
    {
        return idEl.ValueKind switch
        {
            JsonValueKind.Number => idEl.TryGetInt64(out var v) && v == expected,
            JsonValueKind.String => long.TryParse(idEl.GetString(), out var v) && v == expected,
            _ => false,
        };
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _http.Dispose();
    }
}
