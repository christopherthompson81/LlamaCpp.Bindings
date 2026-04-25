using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace LlamaCpp.Bindings.LlamaChat.Services.Remote;

/// <summary>
/// Thin client for OpenAI-compatible servers. Handles bearer auth, JSON
/// request bodies, and SSE streaming for chat completions. Owned by
/// <see cref="RemoteChatSession"/>; one instance per loaded session.
/// </summary>
public sealed class OpenAiChatClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly Uri _baseUri;
    private readonly string? _apiKey;
    private bool _disposed;

    public static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = null, // wire names already on attributes
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
    };

    public OpenAiChatClient(string baseUrl, string? apiKey)
        : this(baseUrl, apiKey, handler: null) { }

    /// <summary>
    /// Test seam: pass a custom <see cref="HttpMessageHandler"/> (e.g.
    /// <c>WebApplicationFactory.Server.CreateHandler()</c>) to drive the
    /// client against an in-process test server. <paramref name="handler"/>
    /// is not disposed by this client.
    /// </summary>
    public OpenAiChatClient(string baseUrl, string? apiKey, HttpMessageHandler? handler)
    {
        if (string.IsNullOrWhiteSpace(baseUrl))
            throw new ArgumentException("BaseUrl is empty.", nameof(baseUrl));

        // Normalize: strip trailing slashes so route-joining is unambiguous.
        var trimmed = baseUrl.TrimEnd('/');
        if (!Uri.TryCreate(trimmed, UriKind.Absolute, out var uri))
            throw new ArgumentException($"BaseUrl is not a valid absolute URI: {baseUrl}", nameof(baseUrl));
        _baseUri = uri;
        _apiKey = string.IsNullOrWhiteSpace(apiKey) ? null : apiKey;

        // Streaming generation can run for minutes on a slow CPU server.
        // We don't want HttpClient timing out the read mid-stream.
        _http = handler is null
            ? new HttpClient { Timeout = Timeout.InfiniteTimeSpan }
            : new HttpClient(handler, disposeHandler: false) { Timeout = Timeout.InfiniteTimeSpan };
    }

    public Uri BaseUri => _baseUri;

    private HttpRequestMessage NewRequest(HttpMethod method, string relativePath)
    {
        // Build the URL by string-concat with explicit slash handling:
        // Uri.ToString() always re-adds a trailing slash when the path is empty,
        // so naive concatenation produces "http://host//v1/models" — which
        // ASP.NET Core's router 404s on.
        var baseStr = _baseUri.AbsoluteUri.TrimEnd('/');
        var relStr = relativePath.StartsWith('/') ? relativePath : "/" + relativePath;
        var uri = new Uri(baseStr + relStr);
        var req = new HttpRequestMessage(method, uri);
        if (_apiKey is not null)
        {
            req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
        }
        return req;
    }

    /// <summary>
    /// Fetch the server's available model ids via <c>GET /v1/models</c>.
    /// Throws on non-2xx; returns an empty list on a successful but malformed
    /// response.
    /// </summary>
    public async Task<IReadOnlyList<string>> ListModelsAsync(CancellationToken ct = default)
    {
        using var req = NewRequest(HttpMethod.Get, "/v1/models");
        using var res = await _http.SendAsync(req, HttpCompletionOption.ResponseContentRead, ct).ConfigureAwait(false);
        if (!res.IsSuccessStatusCode)
        {
            var body = await res.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"GET /v1/models failed: HTTP {(int)res.StatusCode} {res.ReasonPhrase}. {body}");
        }
        var parsed = await res.Content.ReadFromJsonAsync<ModelsListResponse>(JsonOpts, ct).ConfigureAwait(false);
        if (parsed?.Data is null) return Array.Empty<string>();
        var ids = new List<string>(parsed.Data.Count);
        foreach (var m in parsed.Data)
            if (!string.IsNullOrEmpty(m.Id)) ids.Add(m.Id);
        return ids;
    }

    /// <summary>
    /// Non-streaming chat completion. Used by the title-generation path.
    /// </summary>
    public async Task<ChatCompletionsResponse> CreateChatCompletionAsync(
        ChatCompletionsRequest body, CancellationToken ct = default)
    {
        body.Stream = false;
        using var req = NewRequest(HttpMethod.Post, "/v1/chat/completions");
        req.Content = JsonContent.Create(body, options: JsonOpts);
        using var res = await _http.SendAsync(req, HttpCompletionOption.ResponseContentRead, ct).ConfigureAwait(false);
        if (!res.IsSuccessStatusCode)
        {
            var err = await res.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"POST /v1/chat/completions failed: HTTP {(int)res.StatusCode} {res.ReasonPhrase}. {err}");
        }
        var parsed = await res.Content.ReadFromJsonAsync<ChatCompletionsResponse>(JsonOpts, ct).ConfigureAwait(false);
        if (parsed is null) throw new InvalidOperationException("server returned an empty response body.");
        return parsed;
    }

    /// <summary>
    /// Streaming chat completion. Forces <c>stream: true</c>, drains the SSE
    /// response line-by-line, and yields one <see cref="ChatCompletionsChunk"/>
    /// per <c>data:</c> event until the <c>[DONE]</c> sentinel. The HTTP
    /// connection closes when the iterator completes or is cancelled.
    /// </summary>
    public async IAsyncEnumerable<ChatCompletionsChunk> StreamChatCompletionAsync(
        ChatCompletionsRequest body,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        body.Stream = true;
        using var req = NewRequest(HttpMethod.Post, "/v1/chat/completions");
        req.Content = JsonContent.Create(body, options: JsonOpts);
        req.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));

        using var res = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
        if (!res.IsSuccessStatusCode)
        {
            var err = await res.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"POST /v1/chat/completions (stream) failed: HTTP {(int)res.StatusCode} {res.ReasonPhrase}. {err}");
        }

        await using var stream = await res.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var reader = new StreamReader(stream, Encoding.UTF8);

        var dataBuf = new StringBuilder();
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            var line = await reader.ReadLineAsync(ct).ConfigureAwait(false);
            if (line is null) yield break;

            if (line.Length == 0)
            {
                if (dataBuf.Length == 0) continue;
                var payload = dataBuf.ToString();
                dataBuf.Clear();

                if (payload == "[DONE]") yield break;

                ChatCompletionsChunk? chunk;
                try
                {
                    chunk = JsonSerializer.Deserialize<ChatCompletionsChunk>(payload, JsonOpts);
                }
                catch (JsonException)
                {
                    // Tolerate one stray malformed chunk; keep draining.
                    continue;
                }
                if (chunk is not null) yield return chunk;
                continue;
            }

            if (line.StartsWith("data:", StringComparison.Ordinal))
            {
                var bodyText = line.Length > 5 && line[5] == ' ' ? line[6..] : line[5..];
                if (dataBuf.Length > 0) dataBuf.Append('\n');
                dataBuf.Append(bodyText);
            }
            // event:, id:, retry:, comments — ignore.
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _http.Dispose();
    }
}
