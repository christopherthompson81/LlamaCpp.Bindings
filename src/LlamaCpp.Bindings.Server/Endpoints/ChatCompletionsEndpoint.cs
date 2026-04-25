using System.Text;
using System.Text.Json;
using LlamaCpp.Bindings.Server.Configuration;
using LlamaCpp.Bindings.Server.Models;
using LlamaCpp.Bindings.Server.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Options;

namespace LlamaCpp.Bindings.Server.Endpoints;

/// <summary>
/// <c>POST /v1/chat/completions</c> — OpenAI-compatible chat endpoint.
/// Applies the model's GGUF-stored chat template to the supplied messages,
/// borrows a session from <see cref="SessionPool"/>, and streams or batches
/// the generator's output. Uses the pool's longest-common-prefix matching
/// so follow-up turns in a stateless OpenAI flow still reuse the KV from
/// previous turns — the server detects that
/// <c>messages[0..N-1]</c> is identical to a prior request and only
/// decodes the new tail.
/// </summary>
public static class ChatCompletionsEndpoint
{
    public static async Task Handle(
        HttpContext http,
        ChatCompletionsRequest req,
        ModelHost host,
        SessionPool pool,
        IOptions<ServerOptions> options,
        ILoggerFactory loggerFactory,
        ServerMetrics metrics,
        CancellationToken cancellationToken)
    {
        var log = loggerFactory.CreateLogger("ChatCompletions");
        var opts = options.Value;

        if (req.Messages.Count == 0)
        {
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new { error = "messages must not be empty" }, cancellationToken);
            return;
        }

        var template = host.Model.GetChatTemplate();
        if (string.IsNullOrEmpty(template))
        {
            http.Response.StatusCode = StatusCodes.Status500InternalServerError;
            await http.Response.WriteAsJsonAsync(
                new { error = "loaded model has no chat template in its GGUF metadata" }, cancellationToken);
            return;
        }

        string prompt;
        try
        {
            var messages = req.Messages.Select(m => new ChatMessage(m.Role, m.Content)).ToArray();
            prompt = LlamaChatTemplate.Apply(template, messages, addAssistantPrefix: true);
        }
        catch (Exception ex)
        {
            log.LogWarning(ex, "chat template render failed");
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new { error = "chat template render failed: " + ex.Message }, cancellationToken);
            return;
        }

        int maxTokens = Math.Clamp(req.MaxTokens ?? opts.MaxOutputTokens, 1, opts.MaxOutputTokens);

        // Parse stop strings up front — malformed inputs reject before we
        // take a pool slot.
        string[]? stops;
        try
        {
            stops = StopNormalizer.Parse(req.Stop);
        }
        catch (ArgumentException ex)
        {
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new { error = ex.Message }, cancellationToken);
            return;
        }

        // Resolve grammar (response_format / grammar / json_schema) before
        // anything expensive runs — malformed schemas return 400 without
        // a pool lease.
        LlamaGrammar? grammar;
        try
        {
            grammar = GrammarFactory.Resolve(req.Grammar, req.JsonSchemaShort, req.ResponseFormat);
        }
        catch (ArgumentException ex)
        {
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new { error = ex.Message }, cancellationToken);
            return;
        }

        // Tokenize up front so the pool can do prefix matching before we
        // pick a slot.
        var promptTokens = host.Model.Vocab.Tokenize(prompt, addSpecial: false, parseSpecial: true);

        // Build the sampler before taking a slot so malformed logit_bias /
        // mirostat / invalid grammar / etc. fail with 400 without holding
        // the pool.
        LlamaSampler sampler;
        try
        {
            var samplerParams = req.ToSamplerParams() with { Grammar = grammar };
            sampler = SamplerFactory.Build(host.Model, samplerParams);
        }
        catch (ArgumentException ex)
        {
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new { error = ex.Message }, cancellationToken);
            return;
        }
        catch (LlamaException ex)
        {
            // GBNF parse errors surface here (the binding's grammar init
            // returns NULL on invalid source, which LlamaSamplerBuilder
            // wraps in LlamaException).
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new { error = "Invalid grammar: " + ex.Message }, cancellationToken);
            return;
        }

        var matcher = new StopMatcher(stops);

        // Lease a slot (may queue). The lease carries FirstNewIndex = how
        // many of these tokens are already in KV from the last request.
        using var lease = await pool.LeaseAsync(promptTokens, cancellationToken);
        using var _ = sampler; // dispose sampler with the request, not earlier.
        var generator = new LlamaGenerator(lease.Session, sampler);

        // Prototype observability header — tells callers how many prompt
        // tokens this request skipped thanks to the cache. llama-server
        // reports the same thing in its streaming metadata.
        http.Response.Headers["X-Cached-Tokens"] = lease.CachedTokens.ToString();

        // Timings + counters. `promptTokensToDecode` = total prompt minus
        // cache hit, so cache-warm follow-ups report realistically low
        // prompt_ms numbers.
        int promptTokensToDecode = lease.PromptTokens.Length - lease.FirstNewIndex;
        var timer = new RequestTimer(promptTokensToDecode, lease.CachedTokens);
        metrics.AddPromptTokensIngested(promptTokensToDecode);
        metrics.AddCachedTokensReused(lease.CachedTokens);

        string completionId = "chatcmpl-" + Guid.NewGuid().ToString("N");
        long createdUnix = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        if (req.Stream)
        {
            await StreamSse(http, generator, lease, matcher, maxTokens, host, completionId, createdUnix, timer, metrics, cancellationToken);
        }
        else
        {
            await WriteSingleJson(http, generator, lease, matcher, maxTokens, host, completionId, createdUnix, timer, metrics, cancellationToken);
        }
    }

    // ----- Non-streaming: collect all pieces into one response body. -----

    private static async Task WriteSingleJson(
        HttpContext http, LlamaGenerator gen, SessionLease lease,
        StopMatcher matcher, int maxTokens,
        ModelHost host, string id, long created,
        RequestTimer timer, ServerMetrics metrics, CancellationToken ct)
    {
        var buf = new StringBuilder();
        bool stoppedOnMatch = false;
        await foreach (var piece in gen.GenerateAsync(
            lease.PromptTokens,
            maxTokens: maxTokens,
            firstNewIndex: lease.FirstNewIndex,
            onTokenDecoded: t =>
            {
                lease.OnTokenDecoded(t);
                timer.IncrementPredicted();
            },
            cancellationToken: ct))
        {
            timer.MarkFirstToken();
            var (emit, stopped) = matcher.Offer(piece);
            if (emit.Length > 0) buf.Append(emit);
            if (stopped) { stoppedOnMatch = true; break; }
        }
        if (!stoppedOnMatch) buf.Append(matcher.Flush());
        timer.Finish();
        metrics.AddTokensGenerated(timer.PredictedTokens);

        var response = new ChatCompletionsResponse
        {
            Id = id,
            Created = created,
            Model = host.ModelId,
            Choices = new List<ChatChoice>
            {
                new() {
                    Index = 0,
                    Message = new ChatMessageDto { Role = "assistant", Content = buf.ToString() },
                    FinishReason = stoppedOnMatch ? "stop" : MapFinishReason(gen.LastStopReason),
                },
            },
            Timings = timer.Snapshot(),
        };

        http.Response.ContentType = "application/json";
        await http.Response.WriteAsJsonAsync(response, cancellationToken: ct);
    }

    // ----- Streaming: SSE with OpenAI-style chunks. -----

    private static async Task StreamSse(
        HttpContext http, LlamaGenerator gen, SessionLease lease,
        StopMatcher matcher, int maxTokens,
        ModelHost host, string id, long created,
        RequestTimer timer, ServerMetrics metrics, CancellationToken ct)
    {
        http.Response.ContentType = "text/event-stream";
        http.Response.Headers.CacheControl = "no-cache";
        // Disable response buffering so tokens land on the wire as they're emitted.
        var feature = http.Features.Get<Microsoft.AspNetCore.Http.Features.IHttpResponseBodyFeature>();
        feature?.DisableBuffering();

        // First chunk sets role=assistant (OpenAI convention).
        await WriteChunk(http, new ChatCompletionsChunk
        {
            Id = id, Created = created, Model = host.ModelId,
            Choices = new()
            {
                new() { Index = 0, Delta = new() { Role = "assistant" }, FinishReason = null },
            },
        }, ct);

        bool stoppedOnMatch = false;
        await foreach (var piece in gen.GenerateAsync(
            lease.PromptTokens,
            maxTokens: maxTokens,
            firstNewIndex: lease.FirstNewIndex,
            onTokenDecoded: t =>
            {
                lease.OnTokenDecoded(t);
                timer.IncrementPredicted();
            },
            cancellationToken: ct))
        {
            timer.MarkFirstToken();
            var (emit, stopped) = matcher.Offer(piece);
            if (!string.IsNullOrEmpty(emit))
            {
                await WriteChunk(http, new ChatCompletionsChunk
                {
                    Id = id, Created = created, Model = host.ModelId,
                    Choices = new()
                    {
                        new() { Index = 0, Delta = new() { Content = emit }, FinishReason = null },
                    },
                }, ct);
            }
            if (stopped) { stoppedOnMatch = true; break; }
        }

        if (!stoppedOnMatch)
        {
            var tail = matcher.Flush();
            if (!string.IsNullOrEmpty(tail))
            {
                await WriteChunk(http, new ChatCompletionsChunk
                {
                    Id = id, Created = created, Model = host.ModelId,
                    Choices = new()
                    {
                        new() { Index = 0, Delta = new() { Content = tail }, FinishReason = null },
                    },
                }, ct);
            }
        }

        timer.Finish();
        metrics.AddTokensGenerated(timer.PredictedTokens);

        // Final chunk: empty delta + finish_reason + timings sidecar.
        await WriteChunk(http, new ChatCompletionsChunk
        {
            Id = id, Created = created, Model = host.ModelId,
            Choices = new()
            {
                new() { Index = 0, Delta = new(),
                    FinishReason = stoppedOnMatch ? "stop" : MapFinishReason(gen.LastStopReason) },
            },
            Timings = timer.Snapshot(),
        }, ct);

        // OpenAI terminates the stream with a literal "[DONE]" sentinel.
        await http.Response.WriteAsync("data: [DONE]\n\n", ct);
    }

    private static async Task WriteChunk<T>(HttpContext http, T payload, CancellationToken ct)
    {
        var json = JsonSerializer.Serialize(payload);
        await http.Response.WriteAsync("data: ", ct);
        await http.Response.WriteAsync(json, ct);
        await http.Response.WriteAsync("\n\n", ct);
        await http.Response.Body.FlushAsync(ct);
    }

    private static string MapFinishReason(LlamaStopReason reason) => reason switch
    {
        LlamaStopReason.EndOfGeneration   => "stop",
        LlamaStopReason.GrammarSatisfied  => "stop",
        LlamaStopReason.MaxTokens         => "length",
        LlamaStopReason.Cancelled         => "cancelled",
        _                                 => "stop",
    };
}
