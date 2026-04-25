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
/// the generator's output.
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

        // Lease a session (may queue). Wrap generation + dispose in a try so
        // the slot always returns to the pool even on client disconnect.
        using var lease = await pool.LeaseAsync(cancellationToken);
        using var sampler = BuildSampler(req);
        var generator = new LlamaGenerator(lease.Session, sampler);

        string completionId = "chatcmpl-" + Guid.NewGuid().ToString("N");
        long createdUnix = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        if (req.Stream)
        {
            await StreamSse(http, generator, prompt, maxTokens, host, completionId, createdUnix, cancellationToken);
        }
        else
        {
            await WriteSingleJson(http, generator, prompt, maxTokens, host, completionId, createdUnix, cancellationToken);
        }
    }

    private static LlamaSampler BuildSampler(ChatCompletionsRequest req)
    {
        var b = new LlamaSamplerBuilder();
        if (req.TopK is int k && k > 0) b = b.WithTopK(k);
        if (req.TopP is float p && p is > 0f and < 1f) b = b.WithTopP(p);
        // A temperature of 0 collapses to greedy — short-circuit so the chain
        // doesn't include a degenerate temp=0 stage before a distribution
        // sampler (which would be slower for the same result).
        float temp = req.Temperature ?? 0f;
        if (temp <= 0f)
        {
            return b.WithGreedy().Build();
        }
        return b.WithTemperature(temp).WithDistribution(req.Seed ?? 0u).Build();
    }

    // ----- Non-streaming: collect all pieces into one response body. -----

    private static async Task WriteSingleJson(
        HttpContext http, LlamaGenerator gen, string prompt, int maxTokens,
        ModelHost host, string id, long created, CancellationToken ct)
    {
        var buf = new StringBuilder();
        await foreach (var piece in gen.GenerateAsync(
            prompt, maxTokens: maxTokens, addSpecial: false, parseSpecial: true,
            cancellationToken: ct))
        {
            buf.Append(piece);
        }

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
                    FinishReason = MapFinishReason(gen.LastStopReason),
                },
            },
        };

        http.Response.ContentType = "application/json";
        await http.Response.WriteAsJsonAsync(response, cancellationToken: ct);
    }

    // ----- Streaming: SSE with OpenAI-style chunks. -----

    private static async Task StreamSse(
        HttpContext http, LlamaGenerator gen, string prompt, int maxTokens,
        ModelHost host, string id, long created, CancellationToken ct)
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

        await foreach (var piece in gen.GenerateAsync(
            prompt, maxTokens: maxTokens, addSpecial: false, parseSpecial: true,
            cancellationToken: ct))
        {
            if (string.IsNullOrEmpty(piece)) continue;
            await WriteChunk(http, new ChatCompletionsChunk
            {
                Id = id, Created = created, Model = host.ModelId,
                Choices = new()
                {
                    new() { Index = 0, Delta = new() { Content = piece }, FinishReason = null },
                },
            }, ct);
        }

        // Final chunk: empty delta + finish_reason.
        await WriteChunk(http, new ChatCompletionsChunk
        {
            Id = id, Created = created, Model = host.ModelId,
            Choices = new()
            {
                new() { Index = 0, Delta = new(), FinishReason = MapFinishReason(gen.LastStopReason) },
            },
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
