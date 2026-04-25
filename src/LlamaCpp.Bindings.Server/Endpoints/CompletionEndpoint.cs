using System.Text;
using System.Text.Json;
using LlamaCpp.Bindings.Server.Configuration;
using LlamaCpp.Bindings.Server.Models;
using LlamaCpp.Bindings.Server.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Options;

namespace LlamaCpp.Bindings.Server.Endpoints;

/// <summary>
/// <c>POST /completion</c> — llama-server's native raw-text endpoint.
/// No chat templating. The caller is responsible for any prompt framing.
/// </summary>
public static class CompletionEndpoint
{
    public static async Task Handle(
        HttpContext http,
        CompletionRequest req,
        ModelHost host,
        SessionPool pool,
        IOptions<ServerOptions> options,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrEmpty(req.Prompt))
        {
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new { error = "prompt must not be empty" }, cancellationToken);
            return;
        }

        var opts = options.Value;
        int maxTokens = Math.Clamp(req.MaxTokens ?? opts.MaxOutputTokens, 1, opts.MaxOutputTokens);

        using var lease = await pool.LeaseAsync(cancellationToken);
        using var sampler = BuildSampler(req);
        var generator = new LlamaGenerator(lease.Session, sampler);

        if (req.Stream)
        {
            http.Response.ContentType = "text/event-stream";
            http.Response.Headers.CacheControl = "no-cache";
            var feature = http.Features.Get<Microsoft.AspNetCore.Http.Features.IHttpResponseBodyFeature>();
            feature?.DisableBuffering();

            await foreach (var piece in generator.GenerateAsync(
                req.Prompt, maxTokens: maxTokens, addSpecial: true, parseSpecial: false,
                cancellationToken: cancellationToken))
            {
                if (string.IsNullOrEmpty(piece)) continue;
                var json = JsonSerializer.Serialize(new { content = piece, stop = false });
                await http.Response.WriteAsync("data: " + json + "\n\n", cancellationToken);
                await http.Response.Body.FlushAsync(cancellationToken);
            }

            var tail = JsonSerializer.Serialize(new
            {
                content = "",
                stop = true,
                stop_reason = MapStopReason(generator.LastStopReason),
                model = host.ModelId,
            });
            await http.Response.WriteAsync("data: " + tail + "\n\n", cancellationToken);
        }
        else
        {
            var buf = new StringBuilder();
            await foreach (var piece in generator.GenerateAsync(
                req.Prompt, maxTokens: maxTokens, addSpecial: true, parseSpecial: false,
                cancellationToken: cancellationToken))
            {
                buf.Append(piece);
            }
            var body = new CompletionResponse
            {
                Content = buf.ToString(),
                StopReason = MapStopReason(generator.LastStopReason),
                Model = host.ModelId,
            };
            http.Response.ContentType = "application/json";
            await http.Response.WriteAsJsonAsync(body, cancellationToken: cancellationToken);
        }
    }

    private static LlamaSampler BuildSampler(CompletionRequest req)
    {
        var b = new LlamaSamplerBuilder();
        if (req.TopK is int k && k > 0) b = b.WithTopK(k);
        if (req.TopP is float p && p is > 0f and < 1f) b = b.WithTopP(p);
        float temp = req.Temperature ?? 0f;
        if (temp <= 0f) return b.WithGreedy().Build();
        return b.WithTemperature(temp).WithDistribution(req.Seed ?? 0u).Build();
    }

    private static string MapStopReason(LlamaStopReason r) => r switch
    {
        LlamaStopReason.EndOfGeneration   => "stop",
        LlamaStopReason.GrammarSatisfied  => "stop",
        LlamaStopReason.MaxTokens         => "length",
        LlamaStopReason.Cancelled         => "cancelled",
        _                                 => "stop",
    };
}
