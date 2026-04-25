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
/// Prefix-caches against prior requests on the same server through the
/// shared <see cref="SessionPool"/>.
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

        // Tokenize up front — required for the pool's prefix-matching pass.
        // addSpecial=true mirrors llama-server's --special-tokens behaviour
        // for raw /completion calls.
        var promptTokens = host.Model.Vocab.Tokenize(req.Prompt, addSpecial: true, parseSpecial: false);

        LlamaSampler sampler;
        try
        {
            sampler = SamplerFactory.Build(
                host.Model.Vocab, req.Temperature, req.TopK, req.TopP, req.Seed, req.LogitBias);
        }
        catch (ArgumentException ex)
        {
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new { error = ex.Message }, cancellationToken);
            return;
        }

        using var lease = await pool.LeaseAsync(promptTokens, cancellationToken);
        using var _ = sampler;
        var generator = new LlamaGenerator(lease.Session, sampler);

        http.Response.Headers["X-Cached-Tokens"] = lease.CachedTokens.ToString();

        if (req.Stream)
        {
            http.Response.ContentType = "text/event-stream";
            http.Response.Headers.CacheControl = "no-cache";
            var feature = http.Features.Get<Microsoft.AspNetCore.Http.Features.IHttpResponseBodyFeature>();
            feature?.DisableBuffering();

            await foreach (var piece in generator.GenerateAsync(
                lease.PromptTokens,
                maxTokens: maxTokens,
                firstNewIndex: lease.FirstNewIndex,
                onTokenDecoded: lease.OnTokenDecoded,
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
                tokens_cached = lease.CachedTokens,
            });
            await http.Response.WriteAsync("data: " + tail + "\n\n", cancellationToken);
        }
        else
        {
            var buf = new StringBuilder();
            await foreach (var piece in generator.GenerateAsync(
                lease.PromptTokens,
                maxTokens: maxTokens,
                firstNewIndex: lease.FirstNewIndex,
                onTokenDecoded: lease.OnTokenDecoded,
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

    private static string MapStopReason(LlamaStopReason r) => r switch
    {
        LlamaStopReason.EndOfGeneration   => "stop",
        LlamaStopReason.GrammarSatisfied  => "stop",
        LlamaStopReason.MaxTokens         => "length",
        LlamaStopReason.Cancelled         => "cancelled",
        _                                 => "stop",
    };
}
