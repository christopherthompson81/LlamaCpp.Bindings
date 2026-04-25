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
        ServerMetrics metrics,
        MmprojHost mmproj,
        ILoggerFactory loggers,
        CancellationToken cancellationToken)
    {
        var opts = options.Value;
        var log = loggers.CreateLogger("Completion");

        var clientAbortedToken = cancellationToken;
        var (linkedCts, timeoutCts) = RequestGuard.CreateLinkedToken(cancellationToken, opts);
        using var _linkedCts = linkedCts;
        using var _timeoutCts = timeoutCts;
        cancellationToken = linkedCts.Token;

        try
        {
            await HandleCore(http, req, host, pool, opts, metrics, mmproj, log, cancellationToken);
        }
        catch (OperationCanceledException) when (
            timeoutCts is not null
            && timeoutCts.IsCancellationRequested
            && !clientAbortedToken.IsCancellationRequested)
        {
            if (!http.Response.HasStarted)
            {
                http.Response.StatusCode = StatusCodes.Status504GatewayTimeout;
                await http.Response.WriteAsJsonAsync(new
                {
                    error = new
                    {
                        message = $"Request exceeded the {opts.RequestTimeoutSeconds}s server-side timeout.",
                        type = "timeout",
                        code = "request_timeout",
                    },
                }, CancellationToken.None);
            }
        }
    }

    private static async Task HandleCore(
        HttpContext http,
        CompletionRequest req,
        ModelHost host,
        SessionPool pool,
        ServerOptions opts,
        ServerMetrics metrics,
        MmprojHost mmproj,
        ILogger log,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrEmpty(req.Prompt))
        {
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new { error = "prompt must not be empty" }, cancellationToken);
            return;
        }

        int maxTokens = Math.Clamp(req.MaxTokens ?? opts.MaxOutputTokens, 1, opts.MaxOutputTokens);

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

        // Tokenize up front — required for the pool's prefix-matching pass.
        // addSpecial=true mirrors llama-server's --special-tokens behaviour
        // for raw /completion calls.
        var promptTokens = host.Model.Vocab.Tokenize(req.Prompt, addSpecial: true, parseSpecial: false);

        int maxPrompt = RequestGuard.EffectiveMaxPromptTokens(opts, host.Context.ContextSize);
        if (promptTokens.Length > maxPrompt)
        {
            http.Response.StatusCode = StatusCodes.Status413PayloadTooLarge;
            await http.Response.WriteAsJsonAsync(new
            {
                error = new
                {
                    message = $"Prompt is {promptTokens.Length} tokens; the server caps prompt length " +
                              $"at {maxPrompt} (context={host.Context.ContextSize}, " +
                              $"max_output_tokens={opts.MaxOutputTokens}).",
                    type = "request_too_large",
                    code = "prompt_too_long",
                },
            }, cancellationToken);
            return;
        }

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
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new { error = "Invalid grammar: " + ex.Message }, cancellationToken);
            return;
        }

        var matcher = new StopMatcher(stops);

        // Multimodal branch — opt-in via multimodal_data. Each entry is a
        // data: URL; the caller is responsible for putting media markers
        // (typically "<__media__>") in the prompt where each image goes.
        // Mirrors ChatCompletionsEndpoint's image branch: lease a fresh
        // slot, clear KV, run EvalPromptAsync to prefill image+text
        // chunks, then stream from the context's current state.
        if (req.MultimodalData is { Count: > 0 } imgUrls)
        {
            if (!mmproj.IsAvailable)
            {
                http.Response.StatusCode = StatusCodes.Status400BadRequest;
                await http.Response.WriteAsJsonAsync(new
                {
                    error = "This server is not configured with a multimodal projector. " +
                            "Set LlamaServer:MmprojPath (or MmprojAuto) to accept multimodal_data.",
                }, cancellationToken);
                return;
            }

            var images = new List<byte[]>(imgUrls.Count);
            try
            {
                foreach (var url in imgUrls)
                {
                    images.Add(ChatContentExtractor.DecodeImageUrl(url));
                }
            }
            catch (ArgumentException ex)
            {
                http.Response.StatusCode = StatusCodes.Status400BadRequest;
                await http.Response.WriteAsJsonAsync(new { error = ex.Message }, cancellationToken);
                return;
            }

            await HandleMultimodalAsync(
                http, req, host, pool, mmproj, log, sampler, matcher, maxTokens, images, cancellationToken);
            return;
        }

        using var lease = await pool.LeaseAsync(
            promptTokens, cancellationToken, useCache: req.CachePrompt ?? true);
        using var _ = sampler;
        var generator = new LlamaGenerator(lease.Session, sampler);

        http.Response.Headers["X-Cached-Tokens"] = lease.CachedTokens.ToString();

        int promptTokensToDecode = lease.PromptTokens.Length - lease.FirstNewIndex;
        var timer = new RequestTimer(promptTokensToDecode, lease.CachedTokens);
        metrics.AddPromptTokensIngested(promptTokensToDecode);
        metrics.AddCachedTokensReused(lease.CachedTokens);

        Action<int> onDecoded = t =>
        {
            lease.OnTokenDecoded(t);
            timer.IncrementPredicted();
        };

        if (req.Stream)
        {
            http.Response.ContentType = "text/event-stream";
            http.Response.Headers.CacheControl = "no-cache";
            var feature = http.Features.Get<Microsoft.AspNetCore.Http.Features.IHttpResponseBodyFeature>();
            feature?.DisableBuffering();

            bool stoppedOnMatch = false;
            await foreach (var piece in generator.GenerateAsync(
                lease.PromptTokens,
                maxTokens: maxTokens,
                firstNewIndex: lease.FirstNewIndex,
                onTokenDecoded: onDecoded,
                cancellationToken: cancellationToken))
            {
                timer.MarkFirstToken();
                var (emit, stopped) = matcher.Offer(piece);
                if (!string.IsNullOrEmpty(emit))
                {
                    var json = JsonSerializer.Serialize(new { content = emit, stop = false });
                    await http.Response.WriteAsync("data: " + json + "\n\n", cancellationToken);
                    await http.Response.Body.FlushAsync(cancellationToken);
                }
                if (stopped) { stoppedOnMatch = true; break; }
            }
            if (!stoppedOnMatch)
            {
                var trailing = matcher.Flush();
                if (!string.IsNullOrEmpty(trailing))
                {
                    var json = JsonSerializer.Serialize(new { content = trailing, stop = false });
                    await http.Response.WriteAsync("data: " + json + "\n\n", cancellationToken);
                    await http.Response.Body.FlushAsync(cancellationToken);
                }
            }
            timer.Finish();
            metrics.AddTokensGenerated(timer.PredictedTokens);

            var tail = JsonSerializer.Serialize(new
            {
                content = "",
                stop = true,
                stop_reason = stoppedOnMatch ? "stop" : MapStopReason(generator.LastStopReason),
                model = host.ModelId,
                tokens_cached = lease.CachedTokens,
                timings = timer.Snapshot(),
            });
            await http.Response.WriteAsync("data: " + tail + "\n\n", cancellationToken);
        }
        else
        {
            var buf = new StringBuilder();
            bool stoppedOnMatch = false;
            await foreach (var piece in generator.GenerateAsync(
                lease.PromptTokens,
                maxTokens: maxTokens,
                firstNewIndex: lease.FirstNewIndex,
                onTokenDecoded: onDecoded,
                cancellationToken: cancellationToken))
            {
                timer.MarkFirstToken();
                var (emit, stopped) = matcher.Offer(piece);
                if (emit.Length > 0) buf.Append(emit);
                if (stopped) { stoppedOnMatch = true; break; }
            }
            if (!stoppedOnMatch) buf.Append(matcher.Flush());
            timer.Finish();
            metrics.AddTokensGenerated(timer.PredictedTokens);

            var body = new CompletionResponse
            {
                Content = buf.ToString(),
                StopReason = stoppedOnMatch ? "stop" : MapStopReason(generator.LastStopReason),
                Model = host.ModelId,
                Timings = timer.Snapshot(),
            };
            http.Response.ContentType = "application/json";
            await http.Response.WriteAsJsonAsync(body, cancellationToken: cancellationToken);
        }
    }

    private static async Task HandleMultimodalAsync(
        HttpContext http, CompletionRequest req, ModelHost host, SessionPool pool,
        MmprojHost mmproj, ILogger log, LlamaSampler sampler, StopMatcher matcher, int maxTokens,
        List<byte[]> images, CancellationToken cancellationToken)
    {
        // Lease with no-prompt + invalidate so the slot starts cold —
        // image tokens going through EvalPromptAsync don't fit the
        // pool's text-prefix-match model.
        using var lease = await pool.LeaseAsync(Array.Empty<int>(), cancellationToken);
        using var _sampler = sampler;
        lease.InvalidateCache();
        lease.Session.ClearHistory();

        var generator = new LlamaGenerator(lease.Session, sampler);
        http.Response.Headers["X-Cached-Tokens"] = "0";
        var timer = new RequestTimer(promptTokensToDecode: 0, cachedTokens: 0);

        var bitmaps = new List<MtmdBitmap>();
        try
        {
            foreach (var bytes in images)
            {
                bitmaps.Add(MtmdBitmap.FromBytes(mmproj.Context!, bytes));
            }
            await mmproj.Context!.EvalPromptAsync(
                lease.Session.Context, req.Prompt!, bitmaps,
                nPast: 0,
                seqId: lease.Session.SequenceId,
                nBatch: lease.Session.Context.LogicalBatchSize,
                logitsLast: true,
                addSpecial: false,
                parseSpecial: true,
                cancellationToken);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            foreach (var b in bitmaps) b.Dispose();
            log.LogWarning(ex, "/completion mmproj prefill failed");
            http.Response.StatusCode = StatusCodes.Status500InternalServerError;
            await http.Response.WriteAsJsonAsync(
                new { error = "multimodal prefill failed: " + ex.Message },
                cancellationToken);
            return;
        }

        try
        {
            if (req.Stream)
            {
                http.Response.ContentType = "text/event-stream";
                http.Response.Headers.CacheControl = "no-cache";
                var feature = http.Features.Get<Microsoft.AspNetCore.Http.Features.IHttpResponseBodyFeature>();
                feature?.DisableBuffering();

                bool stoppedOnMatch = false;
                await foreach (var piece in generator.StreamFromCurrentStateAsync(
                    maxTokens: maxTokens, cancellationToken: cancellationToken))
                {
                    timer.MarkFirstToken();
                    timer.IncrementPredicted();
                    var (emit, stopped) = matcher.Offer(piece);
                    if (!string.IsNullOrEmpty(emit))
                    {
                        var json = System.Text.Json.JsonSerializer.Serialize(new { content = emit, stop = false });
                        await http.Response.WriteAsync("data: " + json + "\n\n", cancellationToken);
                        await http.Response.Body.FlushAsync(cancellationToken);
                    }
                    if (stopped) { stoppedOnMatch = true; break; }
                }
                if (!stoppedOnMatch)
                {
                    var trailing = matcher.Flush();
                    if (!string.IsNullOrEmpty(trailing))
                    {
                        var json = System.Text.Json.JsonSerializer.Serialize(new { content = trailing, stop = false });
                        await http.Response.WriteAsync("data: " + json + "\n\n", cancellationToken);
                        await http.Response.Body.FlushAsync(cancellationToken);
                    }
                }
                timer.Finish();

                var tail = System.Text.Json.JsonSerializer.Serialize(new
                {
                    content = "",
                    stop = true,
                    stop_reason = stoppedOnMatch ? "stop" : MapStopReason(generator.LastStopReason),
                    model = host.ModelId,
                    tokens_cached = 0,
                    timings = timer.Snapshot(),
                });
                await http.Response.WriteAsync("data: " + tail + "\n\n", cancellationToken);
            }
            else
            {
                var buf = new System.Text.StringBuilder();
                bool stoppedOnMatch = false;
                await foreach (var piece in generator.StreamFromCurrentStateAsync(
                    maxTokens: maxTokens, cancellationToken: cancellationToken))
                {
                    timer.MarkFirstToken();
                    timer.IncrementPredicted();
                    var (emit, stopped) = matcher.Offer(piece);
                    if (emit.Length > 0) buf.Append(emit);
                    if (stopped) { stoppedOnMatch = true; break; }
                }
                if (!stoppedOnMatch) buf.Append(matcher.Flush());
                timer.Finish();

                var body = new CompletionResponse
                {
                    Content = buf.ToString(),
                    StopReason = stoppedOnMatch ? "stop" : MapStopReason(generator.LastStopReason),
                    Model = host.ModelId,
                    Timings = timer.Snapshot(),
                };
                http.Response.ContentType = "application/json";
                await http.Response.WriteAsJsonAsync(body, cancellationToken: cancellationToken);
            }
        }
        finally
        {
            foreach (var b in bitmaps) b.Dispose();
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
