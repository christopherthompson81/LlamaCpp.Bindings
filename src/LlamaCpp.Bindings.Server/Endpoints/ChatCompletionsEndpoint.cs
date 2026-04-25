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
        MmprojHost mmproj,
        DraftHost draft,
        CancellationToken cancellationToken)
    {
        var log = loggerFactory.CreateLogger("ChatCompletions");
        var opts = options.Value;

        // Wrap the client-driven cancellation token with a server-side
        // timeout. Re-bind the parameter so every subsequent
        // `cancellationToken` reference picks up the linked token —
        // surfacing timeouts as 504 below requires checking
        // timeoutCts.IsCancellationRequested in a caught OCE handler.
        var clientAbortedToken = cancellationToken;
        var (linkedCts, timeoutCts) = RequestGuard.CreateLinkedToken(cancellationToken, opts);
        using var _linkedCts = linkedCts;
        using var _timeoutCts = timeoutCts;
        cancellationToken = linkedCts.Token;

        try
        {
            await HandleCore(http, req, host, pool, opts, log, metrics, mmproj, draft, cancellationToken);
        }
        catch (OperationCanceledException) when (
            timeoutCts is not null
            && timeoutCts.IsCancellationRequested
            && !clientAbortedToken.IsCancellationRequested)
        {
            // Timeout fired and the client did NOT abort — surface as 504
            // for non-streaming. Streaming responses will already have
            // headers on the wire, so the best we can do is let the
            // connection close (clients observe end-of-stream early).
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
        ChatCompletionsRequest req,
        ModelHost host,
        SessionPool pool,
        ServerOptions opts,
        ILogger log,
        ServerMetrics metrics,
        MmprojHost mmproj,
        DraftHost draft,
        CancellationToken cancellationToken)
    {
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

        // Flatten polymorphic content (string or array of parts) into plain
        // text + a separate list of image bytes. This runs before template
        // application so the Jinja renderer just sees strings. Multi-part
        // input that includes images needs the mmproj path; we check that
        // after extraction so malformed parts reject with 400 first.
        string mediaMarker = mmproj.IsAvailable ? mmproj.Context!.DefaultMediaMarker : "<__media__>";
        ChatContentExtractor.Result contentResult;
        try
        {
            contentResult = ChatContentExtractor.FlattenAndExtract(req.Messages, mediaMarker);
        }
        catch (ArgumentException ex)
        {
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new { error = ex.Message }, cancellationToken);
            return;
        }

        if (contentResult.HasImages && !mmproj.IsAvailable)
        {
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new
            {
                error = "This server is not configured with a multimodal projector. " +
                        "Set LlamaServer:MmprojPath to accept image_url content parts.",
            }, cancellationToken);
            return;
        }

        // Resolve tool_choice first — it's the most specific grammar
        // source and overrides response_format / grammar / json_schema
        // when it forces a specific tool. Also drives whether tools are
        // passed to the chat template.
        ToolChoiceDescriptor toolChoice;
        try
        {
            toolChoice = ToolChoiceResolver.Resolve(req.ToolChoice, req.Tools);
        }
        catch (ArgumentException ex)
        {
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new { error = ex.Message }, cancellationToken);
            return;
        }

        if (toolChoice.Kind == ToolChoiceKind.RequiredAny)
        {
            // V1 would need a GBNF union across every tool's schema;
            // JsonSchemaToGbnf doesn't currently support that cleanly.
            // Filed as a follow-up (see tool-calling issue notes).
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new
            {
                error = "tool_choice='required' (any-tool) is not supported in V1. " +
                        "Specify a single tool via " +
                        "{\"type\":\"function\",\"function\":{\"name\":\"X\"}} to force it.",
            }, cancellationToken);
            return;
        }

        string prompt;
        try
        {
            var messages = req.Messages.Select(m =>
                new ChatMessage(m.Role, FlattenMessageContent(m))).ToArray();
            // Only pass tools to the template when we actually want them
            // rendered into the prompt — "none" means no mention.
            var toolsForTemplate = toolChoice.Kind == ToolChoiceKind.None
                ? null
                : ToolDefsToTemplate(req.Tools);
            prompt = LlamaChatTemplate.Apply(
                template, messages, addAssistantPrefix: true, tools: toolsForTemplate);
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

        // Resolve grammar (response_format / grammar / json_schema) — tool
        // choice takes precedence when it forces a specific tool.
        LlamaGrammar? grammar;
        try
        {
            grammar = ToolChoiceResolver.ForcedGrammar(toolChoice)
                      ?? GrammarFactory.Resolve(req.Grammar, req.JsonSchemaShort, req.ResponseFormat);
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

        // Prompt-length cap. Better than letting llama_decode hit a
        // "no KV slot" error halfway through. 413 = request entity too
        // large; the body explains how the limit was derived.
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

        // Speculative branch: opt-in via `speculative=true` AND a draft host
        // configured at startup. Falls back to the normal path when the
        // request uses features the speculative generator doesn't carry
        // (multimodal images, forced tool calls, per-token logprobs).
        bool wantsSpeculative = req.Speculative == true
            && draft.IsAvailable
            && !contentResult.HasImages
            && toolChoice.Kind != ToolChoiceKind.ForcedFunction
            && req.Logprobs != true;
        if (wantsSpeculative)
        {
            using var draftLease = await draft.LeaseAsync(cancellationToken);
            using var _samplerSpec = sampler;

            // Greedy draft sampler — V1 keeps it simple. The draft only needs
            // to be a fast guesser; its picks are verified against the main's
            // (sampler-applied) logits, so a probabilistic draft sampler
            // doesn't add correctness, only acceptance-rate variance.
            using var draftSampler = new LlamaSamplerBuilder().WithGreedy().Build();

            using var specGen = new LlamaSpeculativeGenerator(
                draftLease.MainContext, draftLease.DraftContext,
                sampler, draftSampler,
                draftLookahead: draftLease.DraftLookahead);

            // Speculative bypasses the SessionPool, so X-Cached-Tokens is
            // always 0 for these requests. Operators tracking cache-hit
            // rate per-endpoint will see the difference in /metrics.
            http.Response.Headers["X-Cached-Tokens"] = "0";

            var specTimer = new RequestTimer(
                promptTokensToDecode: promptTokens.Length, cachedTokens: 0);
            metrics.AddPromptTokensIngested(promptTokens.Length);

            string specId = "chatcmpl-" + Guid.NewGuid().ToString("N");
            long specCreated = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

            if (req.Stream)
            {
                await StreamSseSpeculative(
                    http, specGen, promptTokens, matcher, maxTokens,
                    host, specId, specCreated, specTimer, metrics, cancellationToken);
            }
            else
            {
                await WriteSingleJsonSpeculative(
                    http, specGen, promptTokens, matcher, maxTokens,
                    host, specId, specCreated, specTimer, metrics, cancellationToken);
            }
            return;
        }

        // Multimodal branch: image tokens go into the KV via the mtmd
        // helper rather than the normal prompt-tokens path, so the pool's
        // prefix-cache matching doesn't apply. We lease a slot with
        // no-prompt (forcing zero cache reuse), clear its KV, run
        // EvalPromptAsync to prefill image + text chunks, then stream
        // from the context's current state.
        if (contentResult.HasImages)
        {
            using var lease0 = await pool.LeaseAsync(Array.Empty<int>(), cancellationToken);
            using var _0 = sampler;
            lease0.InvalidateCache();
            lease0.Session.ClearHistory();

            var generator0 = new LlamaGenerator(lease0.Session, sampler);
            http.Response.Headers["X-Cached-Tokens"] = "0";

            var timer0 = new RequestTimer(promptTokensToDecode: 0, cachedTokens: 0);

            string id0 = "chatcmpl-" + Guid.NewGuid().ToString("N");
            long created0 = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

            var bitmaps = new List<MtmdBitmap>();
            try
            {
                foreach (var bytes in contentResult.Images)
                {
                    bitmaps.Add(MtmdBitmap.FromBytes(mmproj.Context!, bytes));
                }
                await mmproj.Context!.EvalPromptAsync(
                    lease0.Session.Context, prompt, bitmaps,
                    nPast: 0,
                    seqId: lease0.Session.SequenceId,
                    nBatch: lease0.Session.Context.LogicalBatchSize,
                    logitsLast: true,
                    addSpecial: false,
                    parseSpecial: true,
                    cancellationToken);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                foreach (var b in bitmaps) b.Dispose();
                log.LogWarning(ex, "mmproj prefill failed");
                http.Response.StatusCode = StatusCodes.Status500InternalServerError;
                await http.Response.WriteAsJsonAsync(new { error = "multimodal prefill failed: " + ex.Message }, cancellationToken);
                return;
            }
            try
            {
                bool forcedTool0 = toolChoice.Kind == ToolChoiceKind.ForcedFunction;
                if (req.Stream && !forcedTool0)
                {
                    await StreamSseFromCurrent(http, generator0, matcher, maxTokens, host, id0, created0, timer0, metrics, cancellationToken);
                }
                else
                {
                    await WriteSingleJsonFromCurrent(http, generator0, matcher, maxTokens, host, id0, created0, timer0, metrics, toolChoice, cancellationToken);
                }
            }
            finally
            {
                foreach (var b in bitmaps) b.Dispose();
            }
            return;
        }

        // Lease a slot (may queue). The lease carries FirstNewIndex = how
        // many of these tokens are already in KV from the last request.
        using var lease = await pool.LeaseAsync(
            promptTokens, cancellationToken, useCache: req.CachePrompt ?? true);
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

        // Logprobs config — capped at OpenAI's documented top_logprobs
        // upper bound (20). The cap protects the per-token cost when a
        // caller asks for an unreasonably large alternative set.
        bool wantLogprobs = req.Logprobs == true;
        int topN = wantLogprobs ? Math.Clamp(req.TopLogprobs ?? 0, 0, 20) : 0;

        // Forced tool-call paths go through the non-streaming writer:
        // tool-call streaming has its own delta format and V1 doesn't
        // implement it. Plain text requests honour stream=true as usual.
        bool forcedTool = toolChoice.Kind == ToolChoiceKind.ForcedFunction;
        if (req.Stream && !forcedTool)
        {
            await StreamSse(http, generator, lease, matcher, maxTokens, host, completionId, createdUnix, timer, metrics, wantLogprobs, topN, cancellationToken);
        }
        else
        {
            await WriteSingleJson(http, generator, lease, matcher, maxTokens, host, completionId, createdUnix, timer, metrics, toolChoice, wantLogprobs, topN, cancellationToken);
        }
    }

    // ----- Speculative path: draft + main, no SessionPool, no prefix cache -----

    private static async Task WriteSingleJsonSpeculative(
        HttpContext http, LlamaSpeculativeGenerator gen,
        int[] promptTokens, StopMatcher matcher, int maxTokens,
        ModelHost host, string id, long created,
        RequestTimer timer, ServerMetrics metrics, CancellationToken ct)
    {
        var buf = new StringBuilder();
        bool stoppedOnMatch = false;
        int predicted = 0;
        await foreach (var piece in gen.GenerateAsync(
            promptTokens, maxTokens: maxTokens, cancellationToken: ct))
        {
            timer.MarkFirstToken();
            predicted++;
            timer.IncrementPredicted();
            var (emit, stopped) = matcher.Offer(piece);
            if (emit.Length > 0) buf.Append(emit);
            if (stopped) { stoppedOnMatch = true; break; }
        }
        if (!stoppedOnMatch) buf.Append(matcher.Flush());
        timer.Finish();
        metrics.AddTokensGenerated(predicted);

        var response = new ChatCompletionsResponse
        {
            Id = id, Created = created, Model = host.ModelId,
            Choices = new()
            {
                new ChatChoice
                {
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

    private static async Task StreamSseSpeculative(
        HttpContext http, LlamaSpeculativeGenerator gen,
        int[] promptTokens, StopMatcher matcher, int maxTokens,
        ModelHost host, string id, long created,
        RequestTimer timer, ServerMetrics metrics, CancellationToken ct)
    {
        http.Response.ContentType = "text/event-stream";
        http.Response.Headers.CacheControl = "no-cache";
        var feature = http.Features.Get<Microsoft.AspNetCore.Http.Features.IHttpResponseBodyFeature>();
        feature?.DisableBuffering();

        await WriteChunk(http, new ChatCompletionsChunk
        {
            Id = id, Created = created, Model = host.ModelId,
            Choices = new() { new() { Index = 0, Delta = new() { Role = "assistant" }, FinishReason = null } },
        }, ct);

        bool stoppedOnMatch = false;
        int predicted = 0;
        await foreach (var piece in gen.GenerateAsync(
            promptTokens, maxTokens: maxTokens, cancellationToken: ct))
        {
            timer.MarkFirstToken();
            predicted++;
            timer.IncrementPredicted();
            var (emit, stopped) = matcher.Offer(piece);
            if (!string.IsNullOrEmpty(emit))
            {
                await WriteChunk(http, new ChatCompletionsChunk
                {
                    Id = id, Created = created, Model = host.ModelId,
                    Choices = new() { new() { Index = 0, Delta = new() { Content = emit }, FinishReason = null } },
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
                    Choices = new() { new() { Index = 0, Delta = new() { Content = tail }, FinishReason = null } },
                }, ct);
            }
        }
        timer.Finish();
        metrics.AddTokensGenerated(predicted);

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
        await http.Response.WriteAsync("data: [DONE]\n\n", ct);
    }

    // ----- Multimodal path: sampling starts from logits already in place -----

    private static async Task WriteSingleJsonFromCurrent(
        HttpContext http, LlamaGenerator gen,
        StopMatcher matcher, int maxTokens,
        ModelHost host, string id, long created,
        RequestTimer timer, ServerMetrics metrics,
        ToolChoiceDescriptor toolChoice, CancellationToken ct)
    {
        var buf = new StringBuilder();
        bool stoppedOnMatch = false;
        await foreach (var piece in gen.StreamFromCurrentStateAsync(
            maxTokens: maxTokens, cancellationToken: ct))
        {
            timer.MarkFirstToken();
            timer.IncrementPredicted();
            var (emit, stopped) = matcher.Offer(piece);
            if (emit.Length > 0) buf.Append(emit);
            if (stopped) { stoppedOnMatch = true; break; }
        }
        if (!stoppedOnMatch) buf.Append(matcher.Flush());
        timer.Finish();
        metrics.AddTokensGenerated(timer.PredictedTokens);

        var choice = toolChoice.Kind == ToolChoiceKind.ForcedFunction
            ? BuildForcedToolChoice(toolChoice, buf.ToString())
            : new ChatChoice
            {
                Index = 0,
                Message = new ChatMessageDto { Role = "assistant", Content = buf.ToString() },
                FinishReason = stoppedOnMatch ? "stop" : MapFinishReason(gen.LastStopReason),
            };

        var response = new ChatCompletionsResponse
        {
            Id = id, Created = created, Model = host.ModelId,
            Choices = new() { choice },
            Timings = timer.Snapshot(),
        };
        http.Response.ContentType = "application/json";
        await http.Response.WriteAsJsonAsync(response, cancellationToken: ct);
    }

    private static async Task StreamSseFromCurrent(
        HttpContext http, LlamaGenerator gen,
        StopMatcher matcher, int maxTokens,
        ModelHost host, string id, long created,
        RequestTimer timer, ServerMetrics metrics, CancellationToken ct)
    {
        http.Response.ContentType = "text/event-stream";
        http.Response.Headers.CacheControl = "no-cache";
        var feature = http.Features.Get<Microsoft.AspNetCore.Http.Features.IHttpResponseBodyFeature>();
        feature?.DisableBuffering();

        await WriteChunk(http, new ChatCompletionsChunk
        {
            Id = id, Created = created, Model = host.ModelId,
            Choices = new() { new() { Index = 0, Delta = new() { Role = "assistant" }, FinishReason = null } },
        }, ct);

        bool stoppedOnMatch = false;
        await foreach (var piece in gen.StreamFromCurrentStateAsync(
            maxTokens: maxTokens, cancellationToken: ct))
        {
            timer.MarkFirstToken();
            timer.IncrementPredicted();
            var (emit, stopped) = matcher.Offer(piece);
            if (!string.IsNullOrEmpty(emit))
            {
                await WriteChunk(http, new ChatCompletionsChunk
                {
                    Id = id, Created = created, Model = host.ModelId,
                    Choices = new() { new() { Index = 0, Delta = new() { Content = emit }, FinishReason = null } },
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
                    Choices = new() { new() { Index = 0, Delta = new() { Content = tail }, FinishReason = null } },
                }, ct);
            }
        }
        timer.Finish();
        metrics.AddTokensGenerated(timer.PredictedTokens);

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
        await http.Response.WriteAsync("data: [DONE]\n\n", ct);
    }

    // ----- Non-streaming: collect all pieces into one response body. -----

    private static async Task WriteSingleJson(
        HttpContext http, LlamaGenerator gen, SessionLease lease,
        StopMatcher matcher, int maxTokens,
        ModelHost host, string id, long created,
        RequestTimer timer, ServerMetrics metrics,
        ToolChoiceDescriptor toolChoice,
        bool wantLogprobs, int topN, CancellationToken ct)
    {
        var buf = new StringBuilder();
        bool stoppedOnMatch = false;
        var logprobs = wantLogprobs ? new List<TokenLogprobInfo>() : null;
        await foreach (var piece in gen.GenerateAsync(
            lease.PromptTokens,
            maxTokens: maxTokens,
            firstNewIndex: lease.FirstNewIndex,
            onTokenDecoded: t =>
            {
                lease.OnTokenDecoded(t);
                timer.IncrementPredicted();
            },
            logprobsTopN: topN,
            onLogprobs: logprobs is null ? null : info => logprobs.Add(info),
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

        var choice = toolChoice.Kind == ToolChoiceKind.ForcedFunction
            ? BuildForcedToolChoice(toolChoice, buf.ToString())
            : new ChatChoice
            {
                Index = 0,
                Message = new ChatMessageDto { Role = "assistant", Content = buf.ToString() },
                FinishReason = stoppedOnMatch ? "stop" : MapFinishReason(gen.LastStopReason),
            };
        if (logprobs is not null)
        {
            choice.Logprobs = BuildLogprobs(host.Model.Vocab, logprobs);
        }

        var response = new ChatCompletionsResponse
        {
            Id = id,
            Created = created,
            Model = host.ModelId,
            Choices = new() { choice },
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
        RequestTimer timer, ServerMetrics metrics,
        bool wantLogprobs, int topN, CancellationToken ct)
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

        // Streaming logprobs accumulate into a per-chunk queue so each
        // SSE chunk carries the logprobs for the tokens that produced
        // its delta.content. The generator's onLogprobs callback fires
        // per-token; pieces aggregate multiple tokens via the UTF-8
        // decoder before they reach us, so we buffer until the next
        // chunk emit.
        var pendingLogprobs = wantLogprobs ? new Queue<TokenLogprobInfo>() : null;
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
            logprobsTopN: topN,
            onLogprobs: pendingLogprobs is null ? null : info => pendingLogprobs.Enqueue(info),
            cancellationToken: ct))
        {
            timer.MarkFirstToken();
            var (emit, stopped) = matcher.Offer(piece);
            if (!string.IsNullOrEmpty(emit))
            {
                LogprobsContent? chunkLogprobs = null;
                if (pendingLogprobs is { Count: > 0 })
                {
                    var drained = new List<TokenLogprobInfo>(pendingLogprobs.Count);
                    while (pendingLogprobs.Count > 0) drained.Add(pendingLogprobs.Dequeue());
                    chunkLogprobs = BuildLogprobs(host.Model.Vocab, drained);
                }
                await WriteChunk(http, new ChatCompletionsChunk
                {
                    Id = id, Created = created, Model = host.ModelId,
                    Choices = new()
                    {
                        new() { Index = 0, Delta = new() { Content = emit }, FinishReason = null, Logprobs = chunkLogprobs },
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

    /// <summary>
    /// Collapse a message's rich content / tool-related fields into the
    /// plain string the chat template sees. The binding's
    /// <see cref="ChatMessage"/> is <c>(role, content)</c> only, so
    /// assistant messages that carry tool_calls have their calls
    /// serialised into the content body here — the template will render
    /// them as prose in V1 (known limitation; some Jinja templates that
    /// check <c>message.tool_calls</c> directly won't see the field).
    /// </summary>
    private static string FlattenMessageContent(ChatMessageDto m)
    {
        if (m.ToolCalls is { Count: > 0 })
        {
            // OpenAI history shape: assistant-turn tool_calls. Serialise
            // into the content body so at least the template sees SOME
            // representation of what the assistant did.
            var prefix = m.Content?.Text ?? "";
            var suffix = System.Text.Json.JsonSerializer.Serialize(m.ToolCalls);
            return string.IsNullOrEmpty(prefix) ? suffix : prefix + "\n" + suffix;
        }
        return m.Content?.Text ?? "";
    }

    /// <summary>
    /// Convert <see cref="ToolDef"/> records into the untyped shape the
    /// Jinja template expects: a list of <c>{type: "function", function:
    /// {name, description, parameters}}</c> dictionaries. Jinja engines
    /// access fields by name, not by JsonPropertyName attribute, so a
    /// direct pass-through of the DTO objects wouldn't work.
    /// </summary>
    private static IReadOnlyList<object?>? ToolDefsToTemplate(IReadOnlyList<ToolDef>? tools)
    {
        if (tools is null || tools.Count == 0) return null;
        var result = new List<object?>(tools.Count);
        foreach (var t in tools)
        {
            if (t.Function is null) continue;
            var fn = new Dictionary<string, object?>
            {
                ["name"] = t.Function.Name,
            };
            if (!string.IsNullOrEmpty(t.Function.Description))
            {
                fn["description"] = t.Function.Description;
            }
            if (t.Function.Parameters is System.Text.Json.JsonElement p &&
                p.ValueKind != System.Text.Json.JsonValueKind.Null &&
                p.ValueKind != System.Text.Json.JsonValueKind.Undefined)
            {
                // JsonElement is serialisable; Jinja engines that care
                // about iteration can walk it as an object tree.
                fn["parameters"] = p;
            }
            result.Add(new Dictionary<string, object?>
            {
                ["type"] = "function",
                ["function"] = fn,
            });
        }
        return result;
    }

    /// <summary>
    /// Wrap the generator's raw output as a single <see cref="ToolCall"/>
    /// when the caller forced a specific function via tool_choice. The
    /// caller generated under a grammar that constrained output to the
    /// function's parameters schema, so <paramref name="generatedJson"/>
    /// is expected to parse as a valid argument object.
    /// </summary>
    private static ChatChoice BuildForcedToolChoice(
        ToolChoiceDescriptor toolChoice, string generatedJson)
    {
        return new ChatChoice
        {
            Index = 0,
            Message = new ChatMessageDto
            {
                Role = "assistant",
                ToolCalls = new()
                {
                    new ToolCall
                    {
                        Id = "call_" + Guid.NewGuid().ToString("N")[..12],
                        Type = "function",
                        Function = new()
                        {
                            Name = toolChoice.ForcedTool!.Function!.Name,
                            Arguments = generatedJson,
                        },
                    },
                },
            },
            FinishReason = "tool_calls",
        };
    }

    /// <summary>
    /// Convert the generator's per-token <see cref="TokenLogprobInfo"/>
    /// entries into the OpenAI-shaped <see cref="LogprobsContent"/>
    /// envelope. The token's piece-text + UTF-8 bytes come from
    /// <see cref="LlamaVocab.TokenToPiece"/>; tokens that don't roundtrip
    /// through UTF-8 cleanly (e.g. byte-fallback fragments) get
    /// <c>bytes = null</c>, matching OpenAI's convention.
    /// </summary>
    private static LogprobsContent BuildLogprobs(
        LlamaVocab vocab, IReadOnlyList<TokenLogprobInfo> tokens)
    {
        var content = new List<LogprobToken>(tokens.Count);
        foreach (var info in tokens)
        {
            var piece = vocab.TokenToPiece(info.TokenId, renderSpecial: true);
            var entry = new LogprobToken
            {
                Token = piece,
                Logprob = info.Logprob,
                Bytes = TokenBytes(piece),
            };
            foreach (var alt in info.TopAlternatives)
            {
                var altPiece = vocab.TokenToPiece(alt.TokenId, renderSpecial: true);
                entry.TopLogprobs.Add(new TopLogprob
                {
                    Token = altPiece,
                    Logprob = alt.Logprob,
                    Bytes = TokenBytes(altPiece),
                });
            }
            content.Add(entry);
        }
        return new LogprobsContent { Content = content };
    }

    private static int[]? TokenBytes(string piece)
    {
        if (string.IsNullOrEmpty(piece)) return Array.Empty<int>();
        // OpenAI emits bytes as an array of integer code-units. A
        // round-tripable UTF-8 encoding is the common case; if the piece
        // contains a U+FFFD replacement (i.e. came in as a partial
        // byte-fallback fragment), report null per OpenAI's convention.
        if (piece.Contains('�')) return null;
        var bytes = System.Text.Encoding.UTF8.GetBytes(piece);
        var ints = new int[bytes.Length];
        for (int i = 0; i < bytes.Length; i++) ints[i] = bytes[i];
        return ints;
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
