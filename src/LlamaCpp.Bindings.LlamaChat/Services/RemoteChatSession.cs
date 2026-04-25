using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.Services.Remote;
using LlamaCpp.Bindings.LlamaChat.Services.ToolCall;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// HTTP-backed chat session targeting an OpenAI-compatible
/// <c>/v1/chat/completions</c> endpoint. The chat template is applied
/// server-side; we just send the transcript as <c>{role,content}</c>
/// messages, drain the SSE stream, and emit the same
/// <see cref="StreamEvent"/> shapes a local session would.
/// </summary>
public sealed class RemoteChatSession : IChatSession
{
    private readonly OpenAiChatClient _client;
    private readonly RemoteSettings _settings;

    public bool SupportsImages => true;   // OpenAI image_url parts; servers without an mmproj will reject the call.
    public bool SupportsAudio => false;   // Out of scope for v1; the standard image_url part doesn't carry audio.
    public bool SupportsMedia => SupportsImages || SupportsAudio;

    /// <summary>
    /// Title generation works for any reasonable text model; the server
    /// applies the template, so we don't need to pre-screen ASR-only models
    /// the way the local path does.
    /// </summary>
    public bool CanGenerateTitles => true;

    public IToolCallFormat? ToolCallFormat => null;
    public string? ChatTemplate => null;
    public string DisplayModelName => _settings.ModelId;

    public int? EstimatePromptTokens(string prompt) => null;

    public RemoteChatSession(RemoteSettings settings)
    {
        _settings = settings;
        _client = new OpenAiChatClient(settings.BaseUrl, settings.ApiKey);
    }

    /// <summary>
    /// No client-side cache to drop. The server's <c>cache_prompt</c> flag
    /// (driven by <see cref="GenerationSettings.ReusePromptPrefix"/>) is the
    /// equivalent knob and is set per request.
    /// </summary>
    public void ClearKv() { }

    public async Task<string?> GenerateTitleAsync(string userMessage, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(userMessage)) return null;

        var req = new ChatCompletionsRequest
        {
            Model = string.IsNullOrEmpty(_settings.ModelId) ? null : _settings.ModelId,
            Messages = new List<OpenAiChatMessage>
            {
                new() {
                    Role = "system",
                    Content = MessageContent.FromText(
                        "You generate concise conversation titles. Reply with a 3-6 word title that captures the user's topic. Reply with ONLY the title text — no quotes, no trailing punctuation, no \"Title:\" prefix."),
                },
                new() { Role = "user", Content = MessageContent.FromText(userMessage) },
            },
            MaxTokens = 32,
            Temperature = 0.3f,
            TopP = 0.9f,
            TopK = 40,
            CachePrompt = false,
        };

        try
        {
            var resp = await _client.CreateChatCompletionAsync(req, cancellationToken).ConfigureAwait(false);
            var content = resp.Choices.Count > 0 ? resp.Choices[0].Message.Content?.Text : null;
            return CleanTitle(content);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception)
        {
            return null;
        }
    }

    private static string? CleanTitle(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return null;
        var line = raw.Split('\n')[0].Trim().Trim('"').Trim('\'').Trim();
        if (line.StartsWith("Title:", StringComparison.OrdinalIgnoreCase))
            line = line[6..].Trim();
        line = line.TrimEnd('.', '!', '?', ',', ';', ' ');
        if (line.Length == 0) return null;
        if (line.Length > 60) line = line[..60].TrimEnd() + "…";
        return line;
    }

    public async IAsyncEnumerable<StreamEvent> StreamAssistantReplyAsync(
        IReadOnlyList<ChatTurn> transcript,
        SamplerSettings sampler,
        GenerationSettings generation,
        IReadOnlyList<object?>? tools = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var req = BuildRequest(transcript, sampler, generation);

        // The reasoning extractor needs to know whether the chat template
        // pre-opened a <think> block. Locally we read the rendered prompt and
        // pass `startInReasoning` directly; remotely the server does the
        // template, so we sniff the first content bytes to decide:
        //   - First non-whitespace token is `<think>` → model is opening its
        //     own block (start in Content; the open tag flips us to Reasoning).
        //   - Anything else → template pre-opened, we're already inside a
        //     reasoning block (start in Reasoning; wait for `</think>`).
        // Until the decision is made, content bytes are buffered in
        // `pendingHead`. The decision needs at most 7 chars (length of
        // "<think>") + leading whitespace before it commits.
        ReasoningExtractor? extractor = null;
        bool extractorDecided = !generation.ExtractReasoning;
        var pendingHead = generation.ExtractReasoning ? new StringBuilder() : null;
        var asrExtractor = generation.ExtractAsrTranscript ? new AsrTextExtractor() : null;

        var sw = Stopwatch.StartNew();
        TimeSpan? promptTime = null;
        int completionTokens = 0;
        int promptTokens = 0;
        TimeSpan promptTimeFromServer = TimeSpan.Zero;
        TimeSpan generationTimeFromServer = TimeSpan.Zero;
        LlamaStopReason stopReason = LlamaStopReason.EndOfGeneration;

        await foreach (var chunk in _client.StreamChatCompletionAsync(req, cancellationToken).ConfigureAwait(false))
        {
            // First content arrival = end of prefill phase. We use the
            // server's authoritative timings if it sends them on the final
            // chunk; this fallback covers servers that don't.
            if (promptTime is null)
            {
                promptTime = sw.Elapsed;
            }

            string? deltaContent = null;
            string? finishReason = null;
            if (chunk.Choices.Count > 0)
            {
                deltaContent = chunk.Choices[0].Delta.Content;
                finishReason = chunk.Choices[0].FinishReason;
            }

            if (!string.IsNullOrEmpty(deltaContent))
            {
                completionTokens++;

                // Defer extractor creation until the first chunks have given
                // us enough context to decide which mode to start in.
                string? toExtract = deltaContent;
                if (!extractorDecided)
                {
                    pendingHead!.Append(deltaContent);
                    var head = pendingHead.ToString();
                    var trimmed = head.AsSpan().TrimStart();
                    // Decision is made once we have either >= 7 visible chars,
                    // or we can already prove the head doesn't begin with
                    // "<think>" (a non-"<" char appeared past the whitespace).
                    bool canDecide =
                        trimmed.Length >= 7 ||
                        (trimmed.Length > 0 && trimmed[0] != '<');
                    if (!canDecide) continue; // wait for more bytes

                    bool modelOpensThink =
                        trimmed.StartsWith("<think>".AsSpan(), StringComparison.Ordinal);
                    extractor = new ReasoningExtractor(startInReasoning: !modelOpensThink);
                    toExtract = head;
                    pendingHead = null;
                    extractorDecided = true;
                }

                // Push through reasoning then ASR extractors, mirroring the
                // local path so the UI sees identical event shapes regardless
                // of which backend produced the tokens.
                string contentSlice;
                string? reasoningSlice;
                if (extractor is not null)
                {
                    var re = extractor.Push(toExtract);
                    contentSlice = re.Content;
                    reasoningSlice = re.Reasoning.Length > 0 ? re.Reasoning : null;
                }
                else
                {
                    contentSlice = toExtract;
                    reasoningSlice = null;
                }
                if (reasoningSlice is not null) yield return new StreamEvent.Reasoning(reasoningSlice);

                if (contentSlice.Length > 0)
                {
                    if (asrExtractor is not null)
                    {
                        var ae = asrExtractor.Push(contentSlice);
                        if (ae.Language is not null) yield return new StreamEvent.Language(ae.Language);
                        if (ae.Content.Length > 0) yield return new StreamEvent.Content(ae.Content);
                    }
                    else
                    {
                        yield return new StreamEvent.Content(contentSlice);
                    }
                }
            }

            if (chunk.Timings is { } t)
            {
                promptTokens = t.PromptN;
                promptTimeFromServer = TimeSpan.FromMilliseconds(t.PromptMs);
                generationTimeFromServer = TimeSpan.FromMilliseconds(t.PredictedMs);
                if (t.PredictedN > 0) completionTokens = t.PredictedN;
            }

            if (finishReason is not null)
            {
                stopReason = finishReason switch
                {
                    "stop" => LlamaStopReason.EndOfGeneration,
                    "length" => LlamaStopReason.MaxTokens,
                    _ => LlamaStopReason.EndOfGeneration,
                };
            }
        }

        // Stream ended before the extractor-mode heuristic could commit (very
        // short reply: fewer than 7 visible chars, all consistent with a
        // partial "<think>" prefix). Treat the buffered head as content —
        // safer than dropping it.
        if (!extractorDecided && pendingHead is { Length: > 0 })
        {
            extractor = new ReasoningExtractor(startInReasoning: false);
            var re = extractor.Push(pendingHead.ToString());
            if (re.Reasoning.Length > 0) yield return new StreamEvent.Reasoning(re.Reasoning);
            if (re.Content.Length > 0) yield return new StreamEvent.Content(re.Content);
            pendingHead = null;
            extractorDecided = true;
        }

        // Flush extractor tails into the stream.
        if (extractor is not null)
        {
            var tail = extractor.Flush();
            if (tail.Reasoning.Length > 0) yield return new StreamEvent.Reasoning(tail.Reasoning);
            if (tail.Content.Length > 0)
            {
                if (asrExtractor is not null)
                {
                    var ae = asrExtractor.Push(tail.Content);
                    if (ae.Language is not null) yield return new StreamEvent.Language(ae.Language);
                    if (ae.Content.Length > 0) yield return new StreamEvent.Content(ae.Content);
                }
                else
                {
                    yield return new StreamEvent.Content(tail.Content);
                }
            }
        }
        if (asrExtractor is not null)
        {
            var aeTail = asrExtractor.Flush();
            if (aeTail.Language is not null) yield return new StreamEvent.Language(aeTail.Language);
            if (aeTail.Content.Length > 0) yield return new StreamEvent.Content(aeTail.Content);
        }

        sw.Stop();
        var promptTs = promptTimeFromServer != TimeSpan.Zero
            ? promptTimeFromServer
            : (promptTime ?? TimeSpan.Zero);
        var genTs = generationTimeFromServer != TimeSpan.Zero
            ? generationTimeFromServer
            : (promptTime.HasValue ? sw.Elapsed - promptTime.Value : sw.Elapsed);

        yield return new StreamEvent.Done(promptTokens, promptTs, completionTokens, genTs, stopReason);
    }

    public IAsyncEnumerable<StreamEvent> StreamContinuationAsync(
        SamplerSettings sampler,
        GenerationSettings generation,
        bool resumeInReasoning = false,
        CancellationToken cancellationToken = default)
        => throw new NotSupportedException(
            "Continuation isn't supported for remote profiles — the server doesn't expose mid-turn KV state.");

    private ChatCompletionsRequest BuildRequest(
        IReadOnlyList<ChatTurn> transcript,
        SamplerSettings sampler,
        GenerationSettings generation)
    {
        var messages = new List<OpenAiChatMessage>(transcript.Count);
        foreach (var t in transcript)
        {
            var role = t.Role switch
            {
                TurnRole.System    => "system",
                TurnRole.User      => "user",
                TurnRole.Assistant => "assistant",
                TurnRole.Tool      => "tool",
                _                  => "user",
            };

            // User turns with image attachments use the multimodal content-part
            // shape; everything else stays as a plain text body.
            if (t.Role == TurnRole.User && t.Attachments is { Count: > 0 })
            {
                var parts = new List<ContentPart>();
                if (!string.IsNullOrEmpty(t.Content))
                {
                    parts.Add(new ContentPart { Type = "text", Text = t.Content });
                }
                foreach (var a in t.Attachments)
                {
                    if (!a.IsImage) continue; // audio not modelled in OpenAI image_url
                    var dataUrl = "data:" + a.MimeType + ";base64," + Convert.ToBase64String(a.Data);
                    parts.Add(new ContentPart
                    {
                        Type = "image_url",
                        ImageUrl = new ImageUrl { Url = dataUrl },
                    });
                }
                messages.Add(new OpenAiChatMessage
                {
                    Role = role,
                    Content = parts.Count > 0 ? MessageContent.FromParts(parts) : MessageContent.FromText(t.Content),
                });
            }
            else
            {
                messages.Add(new OpenAiChatMessage { Role = role, Content = MessageContent.FromText(t.Content) });
            }
        }

        var req = new ChatCompletionsRequest
        {
            Model = string.IsNullOrEmpty(_settings.ModelId) ? null : _settings.ModelId,
            Messages = messages,
            MaxTokens = generation.MaxTokens,
            CachePrompt = generation.ReusePromptPrefix,
            Temperature = sampler.Temperature,
            Seed = sampler.Seed,
            TopK = sampler.TopK,
            TopP = sampler.TopP,
            MinP = sampler.MinP,
            TypicalP = sampler.Typical,
            TopNSigma = sampler.TopNSigma,
            XtcProbability = sampler.XtcProbability,
            XtcThreshold = sampler.XtcProbability is > 0 ? sampler.XtcThreshold : null,
            DryMultiplier = sampler.DryMultiplier > 0 ? sampler.DryMultiplier : null,
            DryBase = sampler.DryMultiplier > 0 ? sampler.DryBase : null,
            DryAllowedLength = sampler.DryMultiplier > 0 ? sampler.DryAllowedLength : null,
            DryPenaltyLastN = sampler.DryMultiplier > 0 ? sampler.DryPenaltyLastN : null,
            DynatempRange = sampler.DynaTempRange > 0 ? sampler.DynaTempRange : null,
            DynatempExponent = sampler.DynaTempRange > 0 ? sampler.DynaTempExponent : null,
            RepeatPenalty = Math.Abs(sampler.PenaltyRepeat - 1f) > 1e-6f ? sampler.PenaltyRepeat : null,
            FrequencyPenalty = sampler.PenaltyFrequency != 0f ? sampler.PenaltyFrequency : null,
            PresencePenalty = sampler.PenaltyPresence != 0f ? sampler.PenaltyPresence : null,
            RepeatLastN = sampler.PenaltyLastN,
        };
        if (sampler.Mirostat != MirostatMode.Off)
        {
            req.Mirostat = sampler.Mirostat == MirostatMode.V1 ? 1 : 2;
            req.MirostatTau = sampler.MirostatTau;
            req.MirostatEta = sampler.MirostatEta;
        }
        return req;
    }

    public void Dispose() => _client.Dispose();
}
