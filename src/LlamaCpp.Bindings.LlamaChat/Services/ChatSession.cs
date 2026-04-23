using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// One loaded model + context. Stateless w.r.t. the chat transcript —
/// callers (ConversationViewModel) own the turn list and pass a snapshot
/// into <see cref="StreamAssistantReplyAsync"/>. This lets one session
/// service multiple conversations by swapping transcripts in.
/// Not thread-safe: UI marshals send/cancel operations serially.
/// </summary>
public sealed class ChatSession : IDisposable
{
    private readonly LlamaModel _model;
    private readonly LlamaContext _context;
    private readonly MtmdContext? _mtmd;
    private readonly string? _chatTemplate;

    /// <summary>
    /// Tokens currently decoded into the KV cache for seq 0 (including both
    /// prompt and previously-generated assistant tokens). Used to compute the
    /// longest common prefix with each new turn's prompt so we only decode
    /// the delta rather than the full transcript every turn. <c>null</c>
    /// when we can't reason about the cache contents (fresh load, after a
    /// multimodal turn whose image chunks we didn't token-record, after an
    /// explicit <see cref="ClearKv"/>).
    /// </summary>
    private List<int>? _cachedTokens;

    public LlamaModel Model => _model;
    public LlamaContext Context => _context;
    public MtmdContext? Mtmd => _mtmd;
    public string? ChatTemplate => _chatTemplate;

    /// <summary>True if this session can accept image attachments on user turns.</summary>
    public bool SupportsImages => _mtmd?.SupportsVision == true;

    private ChatSession(LlamaModel model, LlamaContext context, MtmdContext? mtmd, string? template)
    {
        _model = model;
        _context = context;
        _mtmd = mtmd;
        _chatTemplate = template;
    }

    public static ChatSession Load(ModelLoadSettings settings, Action<LlamaLogLevel, string>? logSink = null)
    {
        LlamaBackend.Initialize(logSink);

        var model = new LlamaModel(settings.ModelPath, new LlamaModelParameters
        {
            GpuLayerCount = settings.GpuLayerCount,
            UseMmap = settings.UseMmap,
            UseMlock = settings.UseMlock,
        });

        LlamaContext? context = null;
        MtmdContext? mtmd = null;
        try
        {
            context = new LlamaContext(model, new LlamaContextParameters
            {
                ContextSize = settings.ContextSize,
                LogicalBatchSize = settings.LogicalBatchSize,
                PhysicalBatchSize = settings.PhysicalBatchSize,
                MaxSequenceCount = 1,
                OffloadKQV = settings.OffloadKQV,
                FlashAttention = settings.FlashAttention,
            });

            if (!string.IsNullOrWhiteSpace(settings.MmprojPath) && System.IO.File.Exists(settings.MmprojPath))
            {
                var mtmdParams = new MtmdContextParameters
                {
                    UseGpu = !settings.MmprojOnCpu,
                };
                if (settings.MmprojImageMinTokens is int minTokens)
                {
                    mtmdParams.ImageMinTokens = minTokens;
                }
                mtmd = new MtmdContext(model, settings.MmprojPath, mtmdParams);
            }
        }
        catch
        {
            mtmd?.Dispose();
            context?.Dispose();
            model.Dispose();
            throw;
        }

        var template = model.GetChatTemplate();
        DumpChatTemplate(template);
        return new ChatSession(model, context, mtmd, template);
    }

    /// <summary>
    /// Write the model's embedded chat template verbatim to a sibling of the
    /// app's other config files, so it can be inspected when diagnosing
    /// template-application issues. Silent on IO failure; this is debug
    /// plumbing, not a load-path dependency.
    /// </summary>
    private static void DumpChatTemplate(string? template)
    {
        if (string.IsNullOrEmpty(template)) return;
        try
        {
            var dir = System.IO.Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "LlamaChat");
            System.IO.Directory.CreateDirectory(dir);
            var path = System.IO.Path.Combine(dir, "last-template.jinja");
            System.IO.File.WriteAllText(path, template);
        }
        catch { }
    }

    /// <summary>
    /// Drop the KV cache and the prompt-token cache. Call when switching
    /// conversations, after an edit/delete/regenerate that reshapes the
    /// transcript, or any other time the caller can't guarantee the cached
    /// tokens still reflect what's in the KV cache.
    /// </summary>
    public void ClearKv()
    {
        _context.ClearKvCache();
        _cachedTokens = null;
    }

    /// <summary>
    /// Render <paramref name="transcript"/> through the model's chat template
    /// (or a bare fallback) and stream the assistant reply. The caller is
    /// responsible for appending the final assistant turn to its own
    /// transcript once <see cref="StreamEvent.Done"/> is observed.
    /// </summary>
    public async IAsyncEnumerable<StreamEvent> StreamAssistantReplyAsync(
        IReadOnlyList<ChatTurn> transcript,
        SamplerSettings sampler,
        GenerationSettings generation,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Gather attachments in transcript order. When any turn has images and
        // the session has a loaded MtmdContext, we take the multimodal branch:
        // prefill via mtmd_helper_eval_chunks (which encodes images + text),
        // then generate from the primed KV. Otherwise fall through to the
        // text-only tokenize+decode path.
        var bitmaps = new List<MtmdBitmap>();
        bool hasAttachments = false;
        foreach (var t in transcript)
        {
            if (t.Attachments is { Count: > 0 }) { hasAttachments = true; break; }
        }
        bool multimodal = hasAttachments && _mtmd is not null && _mtmd.SupportsVision;

        var prompt = RenderPromptForCompletion(transcript, multimodal ? _mtmd!.DefaultMediaMarker : null);

        using var llamaSampler = SamplerFactory.Build(_model, _model.Vocab, sampler);
        var generator = new LlamaGenerator(_context, llamaSampler);

        // Qwen3 / DeepSeek-R1 templates pre-open a <think> block at the end of
        // the assistant prefix, so the stream arrives already inside reasoning.
        // Prime the extractor to match. The "</think>\n\n" guard avoids the
        // no-thinking case (enable_thinking=false renders an already-closed
        // pair and we should stay in Content mode).
        var startInReasoning =
            prompt.EndsWith("<think>\n", StringComparison.Ordinal) &&
            !prompt.EndsWith("</think>\n\n", StringComparison.Ordinal);
        var extractor = generation.ExtractReasoning
            ? new ReasoningExtractor(startInReasoning)
            : null;

        var sw = Stopwatch.StartNew();
        var promptStartTicks = sw.ElapsedTicks;
        TimeSpan? promptTime = null;
        int completionTokens = 0;

        IAsyncEnumerable<string> stream;
        // Grows with each emitted token so the next turn can diff against it.
        // Null when we lose track of what's in the cache (multimodal prefill
        // writes image tokens we can't reconstruct by tokenising text).
        List<int>? newCachedTokens;

        if (multimodal)
        {
            // Image turns: full prefill through mtmd_helper_eval_chunks. We
            // can't build a text-token list that matches what mtmd wrote into
            // the KV (image chunks have bespoke positions), so we clear the
            // cache and mark it null — next turn will start fresh.
            _context.ClearKvCache();
            _cachedTokens = null;
            newCachedTokens = null;

            // Decode image/audio bytes into native bitmaps on the background
            // thread that eval_chunks runs on — cheap for JPEGs, would be
            // O(seconds) for huge PNGs.
            foreach (var t in transcript)
            {
                if (t.Attachments is null) continue;
                foreach (var a in t.Attachments)
                {
                    if (!a.IsImage) continue;
                    bitmaps.Add(MtmdBitmap.FromBytes(_mtmd!, a.Data));
                }
            }

            try
            {
                await _mtmd!.EvalPromptAsync(
                    _context, prompt, bitmaps,
                    nPast: 0, seqId: 0,
                    nBatch: (int)_context.LogicalBatchSize,
                    logitsLast: true,
                    addSpecial: false, parseSpecial: true,
                    cancellationToken);
            }
            catch
            {
                foreach (var b in bitmaps) b.Dispose();
                throw;
            }

            stream = generator.StreamFromCurrentStateAsync(
                maxTokens: generation.MaxTokens,
                cancellationToken: cancellationToken);
        }
        else
        {
            // Text-only: tokenize the new prompt, diff against what's in the
            // KV cache, re-decode only the delta. Sampler still gets primed
            // with every prompt token so penalty/DRY state is correct.
            var promptTokens = _model.Vocab.Tokenize(prompt, addSpecial: false, parseSpecial: true);
            int firstNew = ComputeCommonPrefixLength(_cachedTokens, promptTokens);

            if (_cachedTokens is not null && firstNew < _cachedTokens.Count)
            {
                // Cache had a tail beyond the common prefix — trim it. If the
                // backend refuses partial removal (some SWA/quantised caches
                // do), fall back to a clean rebuild.
                var trimmed = _context.RemoveSequenceRange(
                    sequenceId: 0, fromPosition: firstNew, toPosition: -1);
                if (!trimmed)
                {
                    _context.ClearKvCache();
                    firstNew = 0;
                }
            }
            else if (_cachedTokens is null)
            {
                _context.ClearKvCache();
                firstNew = 0;
            }

            newCachedTokens = new List<int>(promptTokens.Length + generation.MaxTokens);
            newCachedTokens.AddRange(promptTokens);

            // Capture the local for the closure — newCachedTokens is non-null
            // in this branch and won't be reassigned.
            var cacheToGrow = newCachedTokens;
            stream = generator.GenerateAsync(
                promptTokens,
                maxTokens: generation.MaxTokens,
                firstNewIndex: firstNew,
                onTokenDecoded: t => cacheToGrow.Add(t),
                cancellationToken: cancellationToken);
        }

        try
        {
        await foreach (var piece in stream)
        {
            if (promptTime is null)
            {
                promptTime = TimeSpan.FromSeconds(
                    (sw.ElapsedTicks - promptStartTicks) / (double)Stopwatch.Frequency);
            }

            completionTokens++;

            if (extractor is not null)
            {
                var emit = extractor.Push(piece);
                if (emit.Content.Length > 0) yield return new StreamEvent.Content(emit.Content);
                if (emit.Reasoning.Length > 0) yield return new StreamEvent.Reasoning(emit.Reasoning);
            }
            else
            {
                yield return new StreamEvent.Content(piece);
            }
        }

        if (extractor is not null)
        {
            var tail = extractor.Flush();
            if (tail.Content.Length > 0) yield return new StreamEvent.Content(tail.Content);
            if (tail.Reasoning.Length > 0) yield return new StreamEvent.Reasoning(tail.Reasoning);
        }

        sw.Stop();
        var total = sw.Elapsed;
        var genTime = promptTime.HasValue ? total - promptTime.Value : total;

        yield return new StreamEvent.Done(completionTokens, promptTime ?? TimeSpan.Zero, genTime);
        }
        finally
        {
            // Commit the prompt-token cache so the next turn can prefix-diff
            // against it. Runs even on cancel/exception: whatever tokens the
            // generator did decode before bailing are now in the KV, so the
            // cache needs to reflect that — `newCachedTokens` grew via the
            // onTokenDecoded callback exactly in lockstep with DecodeSingleToken.
            _cachedTokens = newCachedTokens;
            foreach (var b in bitmaps) b.Dispose();
        }
    }

    /// <summary>
    /// Resume the most-recently generated assistant reply from its current
    /// KV state, without re-rendering or re-tokenising. Used to implement
    /// the "Continue" action when a reply hit <c>MaxTokens</c> (or the user
    /// cancelled mid-stream) and the user wants more output conditioned on
    /// exactly what came before. Throws if <see cref="ClearKv"/> has run
    /// since the last generation, because the required KV state is gone.
    /// </summary>
    /// <remarks>
    /// Implementation: trim the last cached token (which is also the token
    /// whose logits drove the current position) and re-decode it via
    /// <see cref="LlamaGenerator.GenerateAsync(IReadOnlyList{int}, int, bool, int, Action{int}?, CancellationToken)"/>'s
    /// back-off-by-one path. That refreshes the logits at the tail, primes
    /// the sampler with the full cached token history, and enters the sample
    /// loop. Identical emit/extractor/timing wiring as
    /// <see cref="StreamAssistantReplyAsync"/>.
    /// </remarks>
    public async IAsyncEnumerable<StreamEvent> StreamContinuationAsync(
        SamplerSettings sampler,
        GenerationSettings generation,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (_cachedTokens is null || _cachedTokens.Count == 0)
        {
            throw new InvalidOperationException(
                "Cannot continue: the KV cache has no prior generation to extend. " +
                "Run a normal turn first, or avoid calling ClearKv between the " +
                "last generation and continuation.");
        }

        using var llamaSampler = SamplerFactory.Build(_model, _model.Vocab, sampler);
        var generator = new LlamaGenerator(_context, llamaSampler);

        // Trim the last cached token from the KV so the back-off-by-one clause
        // in LlamaGenerator decodes it again (fresh logits) without
        // double-writing the position. If the backend refuses partial removal,
        // fall back to a full rebuild against the cached token list.
        int firstNew;
        if (_context.RemoveSequenceRange(0, fromPosition: _cachedTokens.Count - 1, toPosition: -1))
        {
            firstNew = _cachedTokens.Count - 1;
        }
        else
        {
            _context.ClearKvCache();
            firstNew = 0;
        }

        var cacheToGrow = new List<int>(_cachedTokens.Count + generation.MaxTokens);
        cacheToGrow.AddRange(_cachedTokens);

        var stream = generator.GenerateAsync(
            _cachedTokens,
            maxTokens: generation.MaxTokens,
            firstNewIndex: firstNew,
            onTokenDecoded: t => cacheToGrow.Add(t),
            cancellationToken: cancellationToken);

        // Reasoning extractor: only relevant if the reply we're continuing
        // was itself inside a <think> block. The caller already knows the
        // current content; if the model emits </think> during continuation
        // we'll handle the transition naturally. Start in Content mode — the
        // common case is "continue a truncated regular answer."
        var extractor = generation.ExtractReasoning
            ? new ReasoningExtractor(startInReasoning: false)
            : null;

        var sw = Stopwatch.StartNew();
        var promptStartTicks = sw.ElapsedTicks;
        TimeSpan? promptTime = null;
        int completionTokens = 0;

        try
        {
            await foreach (var piece in stream)
            {
                if (promptTime is null)
                {
                    promptTime = TimeSpan.FromSeconds(
                        (sw.ElapsedTicks - promptStartTicks) / (double)Stopwatch.Frequency);
                }

                completionTokens++;

                if (extractor is not null)
                {
                    var emit = extractor.Push(piece);
                    if (emit.Content.Length > 0) yield return new StreamEvent.Content(emit.Content);
                    if (emit.Reasoning.Length > 0) yield return new StreamEvent.Reasoning(emit.Reasoning);
                }
                else
                {
                    yield return new StreamEvent.Content(piece);
                }
            }

            if (extractor is not null)
            {
                var tail = extractor.Flush();
                if (tail.Content.Length > 0) yield return new StreamEvent.Content(tail.Content);
                if (tail.Reasoning.Length > 0) yield return new StreamEvent.Reasoning(tail.Reasoning);
            }

            sw.Stop();
            var total = sw.Elapsed;
            var genTime = promptTime.HasValue ? total - promptTime.Value : total;
            yield return new StreamEvent.Done(completionTokens, promptTime ?? TimeSpan.Zero, genTime);
        }
        finally
        {
            _cachedTokens = cacheToGrow;
        }
    }

    /// <summary>
    /// Apply the chat template. If <paramref name="mediaMarker"/> is non-null
    /// and a turn has attachments, prepends one marker per attachment to that
    /// turn's content — mtmd_tokenize scans the rendered prompt for the marker
    /// and splices the corresponding image chunk into the token stream.
    /// </summary>
    private string RenderPromptForCompletion(IReadOnlyList<ChatTurn> transcript, string? mediaMarker)
    {
        string Expand(ChatTurn t)
        {
            if (mediaMarker is null || t.Attachments is not { Count: > 0 }) return t.Content;
            var sb = new StringBuilder();
            for (int i = 0; i < t.Attachments.Count; i++)
            {
                sb.Append(mediaMarker).Append('\n');
            }
            sb.Append(t.Content);
            return sb.ToString();
        }

        if (string.IsNullOrEmpty(_chatTemplate))
        {
            // Model has no embedded template. Fall back to naked role-prefixed concat.
            var sb = new StringBuilder();
            foreach (var t in transcript)
            {
                sb.Append(RoleLabel(t.Role)).Append(": ").AppendLine(Expand(t));
            }
            sb.Append("assistant: ");
            return sb.ToString();
        }

        var messages = transcript
            .Select(t => new ChatMessage(RoleLabel(t.Role), Expand(t)))
            .ToArray();
        return LlamaChatTemplate.Apply(_chatTemplate, messages, addAssistantPrefix: true);
    }

    /// <summary>
    /// Length of the longest prefix on which both sequences agree token-for-
    /// token. Returns 0 when either side is null/empty. Called by the text-
    /// only generation path to figure out how much of the cached prompt can
    /// be reused for the next turn.
    /// </summary>
    private static int ComputeCommonPrefixLength(IReadOnlyList<int>? cached, IReadOnlyList<int> incoming)
    {
        if (cached is null || cached.Count == 0 || incoming.Count == 0) return 0;
        int max = Math.Min(cached.Count, incoming.Count);
        int i = 0;
        while (i < max && cached[i] == incoming[i]) i++;
        return i;
    }

    private static string RoleLabel(TurnRole role) => role switch
    {
        TurnRole.System => "system",
        TurnRole.User => "user",
        TurnRole.Assistant => "assistant",
        _ => "user",
    };

    public void Dispose()
    {
        _mtmd?.Dispose();
        _context.Dispose();
        _model.Dispose();
    }
}

public abstract record StreamEvent
{
    public sealed record Content(string Text) : StreamEvent;
    public sealed record Reasoning(string Text) : StreamEvent;
    public sealed record Done(int CompletionTokens, TimeSpan PromptTime, TimeSpan GenerationTime) : StreamEvent;
}
