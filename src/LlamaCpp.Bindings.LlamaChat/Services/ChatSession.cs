using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.Services.ToolCall;

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

    /// <summary>
    /// The tool-call wire format this model's chat template expects, sniffed
    /// once at load. Null when the model has no embedded template, or when
    /// the template doesn't match any known format — consumers treat null as
    /// "no tool use" and skip the parse-result loop.
    /// </summary>
    public IToolCallFormat? ToolCallFormat { get; }

    /// <summary>True if this session can accept image attachments on user turns.</summary>
    public bool SupportsImages => _mtmd?.SupportsVision == true;

    /// <summary>True if this session can accept audio attachments on user turns.</summary>
    public bool SupportsAudio => _mtmd?.SupportsAudio == true;

    /// <summary>True if this session can accept any mtmd media (image or audio).</summary>
    public bool SupportsMedia => SupportsImages || SupportsAudio;

    /// <summary>
    /// Coarse "is this model useful for general text generation?" check used
    /// by the auto-title and regenerate-title flows to gate out ASR-only
    /// models (Qwen3-ASR etc.), which accept audio in but aren't trained for
    /// free-form text prompting and produce garbage on a "summarise this
    /// message in 3-6 words" query. Heuristic: accept everything except
    /// audio-in-only models (audio capability but no vision — indicative of
    /// ASR / speech-to-text task specialisation). Omni models (audio + image)
    /// and plain text / vision models are all considered capable.
    /// </summary>
    public bool CanGenerateTitles => !(SupportsAudio && !SupportsImages);

    private ChatSession(LlamaModel model, LlamaContext context, MtmdContext? mtmd, string? template)
    {
        _model = model;
        _context = context;
        _mtmd = mtmd;
        _chatTemplate = template;
        ToolCallFormat = ToolCallFormatRegistry.DetectFromTemplate(template);
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
                KvCacheTypeK = settings.KvCacheTypeK,
                KvCacheTypeV = settings.KvCacheTypeV,
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
    /// Generate a concise (3–6 word) title summarising <paramref name="userMessage"/>,
    /// for the sidebar "auto-title new conversations" + right-click "Regenerate
    /// title" flows. Returns the cleaned title on success, or <c>null</c> if the
    /// model can't do text generation (<see cref="CanGenerateTitles"/>) or the
    /// output was empty after trimming.
    /// </summary>
    /// <remarks>
    /// Uses the main session's context and KV cache — tolerates the collateral
    /// cache wipe this entails (call <see cref="ClearKv"/> before + after) since
    /// title gen happens at most once per conversation and always either right
    /// after an assistant reply that's already committed its tokens to the tree
    /// or from the user's explicit Regenerate Title action. The next real turn
    /// prefills from scratch; acceptable one-off cost.
    /// </remarks>
    public async Task<string?> GenerateTitleAsync(string userMessage, CancellationToken cancellationToken = default)
    {
        if (!CanGenerateTitles) return null;
        if (string.IsNullOrWhiteSpace(userMessage)) return null;

        var transcript = new List<ChatTurn>
        {
            new(Guid.NewGuid(), TurnRole.System,
                "You generate concise conversation titles. Reply with a 3-6 word title that captures the user's topic. Reply with ONLY the title text — no quotes, no trailing punctuation, no \"Title:\" prefix.",
                TurnState.Complete, DateTimeOffset.UtcNow),
            new(Guid.NewGuid(), TurnRole.User, userMessage,
                TurnState.Complete, DateTimeOffset.UtcNow),
        };

        var prompt = RenderPromptForCompletion(transcript, mediaMarker: null, tools: null);

        // Qwen3 / DeepSeek-R1 chat templates inject an opening <think> tag
        // into the assistant-turn prefix so the model starts generating
        // inside a reasoning block. For a one-off title we don't want a
        // reasoning pass — just close the tag ourselves so the model
        // begins in content mode immediately. Fast, deterministic, and
        // works regardless of whether the template recognises /no_think.
        var promptEndsInThink = System.Text.RegularExpressions.Regex.IsMatch(
            prompt, @"<think>\s*$");
        if (promptEndsInThink)
        {
            prompt = System.Text.RegularExpressions.Regex.Replace(
                prompt, @"<think>\s*$", "<think></think>\n");
        }

        var promptTokens = _model.Vocab.Tokenize(prompt, addSpecial: false, parseSpecial: true);

        // Clear cache before and after so neither the title prompt nor any
        // leftover state from previous turns pollute what comes next.
        _context.ClearKvCache();
        _cachedTokens = null;

        var sampler = SamplerFactory.Build(_model, _model.Vocab, new SamplerSettings
        {
            Temperature = 0.3f,
            TopK = 40,
            TopP = 0.9f,
        });
        var generator = new LlamaGenerator(_context, sampler);

        // With the <think> tag pre-closed above, the model starts in
        // content mode; the extractor is still a belt-and-braces guard
        // in case a non-Qwen template opens its own thinking block mid-
        // response, but default-Content-mode is the common case now.
        var extractor = new ReasoningExtractor();
        var rawSb = new StringBuilder();
        var contentSb = new StringBuilder();
        var reasoningSb = new StringBuilder();
        Exception? captured = null;
        string? cleaned = null;
        try
        {
            await foreach (var piece in generator.GenerateAsync(
                promptTokens, maxTokens: 32, firstNewIndex: 0,
                cancellationToken: cancellationToken).ConfigureAwait(false))
            {
                rawSb.Append(piece);
                var em = extractor.Push(piece);
                if (em.Reasoning.Length > 0) reasoningSb.Append(em.Reasoning);
                if (em.Content.Length > 0)
                {
                    contentSb.Append(em.Content);
                    // Content-side newline means the title is complete.
                    if (contentSb.ToString().Contains('\n')) break;
                }
            }
            var tail = extractor.Flush();
            if (tail.Reasoning.Length > 0) reasoningSb.Append(tail.Reasoning);
            if (tail.Content.Length > 0) contentSb.Append(tail.Content);
            cleaned = CleanTitle(contentSb.ToString());
        }
        catch (Exception ex)
        {
            captured = ex;
        }
        finally
        {
            sampler.Dispose();
            _context.ClearKvCache();
            _cachedTokens = null;
        }

        // Always write the log — even if generation threw — so the file
        // trace reflects what actually happened. The "promptEndsInThink"
        // + "/no_think appended" markers make it obvious which code path
        // is live.
        TitleGenLog.Write(
            prompt, rawSb.ToString(), reasoningSb.ToString(), contentSb.ToString(),
            cleaned, promptEndsInThink, captured);

        if (captured is not null) throw captured;
        return cleaned;
    }

    private static string? CleanTitle(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return null;
        // First line only.
        var line = raw.Split('\n')[0].Trim();
        // Strip surrounding quotes, "Title:" prefixes the model sometimes
        // emits despite the instruction, and trailing punctuation.
        line = line.Trim().Trim('"').Trim('\'').Trim();
        if (line.StartsWith("Title:", StringComparison.OrdinalIgnoreCase))
            line = line[6..].Trim();
        line = line.TrimEnd('.', '!', '?', ',', ';', ' ');
        if (line.Length == 0) return null;
        if (line.Length > 60) line = line[..60].TrimEnd() + "…";
        return line;
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
        IReadOnlyList<object?>? tools = null,
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
        // mtmd routes both images and audio through the same tokenize +
        // eval_chunks pipeline. Gate on either modality so audio-only models
        // (Qwen3-ASR) and omni models (Qwen3-Omni) both take this branch.
        bool multimodal = hasAttachments
            && _mtmd is not null
            && (_mtmd.SupportsVision || _mtmd.SupportsAudio);

        var prompt = RenderPromptForCompletion(transcript, multimodal ? _mtmd!.DefaultMediaMarker : null, tools);

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
        var asrExtractor = generation.ExtractAsrTranscript
            ? new AsrTextExtractor()
            : null;

        var sw = Stopwatch.StartNew();
        var promptStartTicks = sw.ElapsedTicks;
        TimeSpan? promptTime = null;
        int completionTokens = 0;
        // Snapshot perf before this request so we can report the prefill
        // token count for THIS turn (PromptTokenCount is cumulative across
        // the context's lifetime; we need the delta).
        var perfBefore = _context.GetPerformance();
        int promptEvalTokens = 0;

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
            // thread that eval_chunks runs on. FromBytes routes through
            // mtmd_helper_bitmap_init_from_buf which auto-detects format
            // (stb_image for images, miniaudio for audio) via magic bytes,
            // so we just need to forward everything that's image-or-audio.
            foreach (var t in transcript)
            {
                if (t.Attachments is null) continue;
                foreach (var a in t.Attachments)
                {
                    if (!a.IsImage && !a.IsAudio) continue;
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

            // Text path knows the exact prefill batch size (prompt length
            // minus the common-prefix cache overlap). Use it directly — the
            // perf-counter diff read at first-token time can show 0 because
            // n_p_eval updates lag the prefill-complete moment on some
            // backends / small batches.
            promptEvalTokens = promptTokens.Length - firstNew;

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

            // Piece flows: raw → reasoning extractor → asr extractor → UI.
            // Either extractor may be null; in that case its stage is a
            // no-op pass-through.
            string contentSlice;
            string? reasoningSlice;
            if (extractor is not null)
            {
                var re = extractor.Push(piece);
                contentSlice = re.Content;
                reasoningSlice = re.Reasoning.Length > 0 ? re.Reasoning : null;
            }
            else
            {
                contentSlice = piece;
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
        var total = sw.Elapsed;
        var genTime = promptTime.HasValue ? total - promptTime.Value : total;

        yield return new StreamEvent.Done(promptEvalTokens, promptTime ?? TimeSpan.Zero, completionTokens, genTime, generator.LastStopReason);
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
    /// exactly what came before. Throws if the KV cache is empty — that's
    /// the only state we truly can't extend from.
    /// </summary>
    /// <remarks>
    /// Two paths:
    /// <list type="bullet">
    ///   <item>
    ///     <c>_cachedTokens</c> is populated (normal text-only turn) — trim
    ///     the last cached token, re-decode it via
    ///     <see cref="LlamaGenerator.GenerateAsync(IReadOnlyList{int}, int, bool, int, Action{int}?, CancellationToken)"/>'s
    ///     back-off-by-one path, and prime the sampler with the full cached
    ///     history. This is the "correct" path: penalty/DRY state sees the
    ///     whole prior transcript.
    ///   </item>
    ///   <item>
    ///     <c>_cachedTokens</c> is null (post-multimodal turn — we can't
    ///     record image-chunk positions as plain token IDs) — sample
    ///     directly from the logits the last decode left at the tail, no
    ///     sampler priming. Penalty/DRY state starts empty for this
    ///     continuation; acceptable because continuations are usually short.
    ///     The cache stays null afterward, matching the current post-multimodal
    ///     invariant.
    ///   </item>
    /// </list>
    /// Emit/extractor/timing wiring is identical to
    /// <see cref="StreamAssistantReplyAsync"/>.
    /// </remarks>
    public async IAsyncEnumerable<StreamEvent> StreamContinuationAsync(
        SamplerSettings sampler,
        GenerationSettings generation,
        bool resumeInReasoning = false,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // KV empty means nothing to continue from — genuinely can't proceed.
        var (_, maxPos) = _context.SequencePositionRange(0);
        if (maxPos is null)
        {
            throw new InvalidOperationException(
                "Cannot continue: the KV cache is empty. Run a turn first.");
        }

        using var llamaSampler = SamplerFactory.Build(_model, _model.Vocab, sampler);
        var generator = new LlamaGenerator(_context, llamaSampler);

        IAsyncEnumerable<string> stream;
        List<int>? cacheToGrow;

        if (_cachedTokens is { Count: > 0 } cached)
        {
            // Fast path: trim the last cached token so LlamaGenerator's
            // back-off-by-one clause decodes it again (fresh logits) without
            // double-writing the position.
            int firstNew;
            if (_context.RemoveSequenceRange(0, fromPosition: cached.Count - 1, toPosition: -1))
            {
                firstNew = cached.Count - 1;
            }
            else
            {
                // Backend refused partial trim — fall back to a full rebuild
                // against the cached token list.
                _context.ClearKvCache();
                firstNew = 0;
            }

            cacheToGrow = new List<int>(cached.Count + generation.MaxTokens);
            cacheToGrow.AddRange(cached);

            stream = generator.GenerateAsync(
                cached,
                maxTokens: generation.MaxTokens,
                firstNewIndex: firstNew,
                onTokenDecoded: t => cacheToGrow.Add(t),
                cancellationToken: cancellationToken);
        }
        else
        {
            // Fallback path (null cache, typically post-multimodal): sample
            // directly from whatever the last decode left. No trim, no
            // re-decode, no sampler priming.
            cacheToGrow = null;
            stream = generator.StreamFromCurrentStateAsync(
                maxTokens: generation.MaxTokens,
                cancellationToken: cancellationToken);
        }

        // Reasoning extractor: if the turn we're continuing was interrupted
        // mid-<think>, the first tokens the model emits will still be
        // reasoning and will include a literal </think>. Starting the
        // extractor in Content mode dumps </think> + a blank line into
        // Content, which CommonMark interprets as an HTML block and breaks
        // downstream markdown rendering (list bullets / bold show as raw
        // syntax). Callers detect the mid-reasoning condition from the
        // message's existing state (has reasoning text, no content yet).
        var extractor = generation.ExtractReasoning
            ? new ReasoningExtractor(startInReasoning: resumeInReasoning)
            : null;
        // Continuation on an ASR turn shouldn't re-parse the preamble —
        // the language chip was already emitted during the original turn,
        // and the continuation starts from wherever decoding left off
        // (typically mid-transcript, past the <asr_text> open tag). Skip
        // the ASR extractor here.
        AsrTextExtractor? asrExtractor = null;

        var sw = Stopwatch.StartNew();
        var promptStartTicks = sw.ElapsedTicks;
        TimeSpan? promptTime = null;
        int completionTokens = 0;
        // Snapshot perf before this request so we can report the prefill
        // token count for THIS turn (PromptTokenCount is cumulative across
        // the context's lifetime; we need the delta).
        var perfBefore = _context.GetPerformance();
        int promptEvalTokens = 0;

        try
        {
            await foreach (var piece in stream)
            {
                if (promptTime is null)
                {
                    promptTime = TimeSpan.FromSeconds(
                        (sw.ElapsedTicks - promptStartTicks) / (double)Stopwatch.Frequency);
                    // Fall back to the perf-counter diff when the text path
                    // didn't pre-populate (multimodal prefill + continuation
                    // paths). Best-effort for those; the text path has the
                    // authoritative count.
                    if (promptEvalTokens == 0)
                    {
                        promptEvalTokens = _context.GetPerformance().PromptTokenCount - perfBefore.PromptTokenCount;
                    }
                }

                completionTokens++;

                string contentSlice;
                if (extractor is not null)
                {
                    var re = extractor.Push(piece);
                    contentSlice = re.Content;
                    if (re.Reasoning.Length > 0) yield return new StreamEvent.Reasoning(re.Reasoning);
                }
                else
                {
                    contentSlice = piece;
                }

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

            if (extractor is not null)
            {
                var tail = extractor.Flush();
                if (tail.Reasoning.Length > 0) yield return new StreamEvent.Reasoning(tail.Reasoning);
                if (tail.Content.Length > 0) yield return new StreamEvent.Content(tail.Content);
            }

            sw.Stop();
            var total = sw.Elapsed;
            var genTime = promptTime.HasValue ? total - promptTime.Value : total;
            yield return new StreamEvent.Done(promptEvalTokens, promptTime ?? TimeSpan.Zero, completionTokens, genTime, generator.LastStopReason);
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
    private string RenderPromptForCompletion(
        IReadOnlyList<ChatTurn> transcript,
        string? mediaMarker,
        IReadOnlyList<object?>? tools = null)
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
        return LlamaChatTemplate.Apply(_chatTemplate, messages,
            addAssistantPrefix: true, tools: tools);
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
        TurnRole.Tool => "tool",
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
    /// <summary>
    /// Detected transcription language from an ASR model (Qwen3-ASR /
    /// Qwen3-Omni audio). Fires once per turn, when the extractor parses
    /// the <c>language &lt;LANG&gt;</c> preamble. Consumers surface it as a
    /// metadata chip next to the content.
    /// </summary>
    public sealed record Language(string Tag) : StreamEvent;
    /// <summary>
    /// End-of-stream marker carrying authoritative performance counters from
    /// <c>llama_context_perf</c>. <see cref="PromptTokens"/> + <see cref="PromptTime"/>
    /// describe the prefill phase; <see cref="CompletionTokens"/> +
    /// <see cref="GenerationTime"/> describe decode. Both pairs are what a
    /// bubble footer shows when the user toggles between prefill and decode
    /// stats views.
    /// </summary>
    /// <summary>
    /// End-of-stream marker carrying authoritative performance counters from
    /// <c>llama_context_perf</c>. <see cref="PromptTokens"/> + <see cref="PromptTime"/>
    /// describe the prefill phase ("Reading" in the UI); <see cref="CompletionTokens"/>
    /// + <see cref="GenerationTime"/> describe decode ("Generation" in the UI).
    /// </summary>
    public sealed record Done(
        int PromptTokens,
        TimeSpan PromptTime,
        int CompletionTokens,
        TimeSpan GenerationTime,
        LlamaStopReason StopReason) : StreamEvent;
}
