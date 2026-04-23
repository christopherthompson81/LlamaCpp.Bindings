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

    /// <summary>Drop the KV cache. Called when switching conversations.</summary>
    public void ClearKv() => _context.ClearKvCache();

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

        // v1 cache policy: clear and re-decode the full transcript every turn.
        // O(total tokens) per turn, correct, trivially debuggable. Prefix-cache
        // reuse is tracked as Run-2 work in docs/webui_parity_investigation.md.
        _context.ClearKvCache();

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

        if (multimodal)
        {
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
            stream = generator.GenerateAsync(
                prompt,
                maxTokens: generation.MaxTokens,
                addSpecial: false,
                parseSpecial: true,
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
            foreach (var b in bitmaps) b.Dispose();
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
