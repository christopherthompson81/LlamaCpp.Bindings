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
    private readonly string? _chatTemplate;

    public LlamaModel Model => _model;
    public LlamaContext Context => _context;
    public string? ChatTemplate => _chatTemplate;

    private ChatSession(LlamaModel model, LlamaContext context, string? template)
    {
        _model = model;
        _context = context;
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
        }
        catch
        {
            model.Dispose();
            throw;
        }

        return new ChatSession(model, context, model.GetChatTemplate());
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
        var prompt = RenderPromptForCompletion(transcript);

        // v1 cache policy: clear and re-decode the full transcript every turn.
        // O(total tokens) per turn, correct, trivially debuggable. Prefix-cache
        // reuse is tracked as Run-2 work in docs/webui_parity_investigation.md.
        _context.ClearKvCache();

        using var llamaSampler = SamplerFactory.Build(_model, _model.Vocab, sampler);
        var generator = new LlamaGenerator(_context, llamaSampler);

        var extractor = generation.ExtractReasoning ? new ReasoningExtractor() : null;

        var sw = Stopwatch.StartNew();
        var promptStartTicks = sw.ElapsedTicks;
        TimeSpan? promptTime = null;
        int completionTokens = 0;

        var stream = generator.GenerateAsync(
            prompt,
            maxTokens: generation.MaxTokens,
            addSpecial: false,
            parseSpecial: true,
            cancellationToken: cancellationToken);

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

    private string RenderPromptForCompletion(IReadOnlyList<ChatTurn> transcript)
    {
        if (string.IsNullOrEmpty(_chatTemplate))
        {
            // Model has no embedded template. Fall back to naked role-prefixed concat.
            var sb = new StringBuilder();
            foreach (var t in transcript)
            {
                sb.Append(RoleLabel(t.Role)).Append(": ").AppendLine(t.Content);
            }
            sb.Append("assistant: ");
            return sb.ToString();
        }

        var messages = transcript
            .Select(t => new ChatMessage(RoleLabel(t.Role), t.Content))
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
