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
/// One loaded model + context bound to a single conversation transcript.
/// Not thread-safe: the UI marshals all send/cancel operations serially.
/// Disposal releases the context and model.
/// </summary>
public sealed class ChatSession : IDisposable
{
    private readonly LlamaModel _model;
    private readonly LlamaContext _context;
    private readonly string? _chatTemplate;

    private readonly List<ChatTurn> _turns = new();

    public IReadOnlyList<ChatTurn> Turns => _turns;
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

    public void AppendUser(string text) =>
        _turns.Add(ChatTurn.NewUser(text));

    public void Reset()
    {
        _turns.Clear();
        _context.ClearKvCache();
    }

    /// <summary>
    /// Generate an assistant reply to the current transcript. Yields stream
    /// events in real time; the caller is responsible for updating the visible
    /// <see cref="ChatTurn"/> (which the session does not track as "live"; call
    /// <see cref="Commit"/> once the stream ends).
    /// </summary>
    public async IAsyncEnumerable<StreamEvent> StreamAssistantTurnAsync(
        SamplerSettings sampler,
        GenerationSettings generation,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var prompt = RenderPromptForCompletion();

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

        var aggregateContent = new StringBuilder();
        var aggregateReasoning = new StringBuilder();

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
                if (emit.Content.Length > 0)
                {
                    aggregateContent.Append(emit.Content);
                    yield return new StreamEvent.Content(emit.Content);
                }
                if (emit.Reasoning.Length > 0)
                {
                    aggregateReasoning.Append(emit.Reasoning);
                    yield return new StreamEvent.Reasoning(emit.Reasoning);
                }
            }
            else
            {
                aggregateContent.Append(piece);
                yield return new StreamEvent.Content(piece);
            }
        }

        if (extractor is not null)
        {
            var tail = extractor.Flush();
            if (tail.Content.Length > 0)
            {
                aggregateContent.Append(tail.Content);
                yield return new StreamEvent.Content(tail.Content);
            }
            if (tail.Reasoning.Length > 0)
            {
                aggregateReasoning.Append(tail.Reasoning);
                yield return new StreamEvent.Reasoning(tail.Reasoning);
            }
        }

        sw.Stop();
        var total = sw.Elapsed;
        var genTime = promptTime.HasValue ? total - promptTime.Value : total;

        // Store the committed turn on the session. TurnStats promptTokens is
        // unknown here without re-tokenizing the prompt; leave as 0 until
        // LlamaGenerator surfaces that count.
        _turns.Add(new ChatTurn(
            Id: Guid.NewGuid(),
            Role: TurnRole.Assistant,
            Content: aggregateContent.ToString(),
            State: TurnState.Complete,
            CreatedAt: DateTimeOffset.UtcNow,
            Reasoning: aggregateReasoning.Length > 0 ? aggregateReasoning.ToString() : null,
            Stats: new TurnStats(0, completionTokens, promptTime ?? TimeSpan.Zero, genTime)));

        yield return new StreamEvent.Done(completionTokens, promptTime ?? TimeSpan.Zero, genTime);
    }

    private string RenderPromptForCompletion()
    {
        if (string.IsNullOrEmpty(_chatTemplate))
        {
            // Model has no embedded template. Fall back to naked role-prefixed concat.
            // Matches samples/LlamaChat's fallback behavior.
            var sb = new StringBuilder();
            foreach (var t in _turns)
            {
                sb.Append(RoleLabel(t.Role)).Append(": ").AppendLine(t.Content);
            }
            sb.Append("assistant: ");
            return sb.ToString();
        }

        var messages = _turns
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
