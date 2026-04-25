using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.Services.ToolCall;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Backend-neutral chat session. Implementations: <see cref="LocalChatSession"/>
/// (in-process llama.cpp) and <see cref="RemoteChatSession"/> (HTTP /v1/chat/completions).
/// The view-model holds <c>IChatSession?</c> and dispatches generation through
/// <see cref="StreamAssistantReplyAsync"/>; both implementations emit the same
/// <see cref="StreamEvent"/> shapes so the UI rendering path is identical.
/// </summary>
public interface IChatSession : IDisposable
{
    bool SupportsImages { get; }
    bool SupportsAudio { get; }
    bool SupportsMedia { get; }
    bool CanGenerateTitles { get; }

    /// <summary>
    /// Wire format for tool-call parsing. Local sessions sniff this from the
    /// model's embedded chat template; remote sessions return null because the
    /// server applies the template — the local-side tool-call extractor isn't
    /// applicable.
    /// </summary>
    IToolCallFormat? ToolCallFormat { get; }

    /// <summary>
    /// The model's embedded chat template (Jinja). Null for remote (server-side).
    /// </summary>
    string? ChatTemplate { get; }

    /// <summary>
    /// Display name for the loaded model. Local: GGUF basename. Remote: the
    /// server-side model id.
    /// </summary>
    string DisplayModelName { get; }

    /// <summary>
    /// Approximate token count for <paramref name="prompt"/> using the local
    /// vocab. Returns null when the session can't tokenize client-side
    /// (remote backend) — callers should hide token-count UI in that case.
    /// </summary>
    int? EstimatePromptTokens(string prompt);

    /// <summary>
    /// Drop the KV cache. No-op for remote sessions (server-managed cache;
    /// per-request <c>cache_prompt</c> handles reuse).
    /// </summary>
    void ClearKv();

    Task<string?> GenerateTitleAsync(string userMessage, CancellationToken cancellationToken = default);

    IAsyncEnumerable<StreamEvent> StreamAssistantReplyAsync(
        IReadOnlyList<ChatTurn> transcript,
        SamplerSettings sampler,
        GenerationSettings generation,
        IReadOnlyList<object?>? tools = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Resume the most-recently generated assistant reply from current KV
    /// state without re-rendering the transcript. Throws
    /// <see cref="NotSupportedException"/> on remote sessions — server-side
    /// continuation isn't part of OpenAI's contract; callers should gate the
    /// Continue button on <c>session is LocalChatSession</c>.
    /// </summary>
    IAsyncEnumerable<StreamEvent> StreamContinuationAsync(
        SamplerSettings sampler,
        GenerationSettings generation,
        bool resumeInReasoning = false,
        CancellationToken cancellationToken = default);
}
