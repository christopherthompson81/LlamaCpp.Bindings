using System;
using System.Collections.Generic;

namespace LlamaCpp.Bindings.LlamaChat.Models;

/// <summary>
/// A single chat thread: title, timestamps, and the ordered transcript.
/// Persisted as JSON by <see cref="Services.ConversationStore"/>. Turn
/// content is stored verbatim (no re-tokenization on reload).
/// </summary>
public sealed record Conversation
{
    public Guid Id { get; init; } = Guid.NewGuid();
    public string Title { get; init; } = "New chat";
    public DateTimeOffset CreatedAt { get; init; } = DateTimeOffset.UtcNow;
    public DateTimeOffset UpdatedAt { get; init; } = DateTimeOffset.UtcNow;
    public List<ChatTurn> Turns { get; init; } = new();
}
