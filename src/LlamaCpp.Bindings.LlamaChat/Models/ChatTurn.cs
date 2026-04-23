using System;
using System.Collections.Generic;

namespace LlamaCpp.Bindings.LlamaChat.Models;

public enum TurnRole { System, User, Assistant, Tool }

public enum TurnState { Pending, Streaming, Complete, Cancelled, Errored }

public sealed record ChatTurn(
    Guid Id,
    TurnRole Role,
    string Content,
    TurnState State,
    DateTimeOffset CreatedAt,
    string? Reasoning = null,
    TurnStats? Stats = null,
    List<Attachment>? Attachments = null,
    /// <summary>
    /// Id of the preceding turn in the conversation tree. Null for the
    /// first turn. Turns sharing a parent are siblings — alternative
    /// branches (retries, edits) the user can switch between.
    /// </summary>
    Guid? ParentId = null)
{
    public static ChatTurn NewUser(string content) =>
        new(Guid.NewGuid(), TurnRole.User, content, TurnState.Complete, DateTimeOffset.UtcNow);

    public static ChatTurn NewAssistantPending() =>
        new(Guid.NewGuid(), TurnRole.Assistant, string.Empty, TurnState.Pending, DateTimeOffset.UtcNow);
}

public sealed record TurnStats(
    int PromptTokens,
    int CompletionTokens,
    TimeSpan PromptTime,
    TimeSpan GenerationTime)
{
    public double TokensPerSecond =>
        GenerationTime.TotalSeconds > 0 ? CompletionTokens / GenerationTime.TotalSeconds : 0;
}
