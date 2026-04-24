using System;
using System.Collections.Generic;
using System.Linq;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services.Exporters;

/// <summary>
/// Helpers that turn the branch-aware <see cref="Conversation"/> tree into
/// the flat ordered transcript that exports actually want. Each exporter
/// serialises the <b>active branch</b> — the path the user sees, root to
/// leaf — rather than every sibling. Siblings are alternative replies and
/// don't belong in a shareable transcript.
/// </summary>
internal static class ConversationBranch
{
    /// <summary>
    /// Walk the active branch from root to leaf. Empty on a conversation
    /// with no active leaf (e.g. freshly-created). Gracefully handles
    /// tree corruption (missing parent id) by stopping the walk.
    /// </summary>
    public static IReadOnlyList<ChatTurn> ActivePath(Conversation conversation)
    {
        if (conversation.ActiveLeafId is null || conversation.Turns.Count == 0)
            return Array.Empty<ChatTurn>();

        var byId = conversation.Turns.ToDictionary(t => t.Id);
        var stack = new Stack<ChatTurn>();
        var cursor = conversation.ActiveLeafId;
        var safety = conversation.Turns.Count + 1;
        while (cursor is not null && safety-- > 0)
        {
            if (!byId.TryGetValue(cursor.Value, out var turn)) break;
            stack.Push(turn);
            cursor = turn.ParentId;
        }
        return stack.ToList();
    }

    /// <summary>
    /// Convenience: format a role label for display (e.g. "User", "Assistant").
    /// </summary>
    public static string RoleLabel(TurnRole role) => role switch
    {
        TurnRole.System    => "System",
        TurnRole.User      => "User",
        TurnRole.Assistant => "Assistant",
        TurnRole.Tool      => "Tool",
        _                  => role.ToString(),
    };
}
