using System;
using System.Collections.Generic;
using LlamaCpp.Bindings.LlamaChat.Services.ToolCall;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Thin entry point that dispatches to the right <see cref="IToolCallFormat"/>
/// in the <see cref="ToolCallFormatRegistry"/>. Callers that know the
/// template's format (<see cref="ChatSession.ToolCallFormat"/>) pass it via
/// <paramref name="preferred"/> to skip the fallback search; callers that
/// don't let the registry try each entry in turn.
/// </summary>
internal static class ToolCallParser
{
    /// <summary>
    /// Alias of <see cref="ToolCall.ParsedToolCall"/> so pre-existing call
    /// sites (<c>MainWindowViewModel</c>) keep compiling after the split.
    /// </summary>
    public sealed record ParsedToolCall(string Name, System.Text.Json.JsonElement Arguments, string RawMatch);

    /// <summary>
    /// Extract every tool call the known formats can find in
    /// <paramref name="text"/>. Empty list on no matches or empty input.
    /// </summary>
    public static IReadOnlyList<ParsedToolCall> Extract(string text, IToolCallFormat? preferred = null)
    {
        if (string.IsNullOrEmpty(text)) return Array.Empty<ParsedToolCall>();

        // Try the template-preferred format first — it's almost always the
        // right one and saves iterating the fallbacks.
        if (preferred is not null && preferred.Contains(text))
        {
            return Wrap(preferred.Extract(text));
        }

        // Fallback: try each known format in registry order. First one that
        // yields at least one call wins; order matters only if a model's
        // output happens to match multiple formats (rare).
        foreach (var f in ToolCallFormatRegistry.All)
        {
            if (f.Contains(text))
            {
                var result = f.Extract(text);
                if (result.Count > 0) return Wrap(result);
            }
        }
        return Array.Empty<ParsedToolCall>();
    }

    /// <summary>Convenience — true if the text looks like it contains any known format.</summary>
    public static bool Contains(string text, IToolCallFormat? preferred = null)
    {
        if (string.IsNullOrEmpty(text)) return false;
        if (preferred is not null) return preferred.Contains(text);
        foreach (var f in ToolCallFormatRegistry.All)
        {
            if (f.Contains(text)) return true;
        }
        return false;
    }

    private static IReadOnlyList<ParsedToolCall> Wrap(IReadOnlyList<ToolCall.ParsedToolCall> inner)
    {
        var list = new List<ParsedToolCall>(inner.Count);
        foreach (var c in inner)
        {
            list.Add(new ParsedToolCall(c.Name, c.Arguments, c.RawMatch));
        }
        return list;
    }
}
