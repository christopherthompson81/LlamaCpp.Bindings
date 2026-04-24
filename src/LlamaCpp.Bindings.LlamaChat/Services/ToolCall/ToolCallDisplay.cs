using System.Collections.Generic;
using System.Text;
using System.Text.Json;

namespace LlamaCpp.Bindings.LlamaChat.Services.ToolCall;

/// <summary>
/// Display-layer helpers for turning raw tool-call markup and raw MCP
/// responses into something the UI can render cleanly. Pure functions —
/// no state, no side effects.
/// </summary>
public static class ToolCallDisplay
{
    /// <summary>
    /// One compact representation of an assistant-emitted tool call, ready
    /// for a chip in the bubble UI. <see cref="RawMatch"/> is the exact
    /// markup the model emitted — surfaced to the user as the chip's
    /// tooltip so the verbose format is available on hover without
    /// cluttering the transcript.
    /// </summary>
    public sealed record Chip(string Name, string ArgsDisplay, string RawMatch);

    /// <summary>
    /// Remove every tool-call block the known formats can find from
    /// <paramref name="content"/>. Leaves the surrounding prose intact so
    /// the bubble's markdown still renders naturally.
    /// </summary>
    /// <remarks>
    /// The RawMatch strings returned by <see cref="ToolCallParser"/> are the
    /// exact substrings matched by each format's wrapper regex, so simple
    /// string <c>Replace</c> is safe — no re-parsing.
    /// </remarks>
    public static string StripMarkup(string content)
    {
        if (string.IsNullOrEmpty(content)) return content;
        var calls = ToolCallParser.Extract(content);
        if (calls.Count == 0) return content;
        var result = content;
        foreach (var c in calls)
        {
            result = result.Replace(c.RawMatch, string.Empty);
        }
        return result.Trim();
    }

    /// <summary>
    /// Extract every tool call from <paramref name="content"/> as
    /// <see cref="Chip"/>s suitable for display. Argument objects are
    /// rendered as compact <c>k=v, k=v</c> strings; non-object argument
    /// payloads fall back to their raw JSON form.
    /// </summary>
    public static IReadOnlyList<Chip> ExtractChips(string content)
    {
        var calls = ToolCallParser.Extract(content);
        if (calls.Count == 0) return System.Array.Empty<Chip>();
        var chips = new List<Chip>(calls.Count);
        foreach (var c in calls)
        {
            chips.Add(new Chip(c.Name, FormatArgs(c.Arguments), c.RawMatch));
        }
        return chips;
    }

    private static string FormatArgs(JsonElement args)
    {
        if (args.ValueKind != JsonValueKind.Object) return args.GetRawText();
        var sb = new StringBuilder();
        bool first = true;
        foreach (var prop in args.EnumerateObject())
        {
            if (!first) sb.Append(", ");
            first = false;
            sb.Append(prop.Name).Append('=');
            // Unquote string values so "hello" reads as hello — easier on
            // the eye and matches the "call like a function" mental model.
            if (prop.Value.ValueKind == JsonValueKind.String)
            {
                sb.Append(prop.Value.GetString());
            }
            else
            {
                sb.Append(prop.Value.GetRawText());
            }
        }
        return sb.ToString();
    }

    /// <summary>
    /// Reduce an MCP <c>tools/call</c> response to its displayable text. MCP
    /// returns <c>{ content: [{ type: "text", text: "..." }, ...] }</c>; we
    /// concatenate all text blocks. Non-text content (blobs, resource refs)
    /// returns a placeholder since the chat UI can't render raw bytes. When
    /// the shape doesn't match at all we fall back to the raw JSON so at
    /// least the user can see what came back.
    /// </summary>
    /// <summary>
    /// True if the MCP response carries <c>isError: true</c>. Mirrors the
    /// check inside <see cref="FormatMcpResult"/> but exposed separately so
    /// callers can tag the tool-message bubble with a destructive state.
    /// </summary>
    public static bool IsErrorResult(JsonElement result) =>
        result.ValueKind == JsonValueKind.Object
        && result.TryGetProperty("isError", out var err)
        && err.ValueKind == JsonValueKind.True;

    public static string FormatMcpResult(JsonElement result)
    {
        if (result.ValueKind == JsonValueKind.Object
            && result.TryGetProperty("content", out var content)
            && content.ValueKind == JsonValueKind.Array)
        {
            var sb = new StringBuilder();
            foreach (var item in content.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.Object) continue;
                var type = item.TryGetProperty("type", out var t) ? t.GetString() : null;
                if (type == "text" && item.TryGetProperty("text", out var txt))
                {
                    if (sb.Length > 0) sb.Append("\n\n");
                    sb.Append(txt.GetString() ?? string.Empty);
                }
                else if (type == "image")
                {
                    if (sb.Length > 0) sb.Append("\n\n");
                    sb.Append("(image content)");
                }
                else if (type == "resource")
                {
                    if (sb.Length > 0) sb.Append("\n\n");
                    sb.Append("(resource reference)");
                }
            }
            if (sb.Length > 0)
            {
                // Surface error flag if the server set isError=true — the
                // content text tends to be descriptive in that case.
                if (result.TryGetProperty("isError", out var err)
                    && err.ValueKind == JsonValueKind.True)
                {
                    return "⚠ " + sb.ToString();
                }
                return sb.ToString();
            }
        }
        return result.GetRawText();
    }
}
