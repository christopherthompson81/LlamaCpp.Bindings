using System.Text.RegularExpressions;

namespace LlamaCpp.Bindings.LlamaChat.Services.Exporters;

/// <summary>
/// Strips model-emitted reasoning/thinking spans from message content
/// before an exporter serialises them. The in-app
/// <see cref="ReasoningExtractor"/> already routes these to a separate
/// channel when <c>GenerationSettings.ExtractReasoning</c> is on, but
/// older turns, models with non-standard tags, or generations run with
/// extraction disabled can still carry inline <c>&lt;think&gt;…&lt;/think&gt;</c>
/// spans in their content. Exports should never leak that noise into
/// shareable documents regardless of how the generation was configured.
/// </summary>
public static class ContentSanitizer
{
    // Covers <think>, <thinking>, <reasoning>, <reflection>, and
    // attribute-carrying variants (e.g. <think id="...">). Single-line
    // mode so . matches newlines; non-greedy so adjacent blocks don't
    // collapse into a single match.
    private static readonly Regex _thinkSpans = new(
        @"<\s*(think|thinking|reasoning|reflection)(\s[^>]*)?>.*?<\s*/\s*\1\s*>",
        RegexOptions.Compiled | RegexOptions.Singleline | RegexOptions.IgnoreCase);

    // Leftover un-closed opens — if the stream stopped mid-block, there's
    // no closing tag. Drop from the open tag to end-of-string. Same tag
    // alternation as above.
    private static readonly Regex _danglingOpen = new(
        @"<\s*(think|thinking|reasoning|reflection)(\s[^>]*)?>.*$",
        RegexOptions.Compiled | RegexOptions.Singleline | RegexOptions.IgnoreCase);

    // Collapse the blank lines that typically remain where a block was.
    private static readonly Regex _excessiveBlankLines = new(
        @"(\r?\n){3,}",
        RegexOptions.Compiled);

    /// <summary>
    /// Return <paramref name="content"/> with reasoning/thinking spans
    /// removed and the gap cleaned up. Safe to call on null/empty —
    /// returns the input unchanged.
    /// </summary>
    public static string StripReasoningSpans(string? content)
    {
        if (string.IsNullOrEmpty(content)) return content ?? string.Empty;

        var cleaned = _thinkSpans.Replace(content, string.Empty);
        cleaned = _danglingOpen.Replace(cleaned, string.Empty);
        cleaned = _excessiveBlankLines.Replace(cleaned, "\n\n");
        return cleaned.Trim();
    }
}
