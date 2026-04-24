using System;
using System.Text;
using System.Threading.Tasks;

namespace LlamaCpp.Bindings.LlamaChat.Services.ToolCall;

/// <summary>
/// Shapes a raw exception (or error condition) into a short, model-friendly
/// message suitable for re-injecting as a tool-role turn. The goal is: the
/// model sees a one-liner it can reason about ("the API returned 404")
/// rather than a stack trace or raw JSON. Full exception detail is logged
/// to <c>ErrorLog</c> so the user can still debug via the log file.
/// </summary>
internal static class ToolCallError
{
    public const int MaxLength = 240;

    public static string Summarize(Exception ex)
    {
        var root = Unwrap(ex);
        var msg = CollapseWhitespace(root.Message ?? root.GetType().Name);
        if (msg.Length > MaxLength) msg = msg[..MaxLength].TrimEnd() + "…";

        // Prefix with exception kind when the message alone is uninformative
        // (e.g. "Object reference not set..." reads better with the type name
        // attached). For canonical "self-describing" exceptions we keep the
        // message only.
        return ShouldIncludeKind(root) ? $"{root.GetType().Name}: {msg}" : msg;
    }

    /// <summary>
    /// Model-facing error string. Format is "Error: &lt;summary&gt;" so both
    /// OpenAI-style and Hermes-style tool-use templates surface it
    /// consistently as an error signal on the tool turn.
    /// </summary>
    public static string FormatForModel(string summary) => $"Error: {summary}";
    public static string FormatForModel(Exception ex) => FormatForModel(Summarize(ex));

    private static Exception Unwrap(Exception ex)
    {
        if (ex is AggregateException agg && agg.InnerExceptions.Count > 0)
            return Unwrap(agg.InnerExceptions[0]);
        return ex.InnerException is not null && ex is not TaskCanceledException
            ? Unwrap(ex.InnerException)
            : ex;
    }

    private static bool ShouldIncludeKind(Exception ex)
    {
        // Self-describing exceptions — the raw message is already a good
        // human-readable sentence, so prefixing the type name is noise.
        if (ex is TaskCanceledException
            or OperationCanceledException
            or TimeoutException) return false;
        // Type-name check avoids importing System.Net.Http here just to
        // reference the type.
        var typeName = ex.GetType().Name;
        if (typeName == "HttpRequestException") return false;
        return true;
    }

    private static string CollapseWhitespace(string s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        var sb = new StringBuilder(s.Length);
        bool prevWs = false;
        foreach (var ch in s)
        {
            if (char.IsWhiteSpace(ch))
            {
                if (!prevWs) sb.Append(' ');
                prevWs = true;
            }
            else
            {
                sb.Append(ch);
                prevWs = false;
            }
        }
        return sb.ToString().Trim();
    }
}
