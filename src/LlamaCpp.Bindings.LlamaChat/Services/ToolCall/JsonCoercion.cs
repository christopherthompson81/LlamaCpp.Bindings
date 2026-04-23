using System.Text.Json;

namespace LlamaCpp.Bindings.LlamaChat.Services.ToolCall;

/// <summary>
/// Shared helpers for building JSON argument objects from loose text the
/// model emitted (XML-wrapped values, unquoted bare atoms, etc.). Keeps the
/// individual format parsers from duplicating the coerce-or-stringify dance.
/// </summary>
internal static class JsonCoercion
{
    /// <summary>
    /// JSON-encode <paramref name="key"/> as a property name (always a quoted
    /// string, with escaping).
    /// </summary>
    public static string Key(string key) => JsonSerializer.Serialize(key);

    /// <summary>
    /// Pick the narrowest valid JSON encoding for <paramref name="value"/>:
    /// pass through untouched if it already parses as JSON (number, bool,
    /// null, object, array, quoted string); otherwise wrap it in JSON quotes.
    /// </summary>
    public static string Value(string value)
    {
        if (value.Length == 0) return "\"\"";
        try
        {
            using var _ = JsonDocument.Parse(value);
            return value;
        }
        catch (JsonException)
        {
            return JsonSerializer.Serialize(value);
        }
    }
}
