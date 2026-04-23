using System.Collections.Generic;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LlamaCpp.Bindings.LlamaChat.Services.ToolCall;

/// <summary>
/// Mistral-family format — <c>[TOOL_CALLS][{"name": "...", "arguments":
/// {...}}, ...]</c>. The body is a JSON array; one envelope can carry several
/// calls. Used by Mistral Large / Small Instruct and downstream finetunes
/// that keep the parent tool-use template.
/// </summary>
public sealed class MistralToolCallFormat : IToolCallFormat
{
    public string Name => "mistral";

    // The wrapper is a literal [TOOL_CALLS] followed immediately by a JSON
    // array. We capture the array (with its brackets) so we can JSON-parse it
    // directly.
    private static readonly Regex Wrapper =
        new(@"\[TOOL_CALLS\]\s*(?<body>\[.*?\])",
            RegexOptions.Singleline | RegexOptions.Compiled);

    public bool LooksCompatible(string jinjaTemplate)
    {
        if (string.IsNullOrEmpty(jinjaTemplate)) return false;
        return jinjaTemplate.Contains("[TOOL_CALLS]");
    }

    public bool Contains(string text) =>
        !string.IsNullOrEmpty(text) && Wrapper.IsMatch(text);

    public IReadOnlyList<ParsedToolCall> Extract(string text)
    {
        var list = new List<ParsedToolCall>();
        foreach (Match m in Wrapper.Matches(text))
        {
            var body = m.Groups["body"].Value;
            JsonDocument? doc = null;
            try
            {
                doc = JsonDocument.Parse(body);
                if (doc.RootElement.ValueKind != JsonValueKind.Array) continue;
                foreach (var call in doc.RootElement.EnumerateArray())
                {
                    var name = call.TryGetProperty("name", out var n) ? n.GetString() : null;
                    if (string.IsNullOrEmpty(name)) continue;
                    JsonElement args = call.TryGetProperty("arguments", out var a)
                        ? a.Clone()
                        : JsonDocument.Parse("{}").RootElement.Clone();
                    list.Add(new ParsedToolCall(name, args, m.Value));
                }
            }
            catch (JsonException)
            {
                // Skip malformed batch.
            }
            finally
            {
                doc?.Dispose();
            }
        }
        return list;
    }
}
