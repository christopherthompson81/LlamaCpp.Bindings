using System.Collections.Generic;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LlamaCpp.Bindings.LlamaChat.Services.ToolCall;

/// <summary>
/// Hermes-style JSON format — <c>&lt;tool_call&gt;{"name": "...", "arguments":
/// {...}}&lt;/tool_call&gt;</c>. Used by Hermes 2/3/4 finetunes, several
/// DeepSeek variants, and some older Qwen templates that predate the XML
/// variant. The body is a single JSON object; consuming one envelope yields
/// exactly one call.
/// </summary>
public sealed class HermesJsonToolCallFormat : IToolCallFormat
{
    public string Name => "hermes-json";

    private static readonly Regex Wrapper =
        new(@"<tool_call>\s*(?<body>\{.*?\})\s*</tool_call>",
            RegexOptions.Singleline | RegexOptions.Compiled);

    public bool LooksCompatible(string jinjaTemplate)
    {
        if (string.IsNullOrEmpty(jinjaTemplate)) return false;
        // Hermes templates reference the JSON shape directly — the "arguments"
        // key is the giveaway. Qwen XML templates use "parameter" instead.
        if (!jinjaTemplate.Contains("<tool_call>")) return false;
        return jinjaTemplate.Contains("\"arguments\"") || jinjaTemplate.Contains("'arguments'");
    }

    public bool Contains(string text) =>
        !string.IsNullOrEmpty(text) && Wrapper.IsMatch(text);

    public IReadOnlyList<ParsedToolCall> Extract(string text)
    {
        var list = new List<ParsedToolCall>();
        foreach (Match m in Wrapper.Matches(text))
        {
            var body = m.Groups["body"].Value;
            try
            {
                using var doc = JsonDocument.Parse(body);
                var root = doc.RootElement;
                var name = root.TryGetProperty("name", out var n) ? n.GetString() : null;
                if (string.IsNullOrEmpty(name)) continue;
                JsonElement args = root.TryGetProperty("arguments", out var a)
                    ? a.Clone()
                    : JsonDocument.Parse("{}").RootElement.Clone();
                list.Add(new ParsedToolCall(name, args, m.Value));
            }
            catch (JsonException)
            {
                // Skip malformed — partial streams occasionally land here.
            }
        }
        return list;
    }
}
