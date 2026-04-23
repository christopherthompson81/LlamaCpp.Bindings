using System.Collections.Generic;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LlamaCpp.Bindings.LlamaChat.Services.ToolCall;

/// <summary>
/// Llama 3.1 / 3.2 function-calling format — <c>&lt;|python_tag|&gt;{"name":
/// "...", "parameters": {...}}</c> optionally terminated by <c>&lt;|eom_id|&gt;</c>.
/// Note that Meta's convention uses <c>parameters</c>, not <c>arguments</c>,
/// so we normalise both here.
///
/// Some Llama-3 finetunes emit raw JSON with no <c>&lt;|python_tag|&gt;</c>
/// prefix when the model decides to call a tool. We don't attempt to parse
/// that variant here — too ambiguous with regular JSON content. Users with
/// such a finetune can extend the registry.
/// </summary>
public sealed class Llama3ToolCallFormat : IToolCallFormat
{
    public string Name => "llama3";

    // JSON payload follows the python_tag marker and runs until either
    // <|eom_id|>, <|eot_id|>, end of line, or end of input. We capture a
    // single balanced {...} by non-greedy matching; malformed/truncated
    // payloads fall through via JSON parse failure.
    private static readonly Regex Wrapper =
        new(@"<\|python_tag\|>\s*(?<body>\{.*?\})(?=\s*(?:<\|eom_id\|>|<\|eot_id\|>|$))",
            RegexOptions.Singleline | RegexOptions.Compiled);

    public bool LooksCompatible(string jinjaTemplate)
    {
        if (string.IsNullOrEmpty(jinjaTemplate)) return false;
        return jinjaTemplate.Contains("<|python_tag|>");
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

                // Meta's Llama-3 spec uses "parameters"; some downstream
                // templates switched to "arguments" to match the OpenAI
                // convention. Accept either.
                JsonElement args;
                if (root.TryGetProperty("parameters", out var p)) args = p.Clone();
                else if (root.TryGetProperty("arguments", out var a)) args = a.Clone();
                else args = JsonDocument.Parse("{}").RootElement.Clone();

                list.Add(new ParsedToolCall(name, args, m.Value));
            }
            catch (JsonException)
            {
                // Malformed / truncated — skip.
            }
        }
        return list;
    }
}
