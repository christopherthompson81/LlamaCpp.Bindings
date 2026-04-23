using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LlamaCpp.Bindings.LlamaChat.Services.ToolCall;

/// <summary>
/// Qwen-family XML format — <c>&lt;tool_call&gt;</c> envelope wrapping one or
/// more <c>&lt;function=NAME&gt;&lt;parameter=KEY&gt;VALUE&lt;/parameter&gt;...&lt;/function&gt;</c>
/// blocks. Used by Qwen2.5-Instruct, Qwen3 (including MoE variants), Kimi-K2,
/// and a growing set of Qwen-derived finetunes.
///
/// Parameter values are coerced to narrow JSON types where possible
/// (numbers, booleans, nested arrays/objects). Anything that doesn't parse
/// as a JSON literal passes through as a string — matching what most tool
/// schemas expect on the input side.
/// </summary>
public sealed class QwenXmlToolCallFormat : IToolCallFormat
{
    public string Name => "qwen-xml";

    private static readonly Regex Wrapper =
        new(@"<tool_call>(?<body>.*?)</tool_call>",
            RegexOptions.Singleline | RegexOptions.Compiled);

    // Tool names may contain alnum, _, -, . — MCP name charset plus our
    // "serverName__tool" prefix convention.
    private static readonly Regex Function =
        new(@"<function=(?<name>[^>\s]+)\s*>(?<args>.*?)</function>",
            RegexOptions.Singleline | RegexOptions.Compiled);

    private static readonly Regex Parameter =
        new(@"<parameter=(?<key>[^>\s]+)\s*>(?<value>.*?)</parameter>",
            RegexOptions.Singleline | RegexOptions.Compiled);

    public bool LooksCompatible(string jinjaTemplate)
    {
        // The Qwen templates explicitly show <function=... and <parameter=...
        // in the tool-use instruction block when tools are passed.
        if (string.IsNullOrEmpty(jinjaTemplate)) return false;
        return jinjaTemplate.Contains("<function=") || jinjaTemplate.Contains("<parameter=");
    }

    public bool Contains(string text) =>
        !string.IsNullOrEmpty(text) && Wrapper.IsMatch(text);

    public IReadOnlyList<ParsedToolCall> Extract(string text)
    {
        var list = new List<ParsedToolCall>();
        foreach (Match wrap in Wrapper.Matches(text))
        {
            foreach (Match fn in Function.Matches(wrap.Groups["body"].Value))
            {
                if (TryParse(fn, wrap.Value, out var call)) list.Add(call);
            }
        }
        return list;
    }

    private static bool TryParse(Match fn, string raw, out ParsedToolCall call)
    {
        call = null!;
        var name = fn.Groups["name"].Value.Trim();
        if (name.Length == 0) return false;

        var sb = new StringBuilder("{");
        bool first = true;
        foreach (Match p in Parameter.Matches(fn.Groups["args"].Value))
        {
            var key = p.Groups["key"].Value.Trim();
            if (key.Length == 0) continue;
            var value = p.Groups["value"].Value.Trim();

            if (!first) sb.Append(',');
            first = false;
            sb.Append(JsonCoercion.Key(key)).Append(':').Append(JsonCoercion.Value(value));
        }
        sb.Append('}');

        try
        {
            var doc = JsonDocument.Parse(sb.ToString());
            call = new ParsedToolCall(name, doc.RootElement.Clone(), raw);
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }
}
