using System.Collections.Generic;

namespace LlamaCpp.Bindings.LlamaChat.Services.ToolCall;

/// <summary>
/// Static registry of the tool-call wire formats the app understands.
/// Adding support for a new model family is a matter of writing another
/// <see cref="IToolCallFormat"/> and appending it to <see cref="All"/>.
///
/// Two consumers:
/// <list type="bullet">
///   <item>
///     <c>ChatSession</c> detects the format once at model-load time by
///     sniffing the Jinja template via <see cref="DetectFromTemplate"/>.
///   </item>
///   <item>
///     <c>ToolCallParser.Extract</c> takes an optional preferred format
///     (from the session) and falls back to trying each entry in
///     <see cref="All"/> order when the template didn't match anything.
///   </item>
/// </list>
/// </summary>
public static class ToolCallFormatRegistry
{
    /// <summary>
    /// All known formats, in fallback-try order. Qwen XML is first since it
    /// covers Qwen3 (the largest open-weight tool-use family today); Hermes
    /// JSON second for Hermes/DeepSeek; Mistral next; Llama-3 last.
    /// </summary>
    public static IReadOnlyList<IToolCallFormat> All { get; } = new IToolCallFormat[]
    {
        new QwenXmlToolCallFormat(),
        new HermesJsonToolCallFormat(),
        new MistralToolCallFormat(),
        new Llama3ToolCallFormat(),
    };

    /// <summary>
    /// Pick the format whose <see cref="IToolCallFormat.LooksCompatible"/>
    /// returns true for <paramref name="jinjaTemplate"/>. Returns null when
    /// either (a) no template was embedded in the GGUF, or (b) the template
    /// doesn't look like any format we know.
    /// </summary>
    public static IToolCallFormat? DetectFromTemplate(string? jinjaTemplate)
    {
        if (string.IsNullOrEmpty(jinjaTemplate)) return null;
        foreach (var f in All)
        {
            if (f.LooksCompatible(jinjaTemplate)) return f;
        }
        return null;
    }
}
