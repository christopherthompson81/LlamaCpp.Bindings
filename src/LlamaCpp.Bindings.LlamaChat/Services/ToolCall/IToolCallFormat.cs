using System.Collections.Generic;
using System.Text.Json;

namespace LlamaCpp.Bindings.LlamaChat.Services.ToolCall;

/// <summary>
/// One parsed tool invocation. <see cref="Arguments"/> is a cloned JSON
/// object independent of whatever document it came from, so the consumer
/// can hold onto it after the parser returns.
/// </summary>
public sealed record ParsedToolCall(string Name, JsonElement Arguments, string RawMatch);

/// <summary>
/// A tool-call wire format — the pairing of (a) how a specific chat template
/// instructs its model to emit tool calls and (b) how we extract them back
/// out of the assistant's stream.
///
/// Different open-weight model families use different wrappers (Qwen XML,
/// Hermes JSON, Mistral <c>[TOOL_CALLS]</c>, Llama-3 <c>&lt;|python_tag|&gt;</c>,
/// …). The format is dictated by the Jinja template baked into the GGUF, so
/// selection is a template-level concern — see
/// <see cref="ToolCallFormatRegistry.DetectFromTemplate"/>.
///
/// Implementations are stateless and safe to share as singletons.
/// </summary>
public interface IToolCallFormat
{
    /// <summary>Human-readable identifier — shown in diagnostics and logs.</summary>
    string Name { get; }

    /// <summary>
    /// True if the given Jinja template's tool-use branch renders instructions
    /// compatible with this format. Used at model-load time to pick the right
    /// parser up front, so the parse loop isn't guessing per token.
    /// </summary>
    bool LooksCompatible(string jinjaTemplate);

    /// <summary>
    /// True if <paramref name="text"/> contains any markup this format would
    /// recognise. Cheap check used to short-circuit the fallback try-all loop.
    /// </summary>
    bool Contains(string text);

    /// <summary>
    /// Extract every tool call the format can find in <paramref name="text"/>.
    /// Returns an empty list (not null) when there are none or all matches
    /// failed to parse cleanly.
    /// </summary>
    IReadOnlyList<ParsedToolCall> Extract(string text);
}
