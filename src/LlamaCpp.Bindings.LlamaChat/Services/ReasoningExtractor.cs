using System;
using System.Text;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Streaming state machine that routes emitted text into either the visible
/// content channel or a "reasoning" channel, based on <c>&lt;think&gt;...&lt;/think&gt;</c>
/// delimiters. Feed arbitrarily-sized pieces to <see cref="Push"/> and it
/// buffers just enough to straddle a potentially split open or close tag.
///
/// Hardcoded to <c>&lt;think&gt;</c>/<c>&lt;/think&gt;</c> in v1. Per-template tag
/// configuration is tracked in <c>docs/webui_parity_investigation.md</c>.
/// </summary>
internal sealed class ReasoningExtractor
{
    private const string OpenTag = "<think>";
    private const string CloseTag = "</think>";

    private enum Mode { Content, Reasoning }

    private Mode _mode;
    private readonly StringBuilder _pending = new();

    public readonly record struct Emission(string Content, string Reasoning);

    /// <summary>
    /// Construct an extractor that starts in the given mode. Pass
    /// <paramref name="startInReasoning"/> = true for models whose chat
    /// template ends the assistant-turn prefix with an *open* <c>&lt;think&gt;</c>
    /// block (Qwen3 family, DeepSeek R1, etc.) — the stream arrives already
    /// inside the reasoning block, and we need to wait for <c>&lt;/think&gt;</c>
    /// before routing bytes to the visible content channel.
    /// </summary>
    public ReasoningExtractor(bool startInReasoning = false) =>
        _mode = startInReasoning ? Mode.Reasoning : Mode.Content;

    /// <summary>
    /// Push a piece of streamed text. Returns what should be appended to the
    /// visible content and to the reasoning channel respectively. Either may
    /// be empty. A trailing byte range that could still grow into an open or
    /// close tag is held back and will appear in a later <see cref="Push"/>
    /// or the final <see cref="Flush"/>.
    /// </summary>
    public Emission Push(string piece)
    {
        _pending.Append(piece);
        var content = new StringBuilder();
        var reasoning = new StringBuilder();

        while (_pending.Length > 0)
        {
            var target = _mode == Mode.Content ? OpenTag : CloseTag;
            var buf = _pending.ToString();

            var idx = buf.IndexOf(target, StringComparison.Ordinal);
            if (idx >= 0)
            {
                (_mode == Mode.Content ? content : reasoning).Append(buf, 0, idx);
                _pending.Clear();
                _pending.Append(buf, idx + target.Length, buf.Length - (idx + target.Length));
                _mode = _mode == Mode.Content ? Mode.Reasoning : Mode.Content;
                continue;
            }

            // No complete tag. Emit every byte that cannot be part of a future
            // tag completion. That is: everything except the longest suffix of
            // buf that is a proper prefix of target.
            var holdBack = LongestProperPrefixSuffix(buf, target);
            var emitLen = buf.Length - holdBack;
            if (emitLen > 0)
            {
                (_mode == Mode.Content ? content : reasoning).Append(buf, 0, emitLen);
                _pending.Remove(0, emitLen);
            }
            break;
        }

        return new Emission(content.ToString(), reasoning.ToString());
    }

    /// <summary>
    /// Drain any remaining buffered text at end of stream. A dangling open
    /// <c>&lt;think&gt;</c> without a close tag leaves the remaining bytes on
    /// the reasoning channel — matches llama-server's generous interpretation.
    /// </summary>
    public Emission Flush()
    {
        var remaining = _pending.ToString();
        _pending.Clear();
        return _mode == Mode.Content
            ? new Emission(remaining, string.Empty)
            : new Emission(string.Empty, remaining);
    }

    private static int LongestProperPrefixSuffix(string buf, string target)
    {
        var max = Math.Min(buf.Length, target.Length - 1);
        for (var k = max; k >= 1; k--)
        {
            if (buf.AsSpan(buf.Length - k).SequenceEqual(target.AsSpan(0, k)))
                return k;
        }
        return 0;
    }
}
