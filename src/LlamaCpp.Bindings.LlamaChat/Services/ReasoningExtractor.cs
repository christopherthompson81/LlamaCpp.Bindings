using System;
using System.Text;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Streaming state machine that routes emitted text into either the visible
/// content channel or a "reasoning" channel, based on <c>&lt;think&gt;...&lt;/think&gt;</c>
/// delimiters. Feed arbitrarily-sized pieces to <see cref="Push"/> and it
/// buffers just enough to straddle a potentially split open or close tag.
///
/// Two entry modes:
/// <list type="bullet">
///   <item>
///     <c>startInReasoning = false</c> (default) — used by the remote path and
///     any case where the template state isn't known ahead of time. Content mode
///     detects both <c>&lt;think&gt;</c> (model-opened block) and <c>&lt;/think&gt;</c>
///     (server-pre-opened block whose open tag was in the assistant prefix and
///     never appeared in the delta stream). The latter routes text before
///     <c>&lt;/think&gt;</c> retroactively to the Reasoning channel, then
///     continues in Content mode.
///   </item>
///   <item>
///     <c>startInReasoning = true</c> — used by the local path when the
///     rendered prompt is known to end with an open <c>&lt;think&gt;</c> tag
///     (Qwen3, DeepSeek-R1 etc.). The stream arrives already inside the
///     reasoning block; wait for <c>&lt;/think&gt;</c> before routing to Content.
///   </item>
/// </list>
/// </summary>
internal sealed class ReasoningExtractor
{
    private const string OpenTag = "<think>";
    private const string CloseTag = "</think>";

    private enum Mode { Content, Reasoning }

    private Mode _mode;
    private readonly StringBuilder _pending = new();

    public readonly record struct Emission(string Content, string Reasoning);

    public ReasoningExtractor(bool startInReasoning = false) =>
        _mode = startInReasoning ? Mode.Reasoning : Mode.Content;

    /// <summary>
    /// Push a piece of streamed text. Returns what should be appended to the
    /// visible content and to the reasoning channel respectively. Either may
    /// be empty. A trailing byte range that could still grow into a tag is
    /// held back and will appear in a later <see cref="Push"/> or
    /// <see cref="Flush"/>.
    /// </summary>
    public Emission Push(string piece)
    {
        _pending.Append(piece);
        var content = new StringBuilder();
        var reasoning = new StringBuilder();

        while (_pending.Length > 0)
        {
            var buf = _pending.ToString();

            if (_mode == Mode.Content)
            {
                var openIdx = buf.IndexOf(OpenTag, StringComparison.Ordinal);
                var closeIdx = buf.IndexOf(CloseTag, StringComparison.Ordinal);

                if (openIdx >= 0 && (closeIdx < 0 || openIdx <= closeIdx))
                {
                    // Standard model-opened block: emit content before tag, switch to Reasoning.
                    content.Append(buf, 0, openIdx);
                    _pending.Clear();
                    _pending.Append(buf, openIdx + OpenTag.Length, buf.Length - (openIdx + OpenTag.Length));
                    _mode = Mode.Reasoning;
                }
                else if (closeIdx >= 0)
                {
                    // Server pre-opened a <think> block in the assistant prefix — the open
                    // tag never appeared in the delta. Text before </think> is reasoning;
                    // stay in Content mode afterward.
                    reasoning.Append(buf, 0, closeIdx);
                    _pending.Clear();
                    _pending.Append(buf, closeIdx + CloseTag.Length, buf.Length - (closeIdx + CloseTag.Length));
                    // _mode stays Content
                }
                else
                {
                    // Neither complete tag yet. Hold back the longest buffer suffix that
                    // could still complete either tag — the max of both candidates.
                    var holdBack = Math.Max(
                        LongestProperPrefixSuffix(buf, OpenTag),
                        LongestProperPrefixSuffix(buf, CloseTag));
                    var emitLen = buf.Length - holdBack;
                    if (emitLen > 0)
                    {
                        content.Append(buf, 0, emitLen);
                        _pending.Remove(0, emitLen);
                    }
                    break;
                }
            }
            else // Mode.Reasoning
            {
                var idx = buf.IndexOf(CloseTag, StringComparison.Ordinal);
                if (idx >= 0)
                {
                    reasoning.Append(buf, 0, idx);
                    _pending.Clear();
                    _pending.Append(buf, idx + CloseTag.Length, buf.Length - (idx + CloseTag.Length));
                    _mode = Mode.Content;
                }
                else
                {
                    var holdBack = LongestProperPrefixSuffix(buf, CloseTag);
                    var emitLen = buf.Length - holdBack;
                    if (emitLen > 0)
                    {
                        reasoning.Append(buf, 0, emitLen);
                        _pending.Remove(0, emitLen);
                    }
                    break;
                }
            }
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
