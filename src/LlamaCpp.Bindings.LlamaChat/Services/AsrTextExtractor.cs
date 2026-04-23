using System;
using System.Text;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Streaming state machine that recognises Qwen3-ASR's output convention
/// and splits it into two channels: a one-shot detected-language tag, and
/// the clean transcription content.
/// </summary>
/// <remarks>
/// Qwen3-ASR (and Qwen3-Omni audio output) starts each reply with a literal
/// preamble of the form <c>language &lt;LANG&gt;&lt;asr_text&gt;transcription…</c>.
/// No closing tag is emitted in practice — the model stops at EOG inside the
/// transcription block. Without post-processing, users see the raw markers
/// in the chat bubble.
///
/// The state machine waits until it sees <c>&lt;asr_text&gt;</c> (or until the
/// buffered preamble exceeds a sane length, at which point we decide the
/// reply isn't ASR-shaped and emit everything verbatim). On match, the text
/// between "language " and <c>&lt;asr_text&gt;</c> is reported as the language
/// tag, and anything after is streamed as content. Any stray
/// <c>&lt;/asr_text&gt;</c> on the way out is swallowed.
/// </remarks>
internal sealed class AsrTextExtractor
{
    private const string LanguagePrefix = "language ";
    private const string OpenTag = "<asr_text>";
    private const string CloseTag = "</asr_text>";

    // Cap the preamble buffer before we give up on detection. Real preambles
    // are ~20–30 chars (`"language English<asr_text>"`), so 128 leaves
    // comfortable headroom for longer language names ("Chinese (Simplified)")
    // without holding back actual transcription text on a non-ASR reply.
    private const int MaxPreambleChars = 128;

    private enum Mode { Preamble, Content, PostClose, Passthrough }

    private Mode _mode = Mode.Preamble;
    private readonly StringBuilder _pending = new();
    private bool _languageEmitted;

    public readonly record struct Emission(string Content, string? Language);

    /// <summary>
    /// Feed a streamed piece through the extractor. <see cref="Emission.Content"/>
    /// is text that should land in the message's normal body;
    /// <see cref="Emission.Language"/> is non-null exactly once, when the
    /// language tag is first parsed.
    /// </summary>
    public Emission Push(string piece)
    {
        _pending.Append(piece);
        var content = new StringBuilder();
        string? language = null;

        while (_pending.Length > 0)
        {
            switch (_mode)
            {
                case Mode.Preamble:
                    if (!TryAdvancePreamble(content, ref language)) return new Emission(content.ToString(), language);
                    break;
                case Mode.Content:
                    if (!TryAdvanceContent(content)) return new Emission(content.ToString(), language);
                    break;
                case Mode.PostClose:
                    // Anything after </asr_text> — rare in practice. Emit
                    // verbatim; some models trail a short coda that's still
                    // useful to the user.
                    content.Append(_pending);
                    _pending.Clear();
                    break;
                case Mode.Passthrough:
                    content.Append(_pending);
                    _pending.Clear();
                    break;
            }
        }

        return new Emission(content.ToString(), language);
    }

    /// <summary>
    /// End-of-stream drain. If we never matched the ASR shape, flush
    /// whatever buffered preamble text remains as content so nothing is
    /// lost.
    /// </summary>
    public Emission Flush()
    {
        if (_pending.Length == 0) return new Emission(string.Empty, null);
        var remaining = _pending.ToString();
        _pending.Clear();
        return new Emission(remaining, null);
    }

    // Preamble: look for <asr_text> in the buffer. Anything before is either
    // "language XYZ" (emit XYZ as language tag) or — if the buffer grows too
    // large without a match — not ASR at all; fall back to passthrough.
    private bool TryAdvancePreamble(StringBuilder content, ref string? language)
    {
        var buf = _pending.ToString();
        var idx = buf.IndexOf(OpenTag, StringComparison.Ordinal);
        if (idx >= 0)
        {
            // Parse what's before <asr_text> as optional "language <LANG>".
            var preamble = buf[..idx];
            if (!_languageEmitted &&
                preamble.StartsWith(LanguagePrefix, StringComparison.OrdinalIgnoreCase))
            {
                language = preamble[LanguagePrefix.Length..].TrimEnd();
                _languageEmitted = true;
            }
            else if (preamble.Length > 0)
            {
                // Non-standard preamble text — don't silently drop it.
                content.Append(preamble);
            }
            _pending.Clear();
            _pending.Append(buf, idx + OpenTag.Length, buf.Length - (idx + OpenTag.Length));
            _mode = Mode.Content;
            // Separate the language chip visually from the transcription.
            content.Append('\n');
            return true;
        }

        // No open tag yet. If we've buffered enough to be confident this isn't
        // ASR-shaped output, give up on detection and dump the buffer.
        if (_pending.Length > MaxPreambleChars)
        {
            content.Append(_pending);
            _pending.Clear();
            _mode = Mode.Passthrough;
            return true;
        }

        // Might still be mid-preamble. Hold the buffer and wait for more.
        return false;
    }

    // Content: flow text through until </asr_text>, holding back only the
    // longest suffix that could still become the close tag.
    private bool TryAdvanceContent(StringBuilder content)
    {
        var buf = _pending.ToString();
        var idx = buf.IndexOf(CloseTag, StringComparison.Ordinal);
        if (idx >= 0)
        {
            content.Append(buf, 0, idx);
            _pending.Clear();
            _pending.Append(buf, idx + CloseTag.Length, buf.Length - (idx + CloseTag.Length));
            _mode = Mode.PostClose;
            return true;
        }

        var holdBack = LongestProperPrefixSuffix(buf, CloseTag);
        var emitLen = buf.Length - holdBack;
        if (emitLen > 0)
        {
            content.Append(buf, 0, emitLen);
            _pending.Remove(0, emitLen);
        }
        return false;
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
