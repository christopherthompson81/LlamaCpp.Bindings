using System;
using System.Collections.Generic;
using System.Text;

namespace LlamaCpp.Bindings.Jinja;

/// <summary>
/// First-pass tokeniser: splits a Jinja source into a flat list of text
/// spans and <c>{{ }}</c>/<c>{% %}</c> blocks. Whitespace-strip flags
/// (<c>{%-</c>, <c>-%}</c>, <c>{{-</c>, <c>-}}</c>) are captured per token
/// so the parser can trim adjacent <see cref="TokenKind.Text"/> tokens.
/// <para>
/// We don't try to tokenise the expression language inside <c>{{ }}</c>
/// and <c>{% %}</c> here — the parser re-scans those payloads with
/// <see cref="ExpressionLexer"/>. This keeps the outer loop simple and
/// lets us surface clean line/col errors per block.
/// </para>
/// </summary>
internal enum TokenKind
{
    Text,
    Expression,   // {{ ... }}
    Statement,    // {% ... %}
    Comment,      // {# ... #}  (never emitted; stripped here)
}

internal readonly record struct RawToken(
    TokenKind Kind,
    string Payload,
    bool StripLeft,
    bool StripRight,
    int Line,
    int Column);

internal static class JinjaLexer
{
    public static List<RawToken> Tokenise(string source)
    {
        var tokens = new List<RawToken>();
        if (string.IsNullOrEmpty(source)) return tokens;

        var i = 0;
        var line = 1;
        var col = 1;
        var textStart = 0;
        var textStartLine = line;
        var textStartCol = col;

        // Helper: advance pointer & line/col tracking across [from, to).
        void Advance(int count)
        {
            for (var k = 0; k < count; k++)
            {
                if (source[i] == '\n') { line++; col = 1; }
                else col++;
                i++;
            }
        }

        void FlushText(int end)
        {
            if (end > textStart)
            {
                tokens.Add(new RawToken(
                    TokenKind.Text,
                    source.Substring(textStart, end - textStart),
                    StripLeft: false, StripRight: false,
                    textStartLine, textStartCol));
            }
        }

        while (i < source.Length)
        {
            if (i + 1 < source.Length && source[i] == '{' &&
                (source[i + 1] == '{' || source[i + 1] == '%' || source[i + 1] == '#'))
            {
                FlushText(i);

                var marker = source[i + 1];
                var openLine = line;
                var openCol = col;

                // Consume '{{' / '{%' / '{#'
                Advance(2);

                // Optional '-' whitespace-strip on the left.
                var stripLeft = false;
                if (i < source.Length && source[i] == '-')
                {
                    stripLeft = true;
                    Advance(1);
                }

                var (close1, close2) = marker switch
                {
                    '{' => ('}', '}'),
                    '%' => ('%', '}'),
                    '#' => ('#', '}'),
                    _ => (' ', ' '),
                };

                var payloadStart = i;
                var payloadEnd = -1;
                var stripRight = false;

                while (i < source.Length)
                {
                    // Look for '-?}}' / '-?%}' / '-?#}'
                    var peek = source[i];
                    if (peek == '-' && i + 2 < source.Length
                        && source[i + 1] == close1 && source[i + 2] == close2)
                    {
                        stripRight = true;
                        payloadEnd = i;
                        Advance(3);
                        break;
                    }
                    if (peek == close1 && i + 1 < source.Length && source[i + 1] == close2)
                    {
                        payloadEnd = i;
                        Advance(2);
                        break;
                    }
                    // Allow newlines inside Jinja tags.
                    Advance(1);
                }

                if (payloadEnd < 0)
                {
                    throw new JinjaException(
                        "Unterminated Jinja tag — expected closing '}}', '%}' or '#}'.",
                        openLine, openCol);
                }

                var payload = source.Substring(payloadStart, payloadEnd - payloadStart);

                var kind = marker switch
                {
                    '{' => TokenKind.Expression,
                    '%' => TokenKind.Statement,
                    _ => TokenKind.Comment,
                };

                if (kind != TokenKind.Comment)
                {
                    tokens.Add(new RawToken(kind, payload, stripLeft, stripRight, openLine, openCol));
                }
                else
                {
                    // Comment tokens still carry strip flags we have to honour on
                    // neighbouring Text tokens. Represent them with an empty
                    // Expression-style marker so Text collapsing can see the
                    // flags, then drop it below.
                    tokens.Add(new RawToken(TokenKind.Comment, "", stripLeft, stripRight, openLine, openCol));
                }

                textStart = i;
                textStartLine = line;
                textStartCol = col;
            }
            else
            {
                Advance(1);
            }
        }

        FlushText(i);

        // Apply whitespace-strip flags to adjacent Text tokens. A `{%- ... %}`
        // strips trailing whitespace on the previous Text; a `{% ... -%}`
        // strips leading whitespace on the next Text. Whitespace here means
        // all of space/tab/CR/LF (Jinja's default, not `trim_blocks`-style).
        for (var n = 0; n < tokens.Count; n++)
        {
            var t = tokens[n];
            if (t.Kind == TokenKind.Text) continue;

            if (t.StripLeft && n > 0 && tokens[n - 1].Kind == TokenKind.Text)
            {
                var prev = tokens[n - 1];
                tokens[n - 1] = prev with { Payload = RTrimWs(prev.Payload) };
            }
            if (t.StripRight && n + 1 < tokens.Count && tokens[n + 1].Kind == TokenKind.Text)
            {
                var next = tokens[n + 1];
                tokens[n + 1] = next with { Payload = LTrimWs(next.Payload) };
            }
        }

        // Drop the synthetic Comment placeholders now that strip flags were applied.
        tokens.RemoveAll(t => t.Kind == TokenKind.Comment);

        return tokens;
    }

    private static string RTrimWs(string s)
    {
        var end = s.Length;
        while (end > 0 && IsWs(s[end - 1])) end--;
        return end == s.Length ? s : s[..end];
    }

    private static string LTrimWs(string s)
    {
        var start = 0;
        while (start < s.Length && IsWs(s[start])) start++;
        return start == 0 ? s : s[start..];
    }

    private static bool IsWs(char c) => c == ' ' || c == '\t' || c == '\r' || c == '\n';
}
