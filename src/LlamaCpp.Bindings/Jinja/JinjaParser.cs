using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace LlamaCpp.Bindings.Jinja;

/// <summary>
/// Second-pass parser: walks the flat <see cref="RawToken"/> stream from
/// <see cref="JinjaLexer"/> and builds a <see cref="Node"/> tree. Expression
/// payloads are re-tokenised by <see cref="ExprLexer"/> and parsed with a
/// recursive-descent expression parser.
/// </summary>
internal static class JinjaParser
{
    public static IReadOnlyList<Node> Parse(string source)
    {
        var raw = JinjaLexer.Tokenise(source);
        var pos = 0;
        var body = ParseBody(raw, ref pos, endKeywords: null, out _);
        if (pos != raw.Count)
        {
            var t = raw[pos];
            throw new JinjaException(
                $"Unexpected '{StatementKeyword(t)}' at top level.", t.Line, t.Column);
        }
        return body;
    }

    // ----- Helpers for the block-level parse --------------------------

    private static string StatementKeyword(RawToken t)
    {
        if (t.Kind != TokenKind.Statement) return t.Kind.ToString();
        var payload = t.Payload.TrimStart();
        var end = 0;
        while (end < payload.Length && (char.IsLetter(payload[end]) || payload[end] == '_')) end++;
        return payload[..end];
    }

    /// <summary>
    /// Parse a sequence of Nodes up to one of <paramref name="endKeywords"/>
    /// (e.g. <c>"endif"</c>, <c>"else"</c>, <c>"elif"</c>). The matching
    /// terminator is consumed by the *caller* via the <paramref name="terminator"/>
    /// out param — we stop *before* it.
    /// </summary>
    private static IReadOnlyList<Node> ParseBody(
        List<RawToken> raw,
        ref int pos,
        HashSet<string>? endKeywords,
        out string? terminator)
    {
        var body = new List<Node>();
        terminator = null;

        while (pos < raw.Count)
        {
            var t = raw[pos];

            if (t.Kind == TokenKind.Text)
            {
                if (t.Payload.Length > 0)
                    body.Add(new TextNode(t.Payload, t.Line, t.Column));
                pos++;
                continue;
            }

            if (t.Kind == TokenKind.Expression)
            {
                var exprLexer = new ExprLexer(t.Payload, t.Line, t.Column);
                var parser = new ExprParser(exprLexer);
                var expr = parser.ParseExpression();
                parser.ExpectEnd();
                body.Add(new OutputNode(expr, t.Line, t.Column));
                pos++;
                continue;
            }

            // Statement token
            var kw = StatementKeyword(t);

            if (endKeywords is not null && endKeywords.Contains(kw))
            {
                terminator = kw;
                return body;
            }

            switch (kw)
            {
                case "if":
                    body.Add(ParseIf(raw, ref pos));
                    break;
                case "for":
                    body.Add(ParseFor(raw, ref pos));
                    break;
                case "set":
                    body.Add(ParseSet(raw, ref pos));
                    break;
                case "macro":
                    body.Add(ParseMacro(raw, ref pos));
                    break;
                default:
                    throw new JinjaException(
                        $"Unexpected statement '{kw}'.", t.Line, t.Column);
            }
        }

        if (endKeywords is not null)
        {
            throw new JinjaException(
                $"Unterminated block — expected one of {{{string.Join(", ", endKeywords)}}}.");
        }

        return body;
    }

    private static IfNode ParseIf(List<RawToken> raw, ref int pos)
    {
        var open = raw[pos];
        var cond = ParseStmtTail(open, "if");
        pos++;

        var branches = new List<(Expr? Condition, IReadOnlyList<Node> Body)>();
        var endSet = new HashSet<string> { "elif", "else", "endif" };

        var body = ParseBody(raw, ref pos, endSet, out var term);
        branches.Add((cond, body));

        while (term == "elif")
        {
            var e = raw[pos];
            var elifCond = ParseStmtTail(e, "elif");
            pos++;
            var elifBody = ParseBody(raw, ref pos, endSet, out term);
            branches.Add((elifCond, elifBody));
        }

        if (term == "else")
        {
            pos++; // consume {% else %}
            var elseBody = ParseBody(raw, ref pos, new HashSet<string> { "endif" }, out term);
            branches.Add((null, elseBody));
        }

        if (term != "endif")
            throw new JinjaException("Expected 'endif'.", open.Line, open.Column);
        pos++; // consume {% endif %}

        return new IfNode(branches, open.Line, open.Column);
    }

    private static ForNode ParseFor(List<RawToken> raw, ref int pos)
    {
        var open = raw[pos];
        var payload = open.Payload.TrimStart();
        // Strip leading 'for'
        payload = payload.Substring(3).TrimStart();

        // Parse `vars` up to the 'in' keyword. Vars can be `x` or `x, y`.
        // Simple scan: split on ',' before hitting ' in '.
        var inIdx = FindKeyword(payload, "in");
        if (inIdx < 0)
            throw new JinjaException("'for' without 'in'.", open.Line, open.Column);
        var varsText = payload[..inIdx].Trim();
        var iterText = payload[(inIdx + 2)..].Trim();

        var vars = new List<string>();
        foreach (var v in varsText.Split(','))
        {
            var name = v.Trim();
            if (name.Length == 0)
                throw new JinjaException("Empty loop variable.", open.Line, open.Column);
            vars.Add(name);
        }

        var iter = ParseExpressionString(iterText, open.Line, open.Column);
        pos++;

        var body = ParseBody(raw, ref pos,
            new HashSet<string> { "else", "endfor" }, out var term);

        IReadOnlyList<Node>? elseBody = null;
        if (term == "else")
        {
            pos++;
            elseBody = ParseBody(raw, ref pos, new HashSet<string> { "endfor" }, out term);
        }
        if (term != "endfor")
            throw new JinjaException("Expected 'endfor'.", open.Line, open.Column);
        pos++;

        return new ForNode(vars, iter, body, elseBody, open.Line, open.Column);
    }

    private static Node ParseSet(List<RawToken> raw, ref int pos)
    {
        var open = raw[pos];
        var payload = open.Payload.TrimStart();
        payload = payload.Substring(3).TrimStart(); // strip 'set'

        // Two forms: `set target = value` or `set target` (block form).
        var eqIdx = FindTopLevelEquals(payload);
        if (eqIdx >= 0)
        {
            var targetText = payload[..eqIdx].Trim();
            var valueText = payload[(eqIdx + 1)..].Trim();
            var targetPath = ParseDottedTarget(targetText, open);
            var value = ParseExpressionString(valueText, open.Line, open.Column);
            pos++;
            return new SetNode(targetPath, value, open.Line, open.Column);
        }

        // Block form — Jinja allows `{% set x %}...{% endset %}`.
        var targetPath2 = ParseDottedTarget(payload.Trim(), open);
        if (targetPath2.Count != 1)
            throw new JinjaException(
                "Block-form 'set' supports a single identifier target only.",
                open.Line, open.Column);
        pos++;
        var body = ParseBody(raw, ref pos, new HashSet<string> { "endset" }, out var term);
        if (term != "endset")
            throw new JinjaException("Expected 'endset'.", open.Line, open.Column);
        pos++;
        return new SetBlockNode(targetPath2[0], body, open.Line, open.Column);
    }

    private static MacroNode ParseMacro(List<RawToken> raw, ref int pos)
    {
        var open = raw[pos];
        var payload = open.Payload.TrimStart();
        payload = payload.Substring(5).TrimStart(); // 'macro'

        // name(param, param=default)
        var parenIdx = payload.IndexOf('(');
        if (parenIdx < 0)
            throw new JinjaException("'macro' missing '('.", open.Line, open.Column);
        var name = payload[..parenIdx].Trim();
        var rest = payload[(parenIdx + 1)..];
        var closeIdx = FindMatchingClose(rest, '(', ')');
        if (closeIdx < 0)
            throw new JinjaException("'macro' missing ')'.", open.Line, open.Column);
        var paramsText = rest[..closeIdx];

        var parameters = new List<MacroParam>();
        foreach (var part in SplitTopLevelCommas(paramsText))
        {
            var p = part.Trim();
            if (p.Length == 0) continue;
            var eq = FindTopLevelEquals(p);
            if (eq < 0)
            {
                parameters.Add(new MacroParam(p, null));
            }
            else
            {
                var pname = p[..eq].Trim();
                var pdef = p[(eq + 1)..].Trim();
                var defExpr = ParseExpressionString(pdef, open.Line, open.Column);
                parameters.Add(new MacroParam(pname, defExpr));
            }
        }

        pos++;
        var body = ParseBody(raw, ref pos, new HashSet<string> { "endmacro" }, out var term);
        if (term != "endmacro")
            throw new JinjaException("Expected 'endmacro'.", open.Line, open.Column);
        pos++;
        return new MacroNode(name, parameters, body, open.Line, open.Column);
    }

    // ----- Scanning helpers for raw payloads --------------------------

    private static Expr ParseStmtTail(RawToken t, string keyword)
    {
        var payload = t.Payload.TrimStart();
        if (!payload.StartsWith(keyword))
            throw new JinjaException($"Expected '{keyword}'.", t.Line, t.Column);
        var rest = payload[keyword.Length..].TrimStart();
        return ParseExpressionString(rest, t.Line, t.Column);
    }

    internal static Expr ParseExpressionString(string source, int baseLine, int baseCol)
    {
        var lex = new ExprLexer(source, baseLine, baseCol);
        var p = new ExprParser(lex);
        var e = p.ParseExpression();
        p.ExpectEnd();
        return e;
    }

    private static IReadOnlyList<string> ParseDottedTarget(string text, RawToken t)
    {
        var parts = new List<string>();
        var start = 0;
        while (start < text.Length)
        {
            var end = start;
            while (end < text.Length && (char.IsLetterOrDigit(text[end]) || text[end] == '_')) end++;
            if (end == start)
                throw new JinjaException(
                    $"Bad assignment target '{text}'.", t.Line, t.Column);
            parts.Add(text[start..end]);
            if (end == text.Length) break;
            if (text[end] != '.')
                throw new JinjaException(
                    $"Bad assignment target '{text}'.", t.Line, t.Column);
            start = end + 1;
        }
        return parts;
    }

    /// <summary>Locate the position of a whole-word keyword at depth 0.</summary>
    private static int FindKeyword(string s, string kw)
    {
        var depth = 0;
        char? quote = null;
        for (var i = 0; i + kw.Length <= s.Length; i++)
        {
            var c = s[i];
            if (quote is not null)
            {
                if (c == '\\' && i + 1 < s.Length) { i++; continue; }
                if (c == quote) quote = null;
                continue;
            }
            if (c == '\'' || c == '"') { quote = c; continue; }
            if (c == '(' || c == '[' || c == '{') depth++;
            else if (c == ')' || c == ']' || c == '}') depth--;
            else if (depth == 0
                && string.CompareOrdinal(s, i, kw, 0, kw.Length) == 0
                && (i == 0 || !IsIdentChar(s[i - 1]))
                && (i + kw.Length == s.Length || !IsIdentChar(s[i + kw.Length])))
            {
                return i;
            }
        }
        return -1;
    }

    private static int FindTopLevelEquals(string s)
    {
        var depth = 0;
        char? quote = null;
        for (var i = 0; i < s.Length; i++)
        {
            var c = s[i];
            if (quote is not null)
            {
                if (c == '\\' && i + 1 < s.Length) { i++; continue; }
                if (c == quote) quote = null;
                continue;
            }
            if (c == '\'' || c == '"') { quote = c; continue; }
            if (c == '(' || c == '[' || c == '{') depth++;
            else if (c == ')' || c == ']' || c == '}') depth--;
            else if (depth == 0 && c == '=')
            {
                // Skip ==, !=, <=, >=
                if (i + 1 < s.Length && s[i + 1] == '=') { i++; continue; }
                if (i > 0 && (s[i - 1] == '!' || s[i - 1] == '<' || s[i - 1] == '>')) continue;
                return i;
            }
        }
        return -1;
    }

    private static int FindMatchingClose(string s, char open, char close)
    {
        var depth = 1;
        char? quote = null;
        for (var i = 0; i < s.Length; i++)
        {
            var c = s[i];
            if (quote is not null)
            {
                if (c == '\\' && i + 1 < s.Length) { i++; continue; }
                if (c == quote) quote = null;
                continue;
            }
            if (c == '\'' || c == '"') { quote = c; continue; }
            if (c == open) depth++;
            else if (c == close) { depth--; if (depth == 0) return i; }
        }
        return -1;
    }

    private static IEnumerable<string> SplitTopLevelCommas(string s)
    {
        var depth = 0;
        char? quote = null;
        var start = 0;
        for (var i = 0; i < s.Length; i++)
        {
            var c = s[i];
            if (quote is not null)
            {
                if (c == '\\' && i + 1 < s.Length) { i++; continue; }
                if (c == quote) quote = null;
                continue;
            }
            if (c == '\'' || c == '"') { quote = c; continue; }
            if (c == '(' || c == '[' || c == '{') depth++;
            else if (c == ')' || c == ']' || c == '}') depth--;
            else if (depth == 0 && c == ',')
            {
                yield return s[start..i];
                start = i + 1;
            }
        }
        if (start <= s.Length) yield return s[start..];
    }

    private static bool IsIdentChar(char c) => char.IsLetterOrDigit(c) || c == '_';
}

// ====================================================================
// ExprLexer — tokenise expression payloads into small, typed pieces.
// ====================================================================

internal enum ExprTokKind
{
    End,
    Ident,
    Number,
    String,
    // Punctuation / operators carry their literal text in Text. We keep a
    // single kind to avoid explosion; the parser switches on Text.
    Punct,
}

internal readonly record struct ExprTok(ExprTokKind Kind, string Text, int Line, int Column);

internal sealed class ExprLexer
{
    private readonly string _src;
    private int _pos;
    private int _line;
    private int _col;
    private readonly int _baseLine;
    private readonly int _baseCol;

    private static readonly string[] TwoCharOps =
        { "==", "!=", "<=", ">=", "//", "**" };

    public ExprLexer(string source, int baseLine, int baseCol)
    {
        _src = source;
        _pos = 0;
        _line = 1;
        _col = 1;
        _baseLine = baseLine;
        _baseCol = baseCol;
    }

    public ExprTok Next()
    {
        SkipWs();
        if (_pos >= _src.Length) return new ExprTok(ExprTokKind.End, "", LineAbs, ColAbs);

        var c = _src[_pos];

        if (char.IsLetter(c) || c == '_') return ReadIdent();
        if (char.IsDigit(c)) return ReadNumber();
        if (c == '\'' || c == '"') return ReadString(c);

        // Two-char then one-char operators.
        foreach (var op in TwoCharOps)
        {
            if (_pos + 1 < _src.Length && _src[_pos] == op[0] && _src[_pos + 1] == op[1])
            {
                var tok = new ExprTok(ExprTokKind.Punct, op, LineAbs, ColAbs);
                Advance(2);
                return tok;
            }
        }

        // Any other single punctuation.
        var punctTok = new ExprTok(ExprTokKind.Punct, c.ToString(), LineAbs, ColAbs);
        Advance(1);
        return punctTok;
    }

    private ExprTok ReadIdent()
    {
        var start = _pos;
        var line = LineAbs; var col = ColAbs;
        while (_pos < _src.Length && (char.IsLetterOrDigit(_src[_pos]) || _src[_pos] == '_'))
            Advance(1);
        return new ExprTok(ExprTokKind.Ident, _src[start.._pos], line, col);
    }

    private ExprTok ReadNumber()
    {
        var start = _pos;
        var line = LineAbs; var col = ColAbs;
        while (_pos < _src.Length && char.IsDigit(_src[_pos])) Advance(1);
        if (_pos < _src.Length && _src[_pos] == '.' &&
            _pos + 1 < _src.Length && char.IsDigit(_src[_pos + 1]))
        {
            Advance(1);
            while (_pos < _src.Length && char.IsDigit(_src[_pos])) Advance(1);
        }
        return new ExprTok(ExprTokKind.Number, _src[start.._pos], line, col);
    }

    private ExprTok ReadString(char q)
    {
        var line = LineAbs; var col = ColAbs;
        Advance(1); // opening quote
        var sb = new StringBuilder();
        while (_pos < _src.Length && _src[_pos] != q)
        {
            var c = _src[_pos];
            if (c == '\\' && _pos + 1 < _src.Length)
            {
                var esc = _src[_pos + 1];
                sb.Append(esc switch
                {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '\'' => '\'',
                    '"' => '"',
                    '0' => '\0',
                    _ => esc,
                });
                Advance(2);
            }
            else
            {
                sb.Append(c);
                Advance(1);
            }
        }
        if (_pos >= _src.Length)
            throw new JinjaException("Unterminated string literal.", line, col);
        Advance(1); // closing quote
        return new ExprTok(ExprTokKind.String, sb.ToString(), line, col);
    }

    private void SkipWs()
    {
        while (_pos < _src.Length && (_src[_pos] == ' ' || _src[_pos] == '\t'
            || _src[_pos] == '\r' || _src[_pos] == '\n'))
        {
            Advance(1);
        }
    }

    private void Advance(int n)
    {
        for (var i = 0; i < n; i++)
        {
            if (_pos < _src.Length)
            {
                if (_src[_pos] == '\n') { _line++; _col = 1; }
                else _col++;
                _pos++;
            }
        }
    }

    // Lines/cols reported as absolute (relative to the original template source).
    private int LineAbs => _baseLine + _line - 1;
    private int ColAbs => _line == 1 ? _baseCol + _col - 1 : _col;
}

// ====================================================================
// ExprParser — recursive descent over the expression grammar.
// ====================================================================

internal sealed class ExprParser
{
    private readonly ExprLexer _lex;
    private ExprTok _current;
    private ExprTok _peek;
    private bool _peekValid;

    public ExprParser(ExprLexer lex)
    {
        _lex = lex;
        _current = _lex.Next();
        _peekValid = false;
        _peek = default;
    }

    public void ExpectEnd()
    {
        if (_current.Kind != ExprTokKind.End)
            throw Err($"Unexpected trailing '{_current.Text}'.");
    }

    public Expr ParseExpression() => ParseTernary();

    // ternary := filterExpr ('if' filterExpr 'else' ternary)?
    private Expr ParseTernary()
    {
        var left = ParseOr();
        if (IsKeyword("if"))
        {
            Consume();
            var cond = ParseOr();
            ExpectKeyword("else");
            var otherwise = ParseTernary();
            return new TernaryExpr(cond, left, otherwise, left.Line, left.Column);
        }
        return left;
    }

    private Expr ParseOr()
    {
        var left = ParseAnd();
        while (IsKeyword("or"))
        {
            var tok = _current; Consume();
            var right = ParseAnd();
            left = new BinOpExpr("or", left, right, tok.Line, tok.Column);
        }
        return left;
    }

    private Expr ParseAnd()
    {
        var left = ParseNot();
        while (IsKeyword("and"))
        {
            var tok = _current; Consume();
            var right = ParseNot();
            left = new BinOpExpr("and", left, right, tok.Line, tok.Column);
        }
        return left;
    }

    private Expr ParseNot()
    {
        if (IsKeyword("not"))
        {
            var tok = _current; Consume();
            var operand = ParseNot();
            return new UnaryExpr("not", operand, tok.Line, tok.Column);
        }
        return ParseCompare();
    }

    private Expr ParseCompare()
    {
        var left = ParseConcat();
        while (true)
        {
            string? op = null;
            if (IsPunct("==")) op = "==";
            else if (IsPunct("!=")) op = "!=";
            else if (IsPunct("<=")) op = "<=";
            else if (IsPunct(">=")) op = ">=";
            else if (IsPunct("<")) op = "<";
            else if (IsPunct(">")) op = ">";
            else if (IsKeyword("in")) op = "in";
            else if (IsKeyword("not") && PeekKeyword("in")) { op = "not in"; Consume(); }

            if (op is null) break;
            var tok = _current; Consume();
            var right = ParseConcat();
            left = new BinOpExpr(op, left, right, tok.Line, tok.Column);
        }
        return left;
    }

    private Expr ParseConcat()
    {
        var left = ParseArith();
        while (IsPunct("~"))
        {
            var tok = _current; Consume();
            var right = ParseArith();
            left = new BinOpExpr("~", left, right, tok.Line, tok.Column);
        }
        return left;
    }

    private Expr ParseArith()
    {
        var left = ParseTerm();
        while (IsPunct("+") || IsPunct("-"))
        {
            var tok = _current; var op = tok.Text; Consume();
            var right = ParseTerm();
            left = new BinOpExpr(op, left, right, tok.Line, tok.Column);
        }
        return left;
    }

    private Expr ParseTerm()
    {
        var left = ParseFactor();
        while (IsPunct("*") || IsPunct("/") || IsPunct("//") || IsPunct("%"))
        {
            var tok = _current; var op = tok.Text; Consume();
            var right = ParseFactor();
            left = new BinOpExpr(op, left, right, tok.Line, tok.Column);
        }
        return left;
    }

    private Expr ParseFactor()
    {
        if (IsPunct("-") || IsPunct("+"))
        {
            var tok = _current; Consume();
            var operand = ParseFactor();
            return new UnaryExpr(tok.Text, operand, tok.Line, tok.Column);
        }
        return ParsePostfix();
    }

    private Expr ParsePostfix()
    {
        var expr = ParsePrimary();
        while (true)
        {
            // Jinja binds `|filter` and `is test` at postfix precedence — the
            // same level as `.member`, `[idx]`, `(args)`. That's why
            // `messages|length - 1` parses as `(messages|length) - 1` and not
            // as `messages|(length - 1)`.
            if (IsPunct("|"))
            {
                Consume();
                if (_current.Kind != ExprTokKind.Ident)
                    throw Err("Expected filter name after '|'.");
                var name = _current.Text;
                var fline = _current.Line; var fcol = _current.Column;
                Consume();
                var fargs = new List<Expr>();
                if (IsPunct("("))
                {
                    Consume();
                    if (!IsPunct(")"))
                    {
                        fargs.Add(ParseExpression());
                        while (IsPunct(",")) { Consume(); fargs.Add(ParseExpression()); }
                    }
                    ExpectPunct(")");
                }
                expr = new FilterExpr(expr, name, fargs, fline, fcol);
            }
            else if (IsKeyword("is"))
            {
                Consume();
                var negated = false;
                if (IsKeyword("not")) { negated = true; Consume(); }
                if (_current.Kind != ExprTokKind.Ident)
                    throw Err("Expected test name after 'is'.");
                var tname = _current.Text;
                var tline = _current.Line; var tcol = _current.Column;
                Consume();

                var targs = new List<Expr>();
                if (IsPunct("("))
                {
                    Consume();
                    if (!IsPunct(")"))
                    {
                        targs.Add(ParseExpression());
                        while (IsPunct(",")) { Consume(); targs.Add(ParseExpression()); }
                    }
                    ExpectPunct(")");
                }

                expr = new TestExpr(expr, tname, negated, targs, tline, tcol);
            }
            else if (IsPunct("."))
            {
                var tok = _current; Consume();
                if (_current.Kind != ExprTokKind.Ident)
                    throw Err("Expected identifier after '.'.");
                var name = _current.Text;
                Consume();
                expr = new MemberExpr(expr, name, tok.Line, tok.Column);
            }
            else if (IsPunct("["))
            {
                var tok = _current; Consume();

                // Either simple index (`x[expr]`) or slice (`x[start:stop:step]`
                // with any piece optional). We tentatively parse a start
                // expression (unless the first token is already `:`, in which
                // case start is implicit), then branch on whether we see `:`
                // (slice) or `]` (index).
                Expr? start = null;
                if (!IsPunct(":")) start = ParseExpression();

                if (IsPunct("]"))
                {
                    ExpectPunct("]");
                    expr = new IndexExpr(expr, start!, tok.Line, tok.Column);
                }
                else
                {
                    ExpectPunct(":");
                    Expr? stop = null;
                    if (!IsPunct(":") && !IsPunct("]")) stop = ParseExpression();
                    Expr? step = null;
                    if (IsPunct(":"))
                    {
                        Consume();
                        if (!IsPunct("]")) step = ParseExpression();
                    }
                    ExpectPunct("]");
                    expr = new SliceExpr(expr, start, stop, step, tok.Line, tok.Column);
                }
            }
            else if (IsPunct("("))
            {
                var tok = _current; Consume();
                var args = new List<Expr>();
                var kwargs = new List<(string Name, Expr Value)>();
                if (!IsPunct(")"))
                {
                    ParseCallArg(args, kwargs);
                    while (IsPunct(","))
                    {
                        Consume();
                        if (IsPunct(")")) break; // trailing comma
                        ParseCallArg(args, kwargs);
                    }
                }
                ExpectPunct(")");
                expr = new CallExpr(expr, args, kwargs, tok.Line, tok.Column);
            }
            else break;
        }
        return expr;
    }

    private void ParseCallArg(List<Expr> args, List<(string, Expr)> kwargs)
    {
        // kwarg form: ident '=' expr — but only if the next token after ident is '='
        // AND '=' isn't part of a comparison operator (== handled as Punct "==", so a bare "=" is safe).
        if (_current.Kind == ExprTokKind.Ident && PeekPunct("="))
        {
            var name = _current.Text;
            Consume(); // ident
            Consume(); // '='
            var value = ParseExpression();
            kwargs.Add((name, value));
        }
        else
        {
            args.Add(ParseExpression());
        }
    }

    private Expr ParsePrimary()
    {
        var t = _current;

        if (t.Kind == ExprTokKind.Number)
        {
            Consume();
            if (t.Text.Contains('.'))
            {
                var dv = double.Parse(t.Text, CultureInfo.InvariantCulture);
                return new LiteralExpr(dv, t.Line, t.Column);
            }
            var iv = long.Parse(t.Text, CultureInfo.InvariantCulture);
            return new LiteralExpr(iv, t.Line, t.Column);
        }
        if (t.Kind == ExprTokKind.String)
        {
            Consume();
            return new LiteralExpr(t.Text, t.Line, t.Column);
        }
        if (t.Kind == ExprTokKind.Ident)
        {
            Consume();
            object? lit = t.Text switch
            {
                "true" or "True" => true,
                "false" or "False" => false,
                "none" or "None" => null,
                _ => _sentinel,
            };
            if (!ReferenceEquals(lit, _sentinel))
            {
                return new LiteralExpr(lit, t.Line, t.Column);
            }
            return new IdentExpr(t.Text, t.Line, t.Column);
        }
        if (t.Text == "(")
        {
            Consume();
            var inner = ParseExpression();
            ExpectPunct(")");
            return inner;
        }
        if (t.Text == "[")
        {
            Consume();
            var items = new List<Expr>();
            if (!IsPunct("]"))
            {
                items.Add(ParseExpression());
                while (IsPunct(","))
                {
                    Consume();
                    if (IsPunct("]")) break;
                    items.Add(ParseExpression());
                }
            }
            ExpectPunct("]");
            return new ListLiteralExpr(items, t.Line, t.Column);
        }
        if (t.Text == "{")
        {
            Consume();
            var pairs = new List<(Expr, Expr)>();
            if (!IsPunct("}"))
            {
                ParseDictEntry(pairs);
                while (IsPunct(","))
                {
                    Consume();
                    if (IsPunct("}")) break;
                    ParseDictEntry(pairs);
                }
            }
            ExpectPunct("}");
            return new DictLiteralExpr(pairs, t.Line, t.Column);
        }

        throw Err($"Unexpected '{t.Text}'.");
    }

    private void ParseDictEntry(List<(Expr, Expr)> pairs)
    {
        var key = ParseExpression();
        ExpectPunct(":");
        var val = ParseExpression();
        pairs.Add((key, val));
    }

    // ----- token helpers ---------------------------------------------

    private static readonly object _sentinel = new();

    private bool IsKeyword(string s) => _current.Kind == ExprTokKind.Ident && _current.Text == s;
    private bool IsPunct(string s) => _current.Kind == ExprTokKind.Punct && _current.Text == s;

    private bool PeekPunct(string s)
    {
        EnsurePeek();
        return _peek.Kind == ExprTokKind.Punct && _peek.Text == s;
    }

    private bool PeekKeyword(string s)
    {
        EnsurePeek();
        return _peek.Kind == ExprTokKind.Ident && _peek.Text == s;
    }

    private void EnsurePeek()
    {
        if (_peekValid) return;
        _peek = _lex.Next();
        _peekValid = true;
    }

    private void Consume()
    {
        if (_peekValid)
        {
            _current = _peek;
            _peekValid = false;
        }
        else
        {
            _current = _lex.Next();
        }
    }

    private void ExpectKeyword(string kw)
    {
        if (!IsKeyword(kw)) throw Err($"Expected '{kw}'.");
        Consume();
    }

    private void ExpectPunct(string p)
    {
        if (!IsPunct(p)) throw Err($"Expected '{p}', got '{_current.Text}'.");
        Consume();
    }

    private JinjaException Err(string msg) => new(msg, _current.Line, _current.Column);
}
