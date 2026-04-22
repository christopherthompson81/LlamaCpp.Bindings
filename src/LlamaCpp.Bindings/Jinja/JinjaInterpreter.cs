using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace LlamaCpp.Bindings.Jinja;

// ====================================================================
// Runtime value model
// ====================================================================
// Values flow as `object?`. The runtime cares about a small fixed set
// of shapes: null, bool, long/double, string, IList<object?>,
// IDictionary<string, object?>, Macro, Undefined, or a callable thunk
// (Func<IReadOnlyList<object?>, IReadOnlyDictionary<string, object?>?, object?>).
// Everything else is coerced when possible.
// ====================================================================

/// <summary>
/// Singleton sentinel for "variable not defined". Distinct from <c>null</c>,
/// which in Jinja is <c>none</c> (defined but empty).
/// </summary>
internal sealed class Undefined
{
    public static readonly Undefined Instance = new();
    private Undefined() { }
    public override string ToString() => "";
}

/// <summary>
/// User-defined callable from a <c>{% macro %}</c> statement. Renders its
/// body against a fresh scope whose parent is the scope the macro was
/// defined in — that's Jinja's lexical scope for macros.
/// </summary>
internal sealed class Macro
{
    public string Name { get; }
    public IReadOnlyList<MacroParam> Params { get; }
    public IReadOnlyList<Node> Body { get; }
    public Scope DefinitionScope { get; }

    public Macro(string name, IReadOnlyList<MacroParam> pars,
                 IReadOnlyList<Node> body, Scope defScope)
    {
        Name = name; Params = pars; Body = body; DefinitionScope = defScope;
    }
}

/// <summary>
/// A callable implemented in C# — exposed as a top-level identifier
/// (e.g. <c>namespace</c>, <c>raise_exception</c>) or as a bound method on
/// a value (e.g. <c>"x".strip</c>).
/// </summary>
internal delegate object? Callable(
    IReadOnlyList<object?> args,
    IReadOnlyDictionary<string, object?>? kwargs);

// ====================================================================
// Scope
// ====================================================================

internal sealed class Scope
{
    public Dictionary<string, object?> Locals { get; } = new();
    public Scope? Parent { get; }
    public Scope(Scope? parent = null) { Parent = parent; }

    public bool TryGet(string name, out object? value)
    {
        for (var s = this; s is not null; s = s.Parent)
        {
            if (s.Locals.TryGetValue(name, out value)) return true;
        }
        value = null;
        return false;
    }

    public void Set(string name, object? value) => Locals[name] = value;

    /// <summary>
    /// Overwrite a binding in whichever ancestor scope owns it; if none does,
    /// set it locally. This is the write-through semantics Jinja uses for
    /// <c>{% set x = ... %}</c> at top-level (not inside a <c>for</c>).
    /// </summary>
    public void Assign(string name, object? value)
    {
        for (var s = this; s is not null; s = s.Parent)
        {
            if (s.Locals.ContainsKey(name)) { s.Locals[name] = value; return; }
        }
        Locals[name] = value;
    }
}

// ====================================================================
// Interpreter
// ====================================================================

internal sealed class Interpreter
{
    private readonly StringBuilder _out = new();
    private Scope _scope;

    public Interpreter(IReadOnlyDictionary<string, object?> context)
    {
        _scope = new Scope();
        foreach (var kv in context) _scope.Set(kv.Key, kv.Value);

        // Built-in callables visible as identifiers.
        _scope.Set("namespace", (Callable)BuiltIn_Namespace);
        _scope.Set("raise_exception", (Callable)BuiltIn_RaiseException);
        _scope.Set("range", (Callable)BuiltIn_Range);
    }

    public string Render(IReadOnlyList<Node> nodes)
    {
        foreach (var n in nodes) ExecuteNode(n);
        return _out.ToString();
    }

    // ----------------- Block execution -----------------

    private void ExecuteNode(Node node)
    {
        switch (node)
        {
            case TextNode t:
                _out.Append(t.Text);
                break;
            case OutputNode o:
                _out.Append(Stringify(EvaluateExpression(o.Expression)));
                break;
            case IfNode ifn:
                ExecuteIf(ifn);
                break;
            case ForNode fn:
                ExecuteFor(fn);
                break;
            case SetNode sn:
                ExecuteSet(sn);
                break;
            case SetBlockNode sbn:
                ExecuteSetBlock(sbn);
                break;
            case MacroNode mn:
                _scope.Set(mn.Name, new Macro(mn.Name, mn.Params, mn.Body, _scope));
                break;
            default:
                throw new JinjaException($"Unhandled node: {node.GetType().Name}",
                    node.Line, node.Column);
        }
    }

    private void ExecuteIf(IfNode node)
    {
        foreach (var (cond, body) in node.Branches)
        {
            if (cond is null || Truthy(EvaluateExpression(cond)))
            {
                foreach (var n in body) ExecuteNode(n);
                return;
            }
        }
    }

    private void ExecuteFor(ForNode node)
    {
        var iterVal = EvaluateExpression(node.Iterable);
        var items = Iterate(iterVal).ToList();

        if (items.Count == 0)
        {
            if (node.ElseBody is not null)
                foreach (var n in node.ElseBody) ExecuteNode(n);
            return;
        }

        var outer = _scope;
        try
        {
            for (var i = 0; i < items.Count; i++)
            {
                _scope = new Scope(outer);

                // Bind loop variables. If multiple, unpack the item.
                if (node.LoopVars.Count == 1)
                {
                    _scope.Set(node.LoopVars[0], items[i]);
                }
                else
                {
                    var tuple = Iterate(items[i]).ToList();
                    for (var v = 0; v < node.LoopVars.Count; v++)
                    {
                        _scope.Set(node.LoopVars[v], v < tuple.Count ? tuple[v] : Undefined.Instance);
                    }
                }

                // Expose `loop` with index/first/last/length/previtem/nextitem.
                var loop = new Dictionary<string, object?>
                {
                    ["index0"] = (long)i,
                    ["index"] = (long)(i + 1),
                    ["first"] = i == 0,
                    ["last"] = i == items.Count - 1,
                    ["length"] = (long)items.Count,
                    ["previtem"] = i > 0 ? items[i - 1] : Undefined.Instance,
                    ["nextitem"] = i < items.Count - 1 ? items[i + 1] : Undefined.Instance,
                };
                _scope.Set("loop", loop);

                foreach (var n in node.Body) ExecuteNode(n);
            }
        }
        finally
        {
            _scope = outer;
        }
    }

    private void ExecuteSet(SetNode node)
    {
        var value = EvaluateExpression(node.Value);

        if (node.TargetPath.Count == 1)
        {
            _scope.Set(node.TargetPath[0], value);
            return;
        }

        // Dotted target: resolve all but the last, then set on the container.
        if (!_scope.TryGet(node.TargetPath[0], out var container))
            throw new JinjaException(
                $"Undefined name '{node.TargetPath[0]}' in assignment.",
                node.Line, node.Column);

        for (var i = 1; i < node.TargetPath.Count - 1; i++)
        {
            container = GetAttribute(container, node.TargetPath[i], node.Line, node.Column);
        }

        var last = node.TargetPath[^1];
        if (container is IDictionary<string, object?> d)
        {
            d[last] = value;
        }
        else
        {
            throw new JinjaException(
                $"Cannot assign to attribute '{last}' on value of type "
                + $"'{container?.GetType().Name ?? "none"}'. Use a namespace(...) for mutable state.",
                node.Line, node.Column);
        }
    }

    private void ExecuteSetBlock(SetBlockNode node)
    {
        // Render into a sub-StringBuilder, then assign the string.
        var saved = _out.Length;
        foreach (var n in node.Body) ExecuteNode(n);
        var rendered = _out.ToString(saved, _out.Length - saved);
        _out.Length = saved;
        _scope.Set(node.Target, rendered);
    }

    // ----------------- Expression evaluation -----------------

    private object? EvaluateExpression(Expr expr)
    {
        switch (expr)
        {
            case LiteralExpr l:
                return l.Value;

            case IdentExpr id:
                return _scope.TryGet(id.Name, out var v) ? v : Undefined.Instance;

            case MemberExpr m:
                return GetAttribute(EvaluateExpression(m.Target), m.Name, m.Line, m.Column);

            case IndexExpr ix:
                return GetIndex(EvaluateExpression(ix.Target), EvaluateExpression(ix.Index),
                    ix.Line, ix.Column);

            case SliceExpr sx:
                return DoSlice(EvaluateExpression(sx.Target),
                    sx.Start is null ? null : EvaluateExpression(sx.Start),
                    sx.Stop is null ? null : EvaluateExpression(sx.Stop),
                    sx.Step is null ? null : EvaluateExpression(sx.Step),
                    sx.Line, sx.Column);

            case CallExpr c:
                return EvaluateCall(c);

            case FilterExpr f:
                return EvaluateFilter(f);

            case TestExpr t:
                return EvaluateTest(t);

            case BinOpExpr b:
                return EvaluateBinOp(b);

            case UnaryExpr u:
                return EvaluateUnary(u);

            case TernaryExpr tern:
                return Truthy(EvaluateExpression(tern.Cond))
                    ? EvaluateExpression(tern.Then)
                    : EvaluateExpression(tern.Else);

            case ListLiteralExpr ll:
                return ll.Items.Select(EvaluateExpression).ToList<object?>();

            case DictLiteralExpr dl:
            {
                var dict = new Dictionary<string, object?>();
                foreach (var (k, v2) in dl.Pairs)
                {
                    var kObj = EvaluateExpression(k);
                    dict[Stringify(kObj)] = EvaluateExpression(v2);
                }
                return dict;
            }

            default:
                throw new JinjaException("Unknown expression kind.", expr.Line, expr.Column);
        }
    }

    private object? EvaluateCall(CallExpr call)
    {
        // Pre-evaluate args + kwargs.
        var args = call.Args.Select(EvaluateExpression).ToList<object?>();
        Dictionary<string, object?>? kwargs = null;
        if (call.Kwargs.Count > 0)
        {
            kwargs = new Dictionary<string, object?>();
            foreach (var (name, val) in call.Kwargs) kwargs[name] = EvaluateExpression(val);
        }

        // Special case: bound method on a string / dict / list — detect via the call target.
        if (call.Target is MemberExpr m)
        {
            var receiver = EvaluateExpression(m.Target);
            if (TryInvokeMethod(receiver, m.Name, args, out var result))
                return result;
            // Fall through — the member access may have yielded a callable value.
        }

        var target = EvaluateExpression(call.Target);
        return Invoke(target, args, kwargs, call.Line, call.Column);
    }

    private object? Invoke(
        object? target,
        IReadOnlyList<object?> args,
        IReadOnlyDictionary<string, object?>? kwargs,
        int line, int col)
    {
        switch (target)
        {
            case Callable fn:
                return fn(args, kwargs);
            case Macro macro:
                return InvokeMacro(macro, args, kwargs);
            default:
                throw new JinjaException(
                    $"Value of type '{target?.GetType().Name ?? "none"}' is not callable.",
                    line, col);
        }
    }

    private string InvokeMacro(Macro macro,
        IReadOnlyList<object?> args, IReadOnlyDictionary<string, object?>? kwargs)
    {
        // Fresh scope parented on the macro's definition site (lexical scope).
        var savedScope = _scope;
        var macroScope = new Scope(macro.DefinitionScope);

        for (var i = 0; i < macro.Params.Count; i++)
        {
            var p = macro.Params[i];
            object? bound;
            if (i < args.Count) bound = args[i];
            else if (kwargs is not null && kwargs.TryGetValue(p.Name, out var kv)) bound = kv;
            else if (p.Default is not null)
            {
                _scope = macroScope;
                try { bound = EvaluateExpression(p.Default); }
                finally { _scope = savedScope; }
            }
            else bound = Undefined.Instance;

            macroScope.Set(p.Name, bound);
        }

        // Capture output emitted during the macro body.
        var saved = _out.Length;
        _scope = macroScope;
        try
        {
            foreach (var n in macro.Body) ExecuteNode(n);
        }
        finally
        {
            _scope = savedScope;
        }

        var rendered = _out.ToString(saved, _out.Length - saved);
        _out.Length = saved;
        return rendered;
    }

    // ----------------- Binary / unary -----------------

    private object? EvaluateBinOp(BinOpExpr b)
    {
        // Short-circuit for and/or.
        if (b.Op == "and")
        {
            var l = EvaluateExpression(b.Left);
            return Truthy(l) ? EvaluateExpression(b.Right) : l;
        }
        if (b.Op == "or")
        {
            var l = EvaluateExpression(b.Left);
            return Truthy(l) ? l : EvaluateExpression(b.Right);
        }

        var left = EvaluateExpression(b.Left);
        var right = EvaluateExpression(b.Right);

        switch (b.Op)
        {
            case "~":
                return Stringify(left) + Stringify(right);

            case "+":
                if (left is string ls && right is string rs) return ls + rs;
                return Arithmetic(left, right, b.Op, b.Line, b.Column);

            case "-":
            case "*":
            case "/":
            case "//":
            case "%":
                return Arithmetic(left, right, b.Op, b.Line, b.Column);

            case "==": return ValueEquals(left, right);
            case "!=": return !ValueEquals(left, right);

            case "<": return Compare(left, right, b.Line, b.Column) < 0;
            case ">": return Compare(left, right, b.Line, b.Column) > 0;
            case "<=": return Compare(left, right, b.Line, b.Column) <= 0;
            case ">=": return Compare(left, right, b.Line, b.Column) >= 0;

            case "in": return Contains(right, left);
            case "not in": return !Contains(right, left);

            default:
                throw new JinjaException($"Unknown operator '{b.Op}'.", b.Line, b.Column);
        }
    }

    private object? EvaluateUnary(UnaryExpr u)
    {
        var v = EvaluateExpression(u.Operand);
        return u.Op switch
        {
            "not" => !Truthy(v),
            "-" => Negate(v, u.Line, u.Column),
            "+" => v,
            _ => throw new JinjaException($"Unknown unary '{u.Op}'.", u.Line, u.Column),
        };
    }

    // ----------------- Tests (`is defined`, `is none`, etc.) -----------------

    private object? EvaluateTest(TestExpr t)
    {
        bool result = t.Name switch
        {
            "defined" => EvaluateExpression(t.Target) is not Undefined,
            "undefined" => EvaluateExpression(t.Target) is Undefined,
            "none" => EvaluateExpression(t.Target) is null,
            "string" => EvaluateExpression(t.Target) is string,
            "number" => EvaluateExpression(t.Target) is long or int or double or float or decimal,
            "integer" => EvaluateExpression(t.Target) is long or int,
            "float" => EvaluateExpression(t.Target) is double or float or decimal,
            "sequence" or "iterable" => IsSequenceOrIterable(EvaluateExpression(t.Target)),
            "mapping" => EvaluateExpression(t.Target) is IDictionary<string, object?>,
            "true" => EvaluateExpression(t.Target) is bool bT && bT,
            "false" => EvaluateExpression(t.Target) is bool bF && !bF,
            "boolean" => EvaluateExpression(t.Target) is bool,
            "callable" => EvaluateExpression(t.Target) is Callable or Macro,
            "odd" => EvaluateExpression(t.Target) is long lo && lo % 2 != 0,
            "even" => EvaluateExpression(t.Target) is long le && le % 2 == 0,
            _ => throw new JinjaException($"Unknown test '{t.Name}'.", t.Line, t.Column),
        };
        return t.Negated ? !result : result;
    }

    private static bool IsSequenceOrIterable(object? v) =>
        v is string or IList<object?> or IEnumerable and not IDictionary<string, object?>;

    // ----------------- Filters -----------------

    private object? EvaluateFilter(FilterExpr f)
    {
        var value = EvaluateExpression(f.Target);
        var args = f.Args.Select(EvaluateExpression).ToList<object?>();

        return f.Name switch
        {
            "trim" => Stringify(value).Trim(),
            "upper" => Stringify(value).ToUpperInvariant(),
            "lower" => Stringify(value).ToLowerInvariant(),
            "length" or "count" => (long)Length(value, f.Line, f.Column),
            "string" => Stringify(value),
            "safe" => value, // no-op: we never escape HTML
            "tojson" => ToJson(value, args),
            "default" => value is Undefined || value is null ? (args.Count > 0 ? args[0] : "") : value,
            "join" => Join(value, args.Count > 0 ? Stringify(args[0]) : ""),
            "replace" => args.Count >= 2 ? Stringify(value).Replace(Stringify(args[0]), Stringify(args[1])) : Stringify(value),
            "first" => FirstLast(value, first: true, f.Line, f.Column),
            "last" => FirstLast(value, first: false, f.Line, f.Column),
            "reverse" => ReverseSeq(value),
            _ => throw new JinjaException($"Unknown filter '{f.Name}'.", f.Line, f.Column),
        };
    }

    private static string ToJson(object? v, IReadOnlyList<object?> args)
    {
        // Pretty-print when indent kwarg present? Qwen3 template just uses
        // `| tojson` without args, so we always emit compact JSON.
        var opts = new JsonSerializerOptions { WriteIndented = false };
        return JsonSerializer.Serialize(v is Undefined ? null : v, opts);
    }

    private static string Join(object? seq, string sep)
    {
        if (seq is null or Undefined) return "";
        var sb = new StringBuilder();
        var first = true;
        foreach (var item in Iterate(seq))
        {
            if (!first) sb.Append(sep);
            sb.Append(Stringify(item));
            first = false;
        }
        return sb.ToString();
    }

    private static object? FirstLast(object? seq, bool first, int line, int col)
    {
        if (seq is null or Undefined) return Undefined.Instance;
        object? result = Undefined.Instance;
        foreach (var item in Iterate(seq))
        {
            result = item;
            if (first) return result;
        }
        return result;
    }

    private static IList<object?> ReverseSeq(object? seq)
    {
        var list = Iterate(seq).ToList<object?>();
        list.Reverse();
        return list;
    }

    // ----------------- Built-in callables -----------------

    private static object? BuiltIn_Namespace(
        IReadOnlyList<object?> args, IReadOnlyDictionary<string, object?>? kwargs)
    {
        var ns = new Dictionary<string, object?>();
        if (kwargs is not null) foreach (var kv in kwargs) ns[kv.Key] = kv.Value;
        return ns;
    }

    private static object? BuiltIn_RaiseException(
        IReadOnlyList<object?> args, IReadOnlyDictionary<string, object?>? kwargs)
    {
        var msg = args.Count > 0 ? Stringify(args[0]) : "template raised";
        throw new JinjaException("Template error: " + msg);
    }

    private static object? BuiltIn_Range(
        IReadOnlyList<object?> args, IReadOnlyDictionary<string, object?>? kwargs)
    {
        long start = 0, stop = 0, step = 1;
        if (args.Count == 1) stop = ToLong(args[0]);
        else if (args.Count == 2) { start = ToLong(args[0]); stop = ToLong(args[1]); }
        else if (args.Count >= 3) { start = ToLong(args[0]); stop = ToLong(args[1]); step = ToLong(args[2]); }

        var list = new List<object?>();
        if (step == 0) return list;
        if (step > 0) for (var i = start; i < stop; i += step) list.Add(i);
        else for (var i = start; i > stop; i += step) list.Add(i);
        return list;
    }

    // ----------------- Member / index / slice -----------------

    private static object? GetAttribute(object? target, string name, int line, int col)
    {
        if (target is null) return Undefined.Instance;
        if (target is Undefined) return Undefined.Instance;

        if (target is IDictionary<string, object?> d)
        {
            return d.TryGetValue(name, out var v) ? v : Undefined.Instance;
        }

        // Bound-method surface for strings. The returned value is a Callable that
        // captures `target` in its closure. Other method-lookups go through
        // TryInvokeMethod on the CallExpr path.
        if (target is string s)
        {
            Callable? m = MakeStringMethod(s, name);
            if (m is not null) return m;
        }

        return Undefined.Instance;
    }

    private static object? GetIndex(object? target, object? index, int line, int col)
    {
        if (target is IList<object?> list)
        {
            var i = ToLong(index);
            if (i < 0) i += list.Count;
            if (i < 0 || i >= list.Count) return Undefined.Instance;
            return list[(int)i];
        }
        if (target is string s)
        {
            var i = ToLong(index);
            if (i < 0) i += s.Length;
            if (i < 0 || i >= s.Length) return Undefined.Instance;
            return s[(int)i].ToString();
        }
        if (target is IDictionary<string, object?> d)
        {
            var key = Stringify(index);
            return d.TryGetValue(key, out var v) ? v : Undefined.Instance;
        }
        throw new JinjaException(
            $"Cannot index value of type '{target?.GetType().Name ?? "none"}'.",
            line, col);
    }

    private static object? DoSlice(
        object? target, object? startV, object? stopV, object? stepV, int line, int col)
    {
        long step = stepV is null ? 1 : ToLong(stepV);
        if (step == 0)
            throw new JinjaException("Slice step cannot be zero.", line, col);

        if (target is string s)
        {
            var (a, b) = ResolveSliceBounds(s.Length, startV, stopV, step);
            var sb = new StringBuilder();
            if (step > 0)
                for (var i = a; i < b; i += (int)step) sb.Append(s[i]);
            else
                for (var i = a; i > b; i += (int)step) sb.Append(s[i]);
            return sb.ToString();
        }
        if (target is IList<object?> list)
        {
            var (a, b) = ResolveSliceBounds(list.Count, startV, stopV, step);
            var outList = new List<object?>();
            if (step > 0)
                for (var i = a; i < b; i += (int)step) outList.Add(list[i]);
            else
                for (var i = a; i > b; i += (int)step) outList.Add(list[i]);
            return outList;
        }
        throw new JinjaException(
            $"Cannot slice value of type '{target?.GetType().Name ?? "none"}'.",
            line, col);
    }

    private static (int a, int b) ResolveSliceBounds(int length, object? startV, object? stopV, long step)
    {
        int a, b;
        if (step > 0)
        {
            a = startV is null ? 0 : (int)Math.Max(0, Normalise(ToLong(startV), length));
            b = stopV is null ? length : (int)Math.Min(length, Normalise(ToLong(stopV), length));
        }
        else
        {
            a = startV is null ? length - 1 : (int)Math.Min(length - 1, Normalise(ToLong(startV), length));
            b = stopV is null ? -1 : (int)Math.Max(-1, Normalise(ToLong(stopV), length));
        }
        return (a, b);

        static long Normalise(long i, int len) => i < 0 ? i + len : i;
    }

    // ----------------- String methods -----------------

    private static Callable? MakeStringMethod(string s, string name) => name switch
    {
        "startswith" => (args, _) =>
            args.Count > 0 && s.StartsWith(Stringify(args[0]), StringComparison.Ordinal),
        "endswith" => (args, _) =>
            args.Count > 0 && s.EndsWith(Stringify(args[0]), StringComparison.Ordinal),
        "strip" => (args, _) => args.Count == 0 ? s.Trim() : s.Trim(Stringify(args[0]).ToCharArray()),
        "lstrip" => (args, _) => args.Count == 0 ? s.TrimStart() : s.TrimStart(Stringify(args[0]).ToCharArray()),
        "rstrip" => (args, _) => args.Count == 0 ? s.TrimEnd() : s.TrimEnd(Stringify(args[0]).ToCharArray()),
        "upper" => (args, _) => s.ToUpperInvariant(),
        "lower" => (args, _) => s.ToLowerInvariant(),
        "replace" => (args, _) => args.Count >= 2 ? s.Replace(Stringify(args[0]), Stringify(args[1])) : s,
        "split" => (args, _) =>
        {
            if (args.Count == 0)
                return (object)s.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Cast<object?>().ToList();
            var sep = Stringify(args[0]);
            var parts = s.Split(new[] { sep }, StringSplitOptions.None);
            return parts.Cast<object?>().ToList();
        },
        "join" => (args, _) =>
        {
            if (args.Count == 0) return s;
            var parts = Iterate(args[0]).Select(Stringify);
            return string.Join(s, parts);
        },
        _ => null,
    };

    private bool TryInvokeMethod(
        object? receiver, string name, IReadOnlyList<object?> args, out object? result)
    {
        if (receiver is string s)
        {
            var m = MakeStringMethod(s, name);
            if (m is not null) { result = m(args, null); return true; }
        }
        result = null;
        return false;
    }

    // ----------------- Helpers -----------------

    internal static string Stringify(object? v) => v switch
    {
        null => "",
        Undefined => "",
        string s => s,
        bool b => b ? "True" : "False",
        double d => d.ToString("R", CultureInfo.InvariantCulture),
        long l => l.ToString(CultureInfo.InvariantCulture),
        int i => i.ToString(CultureInfo.InvariantCulture),
        IList<object?> list => "[" + string.Join(", ", list.Select(Stringify)) + "]",
        IDictionary<string, object?> d => "{" + string.Join(", ",
            d.Select(kv => "'" + kv.Key + "': " + Stringify(kv.Value))) + "}",
        _ => v.ToString() ?? "",
    };

    internal static bool Truthy(object? v) => v switch
    {
        null => false,
        Undefined => false,
        bool b => b,
        long l => l != 0,
        int i => i != 0,
        double d => d != 0 && !double.IsNaN(d),
        string s => s.Length > 0,
        IList<object?> list => list.Count > 0,
        IDictionary<string, object?> d => d.Count > 0,
        _ => true,
    };

    private static bool ValueEquals(object? a, object? b)
    {
        if (a is Undefined) a = null;
        if (b is Undefined) b = null;
        if (a is null || b is null) return a is null && b is null;

        // Numeric cross-compare.
        if (IsNumber(a) && IsNumber(b)) return ToDouble(a) == ToDouble(b);

        if (a is string sa && b is string sb) return sa == sb;
        if (a is bool ba && b is bool bb) return ba == bb;

        return a.Equals(b);
    }

    private static int Compare(object? a, object? b, int line, int col)
    {
        if (IsNumber(a) && IsNumber(b))
            return ToDouble(a).CompareTo(ToDouble(b));
        if (a is string sa && b is string sb)
            return string.CompareOrdinal(sa, sb);
        throw new JinjaException(
            $"Can't compare '{a?.GetType().Name}' and '{b?.GetType().Name}'.", line, col);
    }

    private static bool Contains(object? container, object? needle)
    {
        if (container is string cs)
        {
            if (needle is string ns) return cs.Contains(ns, StringComparison.Ordinal);
            return false;
        }
        if (container is IDictionary<string, object?> d)
        {
            return d.ContainsKey(Stringify(needle));
        }
        if (container is IEnumerable en && container is not string)
        {
            foreach (var item in en)
            {
                if (ValueEquals(item, needle)) return true;
            }
            return false;
        }
        return false;
    }

    private static object? Arithmetic(object? a, object? b, string op, int line, int col)
    {
        if (!IsNumber(a) || !IsNumber(b))
        {
            // Jinja's `+` on two strings is string concat, but we already handled
            // that in EvaluateBinOp. Anything else that reaches here is a type error.
            throw new JinjaException(
                $"Cannot {op} values of type '{a?.GetType().Name}' and '{b?.GetType().Name}'.",
                line, col);
        }

        // Preserve int type when both sides are ints; widen to double otherwise.
        if (a is long la && b is long lb)
        {
            return op switch
            {
                "+" => la + lb,
                "-" => la - lb,
                "*" => la * lb,
                "//" => lb == 0 ? throw new JinjaException("Division by zero.", line, col) : la / lb,
                "%" => lb == 0 ? throw new JinjaException("Division by zero.", line, col) : la % lb,
                "/" => lb == 0 ? throw new JinjaException("Division by zero.", line, col) : (double)la / lb,
                _ => throw new JinjaException($"Unknown op '{op}'.", line, col),
            };
        }
        var da = ToDouble(a); var db = ToDouble(b);
        return op switch
        {
            "+" => da + db,
            "-" => da - db,
            "*" => da * db,
            "/" => db == 0 ? throw new JinjaException("Division by zero.", line, col) : da / db,
            "//" => db == 0 ? throw new JinjaException("Division by zero.", line, col) : Math.Floor(da / db),
            "%" => db == 0 ? throw new JinjaException("Division by zero.", line, col) : da % db,
            _ => throw new JinjaException($"Unknown op '{op}'.", line, col),
        };
    }

    private static object? Negate(object? v, int line, int col)
    {
        if (v is long l) return -l;
        if (v is int i) return -(long)i;
        if (v is double d) return -d;
        throw new JinjaException("Cannot negate non-number.", line, col);
    }

    internal static IEnumerable<object?> Iterate(object? value)
    {
        switch (value)
        {
            case null:
            case Undefined:
                yield break;
            case string s:
                foreach (var c in s) yield return c.ToString();
                break;
            case IList<object?> list:
                foreach (var v in list) yield return v;
                break;
            case IDictionary<string, object?> d:
                foreach (var k in d.Keys) yield return k;
                break;
            case IEnumerable en:
                foreach (var v in en)
                {
                    if (v is KeyValuePair<string, object?> kv) yield return kv.Key;
                    else yield return v;
                }
                break;
            default:
                throw new JinjaException($"Value of type '{value.GetType().Name}' is not iterable.");
        }
    }

    private static int Length(object? v, int line, int col) => v switch
    {
        string s => s.Length,
        IList<object?> list => list.Count,
        IDictionary<string, object?> d => d.Count,
        null or Undefined => 0,
        _ => throw new JinjaException(
            $"No length for value of type '{v.GetType().Name}'.", line, col),
    };

    private static bool IsNumber(object? v) => v is long or int or double or float or decimal;

    private static long ToLong(object? v) => v switch
    {
        long l => l,
        int i => i,
        double d => (long)d,
        float f => (long)f,
        bool b => b ? 1 : 0,
        string s when long.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var r) => r,
        null or Undefined => 0,
        _ => throw new JinjaException($"Cannot convert '{v.GetType().Name}' to integer."),
    };

    private static double ToDouble(object? v) => v switch
    {
        long l => l,
        int i => i,
        double d => d,
        float f => f,
        bool b => b ? 1 : 0,
        _ => 0.0,
    };
}
