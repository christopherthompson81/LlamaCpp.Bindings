using System.Collections.Generic;

namespace LlamaCpp.Bindings.Jinja;

// ================================================================
// AST nodes for a Jinja-subset template engine. Two trees: one for
// block-level template structure (Node) and one for expressions (Expr).
// Both are immutable records — parsed once, rendered many times.
// ================================================================

internal abstract record Node(int Line, int Column);

internal sealed record TextNode(string Text, int Line, int Column) : Node(Line, Column);

internal sealed record OutputNode(Expr Expression, int Line, int Column) : Node(Line, Column);

internal sealed record IfNode(
    IReadOnlyList<(Expr? Condition, IReadOnlyList<Node> Body)> Branches,
    int Line, int Column) : Node(Line, Column);

internal sealed record ForNode(
    IReadOnlyList<string> LoopVars,
    Expr Iterable,
    IReadOnlyList<Node> Body,
    IReadOnlyList<Node>? ElseBody,
    int Line, int Column) : Node(Line, Column);

internal sealed record SetNode(
    IReadOnlyList<string> TargetPath,   // e.g. ["ns", "value"] for `ns.value`; single entry for `x`
    Expr Value,
    int Line, int Column) : Node(Line, Column);

/// <summary>
/// A captured block form: <c>{% set x %}...body...{% endset %}</c>. The body
/// is rendered to a string and stored as <c>x</c>. Qwen3.6 doesn't use this,
/// but it's cheap to include and many templates do.
/// </summary>
internal sealed record SetBlockNode(
    string Target,
    IReadOnlyList<Node> Body,
    int Line, int Column) : Node(Line, Column);

internal sealed record MacroNode(
    string Name,
    IReadOnlyList<MacroParam> Params,
    IReadOnlyList<Node> Body,
    int Line, int Column) : Node(Line, Column);

internal readonly record struct MacroParam(string Name, Expr? Default);

// ================================================================
// Expressions
// ================================================================

internal abstract record Expr(int Line, int Column);

internal sealed record LiteralExpr(object? Value, int Line, int Column) : Expr(Line, Column);

internal sealed record IdentExpr(string Name, int Line, int Column) : Expr(Line, Column);

internal sealed record MemberExpr(Expr Target, string Name, int Line, int Column) : Expr(Line, Column);

internal sealed record IndexExpr(Expr Target, Expr Index, int Line, int Column) : Expr(Line, Column);

internal sealed record SliceExpr(
    Expr Target, Expr? Start, Expr? Stop, Expr? Step, int Line, int Column) : Expr(Line, Column);

internal sealed record CallExpr(
    Expr Target,
    IReadOnlyList<Expr> Args,
    IReadOnlyList<(string Name, Expr Value)> Kwargs,
    int Line, int Column) : Expr(Line, Column);

/// <summary>
/// A filter application: <c>x | name</c> or <c>x | name(args...)</c>. Filters
/// are resolved by name against a built-in table during evaluation.
/// </summary>
internal sealed record FilterExpr(
    Expr Target,
    string Name,
    IReadOnlyList<Expr> Args,
    int Line, int Column) : Expr(Line, Column);

internal sealed record BinOpExpr(string Op, Expr Left, Expr Right, int Line, int Column) : Expr(Line, Column);

internal sealed record UnaryExpr(string Op, Expr Operand, int Line, int Column) : Expr(Line, Column);

internal sealed record TernaryExpr(Expr Cond, Expr Then, Expr Else, int Line, int Column) : Expr(Line, Column);

/// <summary>
/// A test: <c>x is name</c> or <c>x is not name</c>. <paramref name="Name"/>
/// resolves to a built-in (<c>defined</c>, <c>none</c>, <c>string</c>, etc.).
/// </summary>
internal sealed record TestExpr(
    Expr Target,
    string Name,
    bool Negated,
    IReadOnlyList<Expr> Args,
    int Line, int Column) : Expr(Line, Column);

internal sealed record ListLiteralExpr(IReadOnlyList<Expr> Items, int Line, int Column) : Expr(Line, Column);

internal sealed record DictLiteralExpr(
    IReadOnlyList<(Expr Key, Expr Value)> Pairs, int Line, int Column) : Expr(Line, Column);
