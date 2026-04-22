using System.Collections.Generic;

namespace LlamaCpp.Bindings.Jinja;

/// <summary>
/// Parsed Jinja template. Parse once via <see cref="Parse"/>, render many
/// times via <see cref="Render"/>. Not thread-safe across concurrent
/// renders — create one <see cref="JinjaTemplate"/> per source and call it
/// serially, or clone by re-parsing.
/// </summary>
public sealed class JinjaTemplate
{
    private readonly IReadOnlyList<Node> _ast;

    private JinjaTemplate(IReadOnlyList<Node> ast) => _ast = ast;

    /// <summary>
    /// Compile a Jinja source string. Throws <see cref="JinjaException"/>
    /// with line/column information if the source is malformed.
    /// </summary>
    public static JinjaTemplate Parse(string source)
    {
        var ast = JinjaParser.Parse(source);
        return new JinjaTemplate(ast);
    }

    /// <summary>
    /// Render the template against a variable context. Keys map to Jinja
    /// identifiers; values can be <c>string</c>, <c>bool</c>, <c>long</c>,
    /// <c>double</c>, <c>IList&lt;object?&gt;</c>, or
    /// <c>IDictionary&lt;string, object?&gt;</c>. Unknown names evaluate to
    /// the Jinja undefined sentinel (truthy-false, empty when stringified).
    /// </summary>
    public string Render(IReadOnlyDictionary<string, object?> context)
    {
        var interp = new Interpreter(context);
        return interp.Render(_ast);
    }
}
