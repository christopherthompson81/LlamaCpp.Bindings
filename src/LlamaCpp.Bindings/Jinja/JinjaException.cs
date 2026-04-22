using System;

namespace LlamaCpp.Bindings.Jinja;

/// <summary>
/// Raised for any Jinja parse or evaluation error. <see cref="Line"/> and
/// <see cref="Column"/> point into the original template source when known.
/// </summary>
public sealed class JinjaException : Exception
{
    public int Line { get; }
    public int Column { get; }

    public JinjaException(string message, int line = 0, int column = 0)
        : base(FormatMessage(message, line, column))
    {
        Line = line;
        Column = column;
    }

    public JinjaException(string message, int line, int column, Exception? inner)
        : base(FormatMessage(message, line, column), inner)
    {
        Line = line;
        Column = column;
    }

    private static string FormatMessage(string msg, int line, int col) =>
        line > 0 ? $"{msg} (line {line}, col {col})" : msg;
}
