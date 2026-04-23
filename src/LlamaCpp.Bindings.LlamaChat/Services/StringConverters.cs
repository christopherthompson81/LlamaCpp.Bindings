using System;
using System.Globalization;
using Avalonia.Data.Converters;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Returns the first alphanumeric character of a string, uppercased. Used by
/// the MCP active-servers avatar strip to render a one-letter initial inside
/// the circular badge.
/// </summary>
public sealed class FirstLetterConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        var s = value as string;
        if (string.IsNullOrEmpty(s)) return "?";
        foreach (var ch in s)
        {
            if (char.IsLetterOrDigit(ch)) return char.ToUpperInvariant(ch).ToString();
        }
        return "?";
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
