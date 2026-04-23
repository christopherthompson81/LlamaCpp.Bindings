using System;
using System.Globalization;
using System.Text.Json;
using Avalonia.Data.Converters;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// <see cref="IValueConverter"/> that pretty-prints a <see cref="JsonElement"/>
/// (typically an MCP tool's input schema) as indented JSON. Used by the
/// Settings Tools tab to render schemas in a readable form. String inputs
/// are re-parsed to catch and reformat compact JSON blobs.
/// </summary>
public sealed class JsonPrettyPrinter : IValueConverter
{
    private static readonly JsonSerializerOptions Indented = new() { WriteIndented = true };

    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        try
        {
            return value switch
            {
                JsonElement el => JsonSerializer.Serialize(el, Indented),
                string s when !string.IsNullOrWhiteSpace(s)
                    => JsonSerializer.Serialize(JsonDocument.Parse(s).RootElement, Indented),
                null => string.Empty,
                _ => value.ToString() ?? string.Empty,
            };
        }
        catch
        {
            return value?.ToString() ?? string.Empty;
        }
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
