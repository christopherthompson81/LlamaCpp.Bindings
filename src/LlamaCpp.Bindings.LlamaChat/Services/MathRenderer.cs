using System;
using System.Collections.Concurrent;
using System.IO;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using CSharpMath.SkiaSharp;
using SkiaSharp;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Renders a LaTeX math string to an Avalonia <see cref="Bitmap"/> via
/// CSharpMath.SkiaSharp. The path is LaTeX → <see cref="MathPainter"/> →
/// PNG stream → <see cref="Bitmap"/>. Bitmaps are cached on
/// (latex, displayStyle, fontSize, foregroundArgb).
///
/// Theme-flip behaviour: the foreground colour is baked into the bitmap at
/// render time. Messages re-render on the next streaming tick / conversation
/// switch; a theme-flip while a fully static message is on screen leaves
/// stale-coloured glyphs until the containing bubble next re-renders.
/// </summary>
public static class MathRenderer
{
    private record struct CacheKey(string LaTeX, bool Display, float FontSize, uint Argb);
    private static readonly ConcurrentDictionary<CacheKey, Bitmap?> Cache = new();

    /// <summary>
    /// Default inline math rendering size. Display math scales up from this.
    /// Picked empirically to sit comfortably alongside our 13px body text.
    /// </summary>
    public const float InlineFontSize = 16f;
    public const float DisplayFontSize = 20f;

    public static Bitmap? Render(string latex, bool display, Color foreground, float? fontSizeOverride = null)
    {
        if (string.IsNullOrWhiteSpace(latex)) return null;
        var size = fontSizeOverride ?? (display ? DisplayFontSize : InlineFontSize);
        var key = new CacheKey(latex, display, size, foreground.ToUInt32());
        return Cache.GetOrAdd(key, static k => RenderUncached(k.LaTeX, k.Display, k.FontSize, k.Argb));
    }

    private static Bitmap? RenderUncached(string latex, bool display, float fontSize, uint argb)
    {
        try
        {
            var painter = new MathPainter
            {
                LaTeX = latex,
                FontSize = fontSize,
                TextColor = new SKColor(argb),
            };
            // MathPainter defaults to display style; toggle off for inline.
            if (!display)
            {
                painter.LineStyle = CSharpMath.Atom.LineStyle.Text;
            }

            using var stream = painter.DrawAsStream(format: SKEncodedImageFormat.Png);
            if (stream is null) return null;
            stream.Position = 0;
            return new Bitmap(stream);
        }
        catch
        {
            // Invalid LaTeX, unknown command, etc. — caller renders as fallback text.
            return null;
        }
    }

    /// <summary>Resolve the current theme's <c>Foreground</c> brush to a concrete Color.</summary>
    public static Color ResolveThemeForeground()
    {
        var app = Application.Current;
        if (app is not null
            && app.Resources.TryGetResource("Foreground", app.ActualThemeVariant, out var obj)
            && obj is ISolidColorBrush brush)
        {
            return brush.Color;
        }
        return Colors.Black;
    }
}
