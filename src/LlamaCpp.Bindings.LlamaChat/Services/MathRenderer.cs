using System.Collections.Concurrent;
using Avalonia.Media.Imaging;
using CSharpMath.SkiaSharp;
using SkiaSharp;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Renders a LaTeX math string to an Avalonia <see cref="Bitmap"/> via
/// CSharpMath.SkiaSharp. The path is LaTeX → <see cref="MathPainter"/> →
/// PNG stream → <see cref="Bitmap"/>. Bitmaps are cached on
/// (latex, displayStyle, fontSize).
///
/// The bitmap is always rendered in pure white — callers tint it at draw
/// time by using the bitmap as an Avalonia <c>OpacityMask</c> over a
/// <c>DynamicResource Foreground</c> brush. That keeps math glyphs in
/// lockstep with the current theme without re-rendering, and avoids a
/// startup race where the theme variant hadn't yet resolved when a
/// stored conversation was first laid out — under the previous "bake
/// the color into the bitmap" approach the cached PNG would freeze
/// at the fallback color (visibly black on dark backgrounds).
/// </summary>
public static class MathRenderer
{
    private record struct CacheKey(string LaTeX, bool Display, float FontSize);
    private static readonly ConcurrentDictionary<CacheKey, Bitmap?> Cache = new();

    /// <summary>
    /// Default inline math rendering size. Display math scales up from this.
    /// Picked empirically to sit comfortably alongside our 13px body text.
    /// </summary>
    public const float InlineFontSize = 16f;
    public const float DisplayFontSize = 20f;

    public static Bitmap? Render(string latex, bool display, float? fontSizeOverride = null)
    {
        if (string.IsNullOrWhiteSpace(latex)) return null;
        var size = fontSizeOverride ?? (display ? DisplayFontSize : InlineFontSize);
        var key = new CacheKey(latex, display, size);
        return Cache.GetOrAdd(key, static k => RenderUncached(k.LaTeX, k.Display, k.FontSize));
    }

    private static Bitmap? RenderUncached(string latex, bool display, float fontSize)
    {
        try
        {
            var painter = new MathPainter
            {
                LaTeX = latex,
                FontSize = fontSize,
                // Pure white with full alpha — callers use the bitmap's alpha
                // channel as an opacity mask and supply the actual foreground
                // through the theme's DynamicResource.
                TextColor = SKColors.White,
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
}
