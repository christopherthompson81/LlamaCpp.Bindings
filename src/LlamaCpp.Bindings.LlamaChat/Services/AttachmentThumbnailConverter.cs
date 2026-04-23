using System;
using System.Globalization;
using System.IO;
using Avalonia.Data.Converters;
using Avalonia.Media.Imaging;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Binds an <see cref="Attachment"/>'s byte buffer to an Avalonia
/// <see cref="Bitmap"/> for the <c>Image.Source</c> property. Used by the
/// thumbnail row in the compose bar and by the user-bubble template.
/// </summary>
/// <remarks>
/// We decode on every conversion pass — acceptable because attachments are
/// chat-sized (a handful of images per turn). A memoising converter would
/// be the next step if users start attaching dozens of large images per turn.
/// </remarks>
internal sealed class AttachmentThumbnailConverter : IValueConverter
{
    public static readonly AttachmentThumbnailConverter Instance = new();

    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not Attachment a || a.Data.Length == 0) return null;
        try
        {
            using var ms = new MemoryStream(a.Data, writable: false);
            return new Bitmap(ms);
        }
        catch
        {
            // Corrupt image bytes — avoid crashing the UI thread. The user
            // sees a missing thumbnail but the bubble still renders.
            return null;
        }
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}
