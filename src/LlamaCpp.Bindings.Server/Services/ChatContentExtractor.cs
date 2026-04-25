using System.Buffers;
using System.Globalization;
using LlamaCpp.Bindings.Server.Models;

namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Walks the request's message list, resolves each message's
/// <see cref="MessageContent"/> into a plain text body (with mmproj
/// media markers spliced in at image positions), and collects every
/// image's raw bytes in visitation order.
/// </summary>
/// <remarks>
/// <para>OpenAI's array shape interleaves text and image parts in a
/// single message. We lower it to the format llama.cpp's mtmd helper
/// wants: a text string with one marker token per image, plus a
/// parallel list of bitmaps. <see cref="MtmdContext.DefaultMediaMarker"/>
/// (typically <c>&lt;__media__&gt;</c>) is the marker upstream uses.</para>
///
/// <para>Image URLs accepted: <c>data:image/&lt;type&gt;;base64,&lt;payload&gt;</c>
/// URLs decode inline (no network). Any other scheme is rejected — fetching
/// remote URLs opens a whole fetch/timeout/MIME-sniffing surface that's
/// out of scope for V1. Remote-URL support is tracked in issue #19's
/// follow-up.</para>
/// </remarks>
public static class ChatContentExtractor
{
    /// <summary>
    /// Upper bound on a single base64 image payload. 10 MiB is comfortable
    /// for a 4K PNG; anything larger almost certainly isn't a chat image.
    /// </summary>
    public const int MaxImageBytes = 10 * 1024 * 1024;

    public sealed class Result
    {
        /// <summary>Extracted image bytes, one entry per image part, in visitation order.</summary>
        public List<byte[]> Images { get; } = new();

        /// <summary>True if any message contained at least one image part.</summary>
        public bool HasImages => Images.Count > 0;
    }

    /// <summary>
    /// Walk <paramref name="messages"/> in order, flattening every
    /// message's content into a plain string in-place. Returns collected
    /// image bytes. Throws <see cref="ArgumentException"/> for
    /// unsupported part types or malformed data URLs; the caller maps
    /// this to HTTP 400.
    /// </summary>
    public static Result FlattenAndExtract(IList<ChatMessageDto> messages, string mediaMarker)
    {
        var result = new Result();
        foreach (var msg in messages)
        {
            if (msg.Content is null) continue;

            if (msg.Content.Parts is not { Count: > 0 } parts)
            {
                // Either the message had a plain string (Content.Text set
                // by the converter) or it was null. Nothing to flatten.
                continue;
            }

            var sb = new System.Text.StringBuilder();
            foreach (var part in parts)
            {
                switch (part.Type)
                {
                    case "text":
                        if (!string.IsNullOrEmpty(part.Text)) sb.Append(part.Text);
                        break;
                    case "image_url":
                        if (part.ImageUrl is null || string.IsNullOrEmpty(part.ImageUrl.Url))
                        {
                            throw new ArgumentException(
                                "image_url part missing image_url.url field.");
                        }
                        result.Images.Add(DecodeImageUrl(part.ImageUrl.Url));
                        sb.Append(mediaMarker);
                        break;
                    default:
                        throw new ArgumentException(
                            $"Unsupported content part type '{part.Type}'. " +
                            $"V1 accepts 'text' and 'image_url' only.");
                }
            }

            // Replace the parts array with the flattened string so the
            // chat template renderer (which expects string content) sees
            // a drop-in equivalent of the multi-part message.
            msg.Content = new MessageContent { Text = sb.ToString() };
        }
        return result;
    }

    private static byte[] DecodeImageUrl(string url)
    {
        const string DataPrefix = "data:";
        if (!url.StartsWith(DataPrefix, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                "V1 only supports data: URLs for image_url entries " +
                "(e.g. data:image/png;base64,...). Remote URL fetch is tracked in issue #19.");
        }
        int commaIdx = url.IndexOf(',');
        if (commaIdx < 0)
        {
            throw new ArgumentException("Malformed data URL: missing ',' separator.");
        }
        var metadata = url.Substring(DataPrefix.Length, commaIdx - DataPrefix.Length);
        var payload = url[(commaIdx + 1)..];

        if (!metadata.Contains(";base64", StringComparison.OrdinalIgnoreCase))
        {
            // Non-base64 data URLs (e.g. percent-encoded) are valid per
            // RFC 2397 but nobody uses them for images — reject to keep
            // the parser simple.
            throw new ArgumentException(
                "Only base64-encoded data URLs are supported (e.g. " +
                "data:image/png;base64,...).");
        }

        // Guard against a payload that's obviously too big before we
        // allocate the output buffer.
        // Base64 decode length = ceil(encoded_len * 3 / 4) — minus padding.
        long approxBytes = (long)payload.Length * 3 / 4;
        if (approxBytes > MaxImageBytes)
        {
            throw new ArgumentException(
                $"image_url data URL exceeds {MaxImageBytes / (1024 * 1024)} MiB limit " +
                $"(approximate payload size: {approxBytes / (1024 * 1024)} MiB).");
        }

        try
        {
            return Convert.FromBase64String(payload);
        }
        catch (FormatException ex)
        {
            throw new ArgumentException("Malformed base64 payload in image_url data URL.", ex);
        }
    }
}
