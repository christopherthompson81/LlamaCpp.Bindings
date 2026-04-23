namespace LlamaCpp.Bindings.LlamaChat.Models;

/// <summary>
/// A binary payload attached to a <see cref="ChatTurn"/>. Image-only for v1
/// (mime type is "image/*"); the shape is ready for audio and other media
/// once those native paths come online.
/// </summary>
/// <remarks>
/// Serialised in <c>conversations.json</c> as base64 via
/// <see cref="System.Text.Json"/>'s default <c>byte[]</c> handling. Small
/// thumbnails inline cleanly; enormous files will inflate the store, but we
/// intentionally keep the payload with the turn so conversations are
/// self-contained (no external file paths that can go stale).
/// </remarks>
public sealed record Attachment(
    byte[] Data,
    string MimeType,
    string? FileName = null)
{
    public bool IsImage => MimeType.StartsWith("image/", System.StringComparison.OrdinalIgnoreCase);
}
