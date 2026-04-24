using System.IO;
using System.Threading;
using System.Threading.Tasks;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services.Exporters;

/// <summary>
/// Format-agnostic export contract. One implementation per output format;
/// the list of available exporters lives in
/// <see cref="ConversationExporterRegistry"/>.
/// </summary>
public interface IConversationExporter
{
    /// <summary>Stable id used in menu plumbing / commands (e.g. <c>"markdown"</c>).</summary>
    string FormatId { get; }

    /// <summary>Human-readable label shown in the Export submenu (e.g. <c>"Markdown (.md)"</c>).</summary>
    string DisplayName { get; }

    /// <summary>File extension without leading dot (e.g. <c>"md"</c>).</summary>
    string FileExtension { get; }

    /// <summary>
    /// Serialise <paramref name="conversation"/>'s active branch into
    /// <paramref name="output"/> using the shape appropriate to the
    /// target format. Does not close the stream — the caller owns it.
    /// </summary>
    Task ExportAsync(
        Conversation conversation,
        Stream output,
        ExportOptions options,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Toggles for what the export includes. Defaults favour a clean, shareable
/// transcript: reasoning/thinking spans are stripped from content and the
/// separate reasoning channel is omitted by default (exports are typically
/// for sharing, where internal reasoning is noise). System prompt and
/// attachment list are included by default; callers wanting a slimmer
/// export flip pieces off at the call site.
/// </summary>
public sealed record ExportOptions
{
    public bool IncludeSystemPrompt   { get; init; } = true;
    public bool IncludeReasoning      { get; init; } = false;
    public bool IncludeAttachmentList { get; init; } = true;
    public bool IncludeTimestamps     { get; init; } = true;
    public bool IncludeStats          { get; init; } = false;

    public static ExportOptions Default { get; } = new();
}
