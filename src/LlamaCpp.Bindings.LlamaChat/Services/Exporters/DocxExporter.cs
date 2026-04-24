using System.IO;
using System.Threading;
using System.Threading.Tasks;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services.Exporters;

/// <summary>
/// Exports the active branch as a Word <c>.docx</c>. Content is written
/// as plain runs — no attempt to render the model's markdown into Word
/// styling beyond bold role labels and italic reasoning blocks. Users
/// who want rich formatting can import the Markdown export into Word
/// instead; this path is for people who need a .docx drop into their
/// existing workflow.
/// </summary>
public sealed class DocxExporter : IConversationExporter
{
    public string FormatId      => "docx";
    public string DisplayName   => "Word (.docx)";
    public string FileExtension => "docx";

    public Task ExportAsync(
        Conversation conversation,
        Stream output,
        ExportOptions options,
        CancellationToken cancellationToken = default)
    {
        // OpenXml's API is synchronous and pulls hard on the disk; run on
        // the thread pool to keep the UI thread responsive for the ~100ms
        // it takes on a medium conversation.
        return Task.Run(() => WriteDocx(conversation, output, options), cancellationToken);
    }

    private static void WriteDocx(Conversation conversation, Stream output, ExportOptions options)
    {
        using var doc = WordprocessingDocument.Create(output, WordprocessingDocumentType.Document);
        var main = doc.AddMainDocumentPart();
        main.Document = new Document();
        var body = main.Document.AppendChild(new Body());

        // Title
        AppendParagraph(body, conversation.Title, bold: true, sizeHalfPoints: 40);

        if (options.IncludeTimestamps)
        {
            AppendParagraph(body,
                $"Created: {conversation.CreatedAt.ToLocalTime():yyyy-MM-dd HH:mm} · " +
                $"Updated: {conversation.UpdatedAt.ToLocalTime():yyyy-MM-dd HH:mm}",
                italic: true, sizeHalfPoints: 20);
        }
        AppendParagraph(body, string.Empty);

        foreach (var turn in ConversationBranch.ActivePath(conversation))
        {
            if (turn.Role == TurnRole.System && !options.IncludeSystemPrompt) continue;

            var header = ConversationBranch.RoleLabel(turn.Role);
            if (options.IncludeTimestamps)
                header += $"  ·  {turn.CreatedAt.ToLocalTime():yyyy-MM-dd HH:mm:ss}";
            AppendParagraph(body, header, bold: true, sizeHalfPoints: 26);

            if (options.IncludeReasoning && !string.IsNullOrWhiteSpace(turn.Reasoning))
            {
                AppendParagraph(body, "Reasoning:", italic: true, bold: true, sizeHalfPoints: 20);
                AppendMultiline(body, turn.Reasoning!, italic: true);
                AppendParagraph(body, string.Empty);
            }

            var cleanContent = ContentSanitizer.StripReasoningSpans(turn.Content);
            if (!string.IsNullOrEmpty(cleanContent))
            {
                // Render the content's markdown through the OpenXml walker
                // so headings / bold / code / tables / lists survive into
                // Word rather than appearing as source-text noise.
                MarkdigOpenXmlRenderer.Render(body, cleanContent);
            }

            if (options.IncludeAttachmentList && turn.Attachments is { Count: > 0 })
            {
                var names = string.Join(", ", turn.Attachments.ConvertAll(a => a.FileName ?? "(unnamed)"));
                AppendParagraph(body, $"Attachments: {names}", italic: true, sizeHalfPoints: 18);
            }

            AppendParagraph(body, string.Empty);
        }

        main.Document.Save();
    }

    private static void AppendMultiline(Body body, string text, bool italic = false)
    {
        foreach (var line in text.Split('\n'))
            AppendParagraph(body, line.TrimEnd('\r'), italic: italic);
    }

    private static void AppendParagraph(
        Body body, string text,
        bool bold = false, bool italic = false, int sizeHalfPoints = 22)
    {
        var p = body.AppendChild(new Paragraph());
        var r = p.AppendChild(new Run());
        var props = new RunProperties();
        if (bold)   props.Append(new Bold());
        if (italic) props.Append(new Italic());
        props.Append(new FontSize { Val = sizeHalfPoints.ToString() });
        r.Append(props);
        r.Append(new Text(text) { Space = SpaceProcessingModeValues.Preserve });
    }
}
