using System.IO;
using System.Threading;
using System.Threading.Tasks;
using LlamaCpp.Bindings.LlamaChat.Models;
using QuestPDF.Fluent;
using QuestPDF.Helpers;
using QuestPDF.Infrastructure;

namespace LlamaCpp.Bindings.LlamaChat.Services.Exporters;

/// <summary>
/// Exports the active branch as a paginated PDF. Content is rendered as
/// plain text — no markdown expansion. Role labels get coloured headers
/// and reasoning blocks get an indented italic block for visual
/// separation. QuestPDF handles pagination; long messages split across
/// pages cleanly.
/// </summary>
public sealed class PdfExporter : IConversationExporter
{
    public string FormatId      => "pdf";
    public string DisplayName   => "PDF (.pdf)";
    public string FileExtension => "pdf";

    static PdfExporter()
    {
        // QuestPDF defaults to a community license for free-to-use projects
        // (revenue < $1M). Setting it here avoids the one-time startup
        // licence-nag log entry.
        QuestPDF.Settings.License = LicenseType.Community;
    }

    // Chain of font families to search for every glyph, left to right.
    // QuestPDF falls through the list whenever the current font lacks a
    // glyph for the character — so bullet markers, CJK, and emoji find a
    // home even though the headline font (Lato) is Latin-only. Picks are
    // OS-portable: at least one in each tier is installed on Linux,
    // Windows, and macOS.
    private static readonly string[] FontFamilyChain = new[]
    {
        "Lato",                    // QuestPDF default, clean for Latin
        "DejaVu Sans",             // Linux baseline Unicode
        "Segoe UI",                // Windows baseline Unicode
        "Noto Sans",               // broad Unicode coverage
        "Noto Color Emoji",        // colour emoji (Linux/Android)
        "Segoe UI Emoji",          // emoji (Windows)
        "Apple Color Emoji",       // emoji (macOS)
        "Symbola",                 // text symbols fallback
    };

    public Task ExportAsync(
        Conversation conversation,
        Stream output,
        ExportOptions options,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => WritePdf(conversation, output, options), cancellationToken);
    }

    private static void WritePdf(Conversation conversation, Stream output, ExportOptions options)
    {
        var path = ConversationBranch.ActivePath(conversation);

        Document.Create(container =>
        {
            container.Page(page =>
            {
                page.Size(PageSizes.A4);
                page.Margin(2, Unit.Centimetre);
                page.DefaultTextStyle(x => x.FontSize(10).FontFamily(FontFamilyChain));

                page.Header().PaddingBottom(10).Column(col =>
                {
                    col.Item().Text(conversation.Title).FontSize(16).Bold();
                    if (options.IncludeTimestamps)
                    {
                        col.Item().Text(t =>
                        {
                            t.Span($"Created {conversation.CreatedAt.ToLocalTime():yyyy-MM-dd HH:mm}")
                             .FontSize(9).FontColor(Colors.Grey.Darken1);
                            t.Span(" · ").FontSize(9).FontColor(Colors.Grey.Darken1);
                            t.Span($"Updated {conversation.UpdatedAt.ToLocalTime():yyyy-MM-dd HH:mm}")
                             .FontSize(9).FontColor(Colors.Grey.Darken1);
                        });
                    }
                });

                page.Content().Column(col =>
                {
                    col.Spacing(14);
                    foreach (var turn in path)
                    {
                        if (turn.Role == TurnRole.System && !options.IncludeSystemPrompt) continue;
                        col.Item().Element(c => RenderTurn(c, turn, options));
                    }
                });

                page.Footer().AlignCenter().Text(t =>
                {
                    t.Span("Page ").FontSize(8).FontColor(Colors.Grey.Darken1);
                    t.CurrentPageNumber().FontSize(8).FontColor(Colors.Grey.Darken1);
                    t.Span(" of ").FontSize(8).FontColor(Colors.Grey.Darken1);
                    t.TotalPages().FontSize(8).FontColor(Colors.Grey.Darken1);
                });
            });
        }).GeneratePdf(output);
    }

    private static void RenderTurn(IContainer container, ChatTurn turn, ExportOptions options)
    {
        var accent = turn.Role switch
        {
            TurnRole.System    => Colors.Amber.Darken2,
            TurnRole.User      => Colors.Blue.Darken2,
            TurnRole.Assistant => Colors.Grey.Darken3,
            TurnRole.Tool      => Colors.Purple.Darken2,
            _                  => Colors.Grey.Darken2,
        };

        container.BorderLeft(2).BorderColor(accent).PaddingLeft(8).Column(col =>
        {
            col.Spacing(4);
            col.Item().Text(t =>
            {
                t.Span(ConversationBranch.RoleLabel(turn.Role)).Bold().FontColor(accent);
                if (options.IncludeTimestamps)
                {
                    t.Span($"  ·  {turn.CreatedAt.ToLocalTime():yyyy-MM-dd HH:mm:ss}")
                     .FontSize(9).FontColor(Colors.Grey.Darken1);
                }
            });

            if (options.IncludeReasoning && !string.IsNullOrWhiteSpace(turn.Reasoning))
            {
                col.Item().PaddingLeft(6).BorderLeft(1).BorderColor(Colors.Grey.Medium)
                   .PaddingLeft(6).Text(turn.Reasoning!)
                   .Italic().FontColor(Colors.Grey.Darken2).FontSize(9);
            }

            var cleanContent = ContentSanitizer.StripReasoningSpans(turn.Content);
            if (!string.IsNullOrEmpty(cleanContent))
            {
                col.Item().Element(cc => MarkdigPdfRenderer.Render(cc, cleanContent));
            }

            if (options.IncludeAttachmentList && turn.Attachments is { Count: > 0 })
            {
                var names = string.Join(", ",
                    turn.Attachments.ConvertAll(a => a.FileName ?? "(unnamed)"));
                col.Item().Text($"Attachments: {names}")
                   .FontSize(9).Italic().FontColor(Colors.Grey.Darken1);
            }
        });
    }
}
