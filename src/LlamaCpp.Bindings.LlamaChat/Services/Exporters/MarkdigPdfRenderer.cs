using System.Linq;
using Markdig;
using Markdig.Extensions.Tables;
using Markdig.Syntax;
using Markdig.Syntax.Inlines;
using QuestPDF.Fluent;
using QuestPDF.Helpers;
using QuestPDF.Infrastructure;

namespace LlamaCpp.Bindings.LlamaChat.Services.Exporters;

/// <summary>
/// Walks a Markdig AST and emits QuestPDF layout. Handles the markdown
/// subset models actually emit: headings, paragraphs, fenced code, lists,
/// pipe tables, blockquotes, plus inline bold/italic/code/links. Unknown
/// blocks fall through as their literal text — better than skipping so
/// nothing silently disappears from the export.
/// </summary>
internal static class MarkdigPdfRenderer
{
    private static readonly MarkdownPipeline _pipeline = new MarkdownPipelineBuilder()
        .UsePipeTables()
        .UseAdvancedExtensions()
        .DisableHtml()
        .Build();

    public static void Render(IContainer container, string markdown)
    {
        if (string.IsNullOrEmpty(markdown))
        {
            container.Text(string.Empty);
            return;
        }

        var doc = Markdown.Parse(markdown, _pipeline);
        container.Column(col =>
        {
            col.Spacing(5);
            foreach (var block in doc)
                col.Item().Element(c => RenderBlock(c, block));
        });
    }

    private static void RenderBlock(IContainer c, Block block)
    {
        switch (block)
        {
            case HeadingBlock h:
                var size = h.Level switch
                {
                    1 => 16, 2 => 14, 3 => 12, 4 => 11, _ => 10,
                };
                c.Text(t =>
                {
                    t.DefaultTextStyle(x => x.FontSize(size).Bold());
                    RenderInlines(t, h.Inline);
                });
                break;

            case ParagraphBlock p:
                c.Text(t => RenderInlines(t, p.Inline));
                break;

            case FencedCodeBlock fc:
                c.Background(Colors.Grey.Lighten3).Padding(6).Text(t =>
                {
                    t.DefaultTextStyle(x => x.FontFamily(Fonts.CourierNew).FontSize(9));
                    t.Span(fc.Lines.ToString());
                });
                break;

            case CodeBlock cb:
                c.Background(Colors.Grey.Lighten3).Padding(6).Text(t =>
                {
                    t.DefaultTextStyle(x => x.FontFamily(Fonts.CourierNew).FontSize(9));
                    t.Span(cb.Lines.ToString());
                });
                break;

            case QuoteBlock qb:
                c.BorderLeft(2).BorderColor(Colors.Grey.Medium).PaddingLeft(8)
                 .Column(col =>
                 {
                     col.Spacing(3);
                     foreach (var inner in qb)
                         col.Item().Element(cc => RenderBlock(cc, inner));
                 });
                break;

            case ListBlock lb:
                c.Column(col =>
                {
                    col.Spacing(2);
                    int idx = 1;
                    foreach (var item in lb)
                    {
                        if (item is ListItemBlock li)
                        {
                            var marker = lb.IsOrdered ? $"{idx}." : "•";
                            col.Item().Row(row =>
                            {
                                row.ConstantItem(18).Text(marker);
                                row.RelativeItem().Column(inner =>
                                {
                                    foreach (var sub in li)
                                        inner.Item().Element(cc => RenderBlock(cc, sub));
                                });
                            });
                            idx++;
                        }
                    }
                });
                break;

            case Table t:
                RenderTable(c, t);
                break;

            case ThematicBreakBlock:
                c.LineHorizontal(0.5f).LineColor(Colors.Grey.Medium);
                break;

            default:
                // Fallback: dump the block's raw text if it has Lines.
                if (block is LeafBlock lb2 && lb2.Lines.Count > 0)
                    c.Text(lb2.Lines.ToString());
                break;
        }
    }

    private static void RenderTable(IContainer c, Table table)
    {
        var colCount = table
            .OfType<TableRow>()
            .Select(r => r.Count)
            .DefaultIfEmpty(0)
            .Max();
        if (colCount == 0) return;

        c.Table(qt =>
        {
            qt.ColumnsDefinition(cols =>
            {
                for (int i = 0; i < colCount; i++)
                    cols.RelativeColumn();
            });

            foreach (var rowBlock in table)
            {
                if (rowBlock is not TableRow row) continue;
                foreach (var cellBlock in row)
                {
                    if (cellBlock is not TableCell cell) continue;
                    var isHeader = row.IsHeader;
                    qt.Cell().Element(cc =>
                    {
                        var container = cc.Border(0.5f).BorderColor(Colors.Grey.Medium).Padding(4);
                        if (isHeader) container = container.Background(Colors.Grey.Lighten3);
                        container.Column(col =>
                        {
                            foreach (var sub in cell)
                                col.Item().Element(sc => RenderBlock(sc, sub));
                        });
                    });
                }
            }
        });
    }

    private static void RenderInlines(TextDescriptor text, ContainerInline? inlines, bool bold = false, bool italic = false)
    {
        if (inlines is null) return;
        foreach (var inline in inlines)
            RenderInline(text, inline, bold, italic);
    }

    private static void RenderInline(TextDescriptor text, Inline inline, bool bold, bool italic)
    {
        switch (inline)
        {
            case LiteralInline lit:
                var span = text.Span(lit.Content.ToString());
                if (bold)   span.Bold();
                if (italic) span.Italic();
                break;

            case EmphasisInline em:
                // Markdig: DelimiterCount 2 = strong (**), 1 = italic (*).
                var nextBold   = bold   || em.DelimiterCount >= 2;
                var nextItalic = italic || em.DelimiterCount == 1;
                RenderInlines(text, em, nextBold, nextItalic);
                break;

            case CodeInline code:
                var codeSpan = text.Span(code.Content).FontFamily(Fonts.CourierNew);
                if (bold)   codeSpan.Bold();
                if (italic) codeSpan.Italic();
                break;

            case LinkInline link:
                // Render link text, followed by the URL in parens so the
                // destination survives a paper-only read. Inline content
                // may itself be bold/italic/etc — recurse.
                RenderInlines(text, link, bold, italic);
                if (!string.IsNullOrEmpty(link.Url))
                    text.Span($" ({link.Url})").FontColor(Colors.Blue.Darken2).FontSize(9);
                break;

            case LineBreakInline:
                text.Line(string.Empty);
                break;

            case HtmlInline:
            case HtmlEntityInline:
                // DisableHtml in pipeline already strips most; ignore the rest.
                break;

            default:
                // Unknown inline — fall back to its Markdig-sourced text if available.
                if (inline is ContainerInline ci)
                    RenderInlines(text, ci, bold, italic);
                break;
        }
    }
}
