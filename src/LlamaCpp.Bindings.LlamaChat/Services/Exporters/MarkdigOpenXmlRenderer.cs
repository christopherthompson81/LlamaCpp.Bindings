using System.Linq;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Wordprocessing;
using Markdig;
using Markdig.Syntax;
using Markdig.Syntax.Inlines;
using MdTable     = Markdig.Extensions.Tables.Table;
using MdTableRow  = Markdig.Extensions.Tables.TableRow;
using MdTableCell = Markdig.Extensions.Tables.TableCell;

namespace LlamaCpp.Bindings.LlamaChat.Services.Exporters;

/// <summary>
/// Walks a Markdig AST and appends OpenXml Wordprocessing elements to a
/// <see cref="Body"/>. Handles headings (via direct run sizing — we avoid
/// custom styles so the output opens cleanly in every Word version),
/// paragraphs with inline bold/italic/code, fenced code blocks (mono
/// font), bulleted and numbered lists (as indented paragraphs prefixed
/// with a marker — not list-styled, but visually correct without needing
/// a numbering definitions part), pipe tables, and blockquotes.
/// </summary>
internal static class MarkdigOpenXmlRenderer
{
    private static readonly MarkdownPipeline _pipeline = new MarkdownPipelineBuilder()
        .UsePipeTables()
        .UseAdvancedExtensions()
        .DisableHtml()
        .Build();

    public static void Render(Body body, string markdown)
    {
        if (string.IsNullOrEmpty(markdown)) return;
        var doc = Markdown.Parse(markdown, _pipeline);
        foreach (var block in doc)
            RenderBlock(body, block, indent: 0);
    }

    private static void RenderBlock(Body body, Block block, int indent, bool inQuote = false)
    {
        switch (block)
        {
            case HeadingBlock h:
                var size = h.Level switch
                {
                    1 => 32, 2 => 28, 3 => 24, 4 => 22, _ => 20,
                };
                var headingP = body.AppendChild(new Paragraph());
                if (indent > 0) ApplyIndent(headingP, indent);
                RenderInlines(headingP, h.Inline, bold: true, sizeHalfPoints: size);
                break;

            case ParagraphBlock p:
                var para = body.AppendChild(new Paragraph());
                if (indent > 0) ApplyIndent(para, indent);
                if (inQuote) ApplyQuoteStyling(para);
                RenderInlines(para, p.Inline);
                break;

            case FencedCodeBlock fc:
                AppendCodeParagraph(body, fc.Lines.ToString(), indent);
                break;

            case CodeBlock cb:
                AppendCodeParagraph(body, cb.Lines.ToString(), indent);
                break;

            case QuoteBlock qb:
                foreach (var inner in qb)
                    RenderBlock(body, inner, indent + 1, inQuote: true);
                break;

            case ListBlock lb:
                RenderList(body, lb, indent);
                break;

            case MdTable t:
                RenderTable(body, t);
                break;

            case ThematicBreakBlock:
                var rule = body.AppendChild(new Paragraph());
                rule.AppendChild(new ParagraphProperties(
                    new ParagraphBorders(
                        new BottomBorder
                        {
                            Val = BorderValues.Single,
                            Size = 6,
                            Color = "CCCCCC",
                        })));
                break;

            default:
                if (block is LeafBlock lb2 && lb2.Lines.Count > 0)
                {
                    var fallback = body.AppendChild(new Paragraph());
                    AppendRun(fallback, lb2.Lines.ToString());
                }
                break;
        }
    }

    private static void RenderList(Body body, ListBlock list, int indent)
    {
        int idx = 1;
        foreach (var item in list)
        {
            if (item is not ListItemBlock li) continue;
            var marker = list.IsOrdered ? $"{idx}. " : "• ";

            bool first = true;
            foreach (var sub in li)
            {
                if (sub is ParagraphBlock pb)
                {
                    var p = body.AppendChild(new Paragraph());
                    ApplyIndent(p, indent + 1);
                    if (first)
                    {
                        AppendRun(p, marker);
                        first = false;
                    }
                    RenderInlines(p, pb.Inline);
                }
                else
                {
                    // nested blocks (code, sub-lists, quotes) — recurse.
                    RenderBlock(body, sub, indent + 1);
                    first = false;
                }
            }
            idx++;
        }
    }

    private static void RenderTable(Body body, MdTable markdigTable)
    {
        var table = body.AppendChild(new Table());

        // Table-level properties: borders + auto-fit.
        var tblProps = new TableProperties(
            new TableBorders(
                new TopBorder     { Val = BorderValues.Single, Size = 4, Color = "AAAAAA" },
                new BottomBorder  { Val = BorderValues.Single, Size = 4, Color = "AAAAAA" },
                new LeftBorder    { Val = BorderValues.Single, Size = 4, Color = "AAAAAA" },
                new RightBorder   { Val = BorderValues.Single, Size = 4, Color = "AAAAAA" },
                new InsideHorizontalBorder { Val = BorderValues.Single, Size = 4, Color = "CCCCCC" },
                new InsideVerticalBorder   { Val = BorderValues.Single, Size = 4, Color = "CCCCCC" }),
            new TableWidth { Width = "5000", Type = TableWidthUnitValues.Pct });
        table.AppendChild(tblProps);

        foreach (var rowBlock in markdigTable)
        {
            if (rowBlock is not MdTableRow row) continue;
            var tr = table.AppendChild(new TableRow());
            foreach (var cellBlock in row)
            {
                if (cellBlock is not MdTableCell cell) continue;
                var tc = tr.AppendChild(new TableCell());
                bool wroteAny = false;
                foreach (var sub in cell)
                {
                    if (sub is ParagraphBlock pb)
                    {
                        var cp = tc.AppendChild(new Paragraph());
                        RenderInlines(cp, pb.Inline, bold: row.IsHeader);
                        wroteAny = true;
                    }
                }
                if (!wroteAny)
                    tc.AppendChild(new Paragraph()); // empty cell stub
            }
        }

        // A trailing empty paragraph after a table is the standard OpenXml
        // pattern — without it, Word refuses to place anything below.
        body.AppendChild(new Paragraph());
    }

    private static void AppendCodeParagraph(Body body, string text, int indent)
    {
        foreach (var rawLine in text.Split('\n'))
        {
            var line = rawLine.TrimEnd('\r');
            var p = body.AppendChild(new Paragraph());
            if (indent > 0) ApplyIndent(p, indent);
            var pp = p.AppendChild(new ParagraphProperties());
            pp.AppendChild(new Shading
            {
                Val = ShadingPatternValues.Clear,
                Color = "auto",
                Fill = "F2F2F2",
            });
            var r = p.AppendChild(new Run());
            var rp = r.AppendChild(new RunProperties());
            rp.AppendChild(new RunFonts
            {
                Ascii = "Consolas",
                HighAnsi = "Consolas",
            });
            rp.AppendChild(new FontSize { Val = "18" });
            r.AppendChild(new Text(line) { Space = SpaceProcessingModeValues.Preserve });
        }
    }

    private static void ApplyIndent(Paragraph paragraph, int levels)
    {
        var pp = paragraph.GetFirstChild<ParagraphProperties>() ?? paragraph.AppendChild(new ParagraphProperties());
        pp.AppendChild(new Indentation { Left = (levels * 360).ToString() });
    }

    private static void ApplyQuoteStyling(Paragraph paragraph)
    {
        var pp = paragraph.GetFirstChild<ParagraphProperties>() ?? paragraph.AppendChild(new ParagraphProperties());
        pp.AppendChild(new ParagraphBorders(
            new LeftBorder
            {
                Val = BorderValues.Single,
                Size = 12,
                Color = "BBBBBB",
                Space = 4,
            }));
    }

    private static void RenderInlines(
        Paragraph paragraph, ContainerInline? inlines,
        bool bold = false, bool italic = false, int sizeHalfPoints = 22)
    {
        if (inlines is null) return;
        foreach (var inline in inlines)
            RenderInline(paragraph, inline, bold, italic, sizeHalfPoints);
    }

    private static void RenderInline(
        Paragraph p, Inline inline, bool bold, bool italic, int sizeHalfPoints)
    {
        switch (inline)
        {
            case LiteralInline lit:
                AppendRun(p, lit.Content.ToString(), bold, italic, sizeHalfPoints);
                break;

            case EmphasisInline em:
                var nextBold   = bold   || em.DelimiterCount >= 2;
                var nextItalic = italic || em.DelimiterCount == 1;
                RenderInlines(p, em, nextBold, nextItalic, sizeHalfPoints);
                break;

            case CodeInline code:
                AppendRun(p, code.Content, bold, italic, sizeHalfPoints, monospace: true);
                break;

            case LinkInline link:
                RenderInlines(p, link, bold, italic, sizeHalfPoints);
                if (!string.IsNullOrEmpty(link.Url))
                    AppendRun(p, $" ({link.Url})", bold, italic,
                              sizeHalfPoints: 18, colorHex: "3060A0");
                break;

            case LineBreakInline:
                p.AppendChild(new Run(new Break()));
                break;

            case HtmlInline:
            case HtmlEntityInline:
                // DisableHtml pipeline already strips most; skip residual.
                break;

            default:
                if (inline is ContainerInline ci)
                    RenderInlines(p, ci, bold, italic, sizeHalfPoints);
                break;
        }
    }

    private static void AppendRun(
        Paragraph p, string text,
        bool bold = false, bool italic = false,
        int sizeHalfPoints = 22,
        bool monospace = false,
        string? colorHex = null)
    {
        var r = p.AppendChild(new Run());
        var props = new RunProperties();
        if (bold)   props.Append(new Bold());
        if (italic) props.Append(new Italic());
        if (monospace)
        {
            props.Append(new RunFonts { Ascii = "Consolas", HighAnsi = "Consolas" });
            props.Append(new Shading
            {
                Val = ShadingPatternValues.Clear,
                Color = "auto",
                Fill = "F2F2F2",
            });
        }
        if (colorHex is not null)
            props.Append(new Color { Val = colorHex });
        props.Append(new FontSize { Val = sizeHalfPoints.ToString() });
        r.Append(props);
        r.Append(new Text(text) { Space = SpaceProcessingModeValues.Preserve });
    }
}
