using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Data;
using Avalonia.Layout;
using Avalonia.Markup.Xaml.MarkupExtensions;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Markdig;
using Markdig.Extensions.Mathematics;
using Markdig.Extensions.TaskLists;
using Markdig.Extensions.Tables;
using Markdig.Syntax;
using Markdig.Syntax.Inlines;
using MdInline = Markdig.Syntax.Inlines.Inline;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Parses a Markdown source with Markdig and walks the AST into an Avalonia
/// control tree. Designed to be cheap enough to re-run on every streaming
/// token: builds a flat <see cref="Panel"/> of block-level children, lets
/// inline-level content stay inside <see cref="TextBlock"/>s (so text flows
/// naturally), and pulls all colors from the app's DynamicResource tokens so
/// rendering tracks the current theme.
///
/// Scope for v1: paragraphs, ATX headings, bold/italic/strikethrough, inline
/// code, fenced/indented code blocks, ordered + unordered lists, blockquotes,
/// thematic breaks, links (coloured but not clickable), line breaks, GFM task
/// lists, pipe tables. Deferred: syntax highlighting, math, images, HTML
/// passthrough, footnotes.
/// </summary>
public static class MarkdownRenderer
{
    private static readonly MarkdownPipeline Pipeline = new MarkdownPipelineBuilder()
        .UseEmphasisExtras()
        .UseAutoLinks()
        .UsePipeTables()
        .UseTaskLists()
        .UseMathematics()
        .Build();

    public static void RenderInto(Panel host, string? markdown)
    {
        host.Children.Clear();
        if (string.IsNullOrEmpty(markdown)) return;

        MarkdownDocument doc;
        try
        {
            doc = Markdown.Parse(markdown, Pipeline);
        }
        catch
        {
            // Malformed mid-stream (e.g. unclosed fence) — show raw text
            // rather than leaving the bubble blank.
            host.Children.Add(new TextBlock { Text = markdown, TextWrapping = TextWrapping.Wrap });
            return;
        }

        foreach (var block in doc)
        {
            var c = RenderBlock(block);
            if (c is not null) host.Children.Add(c);
        }
    }

    // ===============================================================
    // Block-level
    // ===============================================================

    private static Control? RenderBlock(Block block) => block switch
    {
        HeadingBlock h => RenderHeading(h),
        ParagraphBlock p => RenderParagraph(p),
        ListBlock l => RenderList(l),
        QuoteBlock q => RenderQuote(q),
        MathBlock mb => RenderMathBlock(mb),
        FencedCodeBlock fc => RenderFencedCodeBlock(fc),
        CodeBlock cb => RenderIndentedCodeBlock(cb),
        ThematicBreakBlock => RenderThematicBreak(),
        Table t => RenderTable(t),
        _ => null,
    };

    private static Control RenderThematicBreak()
    {
        var b = new Border
        {
            BorderThickness = new Thickness(0, 0, 0, 1),
            Margin = new Thickness(0, 10, 0, 10),
        };
        b[!Border.BorderBrushProperty] = new DynamicResourceExtension("Border");
        return b;
    }

    private static Control RenderHeading(HeadingBlock h)
    {
        var tb = new TextBlock
        {
            FontWeight = FontWeight.SemiBold,
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 8, 0, 4),
            FontSize = h.Level switch { 1 => 20, 2 => 17, 3 => 15, _ => 13 },
        };
        if (h.Inline is not null)
        {
            foreach (var inline in h.Inline) AppendInline(tb.Inlines!, inline);
        }
        return tb;
    }

    private static Control RenderParagraph(ParagraphBlock p)
    {
        var tb = new TextBlock { TextWrapping = TextWrapping.Wrap };
        if (p.Inline is not null)
        {
            foreach (var inline in p.Inline) AppendInline(tb.Inlines!, inline);
        }
        return tb;
    }

    private static Control RenderList(ListBlock list)
    {
        var container = new StackPanel { Spacing = 2, Margin = new Thickness(4, 0, 0, 0) };
        var index = list.OrderedStart is { } s && int.TryParse(s, out var parsed) ? parsed : 1;

        foreach (var child in list)
        {
            if (child is not ListItemBlock item) continue;

            var marker = new TextBlock
            {
                Text = list.IsOrdered ? $"{index++}." : "•",
                Margin = new Thickness(0, 0, 8, 0),
                MinWidth = 20,
                HorizontalAlignment = HorizontalAlignment.Right,
                Opacity = 0.8,
            };

            var content = new StackPanel { Spacing = 4 };
            foreach (var sub in item)
            {
                var rendered = RenderBlock(sub);
                if (rendered is not null) content.Children.Add(rendered);
            }

            // Task-list checkbox: Markdig stores it on the first inline of
            // the item's first paragraph. Replace the numeric marker.
            if (item.Count > 0
                && item[0] is ParagraphBlock firstPar
                && firstPar.Inline?.FirstChild is TaskList tl)
            {
                marker.Text = tl.Checked ? "☑" : "☐";
                marker.Opacity = 1;
            }

            var row = new Grid
            {
                ColumnDefinitions = new ColumnDefinitions("Auto,*"),
                Margin = new Thickness(0, 2, 0, 2),
            };
            Grid.SetColumn(marker, 0);
            Grid.SetColumn(content, 1);
            row.Children.Add(marker);
            row.Children.Add(content);
            container.Children.Add(row);
        }

        return container;
    }

    private static Control RenderQuote(QuoteBlock q)
    {
        var content = new StackPanel { Spacing = 6, Opacity = 0.85 };
        foreach (var sub in q)
        {
            var rendered = RenderBlock(sub);
            if (rendered is not null) content.Children.Add(rendered);
        }
        var border = new Border
        {
            BorderThickness = new Thickness(3, 0, 0, 0),
            Padding = new Thickness(12, 4, 4, 4),
            Margin = new Thickness(0, 4, 0, 4),
            Child = content,
        };
        border[!Border.BorderBrushProperty] = new DynamicResourceExtension("MutedForeground");
        return border;
    }

    private static Control RenderFencedCodeBlock(FencedCodeBlock fc)
    {
        var text = ExtractBlockText(fc);
        if (string.Equals(fc.Info, "mermaid", StringComparison.OrdinalIgnoreCase))
        {
            try { return Mermaid.FlowchartRenderer.Render(text); }
            catch
            {
                // Parser or layout blew up on unexpected syntax — fall back to
                // the raw source so the user can still read it.
                return BuildCodeBlock(text, "mermaid");
            }
        }
        return BuildCodeBlock(text, fc.Info);
    }

    private static Control RenderIndentedCodeBlock(CodeBlock cb)
    {
        var text = ExtractBlockText(cb);
        return BuildCodeBlock(text, language: null);
    }

    private static Control RenderMathBlock(MathBlock mb)
    {
        var latex = ExtractBlockText(mb);
        var bmp = MathRenderer.Render(latex, display: true);
        if (bmp is null)
        {
            // Invalid LaTeX — show raw source in a muted monospace block so
            // the user sees what failed rather than a blank space.
            return BuildCodeBlock(latex, language: "math", showActions: false);
        }

        return BuildMathTintedSurface(bmp, new Thickness(0, 6, 0, 6), HorizontalAlignment.Center);
    }

    /// <summary>
    /// Wraps a math bitmap (rendered by <see cref="MathRenderer"/> in pure
    /// white) in a <see cref="Border"/> whose <c>Background</c> tracks the
    /// theme's <c>Foreground</c> resource and whose <c>OpacityMask</c> is
    /// the bitmap. The composition is "fill with current foreground colour,
    /// keep only the pixels where the math glyphs are opaque" — so the math
    /// always matches the current theme without re-rendering, and a race
    /// where the variant resolves after the bitmap was rendered is harmless.
    /// </summary>
    private static Border BuildMathTintedSurface(
        Avalonia.Media.Imaging.Bitmap bmp,
        Thickness margin,
        HorizontalAlignment hAlign)
    {
        var border = new Border
        {
            Width = bmp.Size.Width,
            Height = bmp.Size.Height,
            OpacityMask = new ImageBrush(bmp) { Stretch = Avalonia.Media.Stretch.None },
            HorizontalAlignment = hAlign,
            Margin = margin,
        };
        border[!Border.BackgroundProperty] = new DynamicResourceExtension("Foreground");
        return border;
    }

    private static string ExtractBlockText(LeafBlock block)
    {
        var lines = block.Lines;
        var sb = new System.Text.StringBuilder();
        for (var i = 0; i < lines.Count; i++)
        {
            if (i > 0) sb.Append('\n');
            sb.Append(lines.Lines[i].ToString());
        }
        return sb.ToString();
    }

    private static Control BuildCodeBlock(string text, string? language) =>
        BuildCodeBlock(text, language, showActions: true);

    /// <summary>
    /// Build a code-block control with syntax highlighting (via
    /// <see cref="CodeHighlighter"/>) and an optional header row of
    /// <c>language</c> + <c>Copy</c> + <c>Expand</c> action buttons. Also
    /// used by the expand-dialog to render the same code block at full size
    /// without the header — hence the <paramref name="showActions"/> flag.
    /// </summary>
    internal static Control BuildCodeBlock(string text, string? language, bool showActions)
    {
        var codeText = new TextBlock
        {
            TextWrapping = TextWrapping.NoWrap,
            FontSize = 12,
        };
        codeText[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("CodeForeground");
        codeText[!TextBlock.FontFamilyProperty] = new DynamicResourceExtension("CodeFontFamily");

        // Tokenise with ColorCode and emit a Run per token. Tokens whose
        // scope doesn't map get the default CodeForeground (set on the
        // containing TextBlock above), so unrecognised languages still
        // render — just monochrome.
        var tokens = CodeHighlighter.Highlight(text, language);
        foreach (var tok in tokens)
        {
            var run = new Run(tok.Text);
            var token = ScopeToTokenBrush(tok.Scope);
            if (token is not null)
            {
                run[!Run.ForegroundProperty] = new DynamicResourceExtension(token);
            }
            codeText.Inlines!.Add(run);
        }

        var scroll = new ScrollViewer
        {
            HorizontalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Auto,
            VerticalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Disabled,
            Content = codeText,
        };

        Control child = scroll;
        if (showActions)
        {
            child = BuildCodeBlockShell(scroll, text, language);
        }

        var border = new Border
        {
            Padding = new Thickness(12, 10),
            Margin = new Thickness(0, 4, 0, 4),
            CornerRadius = new CornerRadius(8),
            Child = child,
        };
        border[!Border.BackgroundProperty] = new DynamicResourceExtension("CodeBackground");
        return border;
    }

    /// <summary>
    /// Wraps the code body in a header row with language label on the left
    /// and Copy / Expand buttons on the right. The buttons use ghost+sm
    /// classes so they blend into the block.
    /// </summary>
    private static Control BuildCodeBlockShell(Control body, string text, string? language)
    {
        var langLabel = new TextBlock
        {
            Text = string.IsNullOrWhiteSpace(language) ? "" : language,
            FontSize = 10,
            Opacity = 0.6,
            VerticalAlignment = VerticalAlignment.Center,
        };
        langLabel[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("MutedForeground");

        var copyBtn = new Button
        {
            Content = "Copy",
            Padding = new Thickness(8, 2),
            FontSize = 10,
            MinHeight = 22,
        };
        copyBtn.Classes.Add("ghost");
        copyBtn.Classes.Add("sm");
        copyBtn.Click += async (_, _) =>
        {
            await DialogService.CopyToClipboardAsync(text);
        };

        var expandBtn = new Button
        {
            Content = "Expand",
            Padding = new Thickness(8, 2),
            FontSize = 10,
            MinHeight = 22,
            Margin = new Thickness(4, 0, 0, 0),
        };
        expandBtn.Classes.Add("ghost");
        expandBtn.Classes.Add("sm");
        expandBtn.Click += async (_, _) =>
        {
            await DialogService.ShowCodePreviewAsync(text, language);
        };

        var headerRow = new StackPanel
        {
            Orientation = Avalonia.Layout.Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
        };
        headerRow.Children.Add(copyBtn);
        headerRow.Children.Add(expandBtn);

        var header = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("*,Auto"),
            Margin = new Thickness(0, 0, 0, 4),
        };
        Grid.SetColumn(langLabel, 0);
        Grid.SetColumn(headerRow, 1);
        header.Children.Add(langLabel);
        header.Children.Add(headerRow);

        var grid = new Grid
        {
            RowDefinitions = new RowDefinitions("Auto,*"),
        };
        Grid.SetRow(header, 0);
        Grid.SetRow(body, 1);
        grid.Children.Add(header);
        grid.Children.Add(body);
        return grid;
    }

    /// <summary>
    /// Map a ColorCode scope name (as found in <c>ColorCode.Common.ScopeName</c>)
    /// to one of our theme's <c>Syntax*</c> brush keys. Unmapped scopes
    /// return null and the token renders with the default code foreground.
    /// </summary>
    private static string? ScopeToTokenBrush(string? scope) => scope switch
    {
        null => null,
        "Keyword" or "Pseudo Keyword" or "Preprocessor Keyword" or "HTML Attribute Name"
            => "SyntaxKeyword",
        "String" or "String C# Verbatim" or "String Character" or "HTML Attribute Value"
            => "SyntaxString",
        "Comment" or "Comment XML Tag" or "Comment XML Attribute"
            or "Comment XML Attribute Value" or "XML Doc Comment" or "XML Doc Tag"
            => "SyntaxComment",
        "Number" => "SyntaxNumber",
        "Type" or "Class Name" or "Built-in Type" or "Type Variable"
            => "SyntaxType",
        "Namespace" or "Module" => "SyntaxType",
        "HTML Element Name" or "HTML Tag Delimiter" or "XML Tag" or "XML Name"
            => "SyntaxTag",
        "Preprocessor" or "HTML Entity" => "SyntaxPreprocessor",
        "Operator" or "Punctuation" => "SyntaxOperator",
        _ => null,
    };

    private static Control RenderTable(Table table)
    {
        // Figure out column count once (ragged rows allowed in markdown).
        var colCount = 0;
        foreach (var rowBlock in table)
        {
            if (rowBlock is TableRow tr && tr.Count > colCount) colCount = tr.Count;
        }

        var grid = new Grid();
        // Star-sized columns: each claims an equal share of the available
        // pane width. Cell TextBlocks (which already set TextWrapping.Wrap)
        // only wrap when they have a constrained width — `Auto` would let
        // cells grow to fit their longest line and the table would overflow
        // the chat pane. A simple equal-share split trades ideal column
        // proportions for content that actually fits the pane.
        for (var i = 0; i < colCount; i++)
            grid.ColumnDefinitions.Add(new ColumnDefinition(new GridLength(1, GridUnitType.Star)));

        var rIdx = 0;
        foreach (var rowBlock in table)
        {
            if (rowBlock is not TableRow row) continue;
            grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));

            var cIdx = 0;
            foreach (var cellBlock in row)
            {
                if (cellBlock is not TableCell cell) continue;

                var cellStack = new StackPanel();
                foreach (var sub in cell)
                {
                    var rendered = RenderBlock(sub);
                    if (rendered is not null)
                    {
                        if (row.IsHeader && rendered is TextBlock rtb)
                            rtb.FontWeight = FontWeight.SemiBold;
                        cellStack.Children.Add(rendered);
                    }
                }

                var cellBorder = new Border
                {
                    BorderThickness = new Thickness(0, 0, 1, 1),
                    Padding = new Thickness(8, 4),
                    Child = cellStack,
                };
                cellBorder[!Border.BorderBrushProperty] = new DynamicResourceExtension("Border");

                Grid.SetRow(cellBorder, rIdx);
                Grid.SetColumn(cellBorder, cIdx);
                grid.Children.Add(cellBorder);
                cIdx++;
            }
            rIdx++;
        }

        var outer = new Border
        {
            BorderThickness = new Thickness(1, 1, 0, 0),
            CornerRadius = new CornerRadius(6),
            Margin = new Thickness(0, 4),
            ClipToBounds = true,
            Child = grid,
        };
        outer[!Border.BorderBrushProperty] = new DynamicResourceExtension("Border");
        return outer;
    }

    // ===============================================================
    // Inline-level
    // ===============================================================

    private static void AppendInline(InlineCollection dst, MdInline inline)
    {
        switch (inline)
        {
            case LiteralInline lit:
                dst.Add(new Run(lit.Content.ToString()));
                break;

            case CodeInline code:
            {
                var run = new Run(code.Content) { FontSize = 12 };
                run[!TextElement.FontFamilyProperty] = new DynamicResourceExtension("CodeFontFamily");
                run[!TextElement.BackgroundProperty] = new DynamicResourceExtension("CodeBackground");
                dst.Add(run);
                break;
            }

            case MathInline math:
            {
                var latex = math.Content.ToString();
                var bmp = MathRenderer.Render(latex, display: false);
                if (bmp is null)
                {
                    // Fallback: render the raw latex in code style so the user
                    // can see what failed.
                    var run = new Run($"${latex}$") { FontSize = 12 };
                    run[!TextElement.FontFamilyProperty] = new DynamicResourceExtension("CodeFontFamily");
                    dst.Add(run);
                }
                else
                {
                    var surface = BuildMathTintedSurface(bmp, default, HorizontalAlignment.Stretch);
                    dst.Add(new InlineUIContainer(surface) { BaselineAlignment = BaselineAlignment.Center });
                }
                break;
            }

            case EmphasisInline em:
            {
                Span wrapper = em.DelimiterChar == '~'
                    ? new Span { TextDecorations = TextDecorations.Strikethrough }
                    : em.DelimiterCount >= 2 ? new Bold() : new Italic();
                foreach (var child in em) AppendInline(wrapper.Inlines, child);
                dst.Add(wrapper);
                break;
            }

            case LinkInline link:
            {
                // Links rendered styled-but-inert in v1. Url is dropped since
                // Avalonia Run has no native href-on-hover tooltip path.
                var linkSpan = new Span { TextDecorations = TextDecorations.Underline };
                linkSpan[!TextElement.ForegroundProperty] = new DynamicResourceExtension("Ring");
                foreach (var child in link) AppendInline(linkSpan.Inlines, child);
                if (linkSpan.Inlines.Count == 0 && !string.IsNullOrEmpty(link.Url))
                    linkSpan.Inlines.Add(new Run(link.Url));
                dst.Add(linkSpan);
                break;
            }

            case AutolinkInline auto:
            {
                var r = new Run(auto.Url) { TextDecorations = TextDecorations.Underline };
                r[!TextElement.ForegroundProperty] = new DynamicResourceExtension("Ring");
                dst.Add(r);
                break;
            }

            case LineBreakInline lb:
                if (lb.IsHard) dst.Add(new LineBreak());
                else dst.Add(new Run(" "));
                break;

            case HtmlInline htmlInline:
                // Don't trust streamed HTML; render the raw tag as text.
                dst.Add(new Run(htmlInline.Tag));
                break;

            case HtmlEntityInline entityInline:
                dst.Add(new Run(entityInline.Transcoded.ToString()));
                break;

            case ContainerInline container:
                foreach (var child in container) AppendInline(dst, child);
                break;
        }
    }
}
