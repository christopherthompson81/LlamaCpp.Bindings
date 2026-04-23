using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Media;
using LlamaCpp.Bindings.LlamaChat.Services;

namespace LlamaCpp.Bindings.LlamaChat.Views;

/// <summary>
/// Partitioned markdown renderer for streamed assistant bubbles.
///
/// Splits the current <see cref="Markdown"/> at the last <c>\n\n</c> that
/// sits outside a fenced code block. Everything before that split is the
/// <b>stable prefix</b> — it's already a complete set of Markdown blocks
/// and can't change — and is handed to <see cref="MarkdownRenderer"/> to
/// rebuild the visual tree once per boundary crossing. Everything after is
/// the <b>live tail</b> — the paragraph currently being written — which is
/// shown as a plain <see cref="TextBlock"/> and updates on every streamed
/// token for the cost of a single measure + arrange.
///
/// For a normal streaming response that's maybe 4-5 full Markdig passes
/// (one per finished paragraph) instead of one per token or one gigantic
/// pass at the end. The live tail shows raw Markdown syntax until the block
/// finalizes — `**bold**` visible as literal chars mid-stream — which is
/// the acceptable trade for getting streaming visuals back.
/// </summary>
public sealed class MarkdownView : UserControl
{
    public static readonly StyledProperty<string?> MarkdownProperty =
        AvaloniaProperty.Register<MarkdownView, string?>(nameof(Markdown));

    public static readonly StyledProperty<bool> IsStreamingProperty =
        AvaloniaProperty.Register<MarkdownView, bool>(nameof(IsStreaming));

    public string? Markdown
    {
        get => GetValue(MarkdownProperty);
        set => SetValue(MarkdownProperty, value);
    }

    /// <summary>
    /// When true, append a pulsing <c>▌</c> cursor to the live tail. The
    /// blink animation is driven by the <c>TextBlock.cursor</c> style in
    /// <c>Theme/Controls.axaml</c> so the behaviour matches the rest of the
    /// theme system.
    /// </summary>
    public bool IsStreaming
    {
        get => GetValue(IsStreamingProperty);
        set => SetValue(IsStreamingProperty, value);
    }

    private readonly StackPanel _stable = new() { Spacing = 6 };
    private readonly TextBlock _live = new() { TextWrapping = TextWrapping.Wrap };

    private string _lastRenderedStable = string.Empty;

    // Incremental state for FindStableBoundary. When the content only grows,
    // we resume scanning from the last position — turning what would be an
    // O(N²) sweep over the full stream into an amortised O(N). Reset when
    // the content shrinks or changes identity (new message, edited bubble).
    private int _scanPos;
    private int _scanLineStart;
    private bool _scanInFence;
    private int _scanLastSafe;

    public MarkdownView()
    {
        var root = new StackPanel { Spacing = 6 };
        root.Children.Add(_stable);
        root.Children.Add(_live);
        Content = root;
    }

    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == MarkdownProperty || change.Property == IsStreamingProperty)
            Update();
    }

    protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
    {
        base.OnAttachedToVisualTree(e);
        Update();
    }

    private void Update()
    {
        var content = Markdown ?? string.Empty;

        // When not streaming, render the whole content through the markdown
        // pipeline — a trailing list (or any block that doesn't end with a
        // blank line) otherwise stays in the raw-text `_live` pane forever,
        // because AdvanceStableBoundary only advances past `\n\n` pairs.
        // We also snap the incremental scan state to the end so a subsequent
        // streaming session (e.g. Continue) treats the current content as
        // the baseline and only newly-appended tokens become the live tail.
        int boundary;
        if (!IsStreaming)
        {
            boundary = content.Length;
            _scanPos = content.Length;
            _scanLineStart = content.Length;
            _scanLastSafe = content.Length;
        }
        else
        {
            boundary = AdvanceStableBoundary(content);
        }
        var newStable = content[..boundary];
        var newLive = content[boundary..];

        // String equality on a growing prefix short-circuits on length check
        // in .NET, so this is O(1) for the common case of tail-only change.
        if (!string.Equals(newStable, _lastRenderedStable, StringComparison.Ordinal))
        {
            MarkdownRenderer.RenderInto(_stable, newStable);
            _lastRenderedStable = newStable;
        }

        // Live tail: plain-text Run for the streaming text + (when
        // IsStreaming) an InlineUIContainer wrapping a TextBlock.cursor
        // whose blink animation lives in the theme. Using Inlines instead
        // of Text lets us mix the two without losing wrap behaviour.
        _live.Inlines!.Clear();

        var hasText = newLive.Length > 0;
        var showCursor = IsStreaming;

        if (hasText || showCursor)
        {
            if (hasText) _live.Inlines.Add(new Run(newLive));
            if (showCursor)
            {
                var cursor = new TextBlock { Text = "▌" };
                cursor.Classes.Add("cursor");
                _live.Inlines.Add(new InlineUIContainer(cursor));
            }
            _live.IsVisible = true;
        }
        else
        {
            _live.IsVisible = false;
        }
    }

    /// <summary>
    /// Incremental boundary scan. Continues from the last position instead
    /// of rewalking the full content every token. If the content shrunk
    /// or is a different prefix than last time (e.g. bubble recycled for a
    /// new message), reset and restart.
    /// </summary>
    private int AdvanceStableBoundary(string content)
    {
        // Detect a "fresh" content: shorter than our cursor, or (cheap check)
        // the remembered lastSafe lies past the end of the new string.
        var resume = _scanPos <= content.Length && _scanLastSafe <= content.Length;
        if (!resume)
        {
            _scanPos = 0;
            _scanLineStart = 0;
            _scanInFence = false;
            _scanLastSafe = 0;
        }

        for (var i = _scanPos; i < content.Length; i++)
        {
            if (content[i] != '\n') continue;

            var line = content.AsSpan(_scanLineStart, i - _scanLineStart);
            var trimmed = line.TrimStart(' ');
            if (trimmed.StartsWith("```".AsSpan()) || trimmed.StartsWith("~~~".AsSpan()))
            {
                _scanInFence = !_scanInFence;
            }

            if (!_scanInFence && i + 1 < content.Length && content[i + 1] == '\n')
            {
                _scanLastSafe = i + 2;
            }
            _scanLineStart = i + 1;
        }

        _scanPos = content.Length;
        return _scanLastSafe;
    }
}
