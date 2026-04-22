using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Threading;
using LlamaCpp.Bindings.LlamaChat.Services;

namespace LlamaCpp.Bindings.LlamaChat.Views;

/// <summary>
/// Lightweight UserControl that renders its <see cref="Markdown"/> string via
/// <see cref="MarkdownRenderer"/>. Binding target for message bubble content.
///
/// Re-renders are throttled to ~40ms so that streaming updates (one per
/// decoded token, ~8ms apart at 120 tok/s) don't thrash the layout pass.
/// That coalesces ~4-5 tokens per render and keeps the visual tree stable.
/// </summary>
public sealed class MarkdownView : UserControl
{
    public static readonly StyledProperty<string?> MarkdownProperty =
        AvaloniaProperty.Register<MarkdownView, string?>(nameof(Markdown));

    public string? Markdown
    {
        get => GetValue(MarkdownProperty);
        set => SetValue(MarkdownProperty, value);
    }

    private readonly StackPanel _host = new() { Spacing = 6 };
    private readonly DispatcherTimer _debounce;

    public MarkdownView()
    {
        Content = _host;
        _debounce = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(40) };
        _debounce.Tick += (_, _) =>
        {
            _debounce.Stop();
            MarkdownRenderer.RenderInto(_host, Markdown);
        };
    }

    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == MarkdownProperty)
        {
            // Restart the debounce window on every change.
            _debounce.Stop();
            _debounce.Start();
        }
    }

    protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
    {
        base.OnAttachedToVisualTree(e);
        // Render synchronously on attach so newly virtualised items don't
        // flash empty for 40ms.
        MarkdownRenderer.RenderInto(_host, Markdown);
    }
}
