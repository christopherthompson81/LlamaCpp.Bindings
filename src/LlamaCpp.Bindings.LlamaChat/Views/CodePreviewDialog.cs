using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Layout;
using Avalonia.Markup.Xaml.MarkupExtensions;
using LlamaCpp.Bindings.LlamaChat.Services;

namespace LlamaCpp.Bindings.LlamaChat.Views;

/// <summary>
/// A lightweight modal that shows a fenced code block full-size — same
/// syntax-highlighted rendering as in a message bubble but without the
/// bubble's width constraint, plus a Copy button and Escape-to-close.
/// Built purely in code (no XAML) because the entire content is just a
/// scroll viewer around <see cref="MarkdownRenderer.BuildCodeBlock"/>.
/// </summary>
public sealed class CodePreviewDialog : Window
{
    private readonly string _code;

    public CodePreviewDialog(string code, string? language)
    {
        _code = code;
        Title = string.IsNullOrWhiteSpace(language) ? "Code" : $"Code — {language}";
        Width = 1000;
        Height = 700;
        MinWidth = 400;
        MinHeight = 300;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        this[!BackgroundProperty] = new DynamicResourceExtension("Background");

        // Use BuildCodeBlock without the header-button row — the dialog has
        // its own Copy / Close footer, and the language is already in the
        // title bar.
        var codeBlock = MarkdownRenderer.BuildCodeBlock(code, language, showActions: false);

        var scroll = new ScrollViewer
        {
            HorizontalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Auto,
            VerticalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Auto,
            Padding = new Thickness(16),
            Content = codeBlock,
        };

        var copyBtn = new Button { Content = "Copy" };
        copyBtn.Click += OnCopyClicked;
        var closeBtn = new Button { Content = "Close" };
        closeBtn.Classes.Add("outline");
        closeBtn.Click += (_, _) => Close();
        closeBtn.Margin = new Thickness(8, 0, 0, 0);

        var footer = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
            Margin = new Thickness(16, 8, 16, 16),
        };
        footer.Children.Add(copyBtn);
        footer.Children.Add(closeBtn);

        var root = new DockPanel();
        DockPanel.SetDock(footer, Dock.Bottom);
        root.Children.Add(footer);
        root.Children.Add(scroll);
        Content = root;

        KeyDown += (_, e) =>
        {
            if (e.Key == Key.Escape) { Close(); e.Handled = true; }
        };
    }

    private async void OnCopyClicked(object? sender, RoutedEventArgs e)
    {
        await DialogService.CopyToClipboardAsync(_code);
    }
}
