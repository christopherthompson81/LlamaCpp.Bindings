using System.Collections.Generic;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Markup.Xaml.MarkupExtensions;
using Avalonia.Media;

namespace LlamaCpp.Bindings.LlamaChat.Views;

/// <summary>
/// A minimal multi-choice dialog: pose a question, show a list of option
/// buttons, return the key of the chosen option (or null if the user
/// closed the window without picking). Used for "delete this" vs
/// "delete + downstream" vs "cancel".
/// </summary>
public sealed class ConfirmDialog : Window
{
    public string? SelectedKey { get; private set; }

    public ConfirmDialog(string title, string message,
                         IReadOnlyList<(string Key, string Label, bool Destructive, bool Primary)> options)
    {
        Title = title;
        Width = 500;
        SizeToContent = SizeToContent.Height;
        MinHeight = 160;
        MinWidth = 360;
        CanResize = false;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        this[!BackgroundProperty] = new DynamicResourceExtension("Background");

        var titleBlock = new TextBlock
        {
            Text = title,
            FontSize = 17,
            FontWeight = FontWeight.SemiBold,
            Margin = new Thickness(20, 18, 20, 4),
        };

        var messageBlock = new TextBlock
        {
            Text = message,
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(20, 0, 20, 16),
        };
        messageBlock[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("MutedForeground");

        var footer = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
            Margin = new Thickness(20, 0, 20, 16),
            Spacing = 8,
        };
        foreach (var (key, label, destructive, primary) in options)
        {
            var btn = new Button { Content = label };
            if (destructive) btn.Classes.Add("destructive");
            else if (!primary) btn.Classes.Add("outline");
            btn.Click += (_, _) =>
            {
                SelectedKey = key;
                Close();
            };
            footer.Children.Add(btn);
        }

        var root = new StackPanel();
        root.Children.Add(titleBlock);
        root.Children.Add(messageBlock);
        root.Children.Add(footer);
        Content = root;

        KeyDown += (_, e) =>
        {
            if (e.Key == Key.Escape) { SelectedKey = null; Close(); e.Handled = true; }
        };
    }
}
