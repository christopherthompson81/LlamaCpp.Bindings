using System.Collections.Generic;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Markup.Xaml.MarkupExtensions;
using Avalonia.Media;

namespace LlamaCpp.Bindings.LlamaChat.Views;

/// <summary>
/// Modal listing the keyboard shortcuts + compose-area key bindings. Pure-
/// code for simplicity; the content is a static table and a close button.
/// </summary>
public sealed class ShortcutsDialog : Window
{
    private static readonly (string Gesture, string Action)[] Shortcuts =
    {
        ("Ctrl+N",         "New chat"),
        ("Ctrl+Shift+O",   "New chat (webui alias)"),
        ("Ctrl+K",         "Focus conversation search"),
        ("Ctrl+B",         "Toggle sidebar"),
        ("Ctrl+L",         "Load model"),
        ("Ctrl+,",         "Open settings"),
        ("",               ""),
        ("Enter",          "Send message"),
        ("Shift+Enter",    "Insert newline in compose"),
        ("Escape",         "Cancel inline rename / close dialog"),
    };

    public ShortcutsDialog()
    {
        Title = "Keyboard shortcuts";
        Width = 480;
        Height = 420;
        MinWidth = 360;
        MinHeight = 280;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        this[!BackgroundProperty] = new DynamicResourceExtension("Background");

        var header = new TextBlock
        {
            Text = "Keyboard shortcuts",
            FontSize = 20,
            FontWeight = FontWeight.SemiBold,
            Margin = new Thickness(24, 20, 24, 12),
        };

        var grid = BuildTable();

        var closeBtn = new Button { Content = "Close" };
        closeBtn.Classes.Add("outline");
        closeBtn.Click += (_, _) => Close();

        var footer = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
            Margin = new Thickness(24, 8, 24, 20),
        };
        footer.Children.Add(closeBtn);

        var root = new DockPanel();
        DockPanel.SetDock(header, Dock.Top);
        DockPanel.SetDock(footer, Dock.Bottom);
        root.Children.Add(header);
        root.Children.Add(footer);
        root.Children.Add(new ScrollViewer
        {
            Padding = new Thickness(24, 0),
            Content = grid,
        });
        Content = root;

        KeyDown += (_, e) =>
        {
            if (e.Key == Key.Escape) { Close(); e.Handled = true; }
        };
    }

    private static Grid BuildTable()
    {
        var grid = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("Auto,12,*"),
            VerticalAlignment = VerticalAlignment.Top,
        };

        for (var i = 0; i < Shortcuts.Length; i++) grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));

        for (var i = 0; i < Shortcuts.Length; i++)
        {
            var (gesture, action) = Shortcuts[i];
            if (string.IsNullOrEmpty(gesture))
            {
                // Blank row separator
                var sep = new Border
                {
                    Height = 12,
                };
                Grid.SetColumn(sep, 0);
                Grid.SetColumnSpan(sep, 3);
                Grid.SetRow(sep, i);
                grid.Children.Add(sep);
                continue;
            }

            var gestureBlock = new TextBlock
            {
                Text = gesture,
                FontFamily = new FontFamily("Consolas,Menlo,DejaVu Sans Mono,monospace"),
                FontWeight = FontWeight.SemiBold,
                Margin = new Thickness(0, 4),
            };
            Grid.SetColumn(gestureBlock, 0);
            Grid.SetRow(gestureBlock, i);
            grid.Children.Add(gestureBlock);

            var actionBlock = new TextBlock
            {
                Text = action,
                Margin = new Thickness(0, 4),
            };
            Grid.SetColumn(actionBlock, 2);
            Grid.SetRow(actionBlock, i);
            grid.Children.Add(actionBlock);
        }
        return grid;
    }
}
