using System.Collections.Generic;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Controls.Templates;
using Avalonia.Layout;
using LlamaCpp.Bindings.LlamaChat.ViewModels;

namespace LlamaCpp.Bindings.LlamaChat.Views;

/// <summary>
/// Modal resource browser. Returns the attached content + URI via
/// <see cref="AttachedContent"/> / <see cref="AttachedUri"/>; the caller
/// formats and inserts into the compose box.
/// </summary>
public sealed class McpResourcePickerDialog : Window
{
    public string? AttachedContent { get; private set; }
    public string? AttachedUri { get; private set; }

    private readonly McpResourcePickerViewModel _vm = new();

    public McpResourcePickerDialog()
    {
        Title = "Browse MCP resources";
        Width = 820;
        Height = 600;
        CanResize = true;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        DataContext = _vm;

        Build();
        _vm.AttachRequested += (_, _) =>
        {
            AttachedContent = _vm.AttachedContent;
            AttachedUri = _vm.AttachedUri;
            Close();
        };
    }

    private void Build()
    {
        var root = new Grid
        {
            RowDefinitions = new RowDefinitions("Auto,*,Auto,Auto"),
            Margin = new Thickness(16),
        };

        // Search
        var search = new TextBox
        {
            PlaceholderText = "Search by name, URI, description…",
            Margin = new Thickness(0, 0, 0, 8),
        };
        search.Bind(TextBox.TextProperty,
            new Avalonia.Data.Binding(nameof(McpResourcePickerViewModel.SearchText))
            { Mode = Avalonia.Data.BindingMode.TwoWay });
        Grid.SetRow(search, 0);
        root.Children.Add(search);

        // Split: list on left, preview on right
        var split = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("320,8,*"),
        };

        var list = new ListBox();
        list.Bind(ItemsControl.ItemsSourceProperty,
            new Avalonia.Data.Binding(nameof(McpResourcePickerViewModel.Filtered)));
        list.Bind(SelectingItemsControl.SelectedItemProperty,
            new Avalonia.Data.Binding(nameof(McpResourcePickerViewModel.Selected))
            { Mode = Avalonia.Data.BindingMode.TwoWay });
        list.ItemTemplate = new FuncDataTemplate<McpResourcePickerViewModel.ResourceOption>(
            (o, _) =>
            {
                if (o is null) return new TextBlock();
                var sp = new StackPanel { Spacing = 2 };
                sp.Children.Add(new TextBlock
                {
                    Text = o.Label,
                    FontWeight = Avalonia.Media.FontWeight.Medium,
                    TextTrimming = Avalonia.Media.TextTrimming.CharacterEllipsis,
                });
                sp.Children.Add(new TextBlock
                {
                    Text = o.ServerLabel,
                    Classes = { "xs", "muted" },
                });
                return sp;
            }, supportsRecycling: true);
        Grid.SetColumn(list, 0);
        split.Children.Add(list);

        var preview = new Border
        {
            Classes = { "card" },
            Padding = new Thickness(12),
        };
        var previewScroll = new ScrollViewer();
        var previewText = new SelectableTextBlock
        {
            TextWrapping = Avalonia.Media.TextWrapping.Wrap,
            FontFamily = (Avalonia.Media.FontFamily)Avalonia.Application.Current!.Resources["CodeFontFamily"]!,
            FontSize = 12,
        };
        previewText.Bind(SelectableTextBlock.TextProperty,
            new Avalonia.Data.Binding(nameof(McpResourcePickerViewModel.PreviewText)));
        previewScroll.Content = previewText;
        preview.Child = previewScroll;
        Grid.SetColumn(preview, 2);
        split.Children.Add(preview);

        Grid.SetRow(split, 1);
        root.Children.Add(split);

        // Status
        var status = new TextBlock
        {
            Classes = { "xs", "muted" },
            Margin = new Thickness(0, 8, 0, 6),
        };
        status.Bind(TextBlock.TextProperty,
            new Avalonia.Data.Binding(nameof(McpResourcePickerViewModel.Status)));
        Grid.SetRow(status, 2);
        root.Children.Add(status);

        // Buttons
        var buttons = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
            Spacing = 6,
        };
        var cancel = new Button { Content = "Close", Classes = { "outline" } };
        cancel.Click += (_, _) => Close();
        var previewBtn = new Button { Content = "Preview", Classes = { "outline" } };
        previewBtn.Bind(Button.CommandProperty,
            new Avalonia.Data.Binding(nameof(McpResourcePickerViewModel.PreviewCommand)));
        var attach = new Button { Content = "Attach" };
        attach.Bind(Button.CommandProperty,
            new Avalonia.Data.Binding(nameof(McpResourcePickerViewModel.AttachCommand)));
        buttons.Children.Add(cancel);
        buttons.Children.Add(previewBtn);
        buttons.Children.Add(attach);
        Grid.SetRow(buttons, 3);
        root.Children.Add(buttons);

        Content = root;
    }
}
