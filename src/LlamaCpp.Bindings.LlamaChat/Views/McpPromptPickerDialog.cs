using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Controls.Templates;
using Avalonia.Layout;
using LlamaCpp.Bindings.LlamaChat.ViewModels;

namespace LlamaCpp.Bindings.LlamaChat.Views;

/// <summary>
/// Small modal that presents the flattened list of MCP prompts, collects
/// arguments, and returns the rendered prompt text. Caller checks
/// <see cref="Result"/> after <c>ShowDialog</c> returns.
/// </summary>
public sealed class McpPromptPickerDialog : Window
{
    public string? Result { get; private set; }

    private readonly McpPromptPickerViewModel _vm = new();

    public McpPromptPickerDialog()
    {
        Title = "Insert MCP prompt";
        Width = 640;
        Height = 520;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        CanResize = true;
        DataContext = _vm;

        Build();
        _vm.InsertCompleted += (_, _) =>
        {
            Result = _vm.PromptText;
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

        // Server / prompt picker
        var picker = new ComboBox
        {
            HorizontalAlignment = HorizontalAlignment.Stretch,
            Margin = new Thickness(0, 0, 0, 8),
            DisplayMemberBinding = new Avalonia.Data.Binding(nameof(McpPromptPickerViewModel.PromptOption.Label)),
        };
        picker.Bind(ItemsControl.ItemsSourceProperty,
            new Avalonia.Data.Binding(nameof(McpPromptPickerViewModel.Options)));
        picker.Bind(SelectingItemsControl.SelectedItemProperty,
            new Avalonia.Data.Binding(nameof(McpPromptPickerViewModel.Selected))
            { Mode = Avalonia.Data.BindingMode.TwoWay });
        Grid.SetRow(picker, 0);
        root.Children.Add(picker);

        // Argument fields
        var argsHost = new ItemsControl
        {
            ItemTemplate = new FuncDataTemplate<McpPromptPickerViewModel.ArgumentField>((field, _) =>
            {
                if (field is null) return new TextBlock();
                var grid = new Grid
                {
                    ColumnDefinitions = new ColumnDefinitions("130,*"),
                    Margin = new Thickness(0, 0, 0, 6),
                };
                var label = new TextBlock
                {
                    Text = field.Label,
                    VerticalAlignment = VerticalAlignment.Center,
                };
                Grid.SetColumn(label, 0);
                var tb = new TextBox();
                tb.Bind(TextBox.TextProperty,
                    new Avalonia.Data.Binding(nameof(field.Value))
                    { Source = field, Mode = Avalonia.Data.BindingMode.TwoWay });
                Grid.SetColumn(tb, 1);
                grid.Children.Add(label);
                grid.Children.Add(tb);
                return grid;
            }, supportsRecycling: true),
        };
        argsHost.Bind(ItemsControl.ItemsSourceProperty,
            new Avalonia.Data.Binding(nameof(McpPromptPickerViewModel.ArgumentFields)));

        var scroll = new ScrollViewer { Content = argsHost };
        Grid.SetRow(scroll, 1);
        root.Children.Add(scroll);

        // Status
        var status = new TextBlock
        {
            Classes = { "xs", "muted" },
            Margin = new Thickness(0, 6, 0, 6),
        };
        status.Bind(TextBlock.TextProperty,
            new Avalonia.Data.Binding(nameof(McpPromptPickerViewModel.Status)));
        Grid.SetRow(status, 2);
        root.Children.Add(status);

        // Buttons
        var buttons = new StackPanel
        {
            Orientation = Avalonia.Layout.Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
            Spacing = 6,
        };
        var cancel = new Button { Content = "Cancel", Classes = { "outline" } };
        cancel.Click += (_, _) => Close();
        var insert = new Button { Content = "Insert" };
        insert.Bind(Button.CommandProperty,
            new Avalonia.Data.Binding(nameof(McpPromptPickerViewModel.InsertCommand)));
        buttons.Children.Add(cancel);
        buttons.Children.Add(insert);
        Grid.SetRow(buttons, 3);
        root.Children.Add(buttons);

        Content = root;
    }
}
