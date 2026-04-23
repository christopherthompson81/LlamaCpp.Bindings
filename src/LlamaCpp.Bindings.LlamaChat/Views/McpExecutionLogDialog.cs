using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Templates;
using Avalonia.Layout;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.Services;

namespace LlamaCpp.Bindings.LlamaChat.Views;

/// <summary>
/// Modal showing the ring buffer of request/response entries emitted by
/// <see cref="McpClientService"/>. Read-only — diagnostic plumbing only.
/// </summary>
public sealed class McpExecutionLogDialog : Window
{
    public McpExecutionLogDialog()
    {
        Title = "MCP execution log";
        Width = 900;
        Height = 640;
        CanResize = true;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;

        Build();
    }

    private void Build()
    {
        var root = new Grid
        {
            RowDefinitions = new RowDefinitions("*,Auto"),
            Margin = new Thickness(16),
        };

        var list = new ListBox
        {
            ItemsSource = McpClientService.Instance.Log,
        };
        list.ItemTemplate = new FuncDataTemplate<McpExecutionLogEntry>((e, _) =>
        {
            if (e is null) return new TextBlock();
            var sp = new StackPanel { Spacing = 2, Margin = new Thickness(0, 2, 0, 2) };

            var header = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                Spacing = 8,
            };
            header.Children.Add(new TextBlock
            {
                Text = e.Timestamp.LocalDateTime.ToString("HH:mm:ss.fff"),
                Classes = { "xs", "muted" },
            });
            header.Children.Add(new TextBlock
            {
                Text = e.Direction,
                FontWeight = Avalonia.Media.FontWeight.SemiBold,
            });
            header.Children.Add(new TextBlock
            {
                Text = e.ServerName,
                Classes = { "xs", "muted" },
            });
            header.Children.Add(new TextBlock
            {
                Text = e.Summary,
                FontWeight = Avalonia.Media.FontWeight.Medium,
            });

            sp.Children.Add(header);
            if (!string.IsNullOrEmpty(e.Payload))
            {
                var body = new TextBlock
                {
                    Text = e.Payload,
                    FontFamily = (Avalonia.Media.FontFamily)Avalonia.Application.Current!.Resources["CodeFontFamily"]!,
                    FontSize = 11,
                    TextWrapping = Avalonia.Media.TextWrapping.Wrap,
                    Classes = { "muted" },
                    MaxHeight = 200,
                };
                sp.Children.Add(body);
            }
            return sp;
        }, supportsRecycling: true);

        Grid.SetRow(list, 0);
        root.Children.Add(list);

        var close = new Button
        {
            Content = "Close",
            Classes = { "outline" },
            HorizontalAlignment = HorizontalAlignment.Right,
            Margin = new Thickness(0, 8, 0, 0),
        };
        close.Click += (_, _) => Close();
        Grid.SetRow(close, 1);
        root.Children.Add(close);

        Content = root;
    }
}
