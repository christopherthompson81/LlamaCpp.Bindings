using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Markup.Xaml.MarkupExtensions;
using Avalonia.Media;

namespace LlamaCpp.Bindings.LlamaChat.Views;

/// <summary>
/// Read-only model metadata browser. Shows the useful fixed fields
/// (parameter count, layer count, context size, sizes, capability flags,
/// template preview) in a top section and the full GGUF key/value bag in a
/// scrollable lower section.
/// </summary>
public sealed class ModelInfoDialog : Window
{
    public ModelInfoDialog(LlamaModel model, string? profileName)
    {
        Title = "Model info";
        Width = 720;
        Height = 700;
        MinWidth = 480;
        MinHeight = 420;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        this[!BackgroundProperty] = new DynamicResourceExtension("Background");

        // Headline: summary chips.
        var summary = BuildSummary(model, profileName);
        var metaGrid = BuildMetadataTable(model.Metadata);

        var metaHeader = new TextBlock
        {
            Text = "Metadata (GGUF key/value)",
            FontWeight = FontWeight.SemiBold,
            Margin = new Thickness(0, 16, 0, 6),
        };

        var metaScroll = new ScrollViewer
        {
            HorizontalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Auto,
            VerticalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Auto,
            Content = metaGrid,
        };

        var closeBtn = new Button { Content = "Close" };
        closeBtn.Classes.Add("outline");
        closeBtn.Click += (_, _) => Close();

        var footer = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
            Margin = new Thickness(20, 8, 20, 16),
        };
        footer.Children.Add(closeBtn);

        var body = new Grid
        {
            RowDefinitions = new RowDefinitions("Auto,Auto,*"),
            Margin = new Thickness(20, 16, 20, 0),
        };
        Grid.SetRow(summary, 0);
        Grid.SetRow(metaHeader, 1);
        Grid.SetRow(metaScroll, 2);
        body.Children.Add(summary);
        body.Children.Add(metaHeader);
        body.Children.Add(metaScroll);

        var root = new DockPanel();
        DockPanel.SetDock(footer, Dock.Bottom);
        root.Children.Add(footer);
        root.Children.Add(body);
        Content = root;

        KeyDown += (_, e) =>
        {
            if (e.Key == Key.Escape) { Close(); e.Handled = true; }
        };
    }

    private static Control BuildSummary(LlamaModel model, string? profileName)
    {
        var rows = new (string Key, string Value)[]
        {
            ("Profile", profileName ?? "(none)"),
            ("Model file", System.IO.Path.GetFileName(model.ModelPath) ?? "(unknown)"),
            ("Description", model.Description ?? ""),
            ("Parameters", FormatParameterCount(model.ParameterCount)),
            ("File size", FormatBytes(model.SizeInBytes)),
            ("Training context", model.TrainingContextSize.ToString(CultureInfo.InvariantCulture) + " tok"),
            ("Layers", model.LayerCount.ToString(CultureInfo.InvariantCulture)),
            ("Embedding dim", model.EmbeddingSize.ToString(CultureInfo.InvariantCulture)),
            ("Architecture", FormatCapabilities(model)),
            ("Chat template", string.IsNullOrWhiteSpace(model.GetChatTemplate())
                ? "(none)" : "embedded"),
            ("Vocab size", model.Vocab.TokenCount.ToString(CultureInfo.InvariantCulture)),
        };

        var grid = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("160,*"),
        };
        for (var i = 0; i < rows.Length; i++)
        {
            grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
            var key = new TextBlock
            {
                Text = rows[i].Key,
                FontWeight = FontWeight.SemiBold,
                Margin = new Thickness(0, 2, 12, 2),
            };
            key[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("MutedForeground");
            var val = new SelectableTextBlock
            {
                Text = rows[i].Value,
                Margin = new Thickness(0, 2),
                TextWrapping = TextWrapping.Wrap,
            };
            Grid.SetRow(key, i); Grid.SetColumn(key, 0);
            Grid.SetRow(val, i); Grid.SetColumn(val, 1);
            grid.Children.Add(key);
            grid.Children.Add(val);
        }
        return grid;
    }

    private static Control BuildMetadataTable(IReadOnlyDictionary<string, string> metadata)
    {
        var grid = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("*,*"),
        };
        var sorted = metadata.OrderBy(kv => kv.Key, System.StringComparer.Ordinal).ToList();
        for (var i = 0; i < sorted.Count; i++)
        {
            grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
            var key = new SelectableTextBlock
            {
                Text = sorted[i].Key,
                FontFamily = new FontFamily("Consolas,Menlo,DejaVu Sans Mono,monospace"),
                FontSize = 11,
                Margin = new Thickness(0, 1, 12, 1),
                TextWrapping = TextWrapping.NoWrap,
            };
            key[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("MutedForeground");

            // Cap very long values so the dialog doesn't blow up on jinja
            // templates or tokenizer models (huge strings).
            var value = sorted[i].Value;
            if (value.Length > 400) value = value[..400] + "…";
            var val = new SelectableTextBlock
            {
                Text = value,
                FontFamily = new FontFamily("Consolas,Menlo,DejaVu Sans Mono,monospace"),
                FontSize = 11,
                Margin = new Thickness(0, 1),
                TextWrapping = TextWrapping.Wrap,
            };

            Grid.SetRow(key, i); Grid.SetColumn(key, 0);
            Grid.SetRow(val, i); Grid.SetColumn(val, 1);
            grid.Children.Add(key);
            grid.Children.Add(val);
        }
        return grid;
    }

    private static string FormatParameterCount(long n)
    {
        if (n <= 0) return "(unknown)";
        if (n >= 1_000_000_000) return $"{n / 1_000_000_000.0:F2} B";
        if (n >= 1_000_000) return $"{n / 1_000_000.0:F2} M";
        if (n >= 1_000) return $"{n / 1_000.0:F2} K";
        return n.ToString(CultureInfo.InvariantCulture);
    }

    private static string FormatBytes(long n)
    {
        if (n <= 0) return "(unknown)";
        const double GB = 1024.0 * 1024.0 * 1024.0;
        const double MB = 1024.0 * 1024.0;
        if (n >= GB) return $"{n / GB:F2} GiB";
        if (n >= MB) return $"{n / MB:F2} MiB";
        return $"{n:N0} B";
    }

    private static string FormatCapabilities(LlamaModel m)
    {
        var parts = new List<string>();
        if (m.HasEncoder) parts.Add("encoder");
        if (m.HasDecoder) parts.Add("decoder");
        if (m.IsRecurrent) parts.Add("recurrent");
        if (m.IsHybrid) parts.Add("hybrid");
        if (m.IsDiffusion) parts.Add("diffusion");
        return parts.Count == 0 ? "decoder" : string.Join(", ", parts);
    }
}
