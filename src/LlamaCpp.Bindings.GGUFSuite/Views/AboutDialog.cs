using System;
using System.Diagnostics;
using System.Reflection;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Markup.Xaml.MarkupExtensions;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Layout;
using Avalonia.Platform;

namespace LlamaCpp.Bindings.GGUFSuite.Views;

/// <summary>
/// Modal "About" window reached from Help → About. Hero image on the left,
/// app + bundled-llama.cpp version + GitHub links on the right. Mirrors
/// the LlamaChat AboutDialog so the two apps feel like the same family.
/// </summary>
public sealed class AboutDialog : Window
{
    // Bundled llama.cpp pin. Source of truth is
    // third_party/llama.cpp/VERSION; duplicated here so the About dialog
    // doesn't need file IO at startup. Update both when bumping.
    private const string LlamaCppVersion = "b8893-1-g86db42e97 (2026-04-23)";

    private const string LlamaCppUrl = "https://github.com/ggml-org/llama.cpp";
    private const string BindingsUrl = "https://github.com/christopherthompson81/LlamaCpp.Bindings";

    public AboutDialog()
    {
        Title = "About GGUF Lab";
        Width = 680;
        Height = 440;
        MinWidth = 560;
        MinHeight = 360;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        CanResize = false;
        this[!BackgroundProperty] = new DynamicResourceExtension("Background");

        Content = BuildBody();

        KeyDown += (_, e) =>
        {
            if (e.Key == Key.Escape) { Close(); e.Handled = true; }
        };
    }

    private Control BuildBody()
    {
        // Left: hero image. Uniform stretch so the near-square 1037x997 is
        // not distorted, capped at 240 DIPs.
        Image hero;
        try
        {
            using var stream = AssetLoader.Open(
                new Uri("avares://LlamaCpp.Bindings.GGUFSuite/Assets/llama_scientist.png"));
            hero = new Image
            {
                Source = new Bitmap(stream),
                Stretch = Stretch.Uniform,
                Width = 240,
                VerticalAlignment = VerticalAlignment.Center,
            };
        }
        catch
        {
            hero = new Image { Width = 0 };
        }

        var appVersion = Assembly.GetExecutingAssembly().GetName().Version?.ToString(3) ?? "dev";

        var title = new TextBlock
        {
            Text = "GGUF Lab",
            FontSize = 28,
            FontWeight = FontWeight.SemiBold,
        };

        var subtitle = new TextBlock
        {
            Text = "Model preparation tools for llama.cpp, built on LlamaCpp.Bindings.",
            Opacity = 0.65,
            FontSize = 12,
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 4, 0, 16),
        };

        var versionTable = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("Auto,12,*"),
            RowDefinitions = new RowDefinitions("Auto,Auto"),
            Margin = new Thickness(0, 0, 0, 16),
        };
        AddRow(versionTable, 0, "App",       appVersion);
        AddRow(versionTable, 1, "llama.cpp", LlamaCppVersion);

        var linksHeader = new TextBlock
        {
            Text = "Links",
            FontWeight = FontWeight.SemiBold,
            Margin = new Thickness(0, 0, 0, 4),
        };

        var linksStack = new StackPanel { Spacing = 2 };
        linksStack.Children.Add(LinkButton("llama.cpp on GitHub",         LlamaCppUrl));
        linksStack.Children.Add(LinkButton("LlamaCpp.Bindings on GitHub", BindingsUrl));

        var closeBtn = new Button { Content = "Close" };
        closeBtn.Click += (_, _) => Close();

        var closeRow = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
            Margin = new Thickness(0, 16, 0, 0),
        };
        closeRow.Children.Add(closeBtn);

        var rightCol = new StackPanel
        {
            Spacing = 0,
            Margin = new Thickness(24, 24, 24, 24),
            VerticalAlignment = VerticalAlignment.Center,
        };
        rightCol.Children.Add(title);
        rightCol.Children.Add(subtitle);
        rightCol.Children.Add(versionTable);
        rightCol.Children.Add(linksHeader);
        rightCol.Children.Add(linksStack);
        rightCol.Children.Add(closeRow);

        var root = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("Auto,*"),
        };
        Grid.SetColumn(hero, 0);
        Grid.SetColumn(rightCol, 1);
        root.Children.Add(hero);
        root.Children.Add(rightCol);
        return root;
    }

    private static Button LinkButton(string text, string url)
    {
        var btn = new Button { Content = text, Padding = new Thickness(0, 2) };
        btn.Click += (_, _) =>
        {
            try
            {
                // UseShellExecute = true so the OS picks the default browser.
                // Headless / sandboxed environments may reject the launch but
                // the dialog stays functional.
                Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
            }
            catch
            {
                // Nothing to recover; user can copy the URL out of the tooltip.
            }
        };
        ToolTip.SetTip(btn, url);
        return btn;
    }

    private static void AddRow(Grid host, int row, string label, string value)
    {
        var lbl = new TextBlock
        {
            Text = label,
            Opacity = 0.6,
            FontSize = 12,
            Margin = new Thickness(0, 2),
        };
        var val = new TextBlock
        {
            Text = value,
            FontFamily = new FontFamily("Consolas,Menlo,DejaVu Sans Mono,monospace"),
            FontSize = 12,
            Margin = new Thickness(0, 2),
            TextTrimming = TextTrimming.CharacterEllipsis,
        };
        Grid.SetColumn(lbl, 0); Grid.SetRow(lbl, row);
        Grid.SetColumn(val, 2); Grid.SetRow(val, row);
        host.Children.Add(lbl);
        host.Children.Add(val);
    }
}
