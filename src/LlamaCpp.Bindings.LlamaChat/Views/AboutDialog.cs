using System;
using System.Diagnostics;
using System.Reflection;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Markup.Xaml.MarkupExtensions;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Platform;

namespace LlamaCpp.Bindings.LlamaChat.Views;

/// <summary>
/// Modal "About" window reached from Help → About. Hero image on the left,
/// app + bundled-llama.cpp version + GitHub links on the right. Pure code
/// for consistency with <see cref="ShortcutsDialog"/>.
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
        Title = "About LlamaChat";
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
        // Left: hero image. Load from the embedded Avalonia resource. Uses
        // Uniform stretch so the tall 742x976 aspect isn't distorted.
        Image hero;
        try
        {
            using var stream = AssetLoader.Open(
                new Uri("avares://LlamaCpp.Bindings.LlamaChat/Assets/llama_hero.png"));
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
            // Missing asset shouldn't break the dialog — degrade to the
            // text-only column.
            hero = new Image { Width = 0 };
        }

        var appVersion = Assembly.GetExecutingAssembly().GetName().Version?.ToString(3) ?? "dev";

        var title = new TextBlock
        {
            Text = "LlamaChat",
            FontSize = 28,
            FontWeight = FontWeight.SemiBold,
        };

        var subtitle = new TextBlock
        {
            Text = "A desktop chat front-end for llama.cpp, built on LlamaCpp.Bindings.",
            Classes = { "muted", "sm" },
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 4, 0, 16),
        };

        var versionTable = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("Auto,12,*"),
            RowDefinitions = new RowDefinitions("Auto,Auto"),
            Margin = new Thickness(0, 0, 0, 16),
        };
        AddRow(versionTable, 0, "App",        appVersion);
        AddRow(versionTable, 1, "llama.cpp",  LlamaCppVersion);

        var linksHeader = new TextBlock
        {
            Text = "Links",
            FontWeight = FontWeight.SemiBold,
            Margin = new Thickness(0, 0, 0, 4),
        };

        var linksStack = new StackPanel { Spacing = 2 };
        linksStack.Children.Add(LinkButton("llama.cpp on GitHub",           LlamaCppUrl));
        linksStack.Children.Add(LinkButton("LlamaCpp.Bindings on GitHub",   BindingsUrl));

        var closeBtn = new Button { Content = "Close" };
        closeBtn.Classes.Add("outline");
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
        var btn = new Button { Content = text };
        btn.Classes.Add("link");
        btn.Click += (_, _) =>
        {
            try
            {
                // UseShellExecute = true so the OS picks the default browser.
                // Wrap in try/catch — a headless / sandboxed environment may
                // reject the launch but the dialog should stay functional.
                Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
            }
            catch
            {
                // Ignore — nothing the user can meaningfully recover from,
                // they can copy the URL out of the tooltip if needed.
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
            Classes = { "muted", "sm" },
            Margin = new Thickness(0, 2),
        };
        var val = new TextBlock
        {
            Text = value,
            FontFamily = new FontFamily("Consolas,Menlo,DejaVu Sans Mono,monospace"),
            Classes = { "sm" },
            Margin = new Thickness(0, 2),
            TextTrimming = TextTrimming.CharacterEllipsis,
        };
        Grid.SetColumn(lbl, 0); Grid.SetRow(lbl, row);
        Grid.SetColumn(val, 2); Grid.SetRow(val, row);
        host.Children.Add(lbl);
        host.Children.Add(val);
    }
}
