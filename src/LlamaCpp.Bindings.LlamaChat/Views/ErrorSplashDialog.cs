using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Markup.Xaml.MarkupExtensions;
using Avalonia.Media;
using LlamaCpp.Bindings.LlamaChat.Services;

namespace LlamaCpp.Bindings.LlamaChat.Views;

/// <summary>
/// Modal shown when an uncaught exception reaches the app's error boundary.
/// Dense but copy-friendly: exception type + message up top, expandable
/// full trace below, three actions — Copy details, Try to continue,
/// Close application.
///
/// Pure-code for simplicity (matches <see cref="ShortcutsDialog"/> /
/// <see cref="AboutDialog"/>). Opened via <c>ShowDialog(owner)</c> so the
/// rest of the app is inert until the user decides.
/// </summary>
public sealed class ErrorSplashDialog : Window
{
    public ErrorSplashDialog(Exception ex, string? context)
    {
        Title = "Unexpected error";
        Width = 640;
        Height = 480;
        MinWidth = 480;
        MinHeight = 320;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        this[!BackgroundProperty] = new DynamicResourceExtension("Background");

        var heading = new TextBlock
        {
            Text = "Something went wrong",
            FontSize = 20,
            FontWeight = FontWeight.SemiBold,
            Margin = new Thickness(24, 20, 24, 4),
        };
        heading[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("Foreground");

        var sub = new TextBlock
        {
            Text = "The app caught an exception it didn't know how to recover from. "
                 + "Its state may or may not be usable — the safe move is to close and reopen.",
            Margin = new Thickness(24, 0, 24, 16),
            TextWrapping = TextWrapping.Wrap,
            FontSize = 12,
        };
        sub[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("MutedForeground");

        var summary = new TextBlock
        {
            Text = $"{ex.GetType().Name}: {ex.Message}",
            Margin = new Thickness(24, 0, 24, 8),
            FontWeight = FontWeight.SemiBold,
            TextWrapping = TextWrapping.Wrap,
        };
        summary[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("Foreground");

        var contextLine = new TextBlock
        {
            Text = string.IsNullOrEmpty(context) ? "" : $"Context: {context}",
            IsVisible = !string.IsNullOrEmpty(context),
            Margin = new Thickness(24, 0, 24, 12),
            FontSize = 11,
        };
        contextLine[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("MutedForeground");

        var traceDetails = BuildTraceBox(ex.ToString());

        var copyBtn = new Button { Content = "Copy details" };
        copyBtn.Classes.Add("outline");
        copyBtn.Click += async (_, _) =>
            await DialogService.CopyToClipboardAsync(BuildClipboardText(ex, context));

        var continueBtn = new Button { Content = "Try to continue" };
        continueBtn.Classes.Add("ghost");
        continueBtn.Click += (_, _) => Close();

        var quitBtn = new Button { Content = "Close application" };
        quitBtn.Classes.Add("destructive");
        quitBtn.Click += (_, _) =>
        {
            if (Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime d)
                d.Shutdown();
            else Environment.Exit(1);
        };

        var logPathHint = new TextBlock
        {
            Text = $"Full trace also written to {ErrorLog.LogPath}",
            Margin = new Thickness(24, 0, 24, 0),
            FontSize = 10,
            TextWrapping = TextWrapping.Wrap,
        };
        logPathHint[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("MutedForeground");

        var footer = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("Auto,*,Auto,Auto"),
            Margin = new Thickness(24, 12, 24, 20),
        };
        Grid.SetColumn(copyBtn, 0);
        Grid.SetColumn(continueBtn, 2);
        Grid.SetColumn(quitBtn, 3);
        continueBtn.Margin = new Thickness(0, 0, 8, 0);
        footer.Children.Add(copyBtn);
        footer.Children.Add(continueBtn);
        footer.Children.Add(quitBtn);

        var root = new Grid { RowDefinitions = new RowDefinitions("Auto,Auto,Auto,Auto,*,Auto,Auto") };
        Grid.SetRow(heading, 0);
        Grid.SetRow(sub, 1);
        Grid.SetRow(summary, 2);
        Grid.SetRow(contextLine, 3);
        Grid.SetRow(traceDetails, 4);
        Grid.SetRow(logPathHint, 5);
        Grid.SetRow(footer, 6);
        root.Children.Add(heading);
        root.Children.Add(sub);
        root.Children.Add(summary);
        root.Children.Add(contextLine);
        root.Children.Add(traceDetails);
        root.Children.Add(logPathHint);
        root.Children.Add(footer);
        Content = root;

        KeyDown += (_, e) =>
        {
            if (e.Key == Key.Escape) Close();
        };
    }

    private static Control BuildTraceBox(string trace)
    {
        var tb = new TextBox
        {
            Text = trace,
            IsReadOnly = true,
            AcceptsReturn = true,
            TextWrapping = TextWrapping.NoWrap,
            FontSize = 11,
        };
        tb[!TextBox.FontFamilyProperty] = new DynamicResourceExtension("CodeFontFamily");

        var scroller = new ScrollViewer
        {
            HorizontalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Auto,
            VerticalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Auto,
            Content = tb,
        };

        var border = new Border
        {
            Margin = new Thickness(24, 0, 24, 12),
            CornerRadius = new CornerRadius(6),
            Padding = new Thickness(8),
            Child = scroller,
        };
        border[!Border.BackgroundProperty] = new DynamicResourceExtension("CodeBackground");
        border[!Border.BorderBrushProperty] = new DynamicResourceExtension("Border");
        border.BorderThickness = new Thickness(1);
        return border;
    }

    private static string BuildClipboardText(Exception ex, string? context)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append("LlamaChat error report\n");
        sb.Append(DateTime.Now.ToString("o")).Append('\n');
        if (!string.IsNullOrEmpty(context)) sb.Append("context: ").Append(context).Append('\n');
        sb.Append('\n').Append(ex).Append('\n');
        return sb.ToString();
    }
}
