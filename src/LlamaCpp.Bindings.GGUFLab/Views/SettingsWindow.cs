using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Markup.Xaml.MarkupExtensions;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using LlamaCpp.Bindings.GGUFLab.Services;

namespace LlamaCpp.Bindings.GGUFLab.Views;

/// <summary>
/// Modal Settings window — currently one section (Workspace), reachable
/// from the Settings menu. Saves on "Save", discards on "Cancel" or Esc.
/// Code-only since the surface is small and stays consistent with
/// <see cref="AboutDialog"/>.
/// </summary>
public sealed class SettingsWindow : Window
{
    private readonly WorkspaceSettings _settings;
    private readonly TextBox _workspaceRoot;
    private readonly TextBox _hfToken;

    public bool Saved { get; private set; }

    public SettingsWindow(WorkspaceSettings settings)
    {
        _settings = settings;

        Title = "GGUF Lab — Settings";
        Width = 680;
        Height = 320;
        MinWidth = 520;
        MinHeight = 280;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        CanResize = false;
        this[!BackgroundProperty] = new DynamicResourceExtension("Background");

        _workspaceRoot = new TextBox
        {
            Text = settings.WorkspaceRoot,
            PlaceholderText = WorkspaceSettings.DefaultWorkspaceRoot,
            HorizontalAlignment = HorizontalAlignment.Stretch,
        };
        _hfToken = new TextBox
        {
            Text = settings.HuggingFaceToken ?? string.Empty,
            PlaceholderText = "(optional) hf_… for gated repos",
            PasswordChar = '•',
            HorizontalAlignment = HorizontalAlignment.Stretch,
        };

        Content = BuildBody();
        KeyDown += (_, e) =>
        {
            if (e.Key == Key.Escape) { Close(); e.Handled = true; }
        };
    }

    private Control BuildBody()
    {
        var heading = new TextBlock
        {
            Text = "Workspace",
            FontSize = 16,
            FontWeight = FontWeight.SemiBold,
            Margin = new Thickness(0, 0, 0, 8),
        };

        var rootHelp = new TextBlock
        {
            Text = "Where the HF Browser saves downloads and the Local Models tool scans.",
            Opacity = 0.65,
            FontSize = 12,
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 0, 0, 6),
        };

        var browse = new Button { Content = "Browse…" };
        browse.Click += async (_, _) =>
        {
            var picked = await PickFolderAsync("Choose workspace root");
            if (picked is not null) _workspaceRoot.Text = picked;
        };

        var rootRow = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("*,8,Auto"),
        };
        Grid.SetColumn(_workspaceRoot, 0); rootRow.Children.Add(_workspaceRoot);
        Grid.SetColumn(browse,         2); rootRow.Children.Add(browse);

        var tokenHelp = new TextBlock
        {
            Text = "HuggingFace token (optional). Required for gated/private repos.",
            Opacity = 0.65,
            FontSize = 12,
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 16, 0, 6),
        };

        var save = new Button { Content = "Save" };
        save.Classes.Add("accent");
        save.Click += (_, _) =>
        {
            var newRoot = string.IsNullOrWhiteSpace(_workspaceRoot.Text)
                ? WorkspaceSettings.DefaultWorkspaceRoot
                : _workspaceRoot.Text!;
            try { Directory.CreateDirectory(newRoot); } catch { /* surface via failed write below */ }
            _settings.WorkspaceRoot = newRoot;
            _settings.HuggingFaceToken = string.IsNullOrWhiteSpace(_hfToken.Text) ? null : _hfToken.Text;
            try { _settings.Save(); Saved = true; }
            catch { /* keep the dialog open so the user can correct */ return; }
            Close();
        };

        var cancel = new Button { Content = "Cancel" };
        cancel.Click += (_, _) => Close();

        var btnRow = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
            Spacing = 8,
            Margin = new Thickness(0, 20, 0, 0),
        };
        btnRow.Children.Add(cancel);
        btnRow.Children.Add(save);

        var col = new StackPanel { Margin = new Thickness(24), Spacing = 0 };
        col.Children.Add(heading);
        col.Children.Add(rootHelp);
        col.Children.Add(rootRow);
        col.Children.Add(tokenHelp);
        col.Children.Add(_hfToken);
        col.Children.Add(btnRow);
        return col;
    }

    private async Task<string?> PickFolderAsync(string title)
    {
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return null;
        var picked = await top.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
        });
        return picked.FirstOrDefault()?.TryGetLocalPath();
    }
}
