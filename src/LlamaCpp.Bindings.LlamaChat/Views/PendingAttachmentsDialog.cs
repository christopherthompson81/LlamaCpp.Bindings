using System;
using System.Collections.Specialized;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Markup.Xaml.MarkupExtensions;
using Avalonia.Media;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.Services;
using LlamaCpp.Bindings.LlamaChat.ViewModels;

namespace LlamaCpp.Bindings.LlamaChat.Views;

/// <summary>
/// Gallery dialog for the pending-attachments queue. Surfaces when there
/// are enough attachments that the inline compose-bar strip stops being
/// comfortable to scan. Shares the backing collection with the compose
/// strip (the same <see cref="MainWindowViewModel.PendingAttachments"/>),
/// so any remove from here immediately updates the strip behind the dialog.
///
/// Code-only, matching <see cref="ShortcutsDialog"/> / <see cref="AboutDialog"/>.
/// </summary>
public sealed class PendingAttachmentsDialog : Window
{
    private readonly MainWindowViewModel _vm;
    private readonly WrapPanel _grid;
    private readonly TextBlock _header;

    public PendingAttachmentsDialog(MainWindowViewModel vm)
    {
        _vm = vm;
        Title = "Pending attachments";
        Width = 680;
        Height = 520;
        MinWidth = 480;
        MinHeight = 320;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        this[!BackgroundProperty] = new DynamicResourceExtension("Background");

        _header = new TextBlock
        {
            FontSize = 18,
            FontWeight = FontWeight.SemiBold,
            Margin = new Thickness(24, 20, 24, 4),
        };
        _header[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("Foreground");

        var sub = new TextBlock
        {
            Text = "These attachments will travel with your next message. "
                 + "Remove the ones you don't want, or clear them all.",
            Margin = new Thickness(24, 0, 24, 12),
            TextWrapping = TextWrapping.Wrap,
            FontSize = 12,
        };
        sub[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("MutedForeground");

        _grid = new WrapPanel
        {
            Orientation = Orientation.Horizontal,
            Margin = new Thickness(16, 0, 16, 0),
            ItemSpacing = 10,
            LineSpacing = 10,
        };

        var scroll = new ScrollViewer
        {
            HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            Content = _grid,
        };

        var clearBtn = new Button { Content = "Clear all" };
        clearBtn.Classes.Add("destructive");
        clearBtn.Click += (_, _) => _vm.ClearPendingAttachmentsCommand.Execute(null);

        var closeBtn = new Button { Content = "Close" };
        closeBtn.Classes.Add("outline");
        closeBtn.Click += (_, _) => Close();

        var footer = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("Auto,*,Auto"),
            Margin = new Thickness(24, 12, 24, 20),
        };
        Grid.SetColumn(clearBtn, 0);
        Grid.SetColumn(closeBtn, 2);
        footer.Children.Add(clearBtn);
        footer.Children.Add(closeBtn);

        var root = new Grid { RowDefinitions = new RowDefinitions("Auto,Auto,*,Auto") };
        Grid.SetRow(_header, 0);
        Grid.SetRow(sub, 1);
        Grid.SetRow(scroll, 2);
        Grid.SetRow(footer, 3);
        root.Children.Add(_header);
        root.Children.Add(sub);
        root.Children.Add(scroll);
        root.Children.Add(footer);
        Content = root;

        // Initial population + live sync — any remove from our own buttons
        // or from the inline strip behind us is reflected immediately.
        Rebuild();
        _vm.PendingAttachments.CollectionChanged += OnCollectionChanged;

        KeyDown += (_, e) =>
        {
            if (e.Key == Key.Escape) { Close(); e.Handled = true; }
        };
        Closed += (_, _) => _vm.PendingAttachments.CollectionChanged -= OnCollectionChanged;
    }

    private void OnCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        Rebuild();
        if (_vm.PendingAttachments.Count == 0) Close();
    }

    private void Rebuild()
    {
        _header.Text = _vm.PendingAttachments.Count switch
        {
            0 => "No pending attachments",
            1 => "1 pending attachment",
            var n => $"{n} pending attachments",
        };

        _grid.Children.Clear();
        foreach (var a in _vm.PendingAttachments)
        {
            _grid.Children.Add(BuildTile(a));
        }
    }

    private Control BuildTile(Attachment attachment)
    {
        const int TileW = 140;
        const int TileH = 140;

        var tile = new Border
        {
            Width = TileW,
            Height = TileH,
            CornerRadius = new CornerRadius(6),
            BorderThickness = new Thickness(1),
            Padding = new Thickness(2),
        };
        tile[!Border.BorderBrushProperty] = new DynamicResourceExtension("Border");
        tile[!Border.BackgroundProperty] = new DynamicResourceExtension("Muted");

        var grid = new Grid();

        if (attachment.IsImage)
        {
            var image = new Image
            {
                Stretch = Stretch.UniformToFill,
                Source = (Avalonia.Media.Imaging.Bitmap?)AttachmentThumbnailConverter.Instance
                    .Convert(attachment, typeof(object), null, System.Globalization.CultureInfo.InvariantCulture),
            };
            grid.Children.Add(image);
        }
        else if (attachment.IsAudio)
        {
            var stack = new StackPanel
            {
                Orientation = Orientation.Vertical,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
                Spacing = 6,
            };
            var iconPath = new Avalonia.Controls.Shapes.Path();
            iconPath.Classes.Add("icon");
            iconPath.Classes.Add("lg");
            iconPath[!Avalonia.Controls.Shapes.Path.DataProperty] =
                new DynamicResourceExtension("IconVolume");
            iconPath.HorizontalAlignment = HorizontalAlignment.Center;
            stack.Children.Add(iconPath);

            var fname = new TextBlock
            {
                Text = attachment.FileName ?? "audio",
                FontSize = 10,
                TextWrapping = TextWrapping.Wrap,
                TextAlignment = TextAlignment.Center,
                MaxWidth = TileW - 16,
            };
            fname[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("MutedForeground");
            stack.Children.Add(fname);
            grid.Children.Add(stack);
        }

        // Remove button — same pattern as the inline strip.
        var remove = new Button
        {
            Width = 20,
            Height = 20,
            HorizontalAlignment = HorizontalAlignment.Right,
            VerticalAlignment = VerticalAlignment.Top,
            Margin = new Thickness(0, -4, -4, 0),
            CornerRadius = new CornerRadius(10),
        };
        remove.Classes.Add("ghost");
        remove.Classes.Add("icon");
        remove.Classes.Add("sm");
        remove[!Button.BackgroundProperty] = new DynamicResourceExtension("Card");
        var xIcon = new Avalonia.Controls.Shapes.Path();
        xIcon.Classes.Add("icon");
        xIcon.Classes.Add("xs");
        xIcon[!Avalonia.Controls.Shapes.Path.DataProperty] = new DynamicResourceExtension("IconX");
        remove.Content = xIcon;
        ToolTip.SetTip(remove, $"Remove {attachment.FileName ?? "attachment"}");
        remove.Click += (_, _) => _vm.RemovePendingAttachmentCommand.Execute(attachment);
        grid.Children.Add(remove);

        tile.Child = grid;
        return tile;
    }
}
