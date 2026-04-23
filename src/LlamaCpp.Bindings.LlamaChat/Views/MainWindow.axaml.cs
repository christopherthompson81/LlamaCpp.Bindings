using System;
using System.Collections.Specialized;
using System.ComponentModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Input.Platform;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using LlamaCpp.Bindings.LlamaChat.ViewModels;

namespace LlamaCpp.Bindings.LlamaChat.Views;

public partial class MainWindow : Window
{
    private MainWindowViewModel? _vm;
    private ConversationViewModel? _subscribedConv;
    private readonly DispatcherTimer _streamScrollTimer;

    public MainWindow()
    {
        InitializeComponent();

        // While the assistant is streaming, poll the scroll-to-end every
        // 100ms. Simpler than subscribing to PropertyChanged on the last
        // assistant message's Content — and since the visual cost is trivial,
        // not worth the per-event bookkeeping.
        _streamScrollTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(100) };
        _streamScrollTimer.Tick += (_, _) => ScrollToEndIfEnabled();

        DataContextChanged += OnDataContextChanged;

        // Drag-drop of image files onto the compose area. Listening on the
        // window (with AllowDrop=true on the compose grid) lets the drop
        // target catch file paths from the OS file manager.
        AddHandler(DragDrop.DragOverEvent, OnComposeDragOver);
        AddHandler(DragDrop.DropEvent, OnComposeDrop);
    }

    // --- Compose (Enter sends; Shift+Enter newline; Ctrl+V may paste image) ---
    private async void OnComposeKeyDown(object? sender, KeyEventArgs e)
    {
        if (sender is not TextBox box) return;

        // Ctrl+V: try to intercept clipboard image before the TextBox handles
        // it as text. If no image data is present, let the default paste run.
        if (e.Key == Key.V && (e.KeyModifiers & KeyModifiers.Control) != 0)
        {
            if (await TryPasteImageFromClipboardAsync())
            {
                e.Handled = true;
            }
            return;
        }

        if (e.Key != Key.Enter) return;

        if ((e.KeyModifiers & KeyModifiers.Shift) != 0)
        {
            var text = box.Text ?? string.Empty;
            var caret = box.CaretIndex;
            if (caret < 0 || caret > text.Length) caret = text.Length;
            box.Text = text.Substring(0, caret) + "\n" + text.Substring(caret);
            box.CaretIndex = caret + 1;
            e.Handled = true;
            return;
        }

        e.Handled = true;
        if (DataContext is MainWindowViewModel vm && vm.SendCommand.CanExecute(null))
        {
            vm.SendCommand.Execute(null);
        }
    }

    // --- Sidebar Ctrl+K focus-search ------------------------------------
    protected override void OnKeyDown(KeyEventArgs e)
    {
        if (e.Key == Key.K && (e.KeyModifiers & KeyModifiers.Control) != 0)
        {
            var search = this.FindControl<TextBox>("SearchBox");
            if (search is not null)
            {
                search.Focus();
                search.SelectAll();
                e.Handled = true;
                return;
            }
        }
        base.OnKeyDown(e);
    }

    // --- Sidebar inline rename ------------------------------------------
    private void OnRenameAttached(object? sender, VisualTreeAttachmentEventArgs e)
    {
        if (sender is TextBox box && box.IsVisible)
        {
            box.Focus();
            box.SelectAll();
        }
    }

    private void OnRenameKeyDown(object? sender, KeyEventArgs e)
    {
        if (sender is not TextBox { DataContext: ConversationViewModel conv }) return;

        if (e.Key == Key.Enter)
        {
            e.Handled = true;
            if (DataContext is MainWindowViewModel vm)
                vm.EndRenameCommand.Execute(conv);
        }
        else if (e.Key == Key.Escape)
        {
            e.Handled = true;
            conv.IsEditing = false;
        }
    }

    private void OnRenameLostFocus(object? sender, RoutedEventArgs e)
    {
        if (sender is TextBox { DataContext: ConversationViewModel conv }
            && conv.IsEditing
            && DataContext is MainWindowViewModel vm)
        {
            vm.EndRenameCommand.Execute(conv);
        }
    }

    // --- Auto-scroll plumbing -------------------------------------------

    private void OnDataContextChanged(object? sender, EventArgs e)
    {
        if (_vm is not null) _vm.PropertyChanged -= OnVmPropertyChanged;
        HookConversation(null);

        _vm = DataContext as MainWindowViewModel;
        if (_vm is null) return;

        _vm.PropertyChanged += OnVmPropertyChanged;
        HookConversation(_vm.SelectedConversation);
    }

    private void OnVmPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        switch (e.PropertyName)
        {
            case nameof(MainWindowViewModel.SelectedConversation):
                HookConversation(_vm?.SelectedConversation);
                break;
            case nameof(MainWindowViewModel.IsGenerating):
                if (_vm?.IsGenerating == true) _streamScrollTimer.Start();
                else _streamScrollTimer.Stop();
                break;
        }
    }

    private void HookConversation(ConversationViewModel? conv)
    {
        if (_subscribedConv is not null)
            _subscribedConv.Messages.CollectionChanged -= OnMessagesChanged;
        _subscribedConv = conv;
        if (conv is not null)
            conv.Messages.CollectionChanged += OnMessagesChanged;
        // Switching conversation jumps the scroll to the bottom so you land
        // on the latest message in the newly-selected chat.
        Dispatcher.UIThread.Post(ScrollToEndIfEnabled, DispatcherPriority.Background);
    }

    private void OnMessagesChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        if (e.Action == NotifyCollectionChangedAction.Add)
            Dispatcher.UIThread.Post(ScrollToEndIfEnabled, DispatcherPriority.Background);
    }

    private void ScrollToEndIfEnabled()
    {
        if (_vm?.AppSettings.AutoScroll != true) return;
        var scroll = this.FindControl<ScrollViewer>("ChatScroll");
        scroll?.ScrollToEnd();
    }

    // --- Compose: drag-drop + clipboard-paste image attachments ---------
    //
    // Drop target is the whole window. We only accept the drop when the model
    // can consume images and the payload contains file paths. Any other
    // payload (plain text, URIs, etc.) is ignored so the default TextBox
    // paste behavior keeps working.

    private void OnComposeDragOver(object? sender, DragEventArgs e)
    {
        if (_vm?.CanAttachImages == true &&
            e.DataTransfer is { } xfer &&
            xfer.Contains(DataFormat.File))
        {
            e.DragEffects = DragDropEffects.Copy;
        }
        else
        {
            e.DragEffects = DragDropEffects.None;
        }
    }

    private void OnComposeDrop(object? sender, DragEventArgs e)
    {
        if (_vm is null || !_vm.CanAttachImages) return;
        var xfer = e.DataTransfer;
        if (xfer is null) return;

        var files = xfer.TryGetFiles();
        if (files is null) return;
        foreach (var file in files)
        {
            var path = file.TryGetLocalPath();
            if (!string.IsNullOrEmpty(path))
            {
                _vm.TryAddPendingImage(path);
            }
        }
        e.Handled = true;
    }

    // Clipboard image paste: intercept Ctrl+V in the compose TextBox. If the
    // clipboard has a bitmap payload AND the model supports attachments, we
    // re-encode it as PNG bytes and add as an attachment. Any other payload
    // falls through to the TextBox's normal text paste.
    private async System.Threading.Tasks.Task<bool> TryPasteImageFromClipboardAsync()
    {
        if (_vm is null || !_vm.CanAttachImages) return false;
        var clipboard = Clipboard;
        if (clipboard is null) return false;

        try
        {
            var bitmap = await clipboard.TryGetBitmapAsync();
            if (bitmap is null) return false;

            using var ms = new System.IO.MemoryStream();
            bitmap.Save(ms);
            var bytes = ms.ToArray();
            if (bytes.Length == 0) return false;

            _vm.AddPendingImageBytes(bytes, "image/png", fileName: null);
            return true;
        }
        catch
        {
            return false;
        }
    }
}
