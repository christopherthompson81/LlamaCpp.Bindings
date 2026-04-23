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

        // Wire compose-box key handling after InitializeComponent builds the tree.
        // Tunnel routing fires BEFORE the TextBox class handler (OnKeyDown), which
        // is what lets us intercept plain Enter and send instead of inserting a
        // newline (AcceptsReturn="True" lets newlines in from paste/Shift+Enter).
        var composeBox = this.FindControl<TextBox>("ComposeBox");
        if (composeBox is not null)
        {
            composeBox.AddHandler(InputElement.KeyDownEvent,
                OnComposeKeyDownTunnel, RoutingStrategies.Tunnel);
            composeBox.AddHandler(InputElement.KeyDownEvent,
                OnComposeKeyDownBubble, RoutingStrategies.Bubble, handledEventsToo: true);
        }
    }

    // --- Compose key handling ---
    // AcceptsReturn="True" so the TextBox never strips newlines (paste or Shift+Enter).
    // Tunnel handler fires BEFORE the TextBox class handler, so we can consume plain
    // Enter for Send before the class handler would insert a newline.
    private void OnComposeKeyDownTunnel(object? sender, KeyEventArgs e)
    {
        if (e.Key != Key.Enter || (e.KeyModifiers & KeyModifiers.Shift) != 0) return;
        e.Handled = true;
        if (DataContext is MainWindowViewModel vm && vm.SendCommand.CanExecute(null))
            vm.SendCommand.Execute(null);
    }

    // Bubble handler (handledEventsToo=true) for Ctrl+V image paste. The TextBox
    // will already have processed Ctrl+V as a text paste; we additionally check
    // for an image payload and attach it if present.
    private async void OnComposeKeyDownBubble(object? sender, KeyEventArgs e)
    {
        if (e.Key == Key.V && (e.KeyModifiers & KeyModifiers.Control) != 0)
            await TryPasteImageFromClipboardAsync();
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
    // can consume media (images or audio) and the payload contains file paths.
    // Any other payload (plain text, URIs, etc.) is ignored so the default
    // TextBox paste behavior keeps working. TryAddPendingMedia filters out
    // non-media extensions per-file.

    private void OnComposeDragOver(object? sender, DragEventArgs e)
    {
        if (_vm?.CanAttachMedia == true &&
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
        if (_vm is null || !_vm.CanAttachMedia) return;
        var xfer = e.DataTransfer;
        if (xfer is null) return;

        var files = xfer.TryGetFiles();
        if (files is null) return;
        foreach (var file in files)
        {
            var path = file.TryGetLocalPath();
            if (!string.IsNullOrEmpty(path))
            {
                _vm.TryAddPendingMedia(path);
            }
        }
        e.Handled = true;
    }

    // clipboard has a bitmap payload AND the model supports image input, we
    // re-encode it as PNG bytes and add as an attachment. Other payloads
    // (plain text, audio clips — Avalonia doesn't expose audio clipboard
    // formats portably) fall through to the TextBox's default paste.
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

            _vm.AddPendingMediaBytes(bytes, "image/png", fileName: null);
            return true;
        }
        catch
        {
            return false;
        }
    }
}
