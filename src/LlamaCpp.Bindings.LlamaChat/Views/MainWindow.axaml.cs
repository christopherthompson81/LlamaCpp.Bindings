using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using LlamaCpp.Bindings.LlamaChat.ViewModels;

namespace LlamaCpp.Bindings.LlamaChat.Views;

public partial class MainWindow : Window
{
    public MainWindow() => InitializeComponent();

    // --- Compose (Enter sends; Shift+Enter newline) ---------------------

    // Enter sends; Shift+Enter inserts a newline. The compose TextBox has
    // AcceptsReturn="False" so its class handler never consumes Enter before
    // us — we'd otherwise be shadowed because TextBox marks Enter handled
    // during its own OnKeyDown before bubbling to attached handlers. We
    // emulate Shift+Enter's newline ourselves by splicing at the caret.
    private void OnComposeKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key != Key.Enter || sender is not TextBox box) return;

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

    // Handled here rather than in Window.KeyBindings because focusing a named
    // element is a view-layer concern that shouldn't leak into the VM.
    // Listens on the Window so the shortcut works regardless of current focus.
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

    // Focus + select-all when the rename TextBox first appears so the user
    // can just start typing. AttachedToVisualTree fires once, after layout.
    private void OnRenameAttached(object? sender, VisualTreeAttachmentEventArgs e)
    {
        if (sender is TextBox box && box.IsVisible)
        {
            box.Focus();
            box.SelectAll();
        }
    }

    // Enter / Escape commit or abort rename; click-away is handled by
    // OnRenameLostFocus below.
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
}
