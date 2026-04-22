using Avalonia.Controls;
using Avalonia.Input;
using LlamaCpp.Bindings.LlamaChat.ViewModels;

namespace LlamaCpp.Bindings.LlamaChat.Views;

public partial class MainWindow : Window
{
    public MainWindow() => InitializeComponent();

    // Enter sends; Shift+Enter inserts a newline. The TextBox has
    // AcceptsReturn="False" so its class handler never consumes Enter before
    // us — we'd otherwise be shadowed because TextBox marks Enter handled
    // during its own OnKeyDown before bubbling to attached handlers. We
    // emulate the Shift+Enter newline ourselves by splicing at the caret.
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
}
