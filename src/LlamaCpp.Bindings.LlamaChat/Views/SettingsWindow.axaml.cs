using Avalonia.Controls;
using Avalonia.Interactivity;

namespace LlamaCpp.Bindings.LlamaChat.Views;

public partial class SettingsWindow : Window
{
    public SettingsWindow() => InitializeComponent();

    private void OnCloseClicked(object? sender, RoutedEventArgs e) => Close();
}
