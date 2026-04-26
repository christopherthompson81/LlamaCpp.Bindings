using Avalonia.Controls;
using Avalonia.Interactivity;

namespace LlamaCpp.Bindings.GGUFSuite.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    private void OnExitClicked(object? sender, RoutedEventArgs e) => Close();

    private async void OnAboutClicked(object? sender, RoutedEventArgs e)
    {
        var dlg = new AboutDialog();
        await dlg.ShowDialog(this);
    }
}
