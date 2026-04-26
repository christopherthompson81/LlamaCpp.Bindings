using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using LlamaCpp.Bindings.GGUFLab.ViewModels;

namespace LlamaCpp.Bindings.GGUFLab.Views;

public partial class LocalModelsView : UserControl
{
    public LocalModelsView()
    {
        InitializeComponent();
    }

    private void OnSetActive(object? sender, RoutedEventArgs e)
    {
        if (DataContext is LocalModelsViewModel vm
            && sender is Button btn
            && btn.Tag is LocalModelsViewModel.LocalModelRow row)
        {
            vm.SetActive(row);
        }
    }

    private void OnReveal(object? sender, RoutedEventArgs e)
    {
        if (DataContext is LocalModelsViewModel vm
            && sender is Button btn
            && btn.Tag is LocalModelsViewModel.LocalModelRow row)
        {
            vm.Reveal(row);
        }
    }

    private async void OnDelete(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not LocalModelsViewModel vm
            || sender is not Button btn
            || btn.Tag is not LocalModelsViewModel.LocalModelRow row)
        {
            return;
        }
        // Confirm before destroying gigabytes — match "Delete" exactly so a
        // mistapped Enter doesn't take the folder. The whole folder is
        // removed, including every format inside.
        var confirm = new Window
        {
            Title = "Delete model folder?",
            Width = 480,
            Height = 170,
            CanResize = false,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
        };
        var msg = new TextBlock
        {
            Text = $"Delete {row.DisplayName} ({row.DisplaySize}) and ALL files inside it?\nThis cannot be undone.",
            TextWrapping = Avalonia.Media.TextWrapping.Wrap,
            Margin = new Avalonia.Thickness(20, 16, 20, 16),
        };
        var yes = new Button { Content = "Delete", Margin = new Avalonia.Thickness(0, 0, 8, 0) };
        var no  = new Button { Content = "Cancel" };
        bool yesClicked = false;
        yes.Click += (_, _) => { yesClicked = true; confirm.Close(); };
        no.Click  += (_, _) => confirm.Close();
        var btnRow = new StackPanel
        {
            Orientation = Avalonia.Layout.Orientation.Horizontal,
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Right,
            Margin = new Avalonia.Thickness(20, 0, 20, 16),
        };
        btnRow.Children.Add(yes);
        btnRow.Children.Add(no);
        var stack = new DockPanel();
        DockPanel.SetDock(btnRow, Dock.Bottom);
        stack.Children.Add(btnRow);
        stack.Children.Add(msg);
        confirm.Content = stack;

        var owner = TopLevel.GetTopLevel(this) as Window;
        if (owner is not null) await confirm.ShowDialog(owner);
        else await Task.Yield();

        if (yesClicked) vm.Delete(row);
    }
}
