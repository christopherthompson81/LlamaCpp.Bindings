using System;
using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.Interactivity;
using LlamaCpp.Bindings.GGUFLab.Services;
using LlamaCpp.Bindings.GGUFLab.ViewModels;

namespace LlamaCpp.Bindings.GGUFLab.Views;

public partial class HfBrowserView : UserControl
{
    public HfBrowserView()
    {
        InitializeComponent();
    }

    private async void OnRepoSelectionChanged(object? sender, SelectionChangedEventArgs e)
    {
        if (DataContext is not HfBrowserViewModel vm) return;
        var picked = vm.SelectedRepo;
        // Bound via SelectedItem already; this hook is just to trigger
        // the file-list fetch when selection lands on something.
        await vm.SelectRepoAsync(picked);
    }

    private async void OnDownloadClicked(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not HfBrowserViewModel vm) return;
        if (sender is Button btn && btn.Tag is HfModelFile file)
        {
            await vm.DownloadAsync(file);
        }
    }

    private void OnOpenRepoLink(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not HfBrowserViewModel vm) return;
        if (string.IsNullOrEmpty(vm.SelectedRepoUrl)) return;
        try
        {
            Process.Start(new ProcessStartInfo(vm.SelectedRepoUrl) { UseShellExecute = true });
        }
        catch
        {
            // Sandboxed environments may reject the launch — nothing
            // useful to surface; the URL is shown in the button text.
        }
    }
}
