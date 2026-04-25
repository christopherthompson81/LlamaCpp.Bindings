using Avalonia.Controls;
using Avalonia.Interactivity;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.ViewModels;

namespace LlamaCpp.Bindings.LlamaChat.Views;

public partial class ProfileEditorView : UserControl
{
    public ProfileEditorView() => InitializeComponent();

    private void OnKindLocalClicked(object? sender, RoutedEventArgs e)
    {
        if (DataContext is ProfileEditorViewModel vm) vm.Kind = ProfileKind.Local;
    }

    private void OnKindRemoteClicked(object? sender, RoutedEventArgs e)
    {
        if (DataContext is ProfileEditorViewModel vm) vm.Kind = ProfileKind.Remote;
    }
}
