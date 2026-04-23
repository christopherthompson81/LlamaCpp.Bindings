using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using LlamaCpp.Bindings.LlamaChat.Services;
using LlamaCpp.Bindings.LlamaChat.ViewModels;
using LlamaCpp.Bindings.LlamaChat.Views;

namespace LlamaCpp.Bindings.LlamaChat;

public partial class App : Application
{
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        // Install global exception handlers before we create any windows so
        // that failures during ctor / initial binding are caught too.
        ErrorBoundary.Install();

        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var vm = new MainWindowViewModel();
            desktop.MainWindow = new MainWindow { DataContext = vm };
            desktop.ShutdownRequested += (_, _) => vm.Dispose();
        }
        base.OnFrameworkInitializationCompleted();
    }
}
