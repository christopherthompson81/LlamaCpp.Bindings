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

        // Kill any child server we spawned, even on abrupt CLR teardown
        // (signals, unhandled exceptions). Avalonia's ShutdownRequested
        // covers the normal-exit path; this is the belt to its braces.
        System.AppDomain.CurrentDomain.ProcessExit += (_, _) =>
            ServerLaunchService.Instance.Dispose();

        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var vm = new MainWindowViewModel();
            desktop.MainWindow = new MainWindow { DataContext = vm };
            desktop.ShutdownRequested += (_, _) =>
            {
                ServerLaunchService.Instance.Dispose();
                vm.Dispose();
            };
        }
        base.OnFrameworkInitializationCompleted();
    }
}
