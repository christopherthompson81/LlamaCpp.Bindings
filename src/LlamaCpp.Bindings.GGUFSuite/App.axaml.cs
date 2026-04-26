using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using LlamaCpp.Bindings.GGUFSuite.Services;
using LlamaCpp.Bindings.GGUFSuite.ViewModels;
using LlamaCpp.Bindings.GGUFSuite.Views;

namespace LlamaCpp.Bindings.GGUFSuite;

public partial class App : Application
{
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var vm = new MainWindowViewModel();
            var window = new MainWindow { DataContext = vm };

            // Restore the prior session's window placement before the
            // first show. First launch (no persisted file) leaves
            // Avalonia's default startup behaviour in place.
            var placement = WindowPlacementStore.Load();
            WindowPlacementStore.Apply(window, placement);

            // Snapshot bounds while the platform window is still alive.
            // ShutdownRequested fires after the X11/Win32 window has
            // been disposed, so any Screens/Position access from there
            // throws ObjectDisposedException — Closing is the last
            // safe point to read live geometry.
            WindowPlacement? captured = null;
            window.Closing += (_, _) =>
            {
                captured = WindowPlacementStore.Capture(window);
            };

            desktop.MainWindow = window;
            desktop.ShutdownRequested += (_, _) =>
            {
                if (captured is not null) WindowPlacementStore.Save(captured);
            };
        }
        base.OnFrameworkInitializationCompleted();
    }
}
