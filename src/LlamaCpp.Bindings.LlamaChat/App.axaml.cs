using System.Linq;
using Avalonia;
using Avalonia.Controls;
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
            var window = new MainWindow { DataContext = vm };
            ApplyPersistedWindowState(window, vm.AppSettings);

            // Snapshot bounds while the platform window is still alive.
            // ShutdownRequested fires after the X11/Win32 window has been
            // disposed, so any Screens/Position access from there throws
            // ObjectDisposedException — Closing is the last safe point.
            window.Closing += (_, _) => CapturePersistedWindowState(window, vm.AppSettings);

            desktop.MainWindow = window;
            desktop.ShutdownRequested += (_, _) =>
            {
                try { AppSettingsStore.Save(vm.AppSettings.ToModel()); } catch { }
                ServerLaunchService.Instance.Dispose();
                vm.Dispose();
            };
        }
        base.OnFrameworkInitializationCompleted();
    }

    /// <summary>
    /// Restore the prior session's window placement. Skipped on first launch
    /// (no persisted size yet) — Avalonia falls back to its default startup
    /// location. Multi-monitor placement: prefer the screen matched by name
    /// so the window lands on the same physical monitor even if the desktop
    /// arrangement changed; fall back to absolute coords (legacy / unknown
    /// monitor) and finally to default placement when neither resolves to
    /// a connected screen.
    /// </summary>
    private static void ApplyPersistedWindowState(Window window, AppSettingsViewModel s)
    {
        if (s.WindowWidth is double w && s.WindowHeight is double h && w > 100 && h > 100)
        {
            window.Width = w;
            window.Height = h;
        }

        var pos = ResolveStartupPosition(window, s);
        if (pos is PixelPoint p)
        {
            window.WindowStartupLocation = WindowStartupLocation.Manual;
            window.Position = p;
        }

        if (s.WindowMaximized)
        {
            window.WindowState = WindowState.Maximized;
        }
    }

    private static PixelPoint? ResolveStartupPosition(Window window, AppSettingsViewModel s)
    {
        var screens = window.Screens?.All;
        if (screens is null || screens.Count == 0)
        {
            // Platform doesn't expose screens (rare). Fall back to absolute
            // coords if we have them; the OS will route them somewhere.
            return s.WindowX is double x && s.WindowY is double y
                ? new PixelPoint((int)x, (int)y)
                : null;
        }

        // Preferred path: same monitor by display name.
        if (s.WindowScreenName is { Length: > 0 } name &&
            s.WindowScreenRelativeX is double relX &&
            s.WindowScreenRelativeY is double relY)
        {
            var match = screens.FirstOrDefault(sc => sc.DisplayName == name);
            if (match is not null)
            {
                return ClampToScreen(
                    new PixelPoint(match.Bounds.X + (int)relX, match.Bounds.Y + (int)relY),
                    match);
            }
        }

        // Fallback: absolute coords. Validate they land on some connected
        // screen — otherwise we'd restore the window off the visible desktop.
        if (s.WindowX is double ax && s.WindowY is double ay)
        {
            var candidate = new PixelPoint((int)ax, (int)ay);
            var hit = screens.FirstOrDefault(sc => sc.Bounds.Contains(candidate));
            if (hit is not null) return ClampToScreen(candidate, hit);
        }

        return null;
    }

    /// <summary>
    /// Keep the window's top-left at least one full window-extent inside the
    /// screen's working area so the title bar stays grabbable. Cheap guard
    /// against minor drift (monitor scale changed, working area shrank).
    /// </summary>
    private static PixelPoint ClampToScreen(PixelPoint p, Avalonia.Platform.Screen screen)
    {
        var wa = screen.WorkingArea;
        var x = System.Math.Clamp(p.X, wa.X, wa.X + System.Math.Max(0, wa.Width - 100));
        var y = System.Math.Clamp(p.Y, wa.Y, wa.Y + System.Math.Max(0, wa.Height - 50));
        return new PixelPoint(x, y);
    }

    /// <summary>
    /// Snapshot the live window into the settings VM so the next save round-
    /// trips it. Captures both the screen identity (display name + offset
    /// within that screen) and the absolute coords so a future launch can
    /// match by monitor name first and fall back to absolute coords.
    /// Avalonia's <see cref="Window"/> doesn't expose pre-maximize bounds the
    /// way WPF does, so closing maximized leaves the prior Normal-state bounds
    /// alone — that's what a future un-maximize should restore to.
    /// </summary>
    private static void CapturePersistedWindowState(Window window, AppSettingsViewModel s)
    {
        s.WindowMaximized = window.WindowState == WindowState.Maximized;

        if (window.WindowState != WindowState.Normal) return;

        var pos = window.Position;
        s.WindowX = pos.X;
        s.WindowY = pos.Y;
        s.WindowWidth = window.Width;
        s.WindowHeight = window.Height;

        var screen = window.Screens?.ScreenFromWindow(window);
        if (screen is not null)
        {
            s.WindowScreenName = screen.DisplayName;
            s.WindowScreenRelativeX = pos.X - screen.Bounds.X;
            s.WindowScreenRelativeY = pos.Y - screen.Bounds.Y;
        }
        else
        {
            s.WindowScreenName = null;
            s.WindowScreenRelativeX = null;
            s.WindowScreenRelativeY = null;
        }
    }
}
