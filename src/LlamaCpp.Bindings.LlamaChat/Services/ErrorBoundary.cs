using System;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Threading;
using LlamaCpp.Bindings.LlamaChat.Views;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Application-wide error boundary. Hooks the three exception funnels that
/// exist on a .NET + Avalonia desktop app:
/// <list type="bullet">
///   <item><see cref="Dispatcher.UnhandledException"/> — exceptions on the
///     UI thread (button click handlers, layout, rendering).</item>
///   <item><see cref="TaskScheduler.UnobservedTaskException"/> — async
///     tasks that faulted and were never awaited / observed.</item>
///   <item><see cref="AppDomain.UnhandledException"/> — last-chance
///     fatal errors on any thread that escape everything else.</item>
/// </list>
/// UI-thread exceptions and explicit fatal reports open an
/// <see cref="ErrorSplashDialog"/> on the main window. Unobserved task
/// exceptions are treated as non-fatal (surface via <see cref="ToastService"/>)
/// — the app is usually still healthy; the offending task just disappeared.
/// Every path funnels through <see cref="ErrorLog.Write"/> so the user
/// always has a copyable trace at <c>~/.config/LlamaChat/last-error.log</c>.
/// </summary>
public static class ErrorBoundary
{
    private static bool _installed;
    private static bool _splashOpen;

    public static void Install()
    {
        if (_installed) return;
        _installed = true;

        Dispatcher.UIThread.UnhandledException += OnDispatcherUnhandled;
        TaskScheduler.UnobservedTaskException += OnUnobservedTask;
        AppDomain.CurrentDomain.UnhandledException += OnAppDomainUnhandled;
    }

    /// <summary>
    /// Route a caught exception through the same boundary a fatal uncaught
    /// exception would take. VMs call this when they hit an error they
    /// don't know how to recover from but don't want to let crash the app.
    /// </summary>
    public static void ReportFatal(Exception ex, string? context = null)
    {
        ErrorLog.Write(ex, context);
        ShowSplash(ex, context);
    }

    /// <summary>
    /// Route a caught exception as non-fatal: log + toast, no splash, no
    /// interruption. Use for recoverable failures where the user still has
    /// a working app (failed file read, dropped MCP connection, etc.).
    /// </summary>
    public static void ReportNonFatal(Exception ex, string title, string? context = null)
    {
        ErrorLog.Write(ex, context);
        ToastService.Error(title, $"{ex.GetType().Name}: {ex.Message}");
    }

    private static void OnDispatcherUnhandled(object? sender, DispatcherUnhandledExceptionEventArgs e)
    {
        ErrorLog.Write(e.Exception, "dispatcher");
        // Mark handled so Avalonia doesn't terminate — the splash lets the
        // user read + copy the trace before deciding whether to close.
        e.Handled = true;
        ShowSplash(e.Exception, "UI thread");
    }

    private static void OnUnobservedTask(object? sender, UnobservedTaskExceptionEventArgs e)
    {
        e.SetObserved();

        // Avalonia on recent Ubuntu fire-and-forgets an appmenu-registrar
        // DBus call that frequently fails because the Unity menubar service
        // doesn't exist on GNOME/KDE. Harmless. Drop it silently rather than
        // spooking the user with a red toast every time they open a menu.
        if (IsKnownBenign(e.Exception)) return;

        ErrorLog.Write(e.Exception, "unobserved-task");
        // Surface on the UI thread so the toast actually appears.
        _ = Dispatcher.UIThread.InvokeAsync(() =>
        {
            ToastService.Error("Background task failed", $"{e.Exception.GetType().Name}: {e.Exception.Message}");
        });
    }

    private static bool IsKnownBenign(Exception ex)
    {
        for (Exception? cur = ex; cur is not null; cur = cur.InnerException)
        {
            var msg = cur.Message ?? string.Empty;
            if (msg.Contains("com.canonical.AppMenu.Registrar", StringComparison.Ordinal))
                return true;
        }
        if (ex is AggregateException agg)
        {
            foreach (var inner in agg.Flatten().InnerExceptions)
                if (IsKnownBenign(inner)) return true;
        }
        return false;
    }

    private static void OnAppDomainUnhandled(object? sender, UnhandledExceptionEventArgs e)
    {
        var ex = e.ExceptionObject as Exception ?? new Exception(e.ExceptionObject?.ToString() ?? "Unknown fatal error");
        ErrorLog.Write(ex, "appdomain-terminating");
        // AppDomain unhandled on a non-UI thread cannot be un-terminated.
        // Best we can do is ensure the log is on disk. If we're lucky and
        // this fires on the UI thread, still try to open the splash.
        if (Dispatcher.UIThread.CheckAccess()) ShowSplash(ex, "fatal (terminating)");
    }

    private static void ShowSplash(Exception ex, string? context)
    {
        if (Dispatcher.UIThread.CheckAccess())
        {
            ShowSplashOnUi(ex, context);
        }
        else
        {
            _ = Dispatcher.UIThread.InvokeAsync(() => ShowSplashOnUi(ex, context));
        }
    }

    private static void ShowSplashOnUi(Exception ex, string? context)
    {
        // Reentrancy guard — if the splash itself throws or the user
        // triggers a cascade of errors, show only the first.
        if (_splashOpen) return;
        _splashOpen = true;
        try
        {
            var owner = MainWindow();
            var dialog = new ErrorSplashDialog(ex, context);
            if (owner is not null) _ = dialog.ShowDialog(owner);
            else dialog.Show();
        }
        catch
        {
            // If even the splash fails, don't loop.
        }
        finally
        {
            _splashOpen = false;
        }
    }

    private static Window? MainWindow() =>
        (Application.Current?.ApplicationLifetime as IClassicDesktopStyleApplicationLifetime)?.MainWindow;
}
