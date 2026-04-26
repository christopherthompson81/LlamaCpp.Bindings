using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using Avalonia;
using Avalonia.Controls;

namespace LlamaCpp.Bindings.GGUFSuite.Services;

/// <summary>
/// Persisted window placement: size, position, maximize state, plus a
/// monitor identity so multi-display setups restore to the same
/// physical screen even when the desktop arrangement changed between
/// sessions.
/// </summary>
public sealed record WindowPlacement
{
    public double? Width  { get; init; }
    public double? Height { get; init; }

    /// <summary>Absolute desktop coordinates. Used as a fallback when the named screen has gone away.</summary>
    public double? X { get; init; }
    public double? Y { get; init; }

    public bool Maximized { get; init; }

    /// <summary>Display name of the screen the window was on. Preferred match path on restore.</summary>
    public string? ScreenName { get; init; }
    public double? ScreenRelativeX { get; init; }
    public double? ScreenRelativeY { get; init; }
}

/// <summary>
/// On-disk persistence + apply/capture helpers for the GGUFSuite
/// main window. Mirrors the pattern in LlamaChat's
/// <c>AppSettingsStore</c> + <c>App.ApplyPersistedWindowState</c> /
/// <c>CapturePersistedWindowState</c>, factored into one
/// self-contained class because GGUFSuite has no other persistent
/// settings yet.
/// </summary>
public static class WindowPlacementStore
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter() },
    };

    /// <summary>Where the placement file lives. <c>%AppData%/GGUFSuite/window-placement.json</c> on Windows; <c>~/.config/GGUFSuite/...</c> on Linux.</summary>
    public static string StorePath
    {
        get
        {
            var dir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "GGUFSuite");
            return Path.Combine(dir, "window-placement.json");
        }
    }

    public static WindowPlacement Load()
    {
        try
        {
            if (!File.Exists(StorePath)) return new WindowPlacement();
            var json = File.ReadAllText(StorePath);
            return JsonSerializer.Deserialize<WindowPlacement>(json, JsonOpts) ?? new WindowPlacement();
        }
        catch
        {
            // A corrupt placement file shouldn't prevent the app from
            // starting — fall back to defaults silently. The next clean
            // close overwrites the file.
            return new WindowPlacement();
        }
    }

    public static void Save(WindowPlacement placement)
    {
        try
        {
            var dir = Path.GetDirectoryName(StorePath)!;
            Directory.CreateDirectory(dir);
            File.WriteAllText(StorePath, JsonSerializer.Serialize(placement, JsonOpts));
        }
        catch
        {
            // Best-effort. We don't want a failed write (read-only fs,
            // permission error, etc.) to break shutdown.
        }
    }

    /// <summary>
    /// Apply <paramref name="placement"/> to <paramref name="window"/>
    /// before it shows. Multi-monitor placement: prefer the screen
    /// matched by name so the window lands on the same physical
    /// monitor even if the desktop arrangement changed; fall back to
    /// absolute coords (legacy / unknown monitor) and finally to
    /// default placement when neither resolves to a connected screen.
    /// </summary>
    public static void Apply(Window window, WindowPlacement placement)
    {
        if (placement.Width is double w && placement.Height is double h && w > 100 && h > 100)
        {
            window.Width = w;
            window.Height = h;
        }

        var pos = ResolveStartupPosition(window, placement);
        if (pos is PixelPoint p)
        {
            window.WindowStartupLocation = WindowStartupLocation.Manual;
            window.Position = p;
        }

        if (placement.Maximized)
        {
            window.WindowState = WindowState.Maximized;
        }
    }

    /// <summary>
    /// Snapshot the live <paramref name="window"/> into a
    /// <see cref="WindowPlacement"/>. Captures both screen identity
    /// (display name + offset within that screen) and absolute coords
    /// so a future launch can match by monitor name first and fall
    /// back to absolute coords. Captures only when the window is in
    /// Normal state — Avalonia doesn't expose pre-maximize bounds the
    /// way WPF does, so closing maximized leaves the prior Normal-
    /// state bounds alone (which is what a future un-maximize should
    /// restore to).
    /// </summary>
    public static WindowPlacement Capture(Window window)
    {
        bool maximized = window.WindowState == WindowState.Maximized;
        if (window.WindowState != WindowState.Normal)
        {
            // Preserve the prior Normal bounds by re-loading from disk
            // and keeping everything except Maximized.
            var prior = Load();
            return prior with { Maximized = maximized };
        }

        var pos = window.Position;
        var snapshot = new WindowPlacement
        {
            Width     = window.Width,
            Height    = window.Height,
            X         = pos.X,
            Y         = pos.Y,
            Maximized = maximized,
        };

        var screen = window.Screens?.ScreenFromWindow(window);
        if (screen is not null)
        {
            snapshot = snapshot with
            {
                ScreenName       = screen.DisplayName,
                ScreenRelativeX  = pos.X - screen.Bounds.X,
                ScreenRelativeY  = pos.Y - screen.Bounds.Y,
            };
        }

        return snapshot;
    }

    private static PixelPoint? ResolveStartupPosition(Window window, WindowPlacement placement)
    {
        var screens = window.Screens?.All;
        if (screens is null || screens.Count == 0)
        {
            // Platform doesn't expose screens (rare). Fall back to absolute
            // coords if we have them; the OS will route them somewhere.
            return placement.X is double x && placement.Y is double y
                ? new PixelPoint((int)x, (int)y)
                : null;
        }

        // Preferred path: same monitor by display name + offset within it.
        if (placement.ScreenName is { Length: > 0 } name &&
            placement.ScreenRelativeX is double relX &&
            placement.ScreenRelativeY is double relY)
        {
            var match = screens.FirstOrDefault(sc => sc.DisplayName == name);
            if (match is not null)
            {
                return ClampToScreen(
                    new PixelPoint(match.Bounds.X + (int)relX, match.Bounds.Y + (int)relY),
                    match);
            }
        }

        // Fallback: absolute coords. Validate they land on some
        // connected screen — otherwise we'd restore the window off the
        // visible desktop.
        if (placement.X is double ax && placement.Y is double ay)
        {
            var candidate = new PixelPoint((int)ax, (int)ay);
            var hit = screens.FirstOrDefault(sc => sc.Bounds.Contains(candidate));
            if (hit is not null) return ClampToScreen(candidate, hit);
        }

        return null;
    }

    /// <summary>
    /// Keep the window's top-left at least one full window-extent
    /// inside the screen's working area so the title bar stays
    /// grabbable. Cheap guard against minor drift (monitor scale
    /// changed, working area shrank).
    /// </summary>
    private static PixelPoint ClampToScreen(PixelPoint p, Avalonia.Platform.Screen screen)
    {
        var wa = screen.WorkingArea;
        var x = Math.Clamp(p.X, wa.X, wa.X + Math.Max(0, wa.Width - 100));
        var y = Math.Clamp(p.Y, wa.Y, wa.Y + Math.Max(0, wa.Height - 50));
        return new PixelPoint(x, y);
    }
}
