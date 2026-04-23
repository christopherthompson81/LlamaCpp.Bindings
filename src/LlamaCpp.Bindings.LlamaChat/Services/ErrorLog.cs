using System;
using System.IO;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Centralized writer for <c>last-error.log</c> under the app's config
/// directory. Replaces the inline log-writing code that used to live in
/// <c>MainWindowViewModel</c>. Safe to call from any thread — writes are
/// self-contained file I/O.
/// </summary>
public static class ErrorLog
{
    public static string LogPath { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "LlamaChat",
        "last-error.log");

    /// <summary>
    /// Write a single timestamped entry to <c>last-error.log</c>, replacing
    /// whatever was there. The file is a single-entry "last one wins" log
    /// by design — recent errors are what the user wants to copy into bug
    /// reports; older context lives in stderr / trace.
    /// </summary>
    public static void Write(Exception ex, string? context = null)
    {
        try
        {
            var dir = Path.GetDirectoryName(LogPath);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

            var sb = new System.Text.StringBuilder();
            sb.Append(DateTime.Now.ToString("o")).Append('\n');
            if (!string.IsNullOrEmpty(context)) sb.Append("context: ").Append(context).Append('\n');
            sb.Append(ex.GetType().FullName).Append(": ").Append(ex.Message).Append("\n\n");
            sb.Append(ex).Append('\n');

            File.WriteAllText(LogPath, sb.ToString());
        }
        catch
        {
            // Never let log-writing failure propagate; the original error is
            // more important.
        }
        System.Diagnostics.Debug.WriteLine($"[error] {(context is null ? "" : $"({context}) ")}{ex}");
    }
}
