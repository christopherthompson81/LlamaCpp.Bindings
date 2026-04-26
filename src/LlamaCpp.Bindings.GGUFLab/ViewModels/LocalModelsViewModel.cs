using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.GGUFLab.Services;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Browses files already on disk under the configured workspace root.
/// Lists *.gguf and *.safetensors found anywhere below the root, with
/// size and last-modified, and a Use-as-active / Delete pair per row.
/// </summary>
public sealed partial class LocalModelsViewModel : ToolPageViewModel
{
    public override string Title => "Local Models";
    public override string Description =>
        "Browse models already on disk under the workspace root. Set one active, or delete to reclaim space.";

    private readonly WorkspaceSettings _settings;
    private readonly ActiveModel _activeModel;

    public ObservableCollection<LocalModelRow> Rows { get; } = new();

    [ObservableProperty]
    private string _statusLine = "Idle.";

    [ObservableProperty]
    private string _workspaceRoot = string.Empty;

    public string TotalDisplaySize
    {
        get
        {
            long sum = 0;
            foreach (var r in Rows) sum += r.SizeBytes;
            return FormatBytes(sum);
        }
    }

    public LocalModelsViewModel(WorkspaceSettings settings, ActiveModel activeModel)
    {
        _settings = settings;
        _activeModel = activeModel;
        WorkspaceRoot = settings.WorkspaceRoot;
        Refresh();
    }

    [RelayCommand]
    public void Refresh()
    {
        WorkspaceRoot = _settings.WorkspaceRoot;
        Rows.Clear();
        if (string.IsNullOrEmpty(WorkspaceRoot) || !Directory.Exists(WorkspaceRoot))
        {
            StatusLine = $"Workspace root not found: {WorkspaceRoot}. Configure it in Settings.";
            OnPropertyChanged(nameof(TotalDisplaySize));
            return;
        }

        try
        {
            var found = new List<FileInfo>();
            foreach (var pat in new[] { "*.gguf", "*.safetensors" })
            {
                foreach (var path in Directory.EnumerateFiles(WorkspaceRoot, pat, SearchOption.AllDirectories))
                {
                    try { found.Add(new FileInfo(path)); }
                    catch { /* skip files that disappeared mid-scan */ }
                }
            }

            // Stable order: most recently modified first — matches the
            // user's mental model after a fresh download.
            foreach (var fi in found.OrderByDescending(f => f.LastWriteTimeUtc))
            {
                Rows.Add(new LocalModelRow(
                    Path:        fi.FullName,
                    DisplayName: GetRelativeDisplay(fi.FullName, WorkspaceRoot),
                    SizeBytes:   fi.Length,
                    LastModified: fi.LastWriteTime));
            }
            StatusLine = Rows.Count == 0
                ? "No GGUF or safetensors files found under the workspace."
                : $"{Rows.Count} model{(Rows.Count == 1 ? "" : "s")}, total {TotalDisplaySize}.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Scan failed: {ex.Message}";
        }
        OnPropertyChanged(nameof(TotalDisplaySize));
    }

    public void SetActive(LocalModelRow row)
    {
        _activeModel.Set(row.Path);
        StatusLine = $"Active model → {row.DisplayName}";
    }

    public void Reveal(LocalModelRow row)
    {
        try
        {
            // Try to highlight the file in the OS file manager. Different
            // platforms need different invocations; fall back to opening
            // the directory if the highlight form isn't supported.
            if (OperatingSystem.IsWindows())
            {
                Process.Start("explorer.exe", $"/select,\"{row.Path}\"");
            }
            else if (OperatingSystem.IsMacOS())
            {
                Process.Start("open", new[] { "-R", row.Path });
            }
            else
            {
                var dir = Path.GetDirectoryName(row.Path) ?? row.Path;
                Process.Start(new ProcessStartInfo(dir) { UseShellExecute = true });
            }
        }
        catch (Exception ex)
        {
            StatusLine = $"Reveal failed: {ex.Message}";
        }
    }

    public void Delete(LocalModelRow row)
    {
        try
        {
            if (File.Exists(row.Path)) File.Delete(row.Path);
            Rows.Remove(row);
            // If the active model was the one we just removed, clear it.
            if (string.Equals(_activeModel.Path, row.Path, StringComparison.Ordinal))
                _activeModel.Clear();
            StatusLine = $"Deleted {row.DisplayName}.";
            OnPropertyChanged(nameof(TotalDisplaySize));
        }
        catch (Exception ex)
        {
            StatusLine = $"Delete failed: {ex.Message}";
        }
    }

    public override void ApplyActiveModel(string? path)
    {
        // No input field to fill — but a refresh on activation surfaces
        // anything that landed via the HF Browser since the last visit.
        Refresh();
    }

    private static string GetRelativeDisplay(string fullPath, string root)
    {
        try
        {
            var rel = Path.GetRelativePath(root, fullPath);
            return string.IsNullOrEmpty(rel) || rel == "." ? fullPath : rel;
        }
        catch { return fullPath; }
    }

    private static string FormatBytes(long bytes) => bytes switch
    {
        < 1024 => $"{bytes} B",
        < 1024L * 1024 => $"{bytes / 1024.0:F1} KB",
        < 1024L * 1024 * 1024 => $"{bytes / (1024.0 * 1024):F1} MB",
        _ => $"{bytes / (1024.0 * 1024 * 1024):F2} GB",
    };

    public sealed record LocalModelRow(
        string Path,
        string DisplayName,
        long SizeBytes,
        DateTime LastModified)
    {
        public string DisplaySize => SizeBytes switch
        {
            < 1024 => $"{SizeBytes} B",
            < 1024L * 1024 => $"{SizeBytes / 1024.0:F1} KB",
            < 1024L * 1024 * 1024 => $"{SizeBytes / (1024.0 * 1024):F1} MB",
            _ => $"{SizeBytes / (1024.0 * 1024 * 1024):F2} GB",
        };
        public string DisplayDate => LastModified.ToString("yyyy-MM-dd HH:mm");
        public bool IsGguf => Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
    }
}
