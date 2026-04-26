using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.GGUFLab.Services;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Browses models on disk under the configured workspace root. A "model"
/// is the on-disk folder produced by the HF replicate / convert flow,
/// which can hold several representations side-by-side (a sharded
/// safetensors checkpoint, an F16 GGUF, one or more quantized GGUFs).
/// One row is shown per folder; each available representation is exposed
/// as a clickable format chip that can be set as the active model so the
/// next tool prefills with the format it needs.
/// </summary>
public sealed partial class LocalModelsViewModel : ToolPageViewModel
{
    public override string Title => "Local Models";
    public override string Description =>
        "Browse models on disk under the workspace root. Each folder may carry several formats — pick the one to set active.";

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
            foreach (var r in Rows) sum += r.TotalBytes;
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
            // Collect every gguf / safetensors file once, then group by
            // their containing directory — that directory is the "model".
            // A directory with both formats is one row exposing both.
            var byDir = new Dictionary<string, List<FileInfo>>(StringComparer.Ordinal);
            foreach (var pat in new[] { "*.gguf", "*.safetensors" })
            {
                foreach (var path in Directory.EnumerateFiles(WorkspaceRoot, pat, SearchOption.AllDirectories))
                {
                    FileInfo fi;
                    try { fi = new FileInfo(path); }
                    catch { continue; }
                    var dir = fi.DirectoryName;
                    if (string.IsNullOrEmpty(dir)) continue;
                    if (!byDir.TryGetValue(dir, out var list))
                    {
                        list = new List<FileInfo>();
                        byDir[dir] = list;
                    }
                    list.Add(fi);
                }
            }

            var rows = new List<LocalModelRow>();
            foreach (var (dir, files) in byDir)
            {
                var formats = new List<LocalFormat>();
                long totalBytes = 0;
                DateTime lastModified = DateTime.MinValue;

                var safetensorsFiles = files
                    .Where(fi => fi.Name.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
                    .ToList();
                if (safetensorsFiles.Count > 0)
                {
                    // The safetensors representation needs config.json,
                    // tokenizer files, etc. — sum every non-gguf file in
                    // the directory tree so the badge reflects what
                    // actually has to be on disk for that format to work.
                    long stBytes = 0;
                    DateTime stTouched = DateTime.MinValue;
                    try
                    {
                        foreach (var path in Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories))
                        {
                            if (path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)) continue;
                            FileInfo fi;
                            try { fi = new FileInfo(path); }
                            catch { continue; }
                            stBytes += fi.Length;
                            if (fi.LastWriteTime > stTouched) stTouched = fi.LastWriteTime;
                        }
                    }
                    catch { /* directory disappeared mid-scan */ continue; }
                    formats.Add(new LocalFormat(
                        Label:     "safetensors",
                        Path:      dir,
                        SizeBytes: stBytes,
                        Kind:      LocalFormatKind.SafetensorsDir));
                    totalBytes += stBytes;
                    if (stTouched > lastModified) lastModified = stTouched;
                }

                foreach (var gg in files
                    .Where(fi => fi.Name.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                    .OrderBy(fi => fi.Name, StringComparer.OrdinalIgnoreCase))
                {
                    formats.Add(new LocalFormat(
                        Label:     InferQuantLabel(gg.Name),
                        Path:      gg.FullName,
                        SizeBytes: gg.Length,
                        Kind:      LocalFormatKind.GgufFile));
                    totalBytes += gg.Length;
                    if (gg.LastWriteTime > lastModified) lastModified = gg.LastWriteTime;
                }

                if (formats.Count == 0) continue;
                rows.Add(new LocalModelRow(
                    DisplayName:   GetRelativeDisplay(dir, WorkspaceRoot),
                    DirectoryPath: dir,
                    TotalBytes:    totalBytes,
                    LastModified:  lastModified == DateTime.MinValue ? DateTime.Now : lastModified,
                    Formats:       formats));
            }

            // Stable order: most recently modified first — matches the
            // user's mental model after a fresh download or convert.
            foreach (var row in rows.OrderByDescending(r => r.LastModified))
                Rows.Add(row);

            StatusLine = Rows.Count == 0
                ? "No models found under the workspace."
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
        // Activate the model FOLDER, not any single file inside it.
        // Each tool resolves the format it needs (e.g. Quantize picks the
        // F16 GGUF inside the folder) so the user can walk the whole
        // pipeline — convert, imatrix, quantize, adaptive — after a
        // single Set-active click.
        _activeModel.Set(row.DirectoryPath);
        StatusLine = $"Active model → {row.DisplayName}";
    }

    public void Reveal(LocalModelRow row)
    {
        try
        {
            // The unified row is always anchored to a directory; just
            // open it. (Previous per-file "highlight in file manager"
            // dance is unnecessary now that the row IS the directory.)
            Process.Start(new ProcessStartInfo(row.DirectoryPath) { UseShellExecute = true });
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
            if (Directory.Exists(row.DirectoryPath))
                Directory.Delete(row.DirectoryPath, recursive: true);
            Rows.Remove(row);
            // If the active model lived inside what we just removed
            // (any of its formats), clear the active slot.
            if (_activeModel.Path is { } active
                && (string.Equals(active, row.DirectoryPath, StringComparison.Ordinal)
                    || active.StartsWith(row.DirectoryPath + Path.DirectorySeparatorChar, StringComparison.Ordinal)))
            {
                _activeModel.Clear();
            }
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

    /// <summary>
    /// Pull a quant label out of a GGUF filename like
    /// <c>model.f16.gguf</c> or <c>Qwen3-0.6B.Q4_K_M.gguf</c>. Walks the
    /// dot-separated parts in reverse and returns the first that looks
    /// like a known llama.cpp ftype token; falls back to <c>"GGUF"</c>
    /// when nothing matches so the chip still has a label.
    /// </summary>
    private static readonly Regex _quantTokenRegex = new(
        @"^(?:F16|F32|BF16|Q\d(?:_[A-Z0-9]+)*|IQ\d(?:_[A-Z0-9]+)*)$",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static string InferQuantLabel(string fileName)
    {
        var stem = Path.GetFileNameWithoutExtension(fileName);
        var parts = stem.Split('.', StringSplitOptions.RemoveEmptyEntries);
        for (int i = parts.Length - 1; i >= 0; i--)
        {
            if (_quantTokenRegex.IsMatch(parts[i]))
                return parts[i].ToUpperInvariant();
        }
        return "GGUF";
    }

    public enum LocalFormatKind { SafetensorsDir, GgufFile }

    /// <summary>One representation of a model that the user can activate.</summary>
    public sealed record LocalFormat(
        string Label,
        string Path,
        long SizeBytes,
        LocalFormatKind Kind)
    {
        public string DisplaySize => SizeBytes switch
        {
            < 1024 => $"{SizeBytes} B",
            < 1024L * 1024 => $"{SizeBytes / 1024.0:F1} KB",
            < 1024L * 1024 * 1024 => $"{SizeBytes / (1024.0 * 1024):F1} MB",
            _ => $"{SizeBytes / (1024.0 * 1024 * 1024):F2} GB",
        };
    }

    /// <summary>
    /// One model folder. Carries the directory path (used for Reveal /
    /// Delete) plus the list of formats discovered inside.
    /// </summary>
    public sealed record LocalModelRow(
        string DisplayName,
        string DirectoryPath,
        long TotalBytes,
        DateTime LastModified,
        IReadOnlyList<LocalFormat> Formats)
    {
        public string DisplaySize => TotalBytes switch
        {
            < 1024 => $"{TotalBytes} B",
            < 1024L * 1024 => $"{TotalBytes / 1024.0:F1} KB",
            < 1024L * 1024 * 1024 => $"{TotalBytes / (1024.0 * 1024):F1} MB",
            _ => $"{TotalBytes / (1024.0 * 1024 * 1024):F2} GB",
        };
        public string DisplayDate => LastModified.ToString("yyyy-MM-dd HH:mm");
    }
}
