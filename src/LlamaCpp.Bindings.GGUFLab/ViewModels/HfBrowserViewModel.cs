using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.GGUFLab.Services;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Drives the HF Browser page: search HuggingFace Hub by query +
/// library + sort; show repo cards; click a repo to load its file
/// list; download a chosen file into <see cref="WorkspaceSettings.WorkspaceRoot"/>
/// and (optionally) set it as the <see cref="ActiveModel"/>.
/// </summary>
public sealed partial class HfBrowserViewModel : ToolPageViewModel
{
    public override string Title => "Huggingface Browser";
    public override string Description =>
        "Search HuggingFace Hub for GGUF and safetensors repos, then download into the workspace.";

    private readonly WorkspaceSettings _settings;
    private readonly ActiveModel _activeModel;
    private readonly HfApi _api;

    [ObservableProperty]
    private string _query = string.Empty;

    [ObservableProperty]
    private HfLibraryFilter _library = HfLibraryFilter.Gguf;

    [ObservableProperty]
    private HfSortKey _sortKey = HfSortKey.Trending;

    public IReadOnlyList<HfLibraryFilter> AvailableLibraries { get; } =
        new[] { HfLibraryFilter.Gguf, HfLibraryFilter.Safetensors, HfLibraryFilter.Any };

    public IReadOnlyList<HfSortKey> AvailableSorts { get; } =
        new[] { HfSortKey.Trending, HfSortKey.Downloads, HfSortKey.RecentlyUpdated, HfSortKey.Likes };

    public ObservableCollection<HfModelSummary> Results { get; } = new();
    public ObservableCollection<HfModelFile> SelectedRepoFiles { get; } = new();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasSelectedRepo))]
    [NotifyPropertyChangedFor(nameof(SelectedRepoUrl))]
    [NotifyPropertyChangedFor(nameof(IsSelectedRepoSafetensors))]
    private HfModelSummary? _selectedRepo;

    public bool HasSelectedRepo => SelectedRepo is not null;
    public string SelectedRepoUrl => SelectedRepo is null ? "" : $"https://huggingface.co/{SelectedRepo.Id}";

    /// <summary>
    /// True when the selected repo carries the <c>safetensors</c> tag.
    /// Drives visibility of the "Download All Files" button — replicating
    /// a whole repo is the standard workflow for safetensors models, where
    /// the weights are sharded and need accompanying config/tokenizer files
    /// to be usable.
    /// </summary>
    public bool IsSelectedRepoSafetensors =>
        SelectedRepo?.Tags is { } tags && tags.Contains("safetensors");

    [ObservableProperty]
    private string _statusLine = "Idle. Type a query (or leave blank) and press Search.";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    private bool _isBusy;

    public bool IsIdle => !IsBusy;

    [ObservableProperty]
    private double _downloadProgressFraction;

    [ObservableProperty]
    private string _downloadProgressText = string.Empty;

    private CancellationTokenSource? _cts;

    public HfBrowserViewModel(WorkspaceSettings settings, ActiveModel activeModel)
    {
        _settings = settings;
        _activeModel = activeModel;
        _api = new HfApi(() => _settings.HuggingFaceToken);
    }

    [RelayCommand]
    private async Task SearchAsync()
    {
        if (IsBusy) return;
        _cts = new CancellationTokenSource();
        IsBusy = true;
        StatusLine = "Searching HuggingFace Hub…";
        Results.Clear();
        SelectedRepo = null;
        SelectedRepoFiles.Clear();
        try
        {
            var results = await _api.SearchAsync(Query, Library, SortKey, limit: 40, _cts.Token);
            foreach (var r in results) Results.Add(r);
            StatusLine = results.Count == 0
                ? "No results."
                : $"Found {results.Count} repo{(results.Count == 1 ? "" : "s")}.";
        }
        catch (OperationCanceledException) { StatusLine = "Cancelled."; }
        catch (Exception ex) { StatusLine = $"Search failed: {ex.Message}"; }
        finally
        {
            IsBusy = false;
            _cts?.Dispose();
            _cts = null;
        }
    }

    [RelayCommand]
    private void Cancel()
    {
        _cts?.Cancel();
        StatusLine = "Cancellation requested…";
    }

    /// <summary>Called from the view code-behind when the user picks a repo card.</summary>
    public async Task SelectRepoAsync(HfModelSummary? repo)
    {
        SelectedRepo = repo;
        SelectedRepoFiles.Clear();
        if (repo is null) return;
        if (IsBusy) return;
        _cts = new CancellationTokenSource();
        IsBusy = true;
        StatusLine = $"Loading file list for {repo.Id}…";
        try
        {
            var files = await _api.ListFilesAsync(repo.Id, ct: _cts.Token);
            foreach (var f in files)
            {
                if (f.IsDirectory) continue;  // browser shows files only
                SelectedRepoFiles.Add(f);
            }
            StatusLine = $"{repo.Id}: {SelectedRepoFiles.Count} file{(SelectedRepoFiles.Count == 1 ? "" : "s")}.";
        }
        catch (OperationCanceledException) { StatusLine = "Cancelled."; }
        catch (Exception ex) { StatusLine = $"List failed: {ex.Message}"; }
        finally
        {
            IsBusy = false;
            _cts?.Dispose();
            _cts = null;
        }
    }

    /// <summary>Called from the view's per-row Download button.</summary>
    public async Task DownloadAsync(HfModelFile file)
    {
        if (SelectedRepo is null || IsBusy) return;
        var repo = SelectedRepo;
        // Lay files out under <workspace>/<author>/<repoName>/<path>. Mirrors
        // the HF tree shape so a later "where did this come from?" lookup
        // is just reading the directory.
        var root = _settings.EnsureWorkspaceRoot();
        var destPath = Path.Combine(root, SafeSegments(repo.Id), SafePath(file.Path));

        if (File.Exists(destPath))
        {
            StatusLine = $"Already present: {destPath}. Setting as active model.";
            _activeModel.Set(destPath);
            return;
        }

        _cts = new CancellationTokenSource();
        IsBusy = true;
        DownloadProgressFraction = 0;
        DownloadProgressText = "0 / ?";
        StatusLine = $"Downloading {file.Path}…";

        var progress = new Progress<(long downloaded, long? total)>(p =>
        {
            if (p.total is long t && t > 0)
            {
                DownloadProgressFraction = (double)p.downloaded / t;
                DownloadProgressText = $"{Mb(p.downloaded)} / {Mb(t)} MB ({DownloadProgressFraction * 100:F1}%)";
            }
            else
            {
                DownloadProgressFraction = 0;
                DownloadProgressText = $"{Mb(p.downloaded)} MB";
            }
        });
        try
        {
            await _api.DownloadAsync(repo.Id, file.Path, destPath, progress, _cts.Token);
            StatusLine = $"Downloaded {file.Path} → {destPath}";
            // Auto-promote .gguf files to active model — that's the
            // common case and it spares the user one click.
            if (destPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                _activeModel.Set(destPath);
        }
        catch (OperationCanceledException) { StatusLine = "Cancelled."; }
        catch (Exception ex) { StatusLine = $"Download failed: {ex.Message}"; }
        finally
        {
            IsBusy = false;
            _cts?.Dispose();
            _cts = null;
        }
    }

    /// <summary>
    /// Download every non-directory file in the currently selected repo
    /// into <c>&lt;workspace&gt;/&lt;author&gt;/&lt;repoName&gt;/</c>, mirroring the HF tree.
    /// Files already present at the expected size are skipped so that a
    /// re-run cheaply resumes a partial replicate.
    /// </summary>
    [RelayCommand]
    private async Task ReplicateRepoAsync()
    {
        if (SelectedRepo is null || IsBusy) return;
        if (SelectedRepoFiles.Count == 0) return;
        var repo = SelectedRepo;
        var root = _settings.EnsureWorkspaceRoot();
        var repoRoot = Path.Combine(root, SafeSegments(repo.Id));

        // Snapshot the file list — the observable collection lives on the
        // UI thread and could be mutated if the user clicks another repo.
        var files = new List<HfModelFile>(SelectedRepoFiles);

        _cts = new CancellationTokenSource();
        IsBusy = true;
        DownloadProgressFraction = 0;
        DownloadProgressText = string.Empty;

        int total = files.Count;
        int done = 0, skipped = 0, downloaded = 0;
        StatusLine = $"Replicating {repo.Id}: 0 of {total}…";

        try
        {
            foreach (var file in files)
            {
                _cts.Token.ThrowIfCancellationRequested();
                done++;
                var destPath = Path.Combine(repoRoot, SafePath(file.Path));

                if (File.Exists(destPath)
                    && file.Size is long expected
                    && new FileInfo(destPath).Length == expected)
                {
                    skipped++;
                    StatusLine = $"Replicating {repo.Id}: {done}/{total} — skipped {file.Path}";
                    continue;
                }

                DownloadProgressFraction = 0;
                DownloadProgressText = "0 / ?";
                StatusLine = $"Replicating {repo.Id}: {done}/{total} — {file.Path}";

                var progress = new Progress<(long downloaded, long? total)>(p =>
                {
                    if (p.total is long t && t > 0)
                    {
                        DownloadProgressFraction = (double)p.downloaded / t;
                        DownloadProgressText = $"{Mb(p.downloaded)} / {Mb(t)} MB ({DownloadProgressFraction * 100:F1}%)";
                    }
                    else
                    {
                        DownloadProgressFraction = 0;
                        DownloadProgressText = $"{Mb(p.downloaded)} MB";
                    }
                });

                await _api.DownloadAsync(repo.Id, file.Path, destPath, progress, _cts.Token);
                downloaded++;
            }
            StatusLine = $"Replicated {repo.Id}: {downloaded} downloaded, {skipped} already present → {repoRoot}";
        }
        catch (OperationCanceledException)
        {
            StatusLine = $"Cancelled. {downloaded} downloaded, {skipped} skipped before stop.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Replicate failed at {done}/{total}: {ex.Message}";
        }
        finally
        {
            IsBusy = false;
            _cts?.Dispose();
            _cts = null;
        }
    }

    private static string SafeSegments(string repoId) =>
        repoId.Replace('\\', '/').Replace("..", "_");

    private static string SafePath(string path) =>
        path.Replace('\\', '/').Replace("..", "_");

    private static string Mb(long bytes) => (bytes / (1024.0 * 1024)).ToString("F1");
}
