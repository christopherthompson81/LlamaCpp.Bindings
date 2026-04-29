using System;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.GGUFLab.Services;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Base class for everything that occupies the main content area. The sidebar
/// holds one of these per tool; <see cref="MainWindowViewModel.SelectedTool"/>
/// drives which one the window shows. Subclasses provide their own view via
/// the data-template hookup in <c>MainWindow.axaml</c>.
/// </summary>
public abstract partial class ToolPageViewModel : ObservableObject
{
    /// <summary>Sidebar label.</summary>
    public abstract string Title { get; }

    /// <summary>One-line subtitle shown under the page header.</summary>
    public abstract string Description { get; }

    /// <summary>
    /// Optional sidebar group label shown above this item to start a new
    /// section. Default null = no header. Used to cluster the evaluation
    /// tools (KL Divergence, HellaSwag) under a "Benchmarks" header
    /// without introducing a hierarchical navigation model.
    /// </summary>
    public virtual string? CategoryLabel => null;

    /// <summary>
    /// Set by the shell on construction so the base class can compute
    /// remedy state (e.g. "active model is a safetensors directory but
    /// this tool needs a GGUF — offer a one-click jump to HF→GGUF").
    /// Null in unit tests / standalone construction.
    /// </summary>
    protected ActiveModel? Active { get; private set; }

    /// <summary>Set by the shell; null when running outside the shell.</summary>
    protected ToolNavigator? Navigator { get; private set; }

    /// <summary>
    /// Wire the tool to the shell-owned services. Called once by
    /// <see cref="MainWindowViewModel"/> just after construction.
    /// </summary>
    public void AttachShell(ActiveModel active, ToolNavigator navigator)
    {
        Active = active;
        Navigator = navigator;
        active.PropertyChanged += (_, e) =>
        {
            // Anything derived from the active path must refresh when
            // it changes; right now that's the remedy visibility.
            if (e.PropertyName == nameof(Services.ActiveModel.Path))
                NotifyRemediesChanged();
        };
    }

    /// <summary>
    /// Called by the main window when the active model changes, or when
    /// this tool becomes the selected page. Default is a no-op; tools
    /// that take a GGUF path override and copy <paramref name="path"/>
    /// into their input field — but only when the field is currently
    /// empty so a typed-in path is never silently overwritten.
    /// </summary>
    public virtual void ApplyActiveModel(string? path) { }

    /// <summary>
    /// True when <paramref name="path"/> looks like a usable GGUF file path
    /// (non-empty, ends in <c>.gguf</c>). Tools whose input slot is a GGUF
    /// file use this to skip auto-fill when the active model is a directory
    /// (a replicated safetensors model) or otherwise non-GGUF.
    /// </summary>
    protected static bool IsGgufPath([NotNullWhen(true)] string? path) =>
        !string.IsNullOrEmpty(path) && path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Resolve <paramref name="path"/> to a usable GGUF file. If
    /// <paramref name="path"/> is already a <c>.gguf</c> file, returns it
    /// unchanged. If it's a directory, scans for <c>.gguf</c> files inside
    /// (top level only) and returns the most useful one — F16 / BF16 /
    /// F32 first since those are the typical conversion outputs that feed
    /// the rest of the pipeline (imatrix, quantize, sensitivity sweep).
    /// Returns null when nothing usable is found, which is the signal
    /// for the safetensors-conversion remedy to surface.
    /// </summary>
    protected static string? ResolveGgufFromActive(string? path)
    {
        if (string.IsNullOrEmpty(path)) return null;
        if (IsGgufPath(path)) return path;
        if (!Directory.Exists(path)) return null;

        string? best = null;
        int bestScore = int.MaxValue;
        foreach (var candidate in Directory.EnumerateFiles(path, "*.gguf", SearchOption.TopDirectoryOnly))
        {
            int score = ScoreGgufPreference(candidate);
            if (score < bestScore)
            {
                bestScore = score;
                best = candidate;
            }
        }
        return best;
    }

    private static int ScoreGgufPreference(string ggufPath)
    {
        // Lower is better. Prefer the lossless / conversion-output ftypes
        // since those are what subsequent tools usually want as the source
        // (you imatrix and quantize FROM an F16, not FROM a Q4_K_M).
        // Anything carrying "imatrix" in the name is excluded outright —
        // those are auxiliary outputs, not source weights.
        var name = Path.GetFileNameWithoutExtension(ggufPath);
        if (Contains(name, "imatrix")) return int.MaxValue;
        if (Contains(name, "f16"))     return 0;
        if (Contains(name, "bf16"))    return 1;
        if (Contains(name, "f32"))     return 2;
        return 3;

        static bool Contains(string haystack, string needle) =>
            haystack.Contains(needle, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Find the imatrix sidecar produced for <paramref name="ggufPath"/>
    /// by our Imatrix tool. Convention is <c>&lt;stem&gt;.imatrix.gguf</c>
    /// next to the source model — see <c>ImatrixViewModel.RunAsync</c>'s
    /// default OutputPath. Returns null when no sidecar exists yet (which
    /// is exactly the signal the build-imatrix remedy uses to surface).
    /// </summary>
    protected static string? ResolveImatrixForGguf(string? ggufPath)
    {
        if (!IsGgufPath(ggufPath)) return null;
        var dir = Path.GetDirectoryName(ggufPath);
        if (string.IsNullOrEmpty(dir)) return null;
        var stem = Path.GetFileNameWithoutExtension(ggufPath);
        var candidate = Path.Combine(dir, $"{stem}.imatrix.gguf");
        return File.Exists(candidate) ? candidate : null;
    }

    /// <summary>
    /// Override in tools whose primary input is a GGUF file path; return
    /// true when the user has already supplied one. The base uses this
    /// to decide whether the safetensors-conversion remedy is relevant.
    /// </summary>
    protected virtual bool HasGgufInputValue => false;

    /// <summary>
    /// Override in tools that take an imatrix file as a secondary input
    /// (e.g. Adaptive Quantization) to opt into the build-imatrix remedy.
    /// </summary>
    protected virtual bool HasImatrixSlot => false;

    /// <summary>True when the imatrix slot is filled. Override paired with <see cref="HasImatrixSlot"/>.</summary>
    protected virtual bool HasImatrixInputValue => false;

    /// <summary>
    /// The GGUF the build-imatrix remedy should hand to the Imatrix tool.
    /// Tools override to return their own input path when set, otherwise
    /// what the active model resolves to. Returning null suppresses the
    /// remedy (we don't know what to imatrix).
    /// </summary>
    protected virtual string? CurrentSourceGguf => null;

    /// <summary>
    /// True when the tool's GGUF input is empty AND the active model is
    /// a directory that has safetensors files but no GGUF the tool could
    /// auto-resolve. Conversion is the meaningful next step under exactly
    /// those conditions; any other state (GGUF available, no safetensors
    /// at all, etc.) makes the remedy noise.
    /// </summary>
    public bool ShowConvertFromSafetensorsRemedy
    {
        get
        {
            if (HasGgufInputValue) return false;
            if (Active?.Path is not { } p) return false;
            if (!Directory.Exists(p)) return false;
            if (ResolveGgufFromActive(p) is not null) return false;
            // Directory exists and carries no GGUF. The remedy is only
            // useful if there are safetensors to convert FROM.
            try
            {
                return Directory.EnumerateFiles(p, "*.safetensors", SearchOption.TopDirectoryOnly).Any();
            }
            catch
            {
                return false;
            }
        }
    }

    /// <summary>
    /// True when the tool has an imatrix slot, the slot is empty, we
    /// know which GGUF to imatrix, and no sidecar exists yet for it.
    /// Surfaces the build-imatrix remedy card in supporting tools.
    /// </summary>
    public bool ShowBuildImatrixRemedy
    {
        get
        {
            if (!HasImatrixSlot) return false;
            if (HasImatrixInputValue) return false;
            if (CurrentSourceGguf is not { } src) return false;
            return ResolveImatrixForGguf(src) is null;
        }
    }

    /// <summary>
    /// Subclasses call this from their own input-field change handler so
    /// the remedy visibility refreshes when the user types or clears the
    /// input. Defined on the base so subclasses don't reach for the
    /// magic property name themselves.
    /// </summary>
    protected void NotifyRemediesChanged()
    {
        OnPropertyChanged(nameof(ShowConvertFromSafetensorsRemedy));
        OnPropertyChanged(nameof(ShowBuildImatrixRemedy));
    }

    /// <summary>
    /// Switch to the HF→GGUF tool with the active safetensors directory
    /// preloaded. Bound to the briefcase-medical remedy button.
    /// </summary>
    [RelayCommand]
    private void RemedyConvertFromSafetensors()
    {
        if (Navigator is null) return;
        if (Active?.Path is not { } dir || !Directory.Exists(dir)) return;
        Navigator.NavigateTo<HfConvertViewModel>(c =>
        {
            // Force-fill the destination — the user explicitly clicked
            // the remedy, so overwriting any prior value is the intent.
            c.HfDirectory = dir;
        });
    }

    /// <summary>
    /// Switch to the Imatrix tool with this tool's source GGUF preloaded.
    /// Returning to this tool after imatrix completes auto-fills the
    /// imatrix slot (the back-fill happens in
    /// <see cref="ApplyActiveModel"/>; see AdaptiveQuantizeViewModel).
    /// </summary>
    [RelayCommand]
    private void RemedyBuildImatrix()
    {
        if (Navigator is null) return;
        if (CurrentSourceGguf is not { } src) return;
        Navigator.NavigateTo<ImatrixViewModel>(c => c.ModelPath = src);
    }
}
