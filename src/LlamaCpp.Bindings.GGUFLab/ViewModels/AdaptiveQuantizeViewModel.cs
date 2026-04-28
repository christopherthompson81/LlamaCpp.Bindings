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
/// Drives the Adaptive Quantization page (v2 profile-based pipeline):
/// pick a target GGUF, resolve a sensitivity profile (auto-match by
/// arch+size, or browse), set a target bpw, preview the recipe with
/// noise-clamp + snap-to-stock guards applied, then quantize via
/// <see cref="LlamaCustomQuantizer"/> so per-tensor types are honored
/// verbatim (the legacy <c>tt_overrides</c> path silently dropped
/// demotions — Run 14).
/// </summary>
public sealed partial class AdaptiveQuantizeViewModel : ToolPageViewModel
{
    public override string Title => "Adaptive Quantization";
    public override string Description =>
        "Apply a per-architecture sensitivity profile to a target model: pick a profile, set target bpw, ship a custom-quantized GGUF.";

    private readonly NativeLogBus _logBus;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ProfileResolutionLine))]
    private string _inputPath = string.Empty;

    [ObservableProperty]
    private string _imatrixPath = string.Empty;

    [ObservableProperty]
    private string _outputPath = string.Empty;

    /// <summary>
    /// Explicit profile path. When empty, auto-match against the
    /// shipped reference profiles in <c>data/profiles/</c> by
    /// architecture + closest parameter count.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ProfileResolutionLine))]
    private string _profilePath = string.Empty;

    /// <summary>Target bpw — drives the recipe builder's budget cap.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(TargetBpwText))]
    private double _targetBitsPerElement = 4.95;

    public string TargetBpwText => $"{TargetBitsPerElement:F3} bpw";

    /// <summary>
    /// Snap-to-stock threshold (0 disables). Below this predicted PPL
    /// gain over stock-equivalent the recipe collapses to stock — see
    /// <see cref="LlamaQuantRecipeFromProfileOptions.MinPredictedGainPpl"/>.
    /// </summary>
    [ObservableProperty]
    private double _minPredictedGainPpl = 0.25;

    [ObservableProperty]
    private double _minPplGainPerBpw = 0.05;

    [ObservableProperty]
    private bool _applyStockBaseline = true;

    [ObservableProperty]
    private double _sizeScalingExponent = 1.0;

    /// <summary>
    /// Use per-tensor data from the profile (when present) to demote
    /// individual tensors below their category-level pick. See
    /// <see cref="LlamaQuantRecipeFromProfileOptions.UsePerTensorData"/>.
    /// </summary>
    [ObservableProperty]
    private bool _usePerTensorData = true;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    private bool _isRunning;

    public bool IsIdle => !IsRunning;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasProfile))]
    [NotifyPropertyChangedFor(nameof(ProfileSummaryLine))]
    [NotifyPropertyChangedFor(nameof(ProfileDetailLine))]
    private LlamaSensitivityProfile? _loadedProfile;

    public bool HasProfile => LoadedProfile is not null;

    /// <summary>
    /// One-line summary of the loaded profile shown in the form
    /// (architecture, source params, F16 PPL baseline). Updates whenever
    /// <see cref="LoadedProfile"/> changes.
    /// </summary>
    public string ProfileSummaryLine
    {
        get
        {
            if (LoadedProfile is not { } p) return string.Empty;
            var src = p.Provenance.SourceModel ?? "?";
            var sourceParams = p.Provenance.SourceParameterCount ?? 0;
            var sizeText = sourceParams > 0
                ? $" ({sourceParams / 1_000_000.0:F0}M params)"
                : "";
            return $"arch={p.ArchitectureId}  layers={p.LayerCount}  source={src}{sizeText}  F16 PPL={p.F16BaselinePerplexity:F3}";
        }
    }

    /// <summary>
    /// Second-line detail: completeness count + per-tensor presence.
    /// Tells the user at a glance whether the profile is fully measured
    /// or partial (a rail against shipping recipes built from gaps).
    /// </summary>
    public string ProfileDetailLine
    {
        get
        {
            if (LoadedProfile is not { } p) return string.Empty;
            var typesMeasured = p.Categories.Values
                .SelectMany(c => c.DeltaPplByType.Keys).Distinct().ToList();
            if (typesMeasured.Count == 0) return "no measurements in profile";
            var completeness = p.ComputeCompleteness(p.Categories.Keys.ToList(), typesMeasured);
            var line = $"{p.Categories.Count} categories × {typesMeasured.Count} types";
            if (!completeness.IsComplete)
                line += $"  ·  {completeness.MeasuredCategoryCells}/{completeness.TotalCategoryCells} cells measured (partial)";
            if (p.PerTensor is { } perTensor && perTensor.Count > 0)
                line += $"  ·  +{perTensor.Count} per-tensor entries";
            return line;
        }
    }

    /// <summary>
    /// Status line for "where is the profile coming from" — shows the
    /// auto-resolved match (or a hint when no profile fits) so the user
    /// always knows what's about to drive the recipe.
    /// </summary>
    [ObservableProperty]
    private string _profileResolutionLine = "Pick an input GGUF to auto-match a profile.";

    /// <summary>Recipe rebuilt whenever any input affecting the recipe changes.</summary>
    public ObservableCollection<RecipeRow> RecipeRows { get; } = new();

    /// <summary>
    /// Apply-time rails: warnings and blocks against avoidable bad
    /// decisions (cross-arch, cross-size, missing profile data, target
    /// bpw out of measured range). Recomputed on every input change.
    /// </summary>
    public ObservableCollection<RailMessage> Rails { get; } = new();

    /// <summary>True when any rail is at <see cref="RailSeverity.Block"/>; disables Quantize.</summary>
    public bool HasBlockingRail => Rails.Any(r => r.Severity == RailSeverity.Block);

    /// <summary>True when at least one rail is currently active — drives the rails panel visibility.</summary>
    public bool HasAnyRail => Rails.Count > 0;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasRecipe))]
    [NotifyPropertyChangedFor(nameof(CanQuantize))]
    private LlamaQuantRecipe? _builtRecipe;

    public bool HasRecipe => BuiltRecipe is not null && BuiltRecipe.Entries.Count > 0;

    /// <summary>Quantize is enabled when we have a recipe AND no rail is blocking.</summary>
    public bool CanQuantize => HasRecipe && !HasBlockingRail;

    [ObservableProperty]
    private string _recipeSummaryLine = string.Empty;

    [ObservableProperty]
    private string _statusLine = "Idle.";

    /// <summary>
    /// Throttled log surface — native llama.cpp emits hundreds of log
    /// lines per quantize, and the naive StringBuilder + PropertyChanged
    /// pattern starves the UI thread of input events. Same pattern as
    /// ImatrixViewModel and ProfileBuilderViewModel.
    /// </summary>
    private readonly ThrottledLogBuffer _log = new();

    public string LogText => _log.Text;

    [ObservableProperty]
    private double _quantizeProgressFraction;

    [ObservableProperty]
    private string _currentTensorLine = string.Empty;

    private CancellationTokenSource? _cts;

    public AdaptiveQuantizeViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
        _log.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(ThrottledLogBuffer.Text))
                OnPropertyChanged(nameof(LogText));
        };
    }

    /// <summary>
    /// Coalescing counter for background refreshes — every property
    /// change kicks a Task.Run; results from earlier (superseded)
    /// runs are dropped. Without this, picking an active model would
    /// freeze the page transition for ~500 ms (three GGUF opens +
    /// profile-directory scan + recipe enumeration on the UI thread).
    /// </summary>
    private int _refreshSeq;

    partial void OnInputPathChanged(string value)
    {
        NotifyRemediesChanged();
        _ = ScheduleRefreshAsync();
    }

    partial void OnImatrixPathChanged(string value) => NotifyRemediesChanged();

    partial void OnProfilePathChanged(string value)             => _ = ScheduleRefreshAsync();
    partial void OnTargetBitsPerElementChanged(double value)    => _ = ScheduleRefreshAsync();
    partial void OnMinPredictedGainPplChanged(double value)     => _ = ScheduleRefreshAsync();
    partial void OnMinPplGainPerBpwChanged(double value)        => _ = ScheduleRefreshAsync();
    partial void OnApplyStockBaselineChanged(bool value)        => _ = ScheduleRefreshAsync();
    partial void OnSizeScalingExponentChanged(double value)     => _ = ScheduleRefreshAsync();
    partial void OnUsePerTensorDataChanged(bool value)          => _ = ScheduleRefreshAsync();

    /// <summary>
    /// Snapshot inputs on the UI thread, run the heavy work
    /// (profile resolution + GGUF read + recipe enumeration + rails
    /// computation) on the thread pool, drop the result if a newer
    /// refresh has been kicked, otherwise apply on the UI thread.
    /// </summary>
    private async Task ScheduleRefreshAsync()
    {
        var seq = System.Threading.Interlocked.Increment(ref _refreshSeq);

        // Capture inputs — these may change while the refresh runs;
        // the sequence counter discards stale results.
        var inputPath   = InputPath;
        var profilePath = ProfilePath;
        var targetBpw   = TargetBitsPerElement;
        var opts = new LlamaQuantRecipeFromProfileOptions
        {
            MinPredictedGainPpl  = MinPredictedGainPpl,
            MinPplGainPerBpw     = MinPplGainPerBpw,
            ApplyStockBaseline   = ApplyStockBaseline,
            SizeScalingExponent  = SizeScalingExponent,
            UsePerTensorData     = UsePerTensorData,
        };

        try
        {
            var result = await Task.Run(() => ComputeRefresh(inputPath, profilePath, targetBpw, opts));
            if (_refreshSeq != seq) return;    // superseded by a newer change
            ApplyRefreshResult(result);
        }
        catch (Exception ex)
        {
            if (_refreshSeq != seq) return;
            RecipeSummaryLine = $"Refresh failed: {ex.Message}";
        }
    }

    /// <summary>
    /// Pure computation: produce the refreshed profile + recipe + rails
    /// from the captured inputs. Runs on a thread pool worker; must not
    /// touch UI state. Reads the source GGUF exactly once.
    /// </summary>
    private static RefreshResult ComputeRefresh(
        string inputPath,
        string profilePath,
        double targetBpw,
        LlamaQuantRecipeFromProfileOptions opts)
    {
        // ---- 1. Resolve profile (explicit > auto-match > none) ----
        LlamaSensitivityProfile? profile = null;
        string profileMsg;

        if (!string.IsNullOrWhiteSpace(profilePath) && File.Exists(profilePath))
        {
            try
            {
                profile = LlamaSensitivityProfile.LoadFromJson(profilePath);
                profileMsg = $"Using explicit profile: {Path.GetFileName(profilePath)}";
            }
            catch (Exception ex)
            {
                profileMsg = $"Failed to load {profilePath}: {ex.Message}";
            }
        }
        else if (!string.IsNullOrWhiteSpace(inputPath) && File.Exists(inputPath))
        {
            (profile, profileMsg) = AutoMatchProfile(inputPath);
        }
        else
        {
            profileMsg = "Pick an input GGUF to auto-match a profile.";
        }

        if (profile is null
            || string.IsNullOrWhiteSpace(inputPath)
            || !File.Exists(inputPath)
            || !(targetBpw > 0))
        {
            return new RefreshResult(profile, profileMsg, null, string.Empty,
                Array.Empty<RecipeRow>(), Array.Empty<RailMessage>());
        }

        // ---- 2. Open GGUF once for everything below ----
        LlamaGgufFile gguf;
        try
        {
            gguf = LlamaGgufFile.Open(inputPath);
        }
        catch (Exception ex)
        {
            return new RefreshResult(profile, profileMsg, null,
                $"Failed to open GGUF: {ex.Message}",
                Array.Empty<RecipeRow>(), Array.Empty<RailMessage>());
        }

        var targetArch = gguf.Metadata
            .FirstOrDefault(m => m.Key == "general.architecture")
            ?.Value.AsString();
        long paramCount = gguf.Tensors.Sum(t => t.Dimensions.Aggregate(1L, (a, b) => a * (long)b));
        var weightLayout = gguf.Tensors
            .Where(t => t.Dimensions.Length > 1 && t.Name.EndsWith(".weight", StringComparison.Ordinal))
            .Select(t => (Name: t.Name, Elements: t.Dimensions.Aggregate(1L, (a, b) => a * (long)b)))
            .ToList();

        // ---- 3. Build recipe (no second GGUF open) ----
        LlamaQuantRecipe? recipe = null;
        string recipeSummary = string.Empty;
        var rows = new List<RecipeRow>();
        try
        {
            recipe = LlamaQuantRecipeFromProfile.BuildFromTensorLayout(
                profile, weightLayout, paramCount, targetBpw, opts);
            foreach (var e in recipe.Entries)
                rows.Add(new RecipeRow(
                    e.TensorName, e.ChosenType.ToString(),
                    e.BitsPerElement, e.RelativeMse));

            var distinct = recipe.Entries.Select(e => e.ChosenType).Distinct().Count();
            var totalGain = recipe.Entries.Sum(e => e.RelativeMse);
            var snapFired = distinct <= 2;
            recipeSummary =
                $"avg {recipe.AverageBitsPerElement:F3} bpw  ·  {recipe.Entries.Count} tensors  ·  " +
                $"{distinct} distinct types  ·  predicted ΔPPL sum = {totalGain:F2}" +
                (snapFired ? "  ·  snap-to-stock fired" : "");
        }
        catch (Exception ex)
        {
            recipeSummary = $"Recipe build failed: {ex.Message}";
        }

        // ---- 4. Compute rails (uses already-read GGUF info) ----
        var rails = ComputeRails(profile, targetArch, paramCount, targetBpw);

        return new RefreshResult(profile, profileMsg, recipe, recipeSummary, rows, rails);
    }

    private static (LlamaSensitivityProfile? profile, string message) AutoMatchProfile(string inputPath)
    {
        try
        {
            var gguf = LlamaGgufFile.Open(inputPath);
            var arch = gguf.Metadata
                .FirstOrDefault(m => m.Key == "general.architecture")
                ?.Value.AsString();
            if (string.IsNullOrEmpty(arch))
                return (null, "Couldn't read general.architecture from the GGUF.");

            long targetParams = gguf.Tensors.Sum(t => t.Dimensions.Aggregate(1L, (a, b) => a * (long)b));
            var profilesDir = ResolveProfilesDirectory();
            if (profilesDir is null || !Directory.Exists(profilesDir))
                return (null, "No data/profiles/ directory found — pick a profile manually.");

            (string Path, LlamaSensitivityProfile Profile, double SizeRatio)? best = null;
            foreach (var jsonPath in Directory.EnumerateFiles(profilesDir, "*.profile.json"))
            {
                try
                {
                    var p = LlamaSensitivityProfile.LoadFromJson(jsonPath);
                    if (!string.Equals(p.ArchitectureId, arch, StringComparison.Ordinal)) continue;
                    var src = p.Provenance.SourceParameterCount ?? 0;
                    if (src <= 0 || targetParams <= 0) continue;
                    var ratio = Math.Abs(Math.Log((double)targetParams / src));
                    if (best is null || ratio < best.Value.SizeRatio)
                        best = (jsonPath, p, ratio);
                }
                catch { /* skip unparseable */ }
            }

            if (best is null)
                return (null, $"No profile matches arch={arch}. Build one (Profile Builder) or pick manually.");

            var srcParams = best.Value.Profile.Provenance.SourceParameterCount ?? 0;
            var sizeRatio = srcParams > 0 ? (double)targetParams / srcParams : 1.0;
            return (best.Value.Profile,
                $"Auto-matched: {Path.GetFileName(best.Value.Path)} (target/source size ratio = {sizeRatio:F2}×)");
        }
        catch (Exception ex)
        {
            return (null, $"Profile resolution failed: {ex.Message}");
        }
    }

    private static IReadOnlyList<RailMessage> ComputeRails(
        LlamaSensitivityProfile profile, string? targetArch, long paramCount, double targetBpw)
    {
        var rails = new List<RailMessage>();

        if (!string.IsNullOrEmpty(targetArch) &&
            !string.Equals(targetArch, profile.ArchitectureId, StringComparison.Ordinal))
        {
            rails.Add(new RailMessage(RailSeverity.Block,
                $"Architecture mismatch: profile is for {profile.ArchitectureId}, model is {targetArch}. " +
                $"Recipe would mostly miss the target's tensors. Pick a profile built for {targetArch}."));
        }

        long sourceParams = profile.Provenance.SourceParameterCount ?? 0;
        if (paramCount > 0 && sourceParams > 0)
        {
            var ratio = (double)paramCount / sourceParams;
            if (ratio > 5.0 || ratio < 0.2)
            {
                rails.Add(new RailMessage(RailSeverity.Warn,
                    $"Cross-size ratio is {ratio:F2}× (target/source). " +
                    "Linear coefficient scaling extrapolates poorly past ~5×; expect predicted ΔPPLs to be " +
                    "either far too optimistic (large→small) or pessimistic (small→large)."));
            }
        }

        var measuredTypes = profile.Categories.Values
            .SelectMany(c => c.DeltaPplByType.Keys)
            .Distinct()
            .OrderBy(LlamaQuantRecipe.GetBitsPerElement)
            .ToList();
        if (measuredTypes.Count > 0)
        {
            var minBpw = LlamaQuantRecipe.GetBitsPerElement(measuredTypes[0]);
            var maxBpw = LlamaQuantRecipe.GetBitsPerElement(measuredTypes[^1]);
            if (targetBpw < minBpw - 0.05)
            {
                rails.Add(new RailMessage(RailSeverity.Block,
                    $"Target {targetBpw:F2} bpw is below the profile's lowest measured rung " +
                    $"({measuredTypes[0]} = {minBpw:F2} bpw). Recipe builder can't extrapolate below; " +
                    "build a profile that covers lower types or raise the target."));
            }
            else if (targetBpw > maxBpw + 0.5)
            {
                rails.Add(new RailMessage(RailSeverity.Warn,
                    $"Target {targetBpw:F2} bpw is well above the profile's highest measured rung " +
                    $"({measuredTypes[^1]} = {maxBpw:F2} bpw). Above this, the recipe pins everything to the highest measured type — no further refinement possible."));
            }

            var completeness = profile.ComputeCompleteness(profile.Categories.Keys.ToList(), measuredTypes);
            if (!completeness.IsComplete)
            {
                var missing = completeness.MissingCategoryCells.Take(3).Select(c => $"{c.Category}@{c.Type}");
                var more = completeness.MissingCategoryCells.Count > 3
                    ? $" (+{completeness.MissingCategoryCells.Count - 3} more)"
                    : "";
                rails.Add(new RailMessage(RailSeverity.Warn,
                    $"Profile is partial: {completeness.MeasuredCategoryCells}/{completeness.TotalCategoryCells} cells measured. " +
                    $"Missing: {string.Join(", ", missing)}{more}. " +
                    "Recipe builder will still work but choices are limited to what's measured."));
            }
        }

        return rails;
    }

    private void ApplyRefreshResult(RefreshResult r)
    {
        LoadedProfile = r.Profile;
        ProfileResolutionLine = r.ProfileMessage;
        RecipeSummaryLine = r.RecipeSummaryLine;
        BuiltRecipe = r.Recipe;

        RecipeRows.Clear();
        foreach (var row in r.Rows) RecipeRows.Add(row);

        Rails.Clear();
        foreach (var rail in r.Rails) Rails.Add(rail);

        OnPropertyChanged(nameof(HasBlockingRail));
        OnPropertyChanged(nameof(HasAnyRail));
        OnPropertyChanged(nameof(CanQuantize));
    }

    private sealed record RefreshResult(
        LlamaSensitivityProfile? Profile,
        string ProfileMessage,
        LlamaQuantRecipe? Recipe,
        string RecipeSummaryLine,
        IReadOnlyList<RecipeRow> Rows,
        IReadOnlyList<RailMessage> Rails);

    /// <summary>
    /// Walk up from the running binary to find the repo's
    /// <c>data/profiles/</c> directory. Returns null if not found —
    /// matches the dev-mode layout; an installed app would override
    /// this with a known data path.
    /// </summary>
    private static string? ResolveProfilesDirectory()
    {
        var dir = AppContext.BaseDirectory;
        while (!string.IsNullOrEmpty(dir))
        {
            var candidate = Path.Combine(dir, "data", "profiles");
            if (Directory.Exists(candidate)) return candidate;
            var slnx = Path.Combine(dir, "LlamaCpp.Bindings.slnx");
            if (File.Exists(slnx))
            {
                // Found repo root but no data/profiles — return null so caller can warn.
                return null;
            }
            dir = Path.GetDirectoryName(dir);
        }
        return null;
    }

    [RelayCommand]
    private async Task QuantizeAsync()
    {
        if (IsRunning) return;
        if (BuiltRecipe is null || BuiltRecipe.Entries.Count == 0)
        {
            StatusLine = "No recipe — load a profile and pick an input GGUF first.";
            return;
        }
        if (string.IsNullOrWhiteSpace(InputPath) || !File.Exists(InputPath))
        {
            StatusLine = "Pick a valid input GGUF.";
            return;
        }
        if (string.IsNullOrWhiteSpace(OutputPath))
        {
            try
            {
                var dir = Path.GetDirectoryName(InputPath) ?? string.Empty;
                var stem = Path.GetFileNameWithoutExtension(InputPath);
                OutputPath = Path.Combine(dir, $"{stem}.profile-{TargetBitsPerElement:F2}.gguf");
            }
            catch
            {
                StatusLine = "Pick an output path.";
                return;
            }
        }

        _log.Clear();
        QuantizeProgressFraction = 0;
        CurrentTensorLine = string.Empty;
        _cts = new CancellationTokenSource();
        IsRunning = true;
        var startedAt = DateTime.Now;
        StatusLine = "Quantizing…";

        var unsubscribe = _logBus.Subscribe(line => _log.Append(line));

        try
        {
            var opts = new LlamaCustomQuantizerOptions
            {
                ImatrixPath = string.IsNullOrWhiteSpace(ImatrixPath) ? null : ImatrixPath,
            };
            var progress = new Progress<LlamaCustomQuantizerProgress>(p =>
            {
                if (p.TotalTensors > 0)
                    QuantizeProgressFraction = (double)p.CompletedTensors / p.TotalTensors;
                CurrentTensorLine = p.CurrentTensor is null
                    ? $"{p.CompletedTensors}/{p.TotalTensors}"
                    : $"{p.CompletedTensors}/{p.TotalTensors} — {p.CurrentTensor} → {p.AppliedType}";
            });
            // Wrap in Task.Run so the synchronous prefix (GGUF open,
            // tensor enumeration, recipe expansion) runs off-thread.
            // The internal awaits are already correctly async; this
            // covers the pre-first-await CPU work that'd otherwise
            // freeze the UI on a multi-GB source.
            var inPath  = InputPath;
            var outPath = OutputPath;
            var recipe  = BuiltRecipe;
            var optsLocal = opts;
            var ct = _cts.Token;
            await Task.Run(
                () => LlamaCustomQuantizer.QuantizeWithRecipeAsync(inPath, outPath, recipe, optsLocal, progress, ct),
                ct);

            var elapsed = DateTime.Now - startedAt;
            StatusLine = $"Wrote {OutputPath} in {elapsed.TotalSeconds:F1}s ({BuiltRecipe.Entries.Count} tensors).";
        }
        catch (OperationCanceledException)
        {
            StatusLine = "Cancelled.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Quantize failed: {ex.Message}";
            _log.Append($"[error] {ex}");
        }
        finally
        {
            unsubscribe();
            _log.Stop();
            IsRunning = false;
            _cts?.Dispose();
            _cts = null;
        }
    }

    [RelayCommand]
    private void Cancel()
    {
        _cts?.Cancel();
        StatusLine = IsRunning ? "Cancellation requested…" : StatusLine;
    }

    public void SaveRecipeJson(string path)
    {
        if (BuiltRecipe is null)
        {
            StatusLine = "No recipe to save.";
            return;
        }
        try
        {
            LlamaQuantRecipe.SaveToJson(BuiltRecipe, path);
            StatusLine = $"Saved recipe to {path}.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Save failed: {ex.Message}";
        }
    }

    public override void ApplyActiveModel(string? path)
    {
        if (string.IsNullOrEmpty(InputPath)
            && ResolveGgufFromActive(path) is { } resolved)
            InputPath = resolved;
        if (string.IsNullOrEmpty(ImatrixPath)
            && ResolveImatrixForGguf(InputPath) is { } imt)
            ImatrixPath = imt;
    }

    protected override bool HasGgufInputValue => !string.IsNullOrEmpty(InputPath);
    protected override bool HasImatrixSlot => true;
    protected override bool HasImatrixInputValue => !string.IsNullOrEmpty(ImatrixPath);
    protected override string? CurrentSourceGguf =>
        string.IsNullOrEmpty(InputPath) ? ResolveGgufFromActive(Active?.Path) : InputPath;

    /// <summary>
    /// Severity of an apply-time rail. <see cref="Block"/> disables the
    /// Quantize button outright; <see cref="Warn"/> only annotates.
    /// </summary>
    public enum RailSeverity { Warn, Block }

    /// <summary>One rail row in the warnings panel.</summary>
    public sealed record RailMessage(RailSeverity Severity, string Text)
    {
        public string Glyph => Severity == RailSeverity.Block ? "✗" : "⚠";
        public bool IsBlock => Severity == RailSeverity.Block;
    }

    public sealed record RecipeRow(
        string TensorName,
        string ChosenType,
        double BitsPerElement,
        double PredictedDeltaPpl)
    {
        public string BitsPerElementText => double.IsNaN(BitsPerElement) ? "—" : $"{BitsPerElement:F2}";
        public string PredictedDeltaPplText => double.IsNaN(PredictedDeltaPpl)
            ? "—"
            : (Math.Abs(PredictedDeltaPpl) < 1e-3 ? "0" : $"{PredictedDeltaPpl:F3}");
    }
}
