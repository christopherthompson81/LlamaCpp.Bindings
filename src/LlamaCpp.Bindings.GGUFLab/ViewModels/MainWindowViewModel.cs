using System.Collections.Generic;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using LlamaCpp.Bindings.GGUFLab.Services;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Owns the navigation list and the currently selected page. Each entry
/// is a real implementation; <see cref="RoadmapViewModel"/> remains
/// available for future tools that need a deferred-stub placeholder.
/// </summary>
public sealed partial class MainWindowViewModel : ObservableObject
{
    public IReadOnlyList<ToolPageViewModel> Tools { get; }

    [ObservableProperty]
    private ToolPageViewModel _selectedTool;

    public NativeLogBus LogBus { get; } = new();
    public WorkspaceSettings Settings { get; }
    public ActiveModel ActiveModel { get; } = new();
    public ToolNavigator Navigator { get; } = new();

    public MainWindowViewModel()
    {
        // One-time backend init wired up to the log bus so every page can
        // observe native log output without re-registering.
        LlamaBackend.Initialize((lvl, msg) => LogBus.Publish(lvl, msg));

        Settings = WorkspaceSettings.Load();

        var hfBrowser = new HfBrowserViewModel(Settings, ActiveModel);
        var localModels = new LocalModelsViewModel(Settings, ActiveModel);
        var quantize = new QuantizeViewModel(LogBus);
        var adaptiveQuantize = new AdaptiveQuantizeViewModel(LogBus);
        var profileBuilder = new ProfileBuilderViewModel(LogBus, Settings);
        var perplexity = new PerplexityViewModel(LogBus);
        var klDivergence = new KlDivergenceViewModel(LogBus);
        var hellaswag = new HellaswagViewModel(LogBus);
        var imatrix = new ImatrixViewModel(LogBus);
        var controlVectors = new ControlVectorViewModel(LogBus);
        var ggufEditor = new GgufEditorViewModel();
        var sharding = new ShardingViewModel(LogBus);
        var loraMerge = new LoraMergeViewModel(LogBus);
        var hfConvert = new HfConvertViewModel(LogBus);

        // Sidebar order is the user-facing pipeline:
        //   discovery (1-2) → import (3) → quality calibration (4) →
        //   produce (5-7) → package/edit (8-9) → runtime adapters (10-11) →
        //   evaluate (12) → benchmarks group (13-14).
        // The Benchmarks header is rendered above KL Divergence via its
        // CategoryLabel override; HellaSwag inherits the section by
        // adjacency.
        Tools = new ToolPageViewModel[]
        {
            localModels,
            hfBrowser,
            hfConvert,
            imatrix,
            quantize,
            profileBuilder,
            adaptiveQuantize,
            sharding,
            ggufEditor,
            controlVectors,
            loraMerge,
            perplexity,
            klDivergence,
            hellaswag,
        };

        // Bind the navigator so any tool can route the user to another
        // tool (used by the briefcase-medical remedy buttons), then
        // hand each tool the shell-owned services.
        Navigator.Bind(
            type => Tools.FirstOrDefault(t => type.IsInstanceOfType(t)),
            tool => SelectedTool = (ToolPageViewModel)tool);
        foreach (var tool in Tools)
            tool.AttachShell(ActiveModel, Navigator);

        _selectedTool = quantize;

        // Push the active model into the visible tool whenever either
        // changes. Tools that don't take a GGUF path inherit the
        // base no-op and ignore it.
        ActiveModel.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(Services.ActiveModel.Path))
                SelectedTool?.ApplyActiveModel(ActiveModel.Path);
        };
    }

    /// <summary>
    /// Re-target the active model when the user picks one in the HF
    /// Browser or Local Models tool. Hooked to the active-model strip
    /// so all tools see the change immediately.
    /// </summary>
    public void SetActiveModel(string? path) => ActiveModel.Set(path);

    partial void OnSelectedToolChanged(ToolPageViewModel value)
    {
        // Newly-focused tool gets a fresh shot at the active model
        // (only if its input is empty — the override guards that).
        value?.ApplyActiveModel(ActiveModel.Path);
    }
}
