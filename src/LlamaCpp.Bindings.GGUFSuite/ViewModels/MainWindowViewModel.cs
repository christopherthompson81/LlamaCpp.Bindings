using System.Collections.Generic;
using CommunityToolkit.Mvvm.ComponentModel;
using LlamaCpp.Bindings.GGUFSuite.Services;

namespace LlamaCpp.Bindings.GGUFSuite.ViewModels;

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

    public MainWindowViewModel()
    {
        // One-time backend init wired up to the log bus so every page can
        // observe native log output without re-registering.
        LlamaBackend.Initialize((lvl, msg) => LogBus.Publish(lvl, msg));

        var quantize = new QuantizeViewModel(LogBus);
        var perplexity = new PerplexityViewModel(LogBus);
        var klDivergence = new KlDivergenceViewModel(LogBus);
        var hellaswag = new HellaswagViewModel(LogBus);
        var imatrix = new ImatrixViewModel(LogBus);
        var controlVectors = new ControlVectorViewModel(LogBus);
        var ggufEditor = new GgufEditorViewModel();
        var sharding = new ShardingViewModel(LogBus);
        var hfConvert = new HfConvertViewModel(LogBus);

        Tools = new ToolPageViewModel[]
        {
            quantize,
            perplexity,
            klDivergence,
            hellaswag,
            ggufEditor,
            sharding,
            imatrix,
            hfConvert,
            controlVectors,
        };

        _selectedTool = quantize;
    }
}
