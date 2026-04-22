using CommunityToolkit.Mvvm.ComponentModel;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.ViewModels;

/// <summary>
/// Editable, observable shadow of a <see cref="ModelProfile"/>. Changes made
/// in the settings UI are picked up live by the main VM because it holds the
/// same instance — no round-trip through the store until the user saves.
/// </summary>
public partial class ProfileEditorViewModel : ObservableObject
{
    [ObservableProperty] private string _name = "New profile";

    // --- Load settings ---
    [ObservableProperty] private string _modelPath = string.Empty;
    [ObservableProperty] private decimal _contextSize = 4096m;
    [ObservableProperty] private decimal _gpuLayerCount = -1m;
    [ObservableProperty] private decimal _logicalBatchSize = 512m;
    [ObservableProperty] private decimal _physicalBatchSize = 512m;
    [ObservableProperty] private bool _useMmap = true;
    [ObservableProperty] private bool _useMlock = false;
    [ObservableProperty] private bool _offloadKQV = true;
    [ObservableProperty] private LlamaFlashAttention _flashAttention = LlamaFlashAttention.Auto;

    // --- Sampling + generation (shared VM so the editor can mutate it live) ---
    public SamplerPanelViewModel SamplerPanel { get; } = new();

    public ProfileEditorViewModel() { }

    public ProfileEditorViewModel(ModelProfile profile)
    {
        Name = profile.Name;
        ModelPath = profile.Load.ModelPath;
        ContextSize = profile.Load.ContextSize;
        GpuLayerCount = profile.Load.GpuLayerCount;
        LogicalBatchSize = profile.Load.LogicalBatchSize;
        PhysicalBatchSize = profile.Load.PhysicalBatchSize;
        UseMmap = profile.Load.UseMmap;
        UseMlock = profile.Load.UseMlock;
        OffloadKQV = profile.Load.OffloadKQV;
        FlashAttention = profile.Load.FlashAttention;
        SamplerPanel.LoadFrom(profile.Sampler, profile.Generation);
    }

    public ModelLoadSettings SnapshotLoad() => new()
    {
        ModelPath = ModelPath,
        ContextSize = (uint)ContextSize,
        GpuLayerCount = (int)GpuLayerCount,
        LogicalBatchSize = (uint)LogicalBatchSize,
        PhysicalBatchSize = (uint)PhysicalBatchSize,
        UseMmap = UseMmap,
        UseMlock = UseMlock,
        OffloadKQV = OffloadKQV,
        FlashAttention = FlashAttention,
    };

    public ModelProfile ToProfile() => new()
    {
        Name = Name,
        Load = SnapshotLoad(),
        Sampler = SamplerPanel.SnapshotSampler(),
        Generation = SamplerPanel.SnapshotGeneration(),
    };

    public override string ToString() => Name;
}
