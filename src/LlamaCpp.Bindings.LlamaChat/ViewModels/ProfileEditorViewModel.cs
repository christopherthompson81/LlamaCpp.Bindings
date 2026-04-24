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
    [ObservableProperty] private string _systemPrompt = string.Empty;

    // --- Load settings ---
    [ObservableProperty] private string _modelPath = string.Empty;
    [ObservableProperty] private string _mmprojPath = string.Empty;
    [ObservableProperty] private bool _mmprojOnCpu;
    [ObservableProperty] private decimal _mmprojImageMinTokens;
    [ObservableProperty] private decimal _contextSize = 4096m;
    [ObservableProperty] private decimal _gpuLayerCount = -1m;
    [ObservableProperty] private decimal _logicalBatchSize = 512m;
    [ObservableProperty] private decimal _physicalBatchSize = 512m;
    [ObservableProperty] private bool _useMmap = true;
    [ObservableProperty] private bool _useMlock = false;
    [ObservableProperty] private bool _offloadKQV = true;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(FlashAttentionHint))]
    private LlamaFlashAttention _flashAttention = LlamaFlashAttention.Auto;

    /// <summary>
    /// Applied to both K and V by default — a single control in the UI keeps
    /// the common case simple. Asymmetric K/V types (e.g. Q8_0 K + Q4_0 V for
    /// extra V compression) still work via the profile JSON but aren't in the
    /// dropdown. Auto-flips <see cref="FlashAttention"/> to Enabled on change
    /// to a quantized type because llama.cpp requires FA for quantized caches.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(FlashAttentionHint))]
    private LlamaKvCacheType _kvCacheType = LlamaKvCacheType.F16;

    partial void OnKvCacheTypeChanged(LlamaKvCacheType value)
    {
        if (IsQuantized(value) && FlashAttention == LlamaFlashAttention.Auto)
            FlashAttention = LlamaFlashAttention.Enabled;
    }

    private static bool IsQuantized(LlamaKvCacheType t) => t switch
    {
        LlamaKvCacheType.F32 or LlamaKvCacheType.F16 or LlamaKvCacheType.BF16 => false,
        _ => true,
    };

    /// <summary>
    /// Helper line shown under the KV cache picker. Warns when a quantized
    /// cache is selected but Flash Attention is explicitly disabled — that
    /// combination triggers a runtime error in llama.cpp.
    /// </summary>
    public string? FlashAttentionHint =>
        IsQuantized(KvCacheType) && FlashAttention == LlamaFlashAttention.Disabled
            ? "Quantized KV cache requires Flash Attention — the model will fail to load while Flash is Disabled."
            : null;

    // --- Sampling + generation (shared VM so the editor can mutate it live) ---
    public SamplerPanelViewModel SamplerPanel { get; } = new();

    public ProfileEditorViewModel() { }

    public ProfileEditorViewModel(ModelProfile profile)
    {
        Name = profile.Name;
        SystemPrompt = profile.SystemPrompt;
        ModelPath = profile.Load.ModelPath;
        MmprojPath = profile.Load.MmprojPath;
        MmprojOnCpu = profile.Load.MmprojOnCpu;
        MmprojImageMinTokens = profile.Load.MmprojImageMinTokens ?? 0;
        ContextSize = profile.Load.ContextSize;
        GpuLayerCount = profile.Load.GpuLayerCount;
        LogicalBatchSize = profile.Load.LogicalBatchSize;
        PhysicalBatchSize = profile.Load.PhysicalBatchSize;
        UseMmap = profile.Load.UseMmap;
        UseMlock = profile.Load.UseMlock;
        OffloadKQV = profile.Load.OffloadKQV;
        FlashAttention = profile.Load.FlashAttention;
        // K/V are persisted independently; the UI picker collapses them into
        // one choice, so we show K's value and write both back on snapshot.
        KvCacheType = profile.Load.KvCacheTypeK;
        SamplerPanel.LoadFrom(profile.Sampler, profile.Generation);
    }

    public ModelLoadSettings SnapshotLoad() => new()
    {
        ModelPath = ModelPath,
        MmprojPath = MmprojPath,
        MmprojOnCpu = MmprojOnCpu,
        MmprojImageMinTokens = MmprojImageMinTokens > 0 ? (int)MmprojImageMinTokens : null,
        ContextSize = (uint)ContextSize,
        GpuLayerCount = (int)GpuLayerCount,
        LogicalBatchSize = (uint)LogicalBatchSize,
        PhysicalBatchSize = (uint)PhysicalBatchSize,
        UseMmap = UseMmap,
        UseMlock = UseMlock,
        OffloadKQV = OffloadKQV,
        FlashAttention = FlashAttention,
        KvCacheTypeK = KvCacheType,
        KvCacheTypeV = KvCacheType,
    };

    public ModelProfile ToProfile() => new()
    {
        Name = Name,
        SystemPrompt = SystemPrompt,
        Load = SnapshotLoad(),
        Sampler = SamplerPanel.SnapshotSampler(),
        Generation = SamplerPanel.SnapshotGeneration(),
    };

    /// <summary>Reset the Sampling sub-panel to code defaults. Load settings are preserved.</summary>
    public void ResetSamplerDefaults() =>
        SamplerPanel.LoadFrom(SamplerSettings.Default, new GenerationSettings());

    public override string ToString() => Name;
}
