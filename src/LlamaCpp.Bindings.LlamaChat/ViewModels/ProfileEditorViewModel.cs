using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.Services;
using LlamaCpp.Bindings.LlamaChat.Services.Remote;

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

    // --- Backend selector ---
    /// <summary>
    /// Local = in-process llama.cpp (Load section). Remote = HTTP server
    /// (Remote section). Other-cluster fields stay populated when toggling
    /// so the user can switch back without losing values.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsLocal), nameof(IsRemote))]
    private ProfileKind _kind = ProfileKind.Local;

    public bool IsLocal => Kind == ProfileKind.Local;
    public bool IsRemote => Kind == ProfileKind.Remote;

    // --- Remote settings ---
    [ObservableProperty] private string _baseUrl = "http://localhost:8080";
    [ObservableProperty] private string _apiKey = string.Empty;
    [ObservableProperty] private string _modelId = string.Empty;
    [ObservableProperty] private string _discoverStatus = string.Empty;
    public ObservableCollection<string> AvailableModels { get; } = new();

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
        Kind = profile.Kind;
        SystemPrompt = profile.SystemPrompt;
        BaseUrl = profile.Remote.BaseUrl;
        ApiKey = profile.Remote.ApiKey ?? string.Empty;
        ModelId = profile.Remote.ModelId;
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

    public RemoteSettings SnapshotRemote() => new()
    {
        BaseUrl = BaseUrl,
        ApiKey = string.IsNullOrEmpty(ApiKey) ? null : ApiKey,
        ModelId = ModelId,
    };

    public ModelProfile ToProfile() => new()
    {
        Name = Name,
        Kind = Kind,
        SystemPrompt = SystemPrompt,
        Load = SnapshotLoad(),
        Remote = SnapshotRemote(),
        Sampler = SamplerPanel.SnapshotSampler(),
        Generation = SamplerPanel.SnapshotGeneration(),
    };

    /// <summary>
    /// Hit <c>GET /v1/models</c> against the configured Remote endpoint and
    /// fill <see cref="AvailableModels"/>. Pre-selects the first id when
    /// <see cref="ModelId"/> is empty so the user can save immediately.
    /// </summary>
    [RelayCommand]
    private async Task DiscoverRemoteModelsAsync()
    {
        if (string.IsNullOrWhiteSpace(BaseUrl))
        {
            DiscoverStatus = "Set a base URL first.";
            return;
        }
        DiscoverStatus = "Discovering...";
        try
        {
            using var client = new OpenAiChatClient(BaseUrl, string.IsNullOrEmpty(ApiKey) ? null : ApiKey);
            var ids = await client.ListModelsAsync().ConfigureAwait(true);
            AvailableModels.Clear();
            foreach (var id in ids) AvailableModels.Add(id);
            if (ids.Count == 0)
            {
                DiscoverStatus = "Connected, but the server reported no models.";
            }
            else
            {
                if (string.IsNullOrEmpty(ModelId)) ModelId = ids[0];
                DiscoverStatus = $"Found {ids.Count} model(s).";
            }
        }
        catch (Exception ex)
        {
            DiscoverStatus = $"Failed: {ex.Message}";
        }
    }

    /// <summary>Reset the Sampling sub-panel to code defaults. Load settings are preserved.</summary>
    public void ResetSamplerDefaults() =>
        SamplerPanel.LoadFrom(SamplerSettings.Default, new GenerationSettings());

    /// <summary>
    /// Overwrite load + sampling fields with the recommendations in
    /// <paramref name="result"/>. The profile name and system prompt are
    /// preserved; the model path is replaced only if the new one is
    /// non-empty (the service always sets it to the probed path, so in
    /// practice it doesn't change).
    /// </summary>
    public void ApplyAutoConfigure(AutoConfigureResult result)
    {
        var load = result.Load;
        if (!string.IsNullOrEmpty(load.ModelPath)) ModelPath = load.ModelPath;
        ContextSize       = load.ContextSize;
        GpuLayerCount     = load.GpuLayerCount;
        LogicalBatchSize  = load.LogicalBatchSize;
        PhysicalBatchSize = load.PhysicalBatchSize;
        UseMmap           = load.UseMmap;
        UseMlock          = load.UseMlock;
        OffloadKQV        = load.OffloadKQV;
        FlashAttention    = load.FlashAttention;
        KvCacheType       = load.KvCacheTypeK;

        // Apply the service's generation settings too — auto-configure
        // picks a higher MaxTokens than the code-default because the
        // default is too low for modern reasoning/thinking models.
        SamplerPanel.LoadFrom(result.Sampler, result.Generation);
    }

    public override string ToString() => Name;
}
