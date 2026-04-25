using System;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.Services;

namespace LlamaCpp.Bindings.LlamaChat.ViewModels;

/// <summary>
/// Drives the Settings → Server tab. Holds editable draft fields mirroring
/// <see cref="LocalServerConfig"/>, plus commands for Start/Stop/Restart that
/// delegate to <see cref="ServerLaunchService.Instance"/>.
/// </summary>
public partial class ServerSettingsViewModel : ObservableObject
{
    public ServerLaunchService Service => ServerLaunchService.Instance;

    // ----- LlamaChat-side -----
    [ObservableProperty] private string _serverExecutablePath = "";
    [ObservableProperty] private bool _launchOnAppStart;
    [ObservableProperty] private bool _autoCreateRemoteProfile = true;
    [ObservableProperty] private bool _autoSelectProfileOnLaunch;
    [ObservableProperty] private string _extraArgs = "";
    [ObservableProperty] private int _startupTimeoutSeconds = 30;
    [ObservableProperty] private string _status = "";

    // ----- Model -----
    [ObservableProperty] private string _modelPath = "";
    [ObservableProperty] private string _modelAlias = "";

    // ----- Context / batching -----
    [ObservableProperty] private int _contextSize = 4096;
    [ObservableProperty] private int _logicalBatchSize = 512;
    [ObservableProperty] private int _physicalBatchSize = 512;
    [ObservableProperty] private int _maxSequenceCount = 4;

    // ----- GPU -----
    [ObservableProperty] private int _gpuLayerCount = -1;
    [ObservableProperty] private bool _offloadKqv = true;
    [ObservableProperty] private int _mainGpu = 0;
    [ObservableProperty] private LlamaSplitMode _splitMode = LlamaSplitMode.Layer;
    [ObservableProperty] private bool _noHost;
    [ObservableProperty] private bool _useExtraBufts = true;
    [ObservableProperty] private bool _cpuMoe;
    [ObservableProperty] private bool _checkTensors;
    [ObservableProperty] private bool _useDirectIo;
    [ObservableProperty] private string _devices = "";       // newline-separated
    [ObservableProperty] private string _tensorSplit = "";   // newline-separated floats

    // ----- CPU / threading -----
    [ObservableProperty] private int _threadCount = -1;
    [ObservableProperty] private int _batchThreadCount = -1;
    [ObservableProperty] private LlamaNumaStrategy _numaStrategy = LlamaNumaStrategy.Disabled;

    // ----- KV cache -----
    [ObservableProperty] private LlamaFlashAttention _flashAttention = LlamaFlashAttention.Auto;
    [ObservableProperty] private LlamaKvCacheType _kvCacheTypeK = LlamaKvCacheType.F16;
    [ObservableProperty] private LlamaKvCacheType _kvCacheTypeV = LlamaKvCacheType.F16;
    [ObservableProperty] private bool _useFullSwaCache = true;

    // ----- File I/O -----
    [ObservableProperty] private bool _useMmap = true;
    [ObservableProperty] private bool _useMlock;

    // ----- HTTP -----
    [ObservableProperty] private string _bindAddress = "127.0.0.1";
    [ObservableProperty] private int _port = 8080;
    [ObservableProperty] private string _httpsCertificatePath = "";
    [ObservableProperty] private string _httpsCertificatePassword = "";

    // ----- CORS -----
    [ObservableProperty] private string _corsAllowedOrigins = "";   // newline-separated
    [ObservableProperty] private bool _corsAllowCredentials;

    // ----- Auth -----
    [ObservableProperty] private string _apiKey = "";
    [ObservableProperty] private string _apiKeyFile = "";

    // ----- Limits -----
    [ObservableProperty] private int _maxOutputTokens = 2048;
    [ObservableProperty] private int _maxPromptTokens;
    [ObservableProperty] private int _requestTimeoutSeconds = 300;
    [ObservableProperty] private int _shutdownDrainSeconds = 30;

    // ----- Endpoints -----
    [ObservableProperty] private bool _exposeMetricsEndpoint = true;
    [ObservableProperty] private bool _exposeSlotsEndpoint = true;

    // ----- RoPE / YARN -----
    [ObservableProperty] private LlamaRopeScalingType _ropeScalingType = LlamaRopeScalingType.Unspecified;
    [ObservableProperty] private float _ropeFreqBase;
    [ObservableProperty] private float _ropeFreqScale;
    [ObservableProperty] private float _yarnExtFactor = -1f;
    [ObservableProperty] private float _yarnAttnFactor = 1f;
    [ObservableProperty] private float _yarnBetaFast = 32f;
    [ObservableProperty] private float _yarnBetaSlow = 1f;
    [ObservableProperty] private int _yarnOriginalContext;

    // ----- Multimodal -----
    [ObservableProperty] private string _mmprojPath = "";
    [ObservableProperty] private bool _mmprojAuto;
    [ObservableProperty] private bool _mmprojOnCpu;
    [ObservableProperty] private int _mmprojImageMinTokens;
    [ObservableProperty] private int _mmprojImageMaxTokens;

    // ----- Embeddings -----
    [ObservableProperty] private string _embeddingModelPath = "";
    [ObservableProperty] private int _embeddingContextSize = 2048;
    [ObservableProperty] private int _embeddingBatchSize = 512;
    [ObservableProperty] private int _embeddingGpuLayerCount = -1;
    [ObservableProperty] private string _embeddingModelAlias = "";

    // ----- Rerank -----
    [ObservableProperty] private string _rerankModelPath = "";
    [ObservableProperty] private int _rerankContextSize = 2048;
    [ObservableProperty] private int _rerankBatchSize = 512;
    [ObservableProperty] private int _rerankGpuLayerCount = -1;
    [ObservableProperty] private string _rerankModelAlias = "";

    // ----- Speculative -----
    [ObservableProperty] private string _draftModelPath = "";
    [ObservableProperty] private int _draftContextSize = 2048;
    [ObservableProperty] private int _draftLogicalBatchSize = 512;
    [ObservableProperty] private int _draftPhysicalBatchSize = 512;
    [ObservableProperty] private int _draftGpuLayerCount = -1;
    [ObservableProperty] private int _draftLookahead = 5;

    public ServerSettingsViewModel()
    {
        LoadFromConfig(Service.CurrentConfig);
        Service.PropertyChanged += OnServicePropertyChanged;
    }

    public bool IsRunning => Service.State == ServerLaunchState.Running;
    public bool IsStarting => Service.State == ServerLaunchState.Starting;
    public string StateLabel => Service.State.ToString();

    private void OnServicePropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        Dispatcher.UIThread.Post(() =>
        {
            if (e.PropertyName == nameof(Service.State))
            {
                OnPropertyChanged(nameof(IsRunning));
                OnPropertyChanged(nameof(IsStarting));
                OnPropertyChanged(nameof(StateLabel));
                StartCommand.NotifyCanExecuteChanged();
                StopCommand.NotifyCanExecuteChanged();
                RestartCommand.NotifyCanExecuteChanged();
            }
        });
    }

    private void LoadFromConfig(LocalServerConfig cfg)
    {
        ServerExecutablePath = cfg.ServerExecutablePath ?? "";
        LaunchOnAppStart = cfg.LaunchOnAppStart;
        AutoCreateRemoteProfile = cfg.AutoCreateRemoteProfile;
        AutoSelectProfileOnLaunch = cfg.AutoSelectProfileOnLaunch;
        ExtraArgs = string.Join("\n", cfg.ExtraArgs);
        StartupTimeoutSeconds = cfg.StartupTimeoutSeconds;

        ModelPath = cfg.ModelPath;
        ModelAlias = cfg.ModelAlias;

        ContextSize = cfg.ContextSize;
        LogicalBatchSize = cfg.LogicalBatchSize;
        PhysicalBatchSize = cfg.PhysicalBatchSize;
        MaxSequenceCount = cfg.MaxSequenceCount;

        GpuLayerCount = cfg.GpuLayerCount;
        OffloadKqv = cfg.OffloadKqv;
        MainGpu = cfg.MainGpu;
        SplitMode = cfg.SplitMode;
        NoHost = cfg.NoHost;
        UseExtraBufts = cfg.UseExtraBufts;
        CpuMoe = cfg.CpuMoe;
        CheckTensors = cfg.CheckTensors;
        UseDirectIo = cfg.UseDirectIo;
        Devices = string.Join("\n", cfg.Devices);
        TensorSplit = string.Join("\n", cfg.TensorSplit
            .Select(f => f.ToString(CultureInfo.InvariantCulture)));

        ThreadCount = cfg.ThreadCount;
        BatchThreadCount = cfg.BatchThreadCount;
        NumaStrategy = cfg.NumaStrategy;

        FlashAttention = cfg.FlashAttention;
        KvCacheTypeK = cfg.KvCacheTypeK;
        KvCacheTypeV = cfg.KvCacheTypeV;
        UseFullSwaCache = cfg.UseFullSwaCache;

        UseMmap = cfg.UseMmap;
        UseMlock = cfg.UseMlock;

        BindAddress = cfg.BindAddress;
        Port = cfg.Port;
        HttpsCertificatePath = cfg.HttpsCertificatePath;
        HttpsCertificatePassword = cfg.HttpsCertificatePassword;

        CorsAllowedOrigins = string.Join("\n", cfg.CorsAllowedOrigins);
        CorsAllowCredentials = cfg.CorsAllowCredentials;

        ApiKey = cfg.ApiKey;
        ApiKeyFile = cfg.ApiKeyFile;

        MaxOutputTokens = cfg.MaxOutputTokens;
        MaxPromptTokens = cfg.MaxPromptTokens;
        RequestTimeoutSeconds = cfg.RequestTimeoutSeconds;
        ShutdownDrainSeconds = cfg.ShutdownDrainSeconds;

        ExposeMetricsEndpoint = cfg.ExposeMetricsEndpoint;
        ExposeSlotsEndpoint = cfg.ExposeSlotsEndpoint;

        RopeScalingType = cfg.RopeScalingType;
        RopeFreqBase = cfg.RopeFreqBase;
        RopeFreqScale = cfg.RopeFreqScale;
        YarnExtFactor = cfg.YarnExtFactor;
        YarnAttnFactor = cfg.YarnAttnFactor;
        YarnBetaFast = cfg.YarnBetaFast;
        YarnBetaSlow = cfg.YarnBetaSlow;
        YarnOriginalContext = (int)cfg.YarnOriginalContext;

        MmprojPath = cfg.MmprojPath;
        MmprojAuto = cfg.MmprojAuto;
        MmprojOnCpu = cfg.MmprojOnCpu;
        MmprojImageMinTokens = cfg.MmprojImageMinTokens;
        MmprojImageMaxTokens = cfg.MmprojImageMaxTokens;

        EmbeddingModelPath = cfg.EmbeddingModelPath;
        EmbeddingContextSize = cfg.EmbeddingContextSize;
        EmbeddingBatchSize = cfg.EmbeddingBatchSize;
        EmbeddingGpuLayerCount = cfg.EmbeddingGpuLayerCount;
        EmbeddingModelAlias = cfg.EmbeddingModelAlias;

        RerankModelPath = cfg.RerankModelPath;
        RerankContextSize = cfg.RerankContextSize;
        RerankBatchSize = cfg.RerankBatchSize;
        RerankGpuLayerCount = cfg.RerankGpuLayerCount;
        RerankModelAlias = cfg.RerankModelAlias;

        DraftModelPath = cfg.DraftModelPath;
        DraftContextSize = cfg.DraftContextSize;
        DraftLogicalBatchSize = cfg.DraftLogicalBatchSize;
        DraftPhysicalBatchSize = cfg.DraftPhysicalBatchSize;
        DraftGpuLayerCount = cfg.DraftGpuLayerCount;
        DraftLookahead = cfg.DraftLookahead;
    }

    private LocalServerConfig BuildConfig() => new()
    {
        ServerExecutablePath = string.IsNullOrWhiteSpace(ServerExecutablePath) ? null : ServerExecutablePath.Trim(),
        LaunchOnAppStart = LaunchOnAppStart,
        AutoCreateRemoteProfile = AutoCreateRemoteProfile,
        AutoSelectProfileOnLaunch = AutoSelectProfileOnLaunch,
        ExtraArgs = SplitLines(ExtraArgs),
        StartupTimeoutSeconds = StartupTimeoutSeconds,

        ModelPath = ModelPath.Trim(),
        ModelAlias = ModelAlias.Trim(),

        ContextSize = ContextSize,
        LogicalBatchSize = LogicalBatchSize,
        PhysicalBatchSize = PhysicalBatchSize,
        MaxSequenceCount = MaxSequenceCount,

        GpuLayerCount = GpuLayerCount,
        OffloadKqv = OffloadKqv,
        MainGpu = MainGpu,
        SplitMode = SplitMode,
        NoHost = NoHost,
        UseExtraBufts = UseExtraBufts,
        CpuMoe = CpuMoe,
        CheckTensors = CheckTensors,
        UseDirectIo = UseDirectIo,
        Devices = SplitLines(Devices),
        TensorSplit = SplitLines(TensorSplit)
            .Select(s => float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var f) ? f : 0f)
            .ToList(),

        ThreadCount = ThreadCount,
        BatchThreadCount = BatchThreadCount,
        NumaStrategy = NumaStrategy,

        FlashAttention = FlashAttention,
        KvCacheTypeK = KvCacheTypeK,
        KvCacheTypeV = KvCacheTypeV,
        UseFullSwaCache = UseFullSwaCache,

        UseMmap = UseMmap,
        UseMlock = UseMlock,

        BindAddress = string.IsNullOrWhiteSpace(BindAddress) ? "127.0.0.1" : BindAddress.Trim(),
        Port = Port,
        HttpsCertificatePath = HttpsCertificatePath.Trim(),
        HttpsCertificatePassword = HttpsCertificatePassword,

        CorsAllowedOrigins = SplitLines(CorsAllowedOrigins),
        CorsAllowCredentials = CorsAllowCredentials,

        ApiKey = ApiKey.Trim(),
        ApiKeyFile = ApiKeyFile.Trim(),

        MaxOutputTokens = MaxOutputTokens,
        MaxPromptTokens = MaxPromptTokens,
        RequestTimeoutSeconds = RequestTimeoutSeconds,
        ShutdownDrainSeconds = ShutdownDrainSeconds,

        ExposeMetricsEndpoint = ExposeMetricsEndpoint,
        ExposeSlotsEndpoint = ExposeSlotsEndpoint,

        RopeScalingType = RopeScalingType,
        RopeFreqBase = RopeFreqBase,
        RopeFreqScale = RopeFreqScale,
        YarnExtFactor = YarnExtFactor,
        YarnAttnFactor = YarnAttnFactor,
        YarnBetaFast = YarnBetaFast,
        YarnBetaSlow = YarnBetaSlow,
        YarnOriginalContext = (uint)Math.Max(0, YarnOriginalContext),

        MmprojPath = MmprojPath.Trim(),
        MmprojAuto = MmprojAuto,
        MmprojOnCpu = MmprojOnCpu,
        MmprojImageMinTokens = MmprojImageMinTokens,
        MmprojImageMaxTokens = MmprojImageMaxTokens,

        EmbeddingModelPath = EmbeddingModelPath.Trim(),
        EmbeddingContextSize = EmbeddingContextSize,
        EmbeddingBatchSize = EmbeddingBatchSize,
        EmbeddingGpuLayerCount = EmbeddingGpuLayerCount,
        EmbeddingModelAlias = EmbeddingModelAlias.Trim(),

        RerankModelPath = RerankModelPath.Trim(),
        RerankContextSize = RerankContextSize,
        RerankBatchSize = RerankBatchSize,
        RerankGpuLayerCount = RerankGpuLayerCount,
        RerankModelAlias = RerankModelAlias.Trim(),

        DraftModelPath = DraftModelPath.Trim(),
        DraftContextSize = DraftContextSize,
        DraftLogicalBatchSize = DraftLogicalBatchSize,
        DraftPhysicalBatchSize = DraftPhysicalBatchSize,
        DraftGpuLayerCount = DraftGpuLayerCount,
        DraftLookahead = DraftLookahead,
    };

    private static System.Collections.Generic.List<string> SplitLines(string text) =>
        text.Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(s => s.Trim())
            .Where(s => s.Length > 0)
            .ToList();

    [RelayCommand]
    private async Task BrowseModelPathAsync()
    {
        var path = await DialogService.PickGgufFileAsync();
        if (!string.IsNullOrEmpty(path)) ModelPath = path;
    }

    [RelayCommand]
    private async Task BrowseMmprojPathAsync()
    {
        var path = await DialogService.PickGgufFileAsync();
        if (!string.IsNullOrEmpty(path)) MmprojPath = path;
    }

    [RelayCommand]
    private async Task BrowseEmbeddingModelPathAsync()
    {
        var path = await DialogService.PickGgufFileAsync();
        if (!string.IsNullOrEmpty(path)) EmbeddingModelPath = path;
    }

    [RelayCommand]
    private async Task BrowseRerankModelPathAsync()
    {
        var path = await DialogService.PickGgufFileAsync();
        if (!string.IsNullOrEmpty(path)) RerankModelPath = path;
    }

    [RelayCommand]
    private async Task BrowseDraftModelPathAsync()
    {
        var path = await DialogService.PickGgufFileAsync();
        if (!string.IsNullOrEmpty(path)) DraftModelPath = path;
    }

    [RelayCommand]
    private async Task BrowseExecutablePathAsync()
    {
        var path = await DialogService.PickServerExecutableAsync();
        if (!string.IsNullOrEmpty(path)) ServerExecutablePath = path;
    }

    [RelayCommand]
    private void GenerateApiKey()
    {
        ApiKey = Guid.NewGuid().ToString("N");
    }

    [RelayCommand]
    private void Save()
    {
        try
        {
            Service.UpdateConfig(BuildConfig());
            Status = "Saved.";
        }
        catch (Exception ex)
        {
            Status = $"Save failed: {ex.Message}";
        }
    }

    [RelayCommand(CanExecute = nameof(CanStart))]
    private async Task StartAsync()
    {
        Service.UpdateConfig(BuildConfig());
        Status = "Starting…";
        try
        {
            await Service.StartAsync().ConfigureAwait(true);
            Status = Service.State == ServerLaunchState.Running
                ? $"Running on {Service.RunningBaseUrl}"
                : Service.Error ?? "Start failed.";
        }
        catch (Exception ex)
        {
            Status = $"Start failed: {ex.Message}";
        }
    }

    [RelayCommand(CanExecute = nameof(CanStop))]
    private async Task StopAsync()
    {
        Status = "Stopping…";
        await Service.StopAsync().ConfigureAwait(true);
        Status = "Stopped.";
    }

    [RelayCommand(CanExecute = nameof(CanStop))]
    private async Task RestartAsync()
    {
        await StopAsync().ConfigureAwait(true);
        await StartAsync().ConfigureAwait(true);
    }

    [RelayCommand]
    private async Task CopyBaseUrlAsync()
    {
        var url = Service.RunningBaseUrl;
        if (!string.IsNullOrEmpty(url)) await DialogService.CopyToClipboardAsync(url);
    }

    [RelayCommand]
    private void ClearLog() => Service.ClearLog();

    private bool CanStart() => Service.State is ServerLaunchState.Stopped or ServerLaunchState.Failed;
    private bool CanStop()  => Service.State is ServerLaunchState.Running or ServerLaunchState.Starting;
}
