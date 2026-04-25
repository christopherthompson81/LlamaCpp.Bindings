using System;
using System.ComponentModel;
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

    [ObservableProperty] private string _serverExecutablePath = "";
    [ObservableProperty] private string _modelPath = "";
    [ObservableProperty] private string _bindAddress = "127.0.0.1";
    [ObservableProperty] private int _port = 8080;
    [ObservableProperty] private int _contextSize = 4096;
    [ObservableProperty] private int _gpuLayerCount = -1;
    [ObservableProperty] private int _maxSequenceCount = 4;
    [ObservableProperty] private LlamaFlashAttention _flashAttention = LlamaFlashAttention.Auto;
    [ObservableProperty] private string _apiKey = "";
    [ObservableProperty] private bool _launchOnAppStart;
    [ObservableProperty] private bool _autoCreateRemoteProfile = true;
    [ObservableProperty] private bool _autoSelectProfileOnLaunch;
    [ObservableProperty] private string _extraArgs = "";
    [ObservableProperty] private int _startupTimeoutSeconds = 30;
    [ObservableProperty] private string _status = "";

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
        ModelPath = cfg.ModelPath;
        BindAddress = cfg.BindAddress;
        Port = cfg.Port;
        ContextSize = cfg.ContextSize;
        GpuLayerCount = cfg.GpuLayerCount;
        MaxSequenceCount = cfg.MaxSequenceCount;
        FlashAttention = cfg.FlashAttention;
        ApiKey = cfg.ApiKey ?? "";
        LaunchOnAppStart = cfg.LaunchOnAppStart;
        AutoCreateRemoteProfile = cfg.AutoCreateRemoteProfile;
        AutoSelectProfileOnLaunch = cfg.AutoSelectProfileOnLaunch;
        ExtraArgs = string.Join("\n", cfg.ExtraArgs);
        StartupTimeoutSeconds = cfg.StartupTimeoutSeconds;
    }

    private LocalServerConfig BuildConfig() => new()
    {
        ServerExecutablePath = string.IsNullOrWhiteSpace(ServerExecutablePath) ? null : ServerExecutablePath.Trim(),
        ModelPath = ModelPath.Trim(),
        BindAddress = string.IsNullOrWhiteSpace(BindAddress) ? "127.0.0.1" : BindAddress.Trim(),
        Port = Port,
        ContextSize = ContextSize,
        GpuLayerCount = GpuLayerCount,
        MaxSequenceCount = MaxSequenceCount,
        FlashAttention = FlashAttention,
        ApiKey = string.IsNullOrWhiteSpace(ApiKey) ? null : ApiKey.Trim(),
        LaunchOnAppStart = LaunchOnAppStart,
        AutoCreateRemoteProfile = AutoCreateRemoteProfile,
        AutoSelectProfileOnLaunch = AutoSelectProfileOnLaunch,
        ExtraArgs = ExtraArgs
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(s => s.Trim())
            .Where(s => s.Length > 0)
            .ToList(),
        StartupTimeoutSeconds = StartupTimeoutSeconds,
    };

    [RelayCommand]
    private async Task BrowseModelPathAsync()
    {
        var path = await DialogService.PickGgufFileAsync();
        if (!string.IsNullOrEmpty(path)) ModelPath = path;
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
