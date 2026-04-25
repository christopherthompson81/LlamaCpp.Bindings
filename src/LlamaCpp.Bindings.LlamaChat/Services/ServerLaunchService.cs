using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.Services.Remote;

namespace LlamaCpp.Bindings.LlamaChat.Services;

public enum ServerLaunchState { Stopped, Starting, Running, Stopping, Failed }

/// <summary>
/// Singleton owning a single child <c>LlamaCpp.Bindings.Server</c> process.
/// One server at a time — the UI guards against double-start. Public surface
/// mirrors <see cref="McpClientService"/>: an <c>Instance</c> property,
/// observable state, a bounded <see cref="Log"/>, lifecycle methods.
/// Auto-wires a remote <c>ModelProfile</c> via the
/// <see cref="RemoteProfileRequested"/> event after the server's
/// <c>/health</c> passes — kept as an event (not a direct call into the VM)
/// so the service stays free of UI coupling.
/// </summary>
public sealed partial class ServerLaunchService : ObservableObject, IDisposable
{
    public static ServerLaunchService Instance { get; } = new();

    private ServerLaunchService()
    {
        CurrentConfig = LocalServerConfigStore.Load();
    }

    public const int LogCap = 500;

    [ObservableProperty] private ServerLaunchState _state = ServerLaunchState.Stopped;
    [ObservableProperty] private string? _error;
    [ObservableProperty] private string? _resolvedExecutablePath;
    [ObservableProperty] private string? _runningBaseUrl;
    [ObservableProperty] private string? _runningModelId;

    public ObservableCollection<string> Log { get; } = new();

    /// <summary>
    /// Raised once after a successful start, when <c>/health</c> has returned
    /// 200 and <c>/v1/models</c> has reported a model id. <see cref="MainWindowViewModel"/>
    /// handles this to upsert the "Local server" profile.
    /// </summary>
    public event EventHandler<RemoteProfileRequestEventArgs>? RemoteProfileRequested;

    public LocalServerConfig CurrentConfig { get; private set; }

    private Process? _process;
    private CancellationTokenSource? _runCts;
    private static readonly HttpClient s_health = new() { Timeout = TimeSpan.FromSeconds(2) };

    public void UpdateConfig(LocalServerConfig cfg)
    {
        CurrentConfig = cfg;
        try { LocalServerConfigStore.Save(cfg); }
        catch (Exception ex) { AppendLog($"[config] save failed: {ex.Message}"); }
    }

    public async Task StartAsync(CancellationToken ct = default)
    {
        if (State is ServerLaunchState.Starting or ServerLaunchState.Running)
        {
            throw new InvalidOperationException("Server is already running.");
        }

        var cfg = CurrentConfig;
        Error = null;
        RunningBaseUrl = null;
        RunningModelId = null;
        State = ServerLaunchState.Starting;

        string exe;
        try
        {
            exe = ResolveExecutable(cfg);
        }
        catch (Exception ex)
        {
            Error = ex.Message;
            State = ServerLaunchState.Failed;
            return;
        }
        ResolvedExecutablePath = exe;
        AppendLog($"[launch] executable: {exe}");

        if (string.IsNullOrWhiteSpace(cfg.ModelPath))
        {
            Error = "Model path is empty.";
            State = ServerLaunchState.Failed;
            return;
        }
        if (!File.Exists(cfg.ModelPath))
        {
            Error = $"Model path does not exist: {cfg.ModelPath}";
            State = ServerLaunchState.Failed;
            return;
        }

        var psi = BuildStartInfo(exe, cfg);
        AppendLog($"[launch] {psi.FileName} {string.Join(" ", psi.ArgumentList)}");

        var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
        proc.OutputDataReceived += (_, e) => { if (e.Data is not null) AppendLog(e.Data); };
        proc.ErrorDataReceived  += (_, e) => { if (e.Data is not null) AppendLog(e.Data); };
        proc.Exited += OnProcessExited;

        try
        {
            if (!proc.Start())
            {
                Error = "Process.Start returned false.";
                State = ServerLaunchState.Failed;
                return;
            }
        }
        catch (Exception ex)
        {
            Error = $"Failed to start process: {ex.Message}";
            State = ServerLaunchState.Failed;
            return;
        }

        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();
        _process = proc;

        _runCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var baseUrl = cfg.BaseUrl;

        // Health-poll loop. Watches both the budget and process exit.
        var deadline = DateTime.UtcNow.AddSeconds(Math.Max(1, cfg.StartupTimeoutSeconds));
        var ok = false;
        while (DateTime.UtcNow < deadline && !_runCts.IsCancellationRequested)
        {
            if (proc.HasExited)
            {
                Error = $"Server exited during startup (code {proc.ExitCode}). See log.";
                State = ServerLaunchState.Failed;
                return;
            }
            try
            {
                using var res = await s_health.GetAsync($"{baseUrl}/health", _runCts.Token).ConfigureAwait(false);
                if (res.IsSuccessStatusCode) { ok = true; break; }
            }
            catch (OperationCanceledException) when (_runCts.IsCancellationRequested) { break; }
            catch
            {
                // Connection refused while Kestrel is still binding — keep polling.
            }
            try { await Task.Delay(200, _runCts.Token).ConfigureAwait(false); }
            catch (OperationCanceledException) { break; }
        }

        if (!ok)
        {
            Error = $"Server did not respond on {baseUrl}/health within {cfg.StartupTimeoutSeconds}s.";
            try { if (!proc.HasExited) proc.Kill(entireProcessTree: true); } catch { }
            State = ServerLaunchState.Failed;
            return;
        }

        // Discover the model id so the auto-wired remote profile picks it up.
        string modelId = "";
        try
        {
            using var client = new OpenAiChatClient(baseUrl, string.IsNullOrEmpty(cfg.ApiKey) ? null : cfg.ApiKey);
            var ids = await client.ListModelsAsync(_runCts.Token).ConfigureAwait(false);
            modelId = ids.FirstOrDefault() ?? "";
        }
        catch (Exception ex)
        {
            AppendLog($"[discover] /v1/models failed: {ex.Message}");
        }

        RunningBaseUrl = baseUrl;
        RunningModelId = modelId;
        State = ServerLaunchState.Running;

        if (cfg.AutoCreateRemoteProfile && !string.IsNullOrEmpty(modelId))
        {
            RemoteProfileRequested?.Invoke(this, new RemoteProfileRequestEventArgs
            {
                ProfileName = "Local server",
                BaseUrl = baseUrl,
                ApiKey = string.IsNullOrEmpty(cfg.ApiKey) ? null : cfg.ApiKey,
                ModelId = modelId,
                AutoSelect = cfg.AutoSelectProfileOnLaunch,
            });
        }
    }

    public async Task StopAsync(int drainSeconds = 5)
    {
        var proc = _process;
        if (proc is null)
        {
            State = ServerLaunchState.Stopped;
            return;
        }

        State = ServerLaunchState.Stopping;
        try { _runCts?.Cancel(); } catch { }
        try
        {
            if (!proc.HasExited)
            {
                proc.Kill(entireProcessTree: true);
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(Math.Max(1, drainSeconds)));
                try { await proc.WaitForExitAsync(cts.Token).ConfigureAwait(false); } catch { }
            }
        }
        catch (Exception ex)
        {
            AppendLog($"[stop] {ex.Message}");
        }
        finally
        {
            try { proc.Dispose(); } catch { }
            _process = null;
            _runCts?.Dispose();
            _runCts = null;
            RunningBaseUrl = null;
            RunningModelId = null;
            if (State != ServerLaunchState.Failed) State = ServerLaunchState.Stopped;
        }
    }

    public void ClearLog()
    {
        if (Dispatcher.UIThread.CheckAccess()) Log.Clear();
        else Dispatcher.UIThread.Post(Log.Clear);
    }

    public void Dispose()
    {
        // Teardown path — sync wait is fine, the alternative is an orphaned child.
        try { StopAsync(2).GetAwaiter().GetResult(); } catch { }
    }

    private void OnProcessExited(object? sender, EventArgs e)
    {
        // Only flip to Failed if we were Running. A Stop-initiated exit is
        // already handled in StopAsync.
        if (State == ServerLaunchState.Running)
        {
            var code = sender is Process p ? p.ExitCode : -1;
            Dispatcher.UIThread.Post(() =>
            {
                Error = $"Server exited unexpectedly (code {code}).";
                State = ServerLaunchState.Failed;
                RunningBaseUrl = null;
                RunningModelId = null;
            });
        }
    }

    private static ProcessStartInfo BuildStartInfo(string exe, LocalServerConfig cfg)
    {
        var isDll = exe.EndsWith(".dll", StringComparison.OrdinalIgnoreCase);
        var psi = new ProcessStartInfo
        {
            FileName = isDll ? "dotnet" : exe,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(exe) ?? Environment.CurrentDirectory,
        };

        if (isDll) psi.ArgumentList.Add(exe);

        void Add(string key, object value) =>
            psi.ArgumentList.Add($"--LlamaServer:{key}={value}");
        void AddIfSet(string key, string value)
        {
            if (!string.IsNullOrEmpty(value)) Add(key, value);
        }
        void AddList(string key, IReadOnlyList<string> values)
        {
            for (var i = 0; i < values.Count; i++)
            {
                if (!string.IsNullOrWhiteSpace(values[i])) Add($"{key}:{i}", values[i]);
            }
        }

        // Model
        Add("ModelPath", cfg.ModelPath);
        AddIfSet("ModelAlias", cfg.ModelAlias);

        // HTTP
        var scheme = cfg.HttpsCertificatePath.Length > 0 ? "https" : "http";
        Add("Urls", $"{scheme}://{cfg.BindAddress}:{cfg.Port}");
        AddIfSet("HttpsCertificatePath", cfg.HttpsCertificatePath);
        AddIfSet("HttpsCertificatePassword", cfg.HttpsCertificatePassword);

        // Context / batching
        Add("ContextSize", cfg.ContextSize);
        Add("LogicalBatchSize", cfg.LogicalBatchSize);
        Add("PhysicalBatchSize", cfg.PhysicalBatchSize);
        Add("MaxSequenceCount", cfg.MaxSequenceCount);

        // GPU
        Add("GpuLayerCount", cfg.GpuLayerCount);
        Add("OffloadKqv", cfg.OffloadKqv);
        Add("MainGpu", cfg.MainGpu);
        Add("SplitMode", cfg.SplitMode);
        Add("NoHost", cfg.NoHost);
        Add("UseExtraBufts", cfg.UseExtraBufts);
        Add("CpuMoe", cfg.CpuMoe);
        Add("CheckTensors", cfg.CheckTensors);
        Add("UseDirectIo", cfg.UseDirectIo);
        AddList("Devices", cfg.Devices);
        for (var i = 0; i < cfg.TensorSplit.Count; i++)
            Add($"TensorSplit:{i}", cfg.TensorSplit[i].ToString(System.Globalization.CultureInfo.InvariantCulture));

        // CPU / threading
        Add("ThreadCount", cfg.ThreadCount);
        Add("BatchThreadCount", cfg.BatchThreadCount);
        Add("NumaStrategy", cfg.NumaStrategy);

        // KV cache
        Add("FlashAttention", cfg.FlashAttention);
        Add("KvCacheTypeK", cfg.KvCacheTypeK);
        Add("KvCacheTypeV", cfg.KvCacheTypeV);
        Add("UseFullSwaCache", cfg.UseFullSwaCache);

        // File I/O
        Add("UseMmap", cfg.UseMmap);
        Add("UseMlock", cfg.UseMlock);

        // CORS
        AddList("CorsAllowedOrigins", cfg.CorsAllowedOrigins);
        Add("CorsAllowCredentials", cfg.CorsAllowCredentials);

        // Auth
        if (!string.IsNullOrEmpty(cfg.ApiKey)) Add("ApiKeys:0", cfg.ApiKey);
        AddIfSet("ApiKeyFile", cfg.ApiKeyFile);

        // Limits
        Add("MaxOutputTokens", cfg.MaxOutputTokens);
        Add("MaxPromptTokens", cfg.MaxPromptTokens);
        Add("RequestTimeoutSeconds", cfg.RequestTimeoutSeconds);
        Add("ShutdownDrainSeconds", cfg.ShutdownDrainSeconds);

        // Endpoints
        Add("ExposeMetricsEndpoint", cfg.ExposeMetricsEndpoint);
        Add("ExposeSlotsEndpoint", cfg.ExposeSlotsEndpoint);

        // RoPE / YARN
        Add("RopeScalingType", cfg.RopeScalingType);
        Add("RopeFreqBase", cfg.RopeFreqBase.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Add("RopeFreqScale", cfg.RopeFreqScale.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Add("YarnExtFactor", cfg.YarnExtFactor.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Add("YarnAttnFactor", cfg.YarnAttnFactor.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Add("YarnBetaFast", cfg.YarnBetaFast.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Add("YarnBetaSlow", cfg.YarnBetaSlow.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Add("YarnOriginalContext", cfg.YarnOriginalContext);

        // Multimodal
        AddIfSet("MmprojPath", cfg.MmprojPath);
        Add("MmprojAuto", cfg.MmprojAuto);
        if (cfg.MmprojOnCpu) Add("MmprojOnCpu", true);
        if (cfg.MmprojImageMinTokens > 0) Add("MmprojImageMinTokens", cfg.MmprojImageMinTokens);
        if (cfg.MmprojImageMaxTokens > 0) Add("MmprojImageMaxTokens", cfg.MmprojImageMaxTokens);

        // Embeddings
        if (!string.IsNullOrEmpty(cfg.EmbeddingModelPath))
        {
            Add("EmbeddingModelPath", cfg.EmbeddingModelPath);
            Add("EmbeddingContextSize", cfg.EmbeddingContextSize);
            Add("EmbeddingBatchSize", cfg.EmbeddingBatchSize);
            Add("EmbeddingGpuLayerCount", cfg.EmbeddingGpuLayerCount);
            AddIfSet("EmbeddingModelAlias", cfg.EmbeddingModelAlias);
        }

        // Rerank
        if (!string.IsNullOrEmpty(cfg.RerankModelPath))
        {
            Add("RerankModelPath", cfg.RerankModelPath);
            Add("RerankContextSize", cfg.RerankContextSize);
            Add("RerankBatchSize", cfg.RerankBatchSize);
            Add("RerankGpuLayerCount", cfg.RerankGpuLayerCount);
            AddIfSet("RerankModelAlias", cfg.RerankModelAlias);
        }

        // Speculative decoding (draft model)
        if (!string.IsNullOrEmpty(cfg.DraftModelPath))
        {
            Add("DraftModelPath", cfg.DraftModelPath);
            Add("DraftContextSize", cfg.DraftContextSize);
            Add("DraftLogicalBatchSize", cfg.DraftLogicalBatchSize);
            Add("DraftPhysicalBatchSize", cfg.DraftPhysicalBatchSize);
            Add("DraftGpuLayerCount", cfg.DraftGpuLayerCount);
            Add("DraftLookahead", cfg.DraftLookahead);
        }

        foreach (var arg in cfg.ExtraArgs)
        {
            if (!string.IsNullOrWhiteSpace(arg)) psi.ArgumentList.Add(arg);
        }
        return psi;
    }

    private static string ResolveExecutable(LocalServerConfig cfg)
    {
        if (!string.IsNullOrWhiteSpace(cfg.ServerExecutablePath))
        {
            if (File.Exists(cfg.ServerExecutablePath)) return cfg.ServerExecutablePath;
            throw new FileNotFoundException($"Configured server path does not exist: {cfg.ServerExecutablePath}");
        }

        var probed = new List<string>();
        var baseDir = AppContext.BaseDirectory;

        // 1. Side-by-side published copy (MSBuild target writes here).
        foreach (var name in new[] { "LlamaCpp.Bindings.Server.dll", "LlamaCpp.Bindings.Server.exe", "LlamaCpp.Bindings.Server" })
        {
            var p = Path.Combine(baseDir, "server", name);
            probed.Add(p);
            if (File.Exists(p)) return p;
        }

        // 2. Dev fallback: sibling project's bin/<cfg>/<tfm>/.
        var roots = new[]
        {
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "LlamaCpp.Bindings.Server", "bin")),
        };
        foreach (var root in roots)
        {
            if (!Directory.Exists(root)) { probed.Add(root); continue; }
            foreach (var configDir in new[] { "Debug", "Release" })
            {
                var tfmRoot = Path.Combine(root, configDir);
                if (!Directory.Exists(tfmRoot)) { probed.Add(tfmRoot); continue; }
                foreach (var tfmDir in Directory.EnumerateDirectories(tfmRoot))
                {
                    var p = Path.Combine(tfmDir, "LlamaCpp.Bindings.Server.dll");
                    probed.Add(p);
                    if (File.Exists(p)) return p;
                }
            }
        }

        throw new FileNotFoundException(
            "Could not find LlamaCpp.Bindings.Server. Set Server path explicitly. Probed:\n  "
            + string.Join("\n  ", probed));
    }

    private void AppendLog(string line)
    {
        if (Dispatcher.UIThread.CheckAccess())
        {
            DoAppend(line);
        }
        else
        {
            Dispatcher.UIThread.Post(() => DoAppend(line));
        }
    }

    private void DoAppend(string line)
    {
        Log.Add(line);
        while (Log.Count > LogCap) Log.RemoveAt(0);
    }
}

public sealed class RemoteProfileRequestEventArgs : EventArgs
{
    public required string ProfileName { get; init; }
    public required string BaseUrl { get; init; }
    public required string? ApiKey { get; init; }
    public required string ModelId { get; init; }
    public required bool AutoSelect { get; init; }
}
