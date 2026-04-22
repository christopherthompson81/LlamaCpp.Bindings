using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings;

namespace LlamaChat.ViewModels;

public partial class MainWindowViewModel : ViewModelBase, IDisposable
{
    // ----- Bindable state -----

    [ObservableProperty]
    private string _modelPath = "/mnt/data/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf";

    // NOTE on the Notify* attributes below: CommunityToolkit.Mvvm has TWO
    // separate notification channels:
    //   - NotifyPropertyChangedFor     -> fires INotifyPropertyChanged for bound display props
    //   - NotifyCanExecuteChangedFor   -> fires ICommand.CanExecuteChanged for a Button's IsEnabled
    // We need BOTH on source-of-truth fields that feed a command's CanExecute,
    // otherwise the button's IsEnabled is evaluated once at bind time and never updates.
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsModelLoaded), nameof(CanSend))]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    private LlamaModel? _loadedModel;

    [ObservableProperty]
    private LlamaContext? _loadedContext;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanSend))]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    private bool _isBusy;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanSend), nameof(CanCancel))]
    [NotifyCanExecuteChangedFor(nameof(SendCommand), nameof(CancelCommand))]
    private bool _isGenerating;

    [ObservableProperty]
    private string _statusText = "Not loaded.";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanSend))]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    private string _userInput = string.Empty;

    [ObservableProperty]
    private float _temperature = 0.7f;

    [ObservableProperty]
    private uint _seed = 42;

    [ObservableProperty]
    private int _maxTokens = 512;

    public ObservableCollection<ChatMessageViewModel> Messages { get; } = new();

    public bool IsModelLoaded => LoadedModel is not null;
    public bool CanSend => IsModelLoaded && !IsGenerating && !IsBusy && !string.IsNullOrWhiteSpace(UserInput);
    public bool CanCancel => IsGenerating;

    // ----- Private state -----

    private CancellationTokenSource? _generationCts;

    // Ring-ish buffer of the last several native log lines — used to enrich
    // load-failure messages (llama.cpp surfaces most real failure causes
    // through the log callback, not through the return status).
    private readonly System.Collections.Generic.List<string> _recentNativeLogLines = new();
    private const int MaxRememberedLogLines = 40;

    // ----- Commands -----

    [RelayCommand]
    private async Task LoadAsync()
    {
        if (IsModelLoaded)
        {
            UnloadInternal();
            return;
        }

        if (!File.Exists(ModelPath))
        {
            StatusText = $"Model file not found: {ModelPath}";
            return;
        }

        IsBusy = true;
        StatusText = "Loading model...";
        _recentNativeLogLines.Clear();

        try
        {
            // Initialize on the UI thread first — safe and idempotent — so the
            // log sink is registered before any background native work starts
            // and can capture every line of the load (important when debugging
            // VRAM failures, which llama.cpp only reports via the log callback).
            LlamaBackend.Initialize(logSink: OnNativeLog);

            // Do ONLY the blocking native work on a background thread. The
            // resulting handles are assigned back on the UI thread so their
            // ObservableProperty setters fire PropertyChanged /
            // CanExecuteChanged on the right SynchronizationContext.
            var (model, context) = await Task.Run(() =>
            {
                var m = new LlamaModel(ModelPath, new LlamaModelParameters
                {
                    GpuLayerCount = -1, // all layers on GPU
                    UseMmap = true,
                });

                // Context size matches the working test fixture; full SWA at
                // 4096 was pushing VRAM over the edge on a 3090 already holding
                // ~20GB of weights. Expose as a user setting later if needed.
                var c = new LlamaContext(m, new LlamaContextParameters
                {
                    ContextSize = 2048,
                    LogicalBatchSize = 512,
                    PhysicalBatchSize = 512,
                    MaxSequenceCount = 1,
                    OffloadKQV = true,
                    UseFullSwaCache = true,
                });
                return (m, c);
            });

            LoadedModel = model;
            LoadedContext = context;
            StatusText = $"Loaded. ctx={LoadedContext!.ContextSize}, layers={LoadedModel!.LayerCount}";
        }
        catch (Exception ex)
        {
            // Surface the last few native log lines alongside the managed
            // exception message — llama.cpp failures are almost always
            // explained by something the native side just logged.
            var tail = string.Join(" | ", _recentNativeLogLines.TakeLast(6));
            StatusText = tail.Length > 0
                ? $"Load failed: {ex.Message}\nNative log tail: {tail}"
                : $"Load failed: {ex.Message}";
            UnloadInternal();
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand(CanExecute = nameof(CanSend))]
    private async Task SendAsync()
    {
        var prompt = UserInput.Trim();
        if (string.IsNullOrEmpty(prompt) || LoadedModel is null || LoadedContext is null)
            return;

        UserInput = string.Empty;
        Messages.Add(new ChatMessageViewModel { Role = "user", Content = prompt });
        var assistantMessage = new ChatMessageViewModel { Role = "assistant", Content = string.Empty };
        Messages.Add(assistantMessage);

        IsGenerating = true;
        StatusText = "Generating...";
        _generationCts = new CancellationTokenSource();

        try
        {
            var template = LoadedModel.GetChatTemplate();
            string renderedPrompt;
            if (string.IsNullOrEmpty(template))
            {
                // Model has no embedded template — fall back to naked concat.
                var sb = new StringBuilder();
                foreach (var m in Messages.SkipLast(1))
                    sb.AppendLine($"{m.Role}: {m.Content}");
                sb.Append("assistant: ");
                renderedPrompt = sb.ToString();
            }
            else
            {
                var history = Messages
                    .SkipLast(1) // don't pass the empty assistant placeholder
                    .Select(m => new ChatMessage(m.Role, m.Content))
                    .ToArray();
                renderedPrompt = LlamaChatTemplate.Apply(template, history, addAssistantPrefix: true);
            }

            // Rebuild-from-scratch strategy: clear KV, re-decode full conversation.
            // Simple, correct, and O(total tokens) per turn. For longer conversations
            // a delta-decode using LlamaContext.SequencePositionRange would be faster.
            LoadedContext.ClearKvCache();

            using var sampler = new LlamaSamplerBuilder()
                .WithPenalties(lastN: 64, repeat: 1.1f)
                .WithTopK(40)
                .WithTopP(0.9f)
                .WithMinP(0.05f)
                .WithTemperature(Temperature)
                .WithDistribution(Seed)
                .Build();

            var generator = new LlamaGenerator(LoadedContext, sampler);

            await foreach (var piece in generator.GenerateAsync(
                renderedPrompt,
                maxTokens: MaxTokens,
                addSpecial: false,
                parseSpecial: true,
                cancellationToken: _generationCts.Token))
            {
                // Generator's native calls run on Task.Run; the await resumes on the
                // UI SynchronizationContext so mutating the observable property is
                // thread-safe as long as Send was invoked from the UI thread.
                assistantMessage.Content += piece;
            }

            StatusText = $"Done. ~{TokensFor(assistantMessage.Content)} words emitted.";
        }
        catch (OperationCanceledException)
        {
            StatusText = "Cancelled.";
            if (assistantMessage.Content.Length == 0)
                assistantMessage.Content = "(cancelled)";
        }
        catch (Exception ex)
        {
            StatusText = $"Error: {ex.Message}";
            if (assistantMessage.Content.Length == 0)
                assistantMessage.Content = $"(error: {ex.Message})";
        }
        finally
        {
            _generationCts?.Dispose();
            _generationCts = null;
            IsGenerating = false;
        }
    }

    [RelayCommand(CanExecute = nameof(CanCancel))]
    private void Cancel() => _generationCts?.Cancel();

    [RelayCommand]
    private void ClearConversation()
    {
        Messages.Clear();
        LoadedContext?.ClearKvCache();
        StatusText = IsModelLoaded ? "Conversation cleared." : "Not loaded.";
    }

    // ----- Internals -----

    private void UnloadInternal()
    {
        _generationCts?.Cancel();
        _generationCts?.Dispose();
        _generationCts = null;

        LoadedContext?.Dispose();
        LoadedContext = null;
        LoadedModel?.Dispose();
        LoadedModel = null;
        Messages.Clear();

        StatusText = "Unloaded.";
    }

    private void OnNativeLog(LlamaLogLevel level, string msg)
    {
        // Native log callback runs on arbitrary threads — marshal to UI before
        // touching bindable state. We keep a tail buffer of recent lines so a
        // load failure can print the native context along with the exception.
        Dispatcher.UIThread.Post(() =>
        {
            _recentNativeLogLines.Add($"[{level}] {msg}");
            if (_recentNativeLogLines.Count > MaxRememberedLogLines)
            {
                _recentNativeLogLines.RemoveAt(0);
            }
            if (level is LlamaLogLevel.Warn or LlamaLogLevel.Error)
            {
                StatusText = $"[{level}] {msg}";
            }
        });
    }

    private static int TokensFor(string s) => s.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;

    public void Dispose() => UnloadInternal();
}

public partial class ChatMessageViewModel : ObservableObject
{
    [ObservableProperty] private string _role = string.Empty;
    [ObservableProperty] private string _content = string.Empty;
}
