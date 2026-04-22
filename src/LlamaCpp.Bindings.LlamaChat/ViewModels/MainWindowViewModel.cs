using System;
using System.Collections.ObjectModel;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.Services;

namespace LlamaCpp.Bindings.LlamaChat.ViewModels;

public partial class MainWindowViewModel : ViewModelBase, IDisposable
{
    public ObservableCollection<ProfileEditorViewModel> Profiles { get; }

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(LoadCommand))]
    private ProfileEditorViewModel? _selectedProfile;

    /// <summary>
    /// The profile that was used for the currently loaded session. Sampler +
    /// generation settings for SendAsync come from here — not from
    /// <see cref="SelectedProfile"/>, which only drives the next Load.
    /// </summary>
    private ProfileEditorViewModel? _activeLoadedProfile;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsModelLoaded), nameof(CanSend))]
    [NotifyCanExecuteChangedFor(nameof(SendCommand), nameof(LoadCommand), nameof(UnloadCommand))]
    private ChatSession? _session;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanSend))]
    [NotifyCanExecuteChangedFor(nameof(SendCommand), nameof(LoadCommand))]
    private bool _isBusy;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanSend), nameof(CanCancel))]
    [NotifyCanExecuteChangedFor(nameof(SendCommand), nameof(CancelCommand), nameof(UnloadCommand))]
    private bool _isGenerating;

    [ObservableProperty] private string _statusText = "Not loaded.";
    [ObservableProperty] private string _modelSummary = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanSend))]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    private string _userInput = string.Empty;

    public ObservableCollection<MessageViewModel> Messages { get; } = new();

    public bool IsModelLoaded => Session is not null;
    public bool CanSend => IsModelLoaded && !IsGenerating && !IsBusy && !string.IsNullOrWhiteSpace(UserInput);
    public bool CanCancel => IsGenerating;

    private CancellationTokenSource? _generationCts;
    private readonly List<string> _recentNativeLogLines = new();
    private const int MaxLogLines = 40;

    public MainWindowViewModel()
    {
        var loaded = ProfileStore.Load()
            .Select(p => new ProfileEditorViewModel(p))
            .ToList();
        Profiles = new ObservableCollection<ProfileEditorViewModel>(loaded);
        SelectedProfile = Profiles.FirstOrDefault();
    }

    [RelayCommand(CanExecute = nameof(CanLoad))]
    private async Task LoadAsync()
    {
        if (SelectedProfile is null) return;
        var profile = SelectedProfile;
        var path = profile.ModelPath;

        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            StatusText = $"Model file not found: {path}";
            return;
        }

        IsBusy = true;
        StatusText = $"Loading '{profile.Name}'...";
        _recentNativeLogLines.Clear();

        try
        {
            var settings = profile.SnapshotLoad();
            var session = await Task.Run(() => ChatSession.Load(settings, OnNativeLog));

            Session = session;
            _activeLoadedProfile = profile;
            ModelSummary = $"{profile.Name}  ·  {Path.GetFileName(session.Model.ModelPath)}  ·  " +
                           $"ctx={session.Context.ContextSize}  ·  layers={session.Model.LayerCount}  ·  " +
                           $"template={(string.IsNullOrEmpty(session.ChatTemplate) ? "(none)" : "embedded")}";
            StatusText = "Loaded.";
        }
        catch (Exception ex)
        {
            var tail = string.Join(" | ", _recentNativeLogLines.TakeLast(6));
            StatusText = tail.Length > 0
                ? $"Load failed: {ex.Message}\nNative log: {tail}"
                : $"Load failed: {ex.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    private bool CanLoad() => !IsBusy && Session is null && SelectedProfile is not null;

    [RelayCommand(CanExecute = nameof(CanUnload))]
    private void Unload()
    {
        _generationCts?.Cancel();
        Session?.Dispose();
        Session = null;
        _activeLoadedProfile = null;
        Messages.Clear();
        ModelSummary = string.Empty;
        StatusText = "Unloaded.";
    }

    private bool CanUnload() => Session is not null && !IsGenerating;

    [RelayCommand(CanExecute = nameof(CanSend))]
    private async Task SendAsync()
    {
        if (Session is null || _activeLoadedProfile is null) return;
        var text = UserInput.Trim();
        if (text.Length == 0) return;

        UserInput = string.Empty;
        Session.AppendUser(text);
        Messages.Add(new MessageViewModel { Role = "user", Content = text });

        var assistant = new MessageViewModel { Role = "assistant", IsStreaming = true };
        Messages.Add(assistant);

        IsGenerating = true;
        StatusText = "Generating...";
        _generationCts = new CancellationTokenSource();

        try
        {
            var sampler = _activeLoadedProfile.SamplerPanel.SnapshotSampler();
            var gen = _activeLoadedProfile.SamplerPanel.SnapshotGeneration();

            await foreach (var evt in Session.StreamAssistantTurnAsync(sampler, gen, _generationCts.Token))
            {
                switch (evt)
                {
                    case StreamEvent.Content c:
                        assistant.Content += c.Text;
                        break;
                    case StreamEvent.Reasoning r:
                        assistant.Reasoning = (assistant.Reasoning ?? string.Empty) + r.Text;
                        break;
                    case StreamEvent.Done d:
                        assistant.IsStreaming = false;
                        var tps = d.GenerationTime.TotalSeconds > 0
                            ? d.CompletionTokens / d.GenerationTime.TotalSeconds : 0;
                        assistant.StatsSummary = $"{d.CompletionTokens} tok · {tps:F1} tok/s";
                        StatusText = $"Done — {assistant.StatsSummary}";
                        break;
                }
            }
        }
        catch (OperationCanceledException)
        {
            assistant.IsStreaming = false;
            StatusText = "Cancelled.";
            if (string.IsNullOrEmpty(assistant.Content)) assistant.Content = "(cancelled)";
        }
        catch (Exception ex)
        {
            assistant.IsStreaming = false;
            StatusText = $"Error: {ex.Message}";
            if (string.IsNullOrEmpty(assistant.Content)) assistant.Content = $"(error: {ex.Message})";
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
        Session?.Reset();
        Messages.Clear();
        StatusText = IsModelLoaded ? "Conversation cleared." : "Not loaded.";
    }

    [RelayCommand]
    private async Task OpenSettingsAsync()
    {
        var vm = new SettingsWindowViewModel(Profiles);
        await DialogService.ShowSettingsAsync(vm);

        // Settings window closes; persist to disk so the edits survive restart.
        // The "Save" button inside the window also persists — this handles the
        // case where the user closed without clicking it. Silent failure here
        // is fine; the user will see stale state on next load and can re-save.
        try { ProfileStore.Save(Profiles.Select(p => p.ToProfile())); }
        catch { /* see note above */ }

        // Ensure SelectedProfile still points at something valid — it may have
        // been deleted in the settings window.
        if (SelectedProfile is not null && !Profiles.Contains(SelectedProfile))
        {
            SelectedProfile = Profiles.FirstOrDefault();
        }
    }

    [RelayCommand]
    private void Exit()
    {
        if (Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime d)
        {
            d.Shutdown();
        }
    }

    private void OnNativeLog(LlamaLogLevel level, string msg)
    {
        Dispatcher.UIThread.Post(() =>
        {
            _recentNativeLogLines.Add($"[{level}] {msg}");
            if (_recentNativeLogLines.Count > MaxLogLines)
                _recentNativeLogLines.RemoveAt(0);
            if (level is LlamaLogLevel.Warn or LlamaLogLevel.Error)
                StatusText = $"[{level}] {msg}";
        });
    }

    public void Dispose()
    {
        _generationCts?.Cancel();
        _generationCts?.Dispose();
        Session?.Dispose();
    }
}
