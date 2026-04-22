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
    // ============================================================
    // Profiles (loaded from ProfileStore in ctor)
    // ============================================================
    public ObservableCollection<ProfileEditorViewModel> Profiles { get; }

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(LoadCommand))]
    private ProfileEditorViewModel? _selectedProfile;

    private ProfileEditorViewModel? _activeLoadedProfile;

    // ============================================================
    // Conversations
    // ============================================================
    public ObservableCollection<ConversationViewModel> Conversations { get; } = new();

    /// <summary>
    /// Filtered slice driven by <see cref="SearchText"/>. Bound to the sidebar
    /// ListBox so typing filters without mutating the master list.
    /// </summary>
    public ObservableCollection<ConversationViewModel> FilteredConversations { get; } = new();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanSend))]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    private ConversationViewModel? _selectedConversation;

    [ObservableProperty] private string _searchText = string.Empty;

    [ObservableProperty] private bool _isSidebarVisible = true;

    // ============================================================
    // Session
    // ============================================================
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

    public bool IsModelLoaded => Session is not null;

    public bool CanSend =>
        IsModelLoaded
        && !IsGenerating
        && !IsBusy
        && !string.IsNullOrWhiteSpace(UserInput)
        && SelectedConversation is not null;

    public bool CanCancel => IsGenerating;

    private CancellationTokenSource? _generationCts;
    private readonly List<string> _recentNativeLogLines = new();
    private const int MaxLogLines = 40;

    // ============================================================
    // Construction
    // ============================================================
    public MainWindowViewModel()
    {
        Profiles = new ObservableCollection<ProfileEditorViewModel>(
            ProfileStore.Load().Select(p => new ProfileEditorViewModel(p)));
        SelectedProfile = Profiles.FirstOrDefault();

        foreach (var c in ConversationStore.Load())
        {
            Conversations.Add(new ConversationViewModel(c));
        }

        // Ensure there is at least one conversation so the UI has a place to
        // append messages on first launch / after last one deleted.
        if (Conversations.Count == 0)
        {
            Conversations.Add(ConversationViewModel.NewEmpty());
        }

        RebuildFilteredConversations();
        SelectedConversation = FilteredConversations.FirstOrDefault();
    }

    partial void OnSearchTextChanged(string value) => RebuildFilteredConversations();

    partial void OnSelectedConversationChanged(ConversationViewModel? oldValue, ConversationViewModel? newValue)
    {
        // Switching conversations invalidates the KV cache — the old prefix
        // belongs to the previous transcript. Clearing now means the next send
        // pays the full re-prefill exactly once (same cost as the current
        // clear-every-turn policy, so no net regression).
        if (oldValue != newValue) Session?.ClearKv();
    }

    private void RebuildFilteredConversations()
    {
        FilteredConversations.Clear();
        IEnumerable<ConversationViewModel> source = Conversations
            .OrderByDescending(c => c.UpdatedAt);
        if (!string.IsNullOrWhiteSpace(SearchText))
        {
            var q = SearchText.Trim();
            source = source.Where(c =>
                c.Title.Contains(q, StringComparison.OrdinalIgnoreCase) ||
                c.Preview.Contains(q, StringComparison.OrdinalIgnoreCase));
        }
        foreach (var c in source) FilteredConversations.Add(c);
    }

    // ============================================================
    // Model load / unload
    // ============================================================
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
        ModelSummary = string.Empty;
        StatusText = "Unloaded.";
    }

    private bool CanUnload() => Session is not null && !IsGenerating;

    // ============================================================
    // Conversation CRUD
    // ============================================================
    [RelayCommand]
    private void NewConversation()
    {
        var conv = ConversationViewModel.NewEmpty();
        Conversations.Add(conv);
        RebuildFilteredConversations();
        SelectedConversation = conv;
        SearchText = string.Empty;
        SaveConversations();
    }

    [RelayCommand]
    private void DeleteConversation(ConversationViewModel? conv)
    {
        conv ??= SelectedConversation;
        if (conv is null) return;
        Conversations.Remove(conv);
        if (Conversations.Count == 0) Conversations.Add(ConversationViewModel.NewEmpty());
        RebuildFilteredConversations();
        SelectedConversation = FilteredConversations.FirstOrDefault();
        SaveConversations();
    }

    [RelayCommand]
    private void BeginRename(ConversationViewModel? conv)
    {
        conv ??= SelectedConversation;
        if (conv is null) return;
        conv.IsEditing = true;
    }

    [RelayCommand]
    private void EndRename(ConversationViewModel? conv)
    {
        conv ??= SelectedConversation;
        if (conv is null) return;
        if (string.IsNullOrWhiteSpace(conv.Title)) conv.Title = "New chat";
        conv.IsEditing = false;
        RebuildFilteredConversations();
        SaveConversations();
    }

    // ============================================================
    // Sidebar + generation
    // ============================================================
    [RelayCommand]
    private void ToggleSidebar() => IsSidebarVisible = !IsSidebarVisible;

    [RelayCommand(CanExecute = nameof(CanSend))]
    private async Task SendAsync()
    {
        if (Session is null || SelectedConversation is null || _activeLoadedProfile is null) return;
        var text = UserInput.Trim();
        if (text.Length == 0) return;
        UserInput = string.Empty;

        var conv = SelectedConversation;
        var user = new MessageViewModel { Role = "user", Content = text };
        conv.Messages.Add(user);
        conv.UpdatedAt = DateTimeOffset.UtcNow;

        // Auto-title off the first user message so the sidebar item is legible.
        if (string.IsNullOrWhiteSpace(conv.Title) ||
            conv.Title == "New chat" || conv.Title == "(untitled)")
        {
            var first = text.Replace('\n', ' ').Trim();
            conv.Title = first.Length > 40 ? first[..40] + "…" : first;
        }

        var assistant = new MessageViewModel { Role = "assistant", IsStreaming = true };
        conv.Messages.Add(assistant);

        RebuildFilteredConversations();

        IsGenerating = true;
        StatusText = "Generating...";
        _generationCts = new CancellationTokenSource();

        try
        {
            var sampler = _activeLoadedProfile.SamplerPanel.SnapshotSampler();
            var gen = _activeLoadedProfile.SamplerPanel.SnapshotGeneration();

            // Feed everything except the empty assistant placeholder to the model.
            var transcript = conv.Messages
                .Take(conv.Messages.Count - 1)
                .Select(m => new ChatTurn(
                    Id: Guid.NewGuid(),
                    Role: m.IsUser ? TurnRole.User : TurnRole.Assistant,
                    Content: m.Content,
                    State: TurnState.Complete,
                    CreatedAt: DateTimeOffset.UtcNow))
                .ToList();

            await foreach (var evt in Session.StreamAssistantReplyAsync(
                               transcript, sampler, gen, _generationCts.Token))
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
            conv.UpdatedAt = DateTimeOffset.UtcNow;
            SaveConversations();
        }
    }

    [RelayCommand(CanExecute = nameof(CanCancel))]
    private void Cancel() => _generationCts?.Cancel();

    [RelayCommand]
    private void ClearConversation()
    {
        if (SelectedConversation is null) return;
        SelectedConversation.Messages.Clear();
        Session?.ClearKv();
        StatusText = IsModelLoaded ? "Conversation cleared." : "Not loaded.";
        SaveConversations();
    }

    [RelayCommand]
    private async Task OpenSettingsAsync()
    {
        var vm = new SettingsWindowViewModel(Profiles);
        await DialogService.ShowSettingsAsync(vm);

        try { ProfileStore.Save(Profiles.Select(p => p.ToProfile())); }
        catch { /* non-fatal; user can re-save from the dialog */ }

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

    // ============================================================
    // Helpers
    // ============================================================
    private void SaveConversations()
    {
        try { ConversationStore.Save(Conversations.Select(c => c.ToModel())); }
        catch { /* non-fatal */ }
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
        SaveConversations();
    }
}
