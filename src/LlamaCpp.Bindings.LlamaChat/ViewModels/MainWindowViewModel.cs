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
    // App settings (global preferences)
    // ============================================================
    public AppSettingsViewModel AppSettings { get; }

    // ============================================================
    // Construction
    // ============================================================
    public MainWindowViewModel()
    {
        AppSettings = new AppSettingsViewModel(AppSettingsStore.Load());

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
        // ObservableCollection.Clear() forces the sidebar ListBox to push
        // null back through its SelectedItem TwoWay binding, which would
        // nuke SelectedConversation mid-send and blank out the chat area.
        // Save here, restore at the end.
        var saved = SelectedConversation;

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

        if (saved is not null && FilteredConversations.Contains(saved))
        {
            SelectedConversation = saved;
        }
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

        await GenerateAssistantReplyAsync(conv);
    }

    /// <summary>
    /// Append an assistant placeholder to <paramref name="conv"/> and stream
    /// a reply into it using the current transcript. Shared by
    /// <see cref="SendAsync"/>, <see cref="RegenerateMessageAsync"/>, and the
    /// user-edit-commit path. Caller must have already mutated the
    /// transcript to reflect "what the model should reply to now".
    /// </summary>
    private async Task GenerateAssistantReplyAsync(ConversationViewModel conv)
    {
        if (Session is null || _activeLoadedProfile is null) return;

        var assistant = new MessageViewModel
        {
            Role = "assistant",
            IsStreaming = true,
            // Auto-open the <think> panel at creation time if the user has
            // opted into watching reasoning trace live. Not re-evaluated after
            // Done, so the user's manual toggle during/after streaming wins.
            IsReasoningExpanded = AppSettings.ShowReasoningInProgress,
        };
        conv.Messages.Add(assistant);
        conv.UpdatedAt = DateTimeOffset.UtcNow;
        RebuildFilteredConversations();

        // Batch content + reasoning updates at ~30 fps. We accumulate on the
        // pool thread (the consumer of Session.StreamAssistantReplyAsync runs
        // there thanks to ConfigureAwait(false) below) and only marshal to
        // the UI thread at flush time via Dispatcher.Post at Background
        // priority — so the decode loop is never blocked on UI thread work.
        // Dispatcher.Post is FIFO at the same priority, which lets us use
        // InvokeAsync(Background) at Done time to guarantee all prior
        // flushes have been applied before the Done-handler runs.
        const int FlushEveryMs = 33;
        var pendingContent = new System.Text.StringBuilder();
        var pendingReasoning = new System.Text.StringBuilder();
        var lastFlushTicks = Environment.TickCount;

        // Drain the pending StringBuilders by posting a UI-thread action that
        // mutates assistant.Content / assistant.Reasoning. Fire-and-forget:
        // the pool thread returns immediately, decode continues.
        void PostFlush()
        {
            if (pendingContent.Length == 0 && pendingReasoning.Length == 0)
            {
                lastFlushTicks = Environment.TickCount;
                return;
            }

            // Snapshot + clear on pool thread — pendingContent is owned by
            // this task, no lock needed.
            string? contentSnap = null, reasoningSnap = null;
            if (pendingContent.Length > 0)
            {
                contentSnap = pendingContent.ToString();
                pendingContent.Clear();
            }
            if (pendingReasoning.Length > 0)
            {
                reasoningSnap = pendingReasoning.ToString();
                pendingReasoning.Clear();
            }
            lastFlushTicks = Environment.TickCount;

            Dispatcher.UIThread.Post(() =>
            {
                if (contentSnap is not null) assistant.Content += contentSnap;
                if (reasoningSnap is not null)
                    assistant.Reasoning = (assistant.Reasoning ?? string.Empty) + reasoningSnap;
            }, DispatcherPriority.Background);
        }

        void TryFlush()
        {
            if (Environment.TickCount - lastFlushTicks < FlushEveryMs) return;
            PostFlush();
        }

        IsGenerating = true;
        StatusText = "Generating...";
        _generationCts = new CancellationTokenSource();

        try
        {
            var sampler = _activeLoadedProfile.SamplerPanel.SnapshotSampler();
            var gen = _activeLoadedProfile.SamplerPanel.SnapshotGeneration();

            // Feed everything except the empty assistant placeholder to the
            // model, prepending the active profile's system prompt if any.
            var transcript = new List<ChatTurn>();
            if (!string.IsNullOrWhiteSpace(_activeLoadedProfile.SystemPrompt))
            {
                transcript.Add(new ChatTurn(
                    Id: Guid.NewGuid(),
                    Role: TurnRole.System,
                    Content: _activeLoadedProfile.SystemPrompt,
                    State: TurnState.Complete,
                    CreatedAt: DateTimeOffset.UtcNow));
            }
            transcript.AddRange(conv.Messages
                .Take(conv.Messages.Count - 1)
                .Select(m => new ChatTurn(
                    Id: Guid.NewGuid(),
                    Role: m.IsUser ? TurnRole.User : TurnRole.Assistant,
                    Content: m.Content,
                    State: TurnState.Complete,
                    CreatedAt: DateTimeOffset.UtcNow)));

            // .ConfigureAwait(false): each MoveNextAsync resumes on a pool
            // thread rather than posting back to the UI Dispatcher. That
            // eliminates a pool→UI→pool round-trip per token, which is
            // ~0.5-1 ms of Dispatcher-queue latency on hot paths — the
            // suspected source of the 8 tok/s gap vs. llama-server at this
            // workload size.
            await foreach (var evt in Session
                .StreamAssistantReplyAsync(transcript, sampler, gen, _generationCts.Token)
                .ConfigureAwait(false))
            {
                switch (evt)
                {
                    case StreamEvent.Content c:
                        pendingContent.Append(c.Text);
                        TryFlush();
                        break;
                    case StreamEvent.Reasoning r:
                        pendingReasoning.Append(r.Text);
                        TryFlush();
                        break;
                    case StreamEvent.Done d:
                        // Force-post the final flush, then InvokeAsync the
                        // Done-handler at the same Background priority. FIFO
                        // semantics guarantee the Done handler runs after all
                        // queued flushes have drained.
                        PostFlush();
                        var doneData = d;
                        await Dispatcher.UIThread.InvokeAsync(() =>
                        {
                            assistant.IsStreaming = false;
                            var tps = doneData.GenerationTime.TotalSeconds > 0
                                ? doneData.CompletionTokens / doneData.GenerationTime.TotalSeconds : 0;
                            assistant.StatsSummary = $"{doneData.CompletionTokens} tok · {tps:F1} tok/s";
                            StatusText = $"Done — {assistant.StatsSummary}";
                        }, DispatcherPriority.Background);
                        break;
                }
            }
        }
        catch (OperationCanceledException)
        {
            PostFlush();
            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                assistant.IsStreaming = false;
                StatusText = "Cancelled.";
                if (string.IsNullOrEmpty(assistant.Content)) assistant.Content = "(cancelled)";
            }, DispatcherPriority.Background);
        }
        catch (Exception ex)
        {
            PostFlush();

            // Dump the full exception (type + message + full stack trace) to a
            // sibling of the app's other config files. Writing a file is safe
            // from any thread — only touching VM state needs the UI thread.
            try
            {
                var dir = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                    "LlamaChat");
                Directory.CreateDirectory(dir);
                File.WriteAllText(
                    Path.Combine(dir, "last-error.log"),
                    $"{DateTime.Now:o}\n" +
                    $"{ex.GetType().FullName}: {ex.Message}\n\n" +
                    $"{ex}\n");
            }
            catch { /* ignore log-write failures */ }
            System.Diagnostics.Debug.WriteLine($"[error] {ex}");

            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                assistant.IsStreaming = false;
                StatusText = $"Error: {ex.GetType().Name} — {ex.Message} (see ~/.config/LlamaChat/last-error.log)";
                if (string.IsNullOrEmpty(assistant.Content))
                {
                    assistant.Content = $"(error: {ex.GetType().Name}: {ex.Message})";
                }
            }, DispatcherPriority.Background);
        }
        finally
        {
            // We arrive here from whichever thread the try/catch last ran on —
            // pool during streaming, UI after InvokeAsync drained. Marshal all
            // VM-observable writes explicitly so we don't depend on state.
            _generationCts?.Dispose();
            _generationCts = null;
            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                IsGenerating = false;
                conv.UpdatedAt = DateTimeOffset.UtcNow;
                SaveConversations();
            }, DispatcherPriority.Background);
        }
    }

    // ============================================================
    // Per-message actions (copy / edit / regenerate / delete)
    // ============================================================

    [RelayCommand]
    private async Task CopyMessageAsync(MessageViewModel? msg)
    {
        if (msg is null) return;
        await DialogService.CopyToClipboardAsync(msg.Content);
        StatusText = "Copied to clipboard.";
    }

    [RelayCommand]
    private void BeginEditMessage(MessageViewModel? msg)
    {
        if (msg is null) return;
        msg.EditDraft = msg.Content;
        msg.IsEditing = true;
    }

    [RelayCommand]
    private void CancelEditMessage(MessageViewModel? msg)
    {
        if (msg is null) return;
        msg.IsEditing = false;
        msg.EditDraft = string.Empty;
    }

    /// <summary>
    /// Commit the inline edit. For a user message we also truncate everything
    /// after it and regenerate — the previous assistant reply was conditioned
    /// on the old prompt, so keeping it would be inconsistent. For an
    /// assistant message we just overwrite its text.
    /// </summary>
    [RelayCommand]
    private async Task CommitEditMessageAsync(MessageViewModel? msg)
    {
        if (msg is null || SelectedConversation is null) return;

        var conv = SelectedConversation;
        var newText = msg.EditDraft;
        msg.Content = newText;
        msg.IsEditing = false;
        msg.EditDraft = string.Empty;

        if (msg.IsUser)
        {
            var idx = conv.Messages.IndexOf(msg);
            if (idx < 0) return;
            // Remove everything after the edited user message.
            while (conv.Messages.Count > idx + 1) conv.Messages.RemoveAt(idx + 1);
            Session?.ClearKv();
            if (Session is not null) await GenerateAssistantReplyAsync(conv);
            else SaveConversations();
        }
        else
        {
            // Assistant edit: overwrite in place, no regeneration.
            conv.UpdatedAt = DateTimeOffset.UtcNow;
            SaveConversations();
        }
    }

    [RelayCommand]
    private void DeleteMessage(MessageViewModel? msg)
    {
        if (msg is null || SelectedConversation is null) return;
        SelectedConversation.Messages.Remove(msg);
        Session?.ClearKv();
        SaveConversations();
    }

    /// <summary>
    /// Remove the given assistant message (and anything after it) and
    /// re-stream a new reply against the preceding transcript. If called on
    /// a user message we re-generate the assistant reply *to* that user turn.
    /// </summary>
    [RelayCommand]
    private async Task RegenerateMessageAsync(MessageViewModel? msg)
    {
        if (msg is null || Session is null || SelectedConversation is null) return;
        var conv = SelectedConversation;
        var idx = conv.Messages.IndexOf(msg);
        if (idx < 0) return;

        // Truncate: drop this message (assistant or user) and everything past
        // it. If it was a user turn we leave nothing for the model to reply
        // to, which we guard against below.
        var truncateFrom = msg.IsUser ? idx + 1 : idx;
        while (conv.Messages.Count > truncateFrom) conv.Messages.RemoveAt(truncateFrom);

        if (conv.Messages.Count == 0 || !conv.Messages[^1].IsUser)
        {
            StatusText = "Nothing to regenerate from.";
            return;
        }

        Session.ClearKv();
        await GenerateAssistantReplyAsync(conv);
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
        var vm = new SettingsWindowViewModel(Profiles, AppSettings);
        await DialogService.ShowSettingsAsync(vm);

        // Persist both stores. Silent catch: neither is fatal, and the
        // Settings window's explicit Save button covers the intentional case.
        try { ProfileStore.Save(Profiles.Select(p => p.ToProfile())); } catch { }
        try { AppSettingsStore.Save(AppSettings.ToModel()); } catch { }

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
