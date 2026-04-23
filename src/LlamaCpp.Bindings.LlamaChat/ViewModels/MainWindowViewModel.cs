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
    [NotifyPropertyChangedFor(nameof(IsModelLoaded), nameof(CanSend), nameof(CanAttachMedia))]
    [NotifyCanExecuteChangedFor(nameof(SendCommand), nameof(LoadCommand),
                                 nameof(UnloadCommand), nameof(ShowModelInfoCommand),
                                 nameof(AttachMediaCommand))]
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

    /// <summary>Token count of <see cref="UserInput"/> for the currently loaded vocab, debounced.</summary>
    [ObservableProperty] private int _userInputTokenCount;

    /// <summary>
    /// Images the user has added via the attach button, drag-drop, or clipboard
    /// paste for the next send. Drained on <see cref="SendAsync"/>; each image
    /// moves from here onto the outgoing user <see cref="MessageViewModel"/>.
    /// </summary>
    public ObservableCollection<Attachment> PendingAttachments { get; } = new();

    public bool HasPendingAttachments => PendingAttachments.Count > 0;

    /// <summary>Disables the paperclip when the loaded model can't consume any media.</summary>
    public bool CanAttachMedia => Session?.SupportsMedia == true;

    /// <summary>
    /// File-picker filter hints. Image models get image filters; audio models
    /// get audio filters; omni models get both.
    /// </summary>
    public bool CanAttachImages => Session?.SupportsImages == true;
    public bool CanAttachAudio => Session?.SupportsAudio == true;

    private readonly Avalonia.Threading.DispatcherTimer _tokenCountTimer = new()
    {
        Interval = TimeSpan.FromMilliseconds(150),
    };

    partial void OnUserInputChanged(string value)
    {
        // Restart the debounce window — tokenisation is cheap but touching
        // native vocab for every keystroke is wasteful.
        _tokenCountTimer.Stop();
        _tokenCountTimer.Start();
    }

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
        PendingAttachments.CollectionChanged += (_, _) =>
        {
            OnPropertyChanged(nameof(HasPendingAttachments));
        };

        AppSettings = new AppSettingsViewModel(AppSettingsStore.Load());
        // Apply theme once on startup based on the persisted setting.
        // Subsequent changes through the Settings panel are caught by
        // AppSettingsViewModel.OnThemeModeChanged.
        ThemeService.Apply(AppSettings.ThemeMode);

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

        _tokenCountTimer.Tick += (_, _) =>
        {
            _tokenCountTimer.Stop();
            RecomputeUserInputTokens();
        };
    }

    private void RecomputeUserInputTokens()
    {
        if (Session is null || string.IsNullOrEmpty(UserInput))
        {
            UserInputTokenCount = 0;
            return;
        }
        try
        {
            var toks = Session.Model.Vocab.Tokenize(UserInput, addSpecial: false, parseSpecial: true);
            UserInputTokenCount = toks.Length;
        }
        catch
        {
            UserInputTokenCount = 0;
        }
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
            .OrderByDescending(c => c.Pinned)
            .ThenByDescending(c => c.UpdatedAt);
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
            ToastService.Success("Model loaded", profile.Name);
        }
        catch (Exception ex)
        {
            var tail = string.Join(" | ", _recentNativeLogLines.TakeLast(6));
            StatusText = tail.Length > 0
                ? $"Load failed: {ex.Message}\nNative log: {tail}"
                : $"Load failed: {ex.Message}";
            ToastService.Error("Load failed", ex.Message);
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
    private void TogglePinned(ConversationViewModel? conv)
    {
        conv ??= SelectedConversation;
        if (conv is null) return;
        conv.Pinned = !conv.Pinned;
        RebuildFilteredConversations();
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

        // Intercept slash commands before sending to the model. Unknown
        // `/commands` fall through to being sent as a normal message.
        if (text.StartsWith('/') && TryHandleSlashCommand(text))
        {
            UserInput = string.Empty;
            return;
        }

        UserInput = string.Empty;

        var conv = SelectedConversation;
        var user = new MessageViewModel { Role = "user", Content = text };
        // Move pending attachments onto the outgoing message. We drain the
        // pending list so the paperclip strip empties and the next compose
        // starts fresh.
        if (PendingAttachments.Count > 0 && CanAttachMedia)
        {
            foreach (var a in PendingAttachments) user.Attachments.Add(a);
        }
        PendingAttachments.Clear();
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
                CreatedAt: DateTimeOffset.UtcNow,
                Attachments: m.Attachments.Count > 0
                    ? new List<Attachment>(m.Attachments)
                    : null)));

        await StreamIntoMessageAsync(
            conv, assistant,
            ct => Session.StreamAssistantReplyAsync(transcript, sampler, gen, ct));
    }

    /// <summary>
    /// Shared stream-consumer: drives the flush pump, marshalling, timing,
    /// and error handling for any <see cref="StreamEvent"/> source that's
    /// targeting a single assistant bubble. Used by both the fresh-turn
    /// path and the "Continue" path — the only difference between them is
    /// what source delegate they hand in.
    /// </summary>
    private async Task StreamIntoMessageAsync(
        ConversationViewModel conv,
        MessageViewModel assistant,
        Func<CancellationToken, IAsyncEnumerable<StreamEvent>> source)
    {
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
            // .ConfigureAwait(false): each MoveNextAsync resumes on a pool
            // thread rather than posting back to the UI Dispatcher. That
            // eliminates a pool→UI→pool round-trip per token, which is
            // ~0.5-1 ms of Dispatcher-queue latency on hot paths — the
            // suspected source of the 8 tok/s gap vs. llama-server at this
            // workload size.
            await foreach (var evt in source(_generationCts.Token).ConfigureAwait(false))
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
            ToastService.Error($"Generation failed: {ex.GetType().Name}",
                $"{ex.Message} — see ~/.config/LlamaChat/last-error.log for the full stack.");
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
        ToastService.Success("Copied", $"{msg.Content.Length} char(s) on the clipboard.");
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
    private async Task DeleteMessageAsync(MessageViewModel? msg)
    {
        if (msg is null || SelectedConversation is null) return;
        var conv = SelectedConversation;
        var idx = conv.Messages.IndexOf(msg);
        if (idx < 0) return;

        var downstreamCount = conv.Messages.Count - idx - 1;

        // Single message with nothing after it: delete straight away, no
        // need to interrupt the user with a dialog for an obvious case.
        if (downstreamCount == 0)
        {
            conv.Messages.RemoveAt(idx);
            Session?.ClearKv();
            SaveConversations();
            return;
        }

        var choice = await DialogService.ConfirmAsync(
            "Delete message",
            $"There {(downstreamCount == 1 ? "is" : "are")} {downstreamCount} message(s) after this one. " +
            "Keeping them would leave the conversation inconsistent with the deleted turn.",
            new (string, string, bool, bool)[]
            {
                ("cancel",     "Cancel",                        false, false),
                ("just",       "Just this",                     true,  false),
                ("downstream", $"This + {downstreamCount} after", true,  true),
            });

        switch (choice)
        {
            case "just":
                conv.Messages.RemoveAt(idx);
                break;
            case "downstream":
                while (conv.Messages.Count > idx) conv.Messages.RemoveAt(idx);
                break;
            default:
                return;   // cancelled
        }

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

    /// <summary>
    /// Extend the last assistant message without re-rendering or re-tokenising —
    /// picks up from the current KV cache. Useful when the reply hit
    /// <c>MaxTokens</c> (truncated) or the user cancelled mid-stream and now
    /// wants more output conditioned on what's already there. Only valid for
    /// the last message in the conversation: older messages don't correspond
    /// to the current KV state.
    /// </summary>
    [RelayCommand]
    private async Task ContinueMessageAsync(MessageViewModel? msg)
    {
        if (msg is null || Session is null || _activeLoadedProfile is null) return;
        if (SelectedConversation is null) return;
        if (!msg.IsAssistant) return;
        // Only the last message can be continued — older ones don't match KV state.
        var conv = SelectedConversation;
        if (conv.Messages.LastOrDefault() != msg)
        {
            ToastService.Warning("Continue", "Can only continue the last message.");
            return;
        }
        if (msg.IsStreaming || IsGenerating)
        {
            return;
        }

        // Strip the "(cancelled)" placeholder we put in Content when a turn
        // was cancelled with zero visible output — otherwise the continuation
        // text appends after it and the bubble reads "(cancelled)X Y Z...".
        if (msg.Content == "(cancelled)") msg.Content = string.Empty;

        // Heuristic: if the previous turn accumulated reasoning text but
        // never produced any regular content, the model was still inside
        // the <think> block when it stopped. Resume the extractor there so
        // the continuation's stray </think> marker is consumed by the state
        // machine instead of landing in the Content stream (which would
        // break markdown rendering downstream — CommonMark reads a loose
        // </think> as an HTML-block opener).
        bool resumeInReasoning =
            !string.IsNullOrEmpty(msg.Reasoning) && string.IsNullOrEmpty(msg.Content);

        msg.IsStreaming = true;

        var sampler = _activeLoadedProfile.SamplerPanel.SnapshotSampler();
        var gen = _activeLoadedProfile.SamplerPanel.SnapshotGeneration();

        await StreamIntoMessageAsync(
            conv, msg,
            ct => Session.StreamContinuationAsync(sampler, gen, resumeInReasoning, ct));
    }

    [RelayCommand(CanExecute = nameof(CanCancel))]
    private void Cancel() => _generationCts?.Cancel();

    /// <summary>
    /// Handle a "/command" typed in the compose box. Returns true if the
    /// command matched; false to let <see cref="SendAsync"/> send the
    /// text through to the model as-is.
    /// </summary>
    private bool TryHandleSlashCommand(string text)
    {
        var space = text.IndexOf(' ');
        var cmd = (space > 0 ? text[..space] : text).ToLowerInvariant();
        switch (cmd)
        {
            case "/clear":
            case "/reset":
                if (SelectedConversation is not null)
                {
                    SelectedConversation.Messages.Clear();
                    Session?.ClearKv();
                    SaveConversations();
                    ToastService.Info("Conversation cleared");
                }
                return true;
            case "/new":
                NewConversation();
                return true;
            case "/settings":
                _ = OpenSettingsAsync();
                return true;
            case "/help":
            case "/?":
                _ = DialogService.ShowShortcutsAsync();
                return true;
            case "/copy":
                // Copy the last assistant message.
                var last = SelectedConversation?.Messages.LastOrDefault(m => m.IsAssistant);
                if (last is not null)
                {
                    _ = CopyMessageAsync(last);
                }
                else
                {
                    ToastService.Warning("/copy", "No assistant message to copy yet.");
                }
                return true;
            default:
                // Unknown slash command — let it flow through as a normal message.
                // Heuristic: if the user clearly typed a command (starts with /
                // + only letters/digits, no spaces), surface it as a warning
                // so they know the command wasn't recognised.
                if (space < 0)
                {
                    ToastService.Warning("Unknown command", $"{cmd} — try /clear, /new, /settings, /help, /copy");
                    return true;
                }
                return false;
        }
    }

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
    private async Task ShowShortcutsAsync() => await DialogService.ShowShortcutsAsync();

    [RelayCommand]
    private async Task ExportConversationsAsync()
    {
        var path = await DialogService.PickExportFileAsync();
        if (string.IsNullOrEmpty(path)) return;
        try
        {
            ConversationStore.ExportToFile(
                Conversations.Select(c => c.ToModel()), path);
            ToastService.Success("Exported", $"{Conversations.Count} conversation(s) → {Path.GetFileName(path)}");
        }
        catch (Exception ex)
        {
            ToastService.Error("Export failed", ex.Message);
        }
    }

    [RelayCommand]
    private async Task ImportConversationsAsync()
    {
        var path = await DialogService.PickImportFileAsync();
        if (string.IsNullOrEmpty(path)) return;
        try
        {
            var imported = ConversationStore.ImportFromFile(path);
            var existing = new HashSet<Guid>(Conversations.Select(c => c.Id));
            var added = 0;
            foreach (var c in imported)
            {
                if (existing.Contains(c.Id)) continue;    // skip duplicates by id
                Conversations.Add(new ConversationViewModel(c));
                added++;
            }
            RebuildFilteredConversations();
            SaveConversations();
            ToastService.Success("Imported",
                added == imported.Count
                    ? $"{added} conversation(s) from {Path.GetFileName(path)}"
                    : $"{added} of {imported.Count} (rest already present)");
        }
        catch (Exception ex)
        {
            ToastService.Error("Import failed", ex.Message);
        }
    }

    [RelayCommand(CanExecute = nameof(IsModelLoaded))]
    private async Task ShowModelInfoAsync()
    {
        if (Session is null) return;
        await DialogService.ShowModelInfoAsync(Session.Model, _activeLoadedProfile?.Name);
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
    // Attachments
    // ============================================================

    /// <summary>
    /// Open a media file picker and queue the selected files as pending
    /// attachments. Filter extensions depend on which modalities the loaded
    /// model advertises (images only, audio only, or both).
    /// </summary>
    [RelayCommand(CanExecute = nameof(CanAttachMedia))]
    private async Task AttachMediaAsync()
    {
        var paths = await DialogService.PickMediaFilesAsync(
            allowImages: CanAttachImages, allowAudio: CanAttachAudio);
        foreach (var p in paths)
        {
            TryAddPendingMedia(p);
        }
    }

    [RelayCommand]
    private void RemovePendingAttachment(Attachment? a)
    {
        if (a is null) return;
        PendingAttachments.Remove(a);
    }

    /// <summary>
    /// Add a media file to the pending list — used by the file picker, drag-
    /// drop, and clipboard paste. Guessed MIME from extension; rejects
    /// anything that isn't image or audio, or that the loaded model can't
    /// consume (an audio-only model dropping an image, or vice versa).
    /// </summary>
    internal void TryAddPendingMedia(string path)
    {
        if (!File.Exists(path)) return;
        try
        {
            var bytes = File.ReadAllBytes(path);
            var ext = Path.GetExtension(path).ToLowerInvariant();
            var mime = ext switch
            {
                ".jpg" or ".jpeg" => "image/jpeg",
                ".png"            => "image/png",
                ".gif"            => "image/gif",
                ".bmp"            => "image/bmp",
                ".webp"           => "image/webp",
                ".wav"            => "audio/wav",
                ".mp3"            => "audio/mpeg",
                ".flac"           => "audio/flac",
                ".ogg"            => "audio/ogg",
                ".m4a"            => "audio/mp4",
                _                 => "application/octet-stream",
            };
            bool isImage = mime.StartsWith("image/");
            bool isAudio = mime.StartsWith("audio/");
            if (!isImage && !isAudio) return;
            if (isImage && !CanAttachImages)
            {
                ToastService.Warning("Attach rejected", "Loaded model doesn't consume images.");
                return;
            }
            if (isAudio && !CanAttachAudio)
            {
                ToastService.Warning("Attach rejected", "Loaded model doesn't consume audio.");
                return;
            }
            PendingAttachments.Add(new Attachment(bytes, mime, Path.GetFileName(path)));
        }
        catch (Exception ex)
        {
            ToastService.Warning("Attach failed", $"{Path.GetFileName(path)}: {ex.Message}");
        }
    }

    internal void AddPendingMediaBytes(byte[] bytes, string mime, string? fileName = null)
    {
        PendingAttachments.Add(new Attachment(bytes, mime, fileName));
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
