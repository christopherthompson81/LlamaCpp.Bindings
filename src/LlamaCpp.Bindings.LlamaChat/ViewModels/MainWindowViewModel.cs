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
    [NotifyPropertyChangedFor(nameof(IsModelLoaded), nameof(CanSend),
                              nameof(CanAttachMedia), nameof(CanAttachImages), nameof(CanAttachAudio),
                              nameof(CanRecordAudio))]
    [NotifyCanExecuteChangedFor(nameof(SendCommand), nameof(LoadCommand),
                                 nameof(UnloadCommand), nameof(ShowModelInfoCommand),
                                 nameof(AttachMediaCommand), nameof(ToggleRecordCommand))]
    private ChatSession? _session;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanSend), nameof(CanRecordAudio))]
    [NotifyCanExecuteChangedFor(nameof(SendCommand), nameof(LoadCommand), nameof(ToggleRecordCommand))]
    private bool _isBusy;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanSend), nameof(CanCancel), nameof(CanRecordAudio))]
    [NotifyCanExecuteChangedFor(nameof(SendCommand), nameof(CancelCommand), nameof(UnloadCommand), nameof(ToggleRecordCommand))]
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

    // ============================================================
    // Microphone capture (compose-bar record button)
    // ============================================================

    private AudioRecorder? _audioRecorder;

    /// <summary>True while the record button is active and the mic is capturing.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SendCommand), nameof(ToggleRecordCommand))]
    [NotifyPropertyChangedFor(nameof(CanSend))]
    private bool _isRecording;

    /// <summary>Live "mm:ss" display updated by <see cref="_recordingTimer"/>.</summary>
    [ObservableProperty] private string _recordingDuration = "0:00";

    /// <summary>Gates the record button — mic only makes sense for audio-capable models.</summary>
    public bool CanRecordAudio => CanAttachAudio && !IsGenerating && !IsBusy;

    private readonly Avalonia.Threading.DispatcherTimer _recordingTimer = new()
    {
        Interval = TimeSpan.FromMilliseconds(250),
    };

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
        && !IsRecording
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
    /// <summary>Active MCP servers — shown as a compact avatar strip in the toolbar.</summary>
    public ObservableCollection<McpServerEntry> ActiveMcpServers { get; } = new();

    public MainWindowViewModel()
    {
        PendingAttachments.CollectionChanged += (_, _) =>
        {
            OnPropertyChanged(nameof(HasPendingAttachments));
        };

        // Start connecting to MCP servers in the background.
        _ = McpClientService.Instance.LoadAndConnectAsync();
        McpClientService.Instance.StateChanged += (_, _) =>
        {
            Dispatcher.UIThread.Post(RefreshActiveMcpServers);
        };
        RefreshActiveMcpServers();

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

        _recordingTimer.Tick += (_, _) =>
        {
            if (_audioRecorder is null) return;
            var t = _audioRecorder.Elapsed;
            RecordingDuration = $"{(int)t.TotalMinutes}:{t.Seconds:D2}";
        };
    }

    private void RefreshActiveMcpServers()
    {
        ActiveMcpServers.Clear();
        foreach (var s in McpClientService.Instance.Servers)
        {
            if (s.State == McpConnectionState.Ready) ActiveMcpServers.Add(s);
        }
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
    private async Task DeleteConversationAsync(ConversationViewModel? conv)
    {
        conv ??= SelectedConversation;
        if (conv is null) return;

        var title = string.IsNullOrWhiteSpace(conv.Title) ? "Untitled conversation" : conv.Title;
        var turnCount = conv.AllMessages.Count;
        var body = turnCount switch
        {
            0 => $"Delete \"{title}\"?",
            1 => $"Delete \"{title}\"? (1 turn)",
            _ => $"Delete \"{title}\"? ({turnCount} turns)",
        };
        var choice = await DialogService.ConfirmAsync("Delete conversation", body, new[]
        {
            ("cancel", "Cancel", false, false),
            ("delete", "Delete", true, true),
        });
        if (choice != "delete") return;

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
        conv.AppendToActivePath(user);
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
        conv.AppendToActivePath(assistant);
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
                Role: m.Role switch
                {
                    "user" => TurnRole.User,
                    "tool" => TurnRole.Tool,
                    _ => TurnRole.Assistant,
                },
                Content: m.Content,
                State: TurnState.Complete,
                CreatedAt: DateTimeOffset.UtcNow,
                Attachments: m.Attachments.Count > 0
                    ? new List<Attachment>(m.Attachments)
                    : null)));

        var tools = McpClientService.Instance.BuildToolsForTemplate();

        await StreamIntoMessageAsync(
            conv, assistant,
            ct => Session.StreamAssistantReplyAsync(transcript, sampler, gen, tools, ct));

        // Tool-calling loop: if the model emitted any <tool_call> blocks,
        // execute them via MCP, append tool-role messages with the results,
        // and re-generate. Bounded by ToolCallMaxRounds to prevent runaway
        // loops on a model that keeps asking for tools forever.
        await MaybeExecuteToolCallsAsync(conv, assistant);
    }

    private const int ToolCallMaxRounds = 6;

    private async Task MaybeExecuteToolCallsAsync(ConversationViewModel conv, MessageViewModel assistant)
    {
        for (int round = 0; round < ToolCallMaxRounds; round++)
        {
            if (Session is null) return;
            var calls = ToolCallParser.Extract(assistant.Content, Session.ToolCallFormat);
            if (calls.Count == 0) return;

            foreach (var call in calls)
            {
                var resolved = McpClientService.Instance.ResolveToolCall(call.Name);
                string resultText;
                if (resolved is null)
                {
                    resultText = $"(error: no MCP server advertises tool '{call.Name}')";
                }
                else
                {
                    try
                    {
                        var res = await McpClientService.Instance.CallToolAsync(
                            resolved.Value.Server, resolved.Value.ToolName, call.Arguments,
                            _generationCts?.Token ?? System.Threading.CancellationToken.None);
                        // Collapse the MCP envelope down to its text content
                        // for the bubble. Re-prompting the model uses this
                        // same text, which matches what most tool-use
                        // templates expect. The full wire envelope is still
                        // visible in the MCP execution-log dialog if the
                        // user needs to debug a response.
                        resultText = Services.ToolCall.ToolCallDisplay.FormatMcpResult(res);
                    }
                    catch (Exception ex)
                    {
                        resultText = $"(error: {ex.Message})";
                    }
                }

                conv.AppendToActivePath(new MessageViewModel
                {
                    Role = "tool",
                    ToolName = call.Name,
                    Content = resultText,
                });
            }

            // Trigger one more generation round with the new tool results in
            // the transcript. The loop re-enters Extract with fresh content,
            // exiting cleanly once the assistant stops requesting tools.
            var next = new MessageViewModel
            {
                Role = "assistant",
                IsStreaming = true,
                IsReasoningExpanded = AppSettings.ShowReasoningInProgress,
            };
            conv.AppendToActivePath(next);

            var sampler = _activeLoadedProfile!.SamplerPanel.SnapshotSampler();
            var gen = _activeLoadedProfile.SamplerPanel.SnapshotGeneration();
            var transcript = BuildTranscriptFor(conv);
            var tools = McpClientService.Instance.BuildToolsForTemplate();

            Session.ClearKv();
            await StreamIntoMessageAsync(
                conv, next,
                ct => Session.StreamAssistantReplyAsync(transcript, sampler, gen, tools, ct));

            assistant = next;
        }

        ToastService.Warning("Tool loop",
            $"Hit {ToolCallMaxRounds}-round cap on tool calls — stopping.");
    }

    private List<ChatTurn> BuildTranscriptFor(ConversationViewModel conv)
    {
        var transcript = new List<ChatTurn>();
        if (!string.IsNullOrWhiteSpace(_activeLoadedProfile?.SystemPrompt))
        {
            transcript.Add(new ChatTurn(
                Id: Guid.NewGuid(),
                Role: TurnRole.System,
                Content: _activeLoadedProfile.SystemPrompt,
                State: TurnState.Complete,
                CreatedAt: DateTimeOffset.UtcNow));
        }
        // Skip the trailing empty/streaming assistant placeholder.
        transcript.AddRange(conv.Messages
            .Take(conv.Messages.Count - 1)
            .Select(m => new ChatTurn(
                Id: Guid.NewGuid(),
                Role: m.Role switch
                {
                    "user" => TurnRole.User,
                    "tool" => TurnRole.Tool,
                    _ => TurnRole.Assistant,
                },
                Content: m.Content,
                State: TurnState.Complete,
                CreatedAt: DateTimeOffset.UtcNow,
                Attachments: m.Attachments.Count > 0
                    ? new List<Attachment>(m.Attachments)
                    : null)));
        return transcript;
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
                    case StreamEvent.Language lang:
                        // Fires at most once per turn. Marshal directly — no
                        // batching needed since there's only one event.
                        var tag = lang.Tag;
                        Dispatcher.UIThread.Post(
                            () => assistant.AsrLanguage = tag,
                            DispatcherPriority.Background);
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

            ErrorLog.Write(ex, "generation");

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
    /// on the old prompt. In the tree model this is a *branch*, not a
    /// destructive truncate — we add the edited text as a sibling user
    /// turn and then regenerate a fresh assistant reply under it. The
    /// original user turn and its assistant reply stay in the tree,
    /// reachable via the sibling-nav control. Assistant edits just
    /// overwrite text in place (no branch).
    /// </summary>
    [RelayCommand]
    private async Task CommitEditMessageAsync(MessageViewModel? msg)
    {
        if (msg is null || SelectedConversation is null) return;

        var conv = SelectedConversation;
        var newText = msg.EditDraft;

        if (msg.IsUser)
        {
            msg.IsEditing = false;
            msg.EditDraft = string.Empty;

            // Sibling user turn with the new content + copied attachments.
            var replacement = new MessageViewModel { Role = "user", Content = newText };
            foreach (var a in msg.Attachments) replacement.Attachments.Add(a);
            conv.AddSibling(msg.Id, replacement);

            Session?.ClearKv();
            if (Session is not null) await GenerateAssistantReplyAsync(conv);
            else SaveConversations();
        }
        else
        {
            // Assistant edit: overwrite in place, no regeneration, no new
            // branch — we're just correcting text the user doesn't like.
            msg.Content = newText;
            msg.IsEditing = false;
            msg.EditDraft = string.Empty;
            conv.UpdatedAt = DateTimeOffset.UtcNow;
            SaveConversations();
        }
    }

    [RelayCommand]
    private async Task DeleteMessageAsync(MessageViewModel? msg)
    {
        if (msg is null || SelectedConversation is null) return;
        var conv = SelectedConversation;

        int siblingCount = conv.GetSiblingCount(msg.Id);
        bool hasSiblings = siblingCount > 1;
        int subtreeSize = CountSubtree(conv, msg.Id);
        int descendants = subtreeSize - 1;

        // Leaf with nothing around it — a single isolated turn. Obvious
        // delete, no prompt worth interrupting the user for.
        if (!hasSiblings && descendants == 0)
        {
            conv.RemoveSubtree(msg.Id);
            Session?.ClearKv();
            SaveConversations();
            return;
        }

        // Build options driven by the tree shape around this message. With
        // siblings, the interesting choice is "this version" (drop this
        // branch) vs "all versions" (drop every alternative at this fork
        // point). Without siblings, only the subtree-delete is meaningful.
        var options = new List<(string Key, string Label, bool Destructive, bool Primary)>
        {
            ("cancel", "Cancel", false, false),
        };

        string msgBody;
        int allVersionsTurnCount = 0;
        if (hasSiblings)
        {
            var plurSelf = subtreeSize == 1 ? "turn" : "turns";
            options.Add(("this", $"Delete this version ({subtreeSize} {plurSelf})", true, true));

            // Footprint of nuking every alternative branch at this fork.
            foreach (var sib in GetSiblings(conv, msg.Id))
            {
                allVersionsTurnCount += CountSubtree(conv, sib.Id);
            }
            var plurAll = allVersionsTurnCount == 1 ? "turn" : "turns";
            options.Add(("all", $"Delete all {siblingCount} versions ({allVersionsTurnCount} {plurAll})", true, false));

            msgBody = descendants > 0
                ? $"This message has {siblingCount - 1} sibling branch(es). Its own branch contains {descendants} descendant(s)."
                : $"This message has {siblingCount - 1} sibling branch(es).";
        }
        else
        {
            var plur = subtreeSize == 1 ? "turn" : "turns";
            options.Add(("subtree", $"Delete subtree ({subtreeSize} {plur})", true, true));
            msgBody = $"This message has {descendants} descendant(s) below it in the tree.";
        }

        var choice = await DialogService.ConfirmAsync("Delete message", msgBody, options);

        switch (choice)
        {
            case "this":
            case "subtree":
                conv.RemoveSubtree(msg.Id);
                break;
            case "all":
                // Snapshot the sibling ids before we start mutating —
                // RemoveSubtree edits AllMessages, which would invalidate
                // a lazy enumeration mid-iteration.
                var siblingIds = GetSiblings(conv, msg.Id)
                    .Select(s => s.Id)
                    .ToList();
                foreach (var id in siblingIds)
                {
                    conv.RemoveSubtree(id);
                }
                break;
            default:
                return;
        }

        Session?.ClearKv();
        SaveConversations();
    }

    private static IEnumerable<MessageViewModel> GetSiblings(ConversationViewModel conv, Guid messageId)
    {
        var node = conv.FindById(messageId);
        if (node is null) yield break;
        foreach (var m in conv.AllMessages)
        {
            if (m.ParentId == node.ParentId) yield return m;
        }
    }

    private static int CountSubtree(ConversationViewModel conv, Guid rootId)
    {
        var ids = new HashSet<Guid> { rootId };
        bool grew;
        do
        {
            grew = false;
            foreach (var m in conv.AllMessages)
            {
                if (m.ParentId is { } p && ids.Contains(p) && ids.Add(m.Id)) grew = true;
            }
        } while (grew);
        return ids.Count;
    }

    /// <summary>
    /// Regenerate a reply. In the tree model this creates a sibling
    /// (parented to the target's parent) rather than truncating, so the
    /// original reply stays reachable via sibling-nav. Called on an
    /// assistant turn: add a new assistant sibling with the same parent
    /// user turn. Called on a user turn: same idea — add an assistant
    /// sibling as a direct child, preserving whichever assistant reply
    /// was already there.
    /// </summary>
    [RelayCommand]
    private async Task RegenerateMessageAsync(MessageViewModel? msg)
    {
        if (msg is null || Session is null || SelectedConversation is null) return;
        var conv = SelectedConversation;

        // Determine the parent-user turn the new assistant reply should
        // hang from. If msg is assistant → its parent is the user. If
        // msg is user → msg itself is the user.
        var userAnchorId = msg.IsAssistant ? msg.ParentId : msg.Id;
        if (userAnchorId is null)
        {
            StatusText = "Nothing to regenerate from.";
            return;
        }

        // Hang the new assistant bubble off the user anchor — if there's
        // already an assistant reply on that anchor, the two become
        // siblings, and the new one wins as the active leaf.
        var newAssistant = new MessageViewModel
        {
            Role = "assistant",
            IsStreaming = true,
            IsReasoningExpanded = AppSettings.ShowReasoningInProgress,
        };
        conv.AddChildOf(userAnchorId, newAssistant);

        Session.ClearKv();
        await StreamIntoNewAssistantAsync(conv, newAssistant);
    }

    /// <summary>
    /// Drive the generation stream into an already-placed assistant
    /// bubble. Peels the nested-await shape of
    /// <see cref="GenerateAssistantReplyAsync"/> apart so the regenerate
    /// path can use it directly on a node it already parented into the tree.
    /// </summary>
    private async Task StreamIntoNewAssistantAsync(
        ConversationViewModel conv, MessageViewModel assistant)
    {
        if (Session is null || _activeLoadedProfile is null) return;

        conv.UpdatedAt = DateTimeOffset.UtcNow;
        RebuildFilteredConversations();

        var sampler = _activeLoadedProfile.SamplerPanel.SnapshotSampler();
        var gen = _activeLoadedProfile.SamplerPanel.SnapshotGeneration();

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
        // Active path minus the streaming assistant placeholder.
        transcript.AddRange(conv.Messages
            .Take(conv.Messages.Count - 1)
            .Select(m => new ChatTurn(
                Id: Guid.NewGuid(),
                Role: m.Role switch
                {
                    "user" => TurnRole.User,
                    "tool" => TurnRole.Tool,
                    _ => TurnRole.Assistant,
                },
                Content: m.Content,
                State: TurnState.Complete,
                CreatedAt: DateTimeOffset.UtcNow,
                Attachments: m.Attachments.Count > 0
                    ? new List<Attachment>(m.Attachments)
                    : null)));

        var tools = McpClientService.Instance.BuildToolsForTemplate();

        await StreamIntoMessageAsync(
            conv, assistant,
            ct => Session.StreamAssistantReplyAsync(transcript, sampler, gen, tools, ct));

        await MaybeExecuteToolCallsAsync(conv, assistant);
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
    /// Generate an assistant reply to a dangling user turn — the case
    /// where the active path's leaf is a user message with no response
    /// (e.g. after deleting the previous assistant reply, or cancelling
    /// mid-stream before any tokens landed). The remedy card in the
    /// compose area binds to this command and is only visible when
    /// <see cref="ConversationViewModel.NeedsAssistantReply"/> is true.
    /// </summary>
    [RelayCommand]
    private async Task GenerateReplyAsync()
    {
        if (Session is null || SelectedConversation is null) return;
        if (IsGenerating) return;
        if (!SelectedConversation.NeedsAssistantReply) return;
        await GenerateAssistantReplyAsync(SelectedConversation);
    }

    /// <summary>
    /// Switch to the previous branch at this point in the tree. Finds the
    /// sibling just before <paramref name="msg"/> (cyclic), then points
    /// ActiveLeafId at that sibling's deepest descendant — restoring the
    /// full path through that branch, not just the single turn.
    /// </summary>
    [RelayCommand]
    private void SwitchPrevSibling(MessageViewModel? msg)
    {
        if (msg is null || SelectedConversation is null) return;
        var prev = SelectedConversation.PrevSibling(msg.Id);
        if (prev is null) return;
        SelectedConversation.SwitchToSibling(prev.Id);
        Session?.ClearKv();
        SaveConversations();
    }

    [RelayCommand]
    private void SwitchNextSibling(MessageViewModel? msg)
    {
        if (msg is null || SelectedConversation is null) return;
        var next = SelectedConversation.NextSibling(msg.Id);
        if (next is null) return;
        SelectedConversation.SwitchToSibling(next.Id);
        Session?.ClearKv();
        SaveConversations();
    }

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
                    SelectedConversation.ClearAll();
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
    private async Task ClearConversationAsync()
    {
        if (SelectedConversation is null) return;
        var count = SelectedConversation.AllMessages.Count;
        if (count == 0) return;

        var body = count == 1
            ? "Clear this conversation? (1 turn)"
            : $"Clear this conversation? ({count} turns)";
        var choice = await DialogService.ConfirmAsync("Clear conversation", body, new[]
        {
            ("cancel", "Cancel", false, false),
            ("clear", "Clear", true, true),
        });
        if (choice != "clear") return;

        SelectedConversation.ClearAll();
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
    private async Task InsertMcpPromptAsync()
    {
        var text = await DialogService.ShowMcpPromptPickerAsync();
        if (string.IsNullOrEmpty(text)) return;
        UserInput = string.IsNullOrEmpty(UserInput) ? text : UserInput + "\n" + text;
    }

    [RelayCommand]
    private async Task AttachMcpResourceAsync()
    {
        var picked = await DialogService.ShowMcpResourceBrowserAsync();
        if (picked is null) return;
        var (uri, content) = picked.Value;
        var block = $"<!-- resource: {uri} -->\n{content}";
        UserInput = string.IsNullOrEmpty(UserInput) ? block : UserInput + "\n\n" + block;
    }

    [RelayCommand]
    private async Task ShowMcpLogAsync() => await DialogService.ShowMcpExecutionLogAsync();

    [RelayCommand]
    private async Task ShowShortcutsAsync() => await DialogService.ShowShortcutsAsync();

    [RelayCommand]
    private async Task ShowAboutAsync() => await DialogService.ShowAboutAsync();

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
    // Mic recording
    // ============================================================

    /// <summary>
    /// Start or stop microphone capture. On stop, the captured PCM is wrapped
    /// in a WAV header and dropped into <see cref="PendingAttachments"/> — from
    /// there it travels the same path as an uploaded audio file, so ASR / omni
    /// models receive it through the existing mtmd pipeline.
    /// </summary>
    [RelayCommand(CanExecute = nameof(CanToggleRecord))]
    private async Task ToggleRecordAsync()
    {
        if (!IsRecording)
        {
            try
            {
                _audioRecorder?.Dispose();
                _audioRecorder = new AudioRecorder();
                _audioRecorder.Start();
                IsRecording = true;
                RecordingDuration = "0:00";
                _recordingTimer.Start();
            }
            catch (Exception ex)
            {
                _audioRecorder?.Dispose();
                _audioRecorder = null;
                IsRecording = false;
                ToastService.Warning("Recording failed", ex.Message);
            }
            return;
        }

        // Stop path — snapshot state first so the timer can stop cleanly.
        _recordingTimer.Stop();
        IsRecording = false;
        var recorder = _audioRecorder;
        _audioRecorder = null;
        if (recorder is null) return;

        try
        {
            var samples = await recorder.StopAsync();
            if (samples.Length == 0)
            {
                ToastService.Warning("Recording empty", "No audio was captured.");
                return;
            }
            var wav = WavWriter.BuildPcm16(samples, AudioRecorder.SampleRate);
            var fileName = $"mic-{DateTime.Now:yyyyMMdd-HHmmss}.wav";
            AddPendingMediaBytes(wav, "audio/wav", fileName);
        }
        catch (Exception ex)
        {
            ToastService.Warning("Recording failed", ex.Message);
        }
        finally
        {
            recorder.Dispose();
        }
    }

    private bool CanToggleRecord() => IsRecording || CanRecordAudio;

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
        _recordingTimer.Stop();
        _audioRecorder?.Dispose();
        _audioRecorder = null;
        Session?.Dispose();
        SaveConversations();
    }
}
