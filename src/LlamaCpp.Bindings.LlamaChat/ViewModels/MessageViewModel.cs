using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.Services.ToolCall;

namespace LlamaCpp.Bindings.LlamaChat.ViewModels;

public partial class MessageViewModel : ObservableObject
{
    /// <summary>
    /// Stable identity for this turn. Used as the source of
    /// <see cref="ParentId"/> links, the <c>ActiveLeafId</c> pointer on
    /// <see cref="ConversationViewModel"/>, and sibling-navigation
    /// commands. Assigned once at construction and never rotated — the
    /// persistence layer round-trips it so branches reconstruct correctly.
    /// </summary>
    public Guid Id { get; init; } = Guid.NewGuid();

    /// <summary>
    /// When this turn was first created. Preserved through load/save round
    /// trips so exports can stamp individual messages with their real wall
    /// time. Display-only — the UI does not currently render per-turn
    /// timestamps.
    /// </summary>
    public DateTimeOffset CreatedAt { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>
    /// Id of the turn immediately preceding this one in the tree. Null for
    /// the root turn of a conversation. A set of turns sharing the same
    /// <see cref="ParentId"/> are siblings — alternative branches the user
    /// can switch between via the sibling-nav control.
    /// </summary>
    [ObservableProperty] private Guid? _parentId;

    /// <summary>
    /// Back-reference to the owning conversation, set by
    /// <see cref="ConversationViewModel"/> when this message is inserted
    /// into the tree. Lets the bubble template ask its conversation about
    /// sibling counts / positions / navigation commands without threading
    /// a conversation reference through every binding path. Nullable only
    /// during the narrow window between construction and insertion.
    /// </summary>
    public ConversationViewModel? Owner { get; internal set; }

    /// <summary>Number of siblings this message has (including itself). 1 when there are no alternatives.</summary>
    public int SiblingCount => Owner?.GetSiblingCount(Id) ?? 1;

    /// <summary>1-based index of this message among its siblings, ordered by creation order.</summary>
    public int SiblingIndex => Owner?.GetSiblingIndex(Id) ?? 1;

    /// <summary>True if this message has at least one sibling branch.</summary>
    public bool HasSiblings => SiblingCount > 1;

    /// <summary>
    /// Emitted by <see cref="ConversationViewModel"/> when the tree changes
    /// in a way that could affect this message's sibling display. Fires on
    /// tree mutations (append, retry-as-sibling, subtree delete) and path
    /// switches. Message subscribers re-raise PropertyChanged for their
    /// sibling-derived bindings.
    /// </summary>
    internal void NotifySiblingInfoChanged()
    {
        OnPropertyChanged(nameof(SiblingCount));
        OnPropertyChanged(nameof(SiblingIndex));
        OnPropertyChanged(nameof(HasSiblings));
    }

    [ObservableProperty] private string _role = string.Empty;
    [ObservableProperty] private string _content = string.Empty;
    [ObservableProperty] private string? _reasoning;
    /// <summary>
    /// Detected transcription language from an ASR model (e.g. "English",
    /// "Mandarin"). Rendered as a compact chip above the content when
    /// present. Null for non-ASR models.
    /// </summary>
    [ObservableProperty] private string? _asrLanguage;
    [ObservableProperty] private bool _isReasoningExpanded = false;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanContinue))]
    private bool _isStreaming = false;
    [ObservableProperty] private string? _statsSummary;

    // Per-message performance stamps. Populated from StreamEvent.Done on
    // the UI thread once the reply finishes. Two pairs — prefill and decode
    // — plus a toggle the bubble exposes so the user can flip between
    // viewing either set (matches the book-icon toggle in llama-server webui).
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DisplayStatTokens), nameof(DisplayStatSeconds),
                              nameof(DisplayStatTokensPerSecond), nameof(HasStats))]
    private int _promptTokens;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DisplayStatTokens), nameof(DisplayStatSeconds),
                              nameof(DisplayStatTokensPerSecond), nameof(HasStats))]
    private double _promptSeconds;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DisplayStatTokens), nameof(DisplayStatSeconds),
                              nameof(DisplayStatTokensPerSecond), nameof(HasStats))]
    private int _completionTokens;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DisplayStatTokens), nameof(DisplayStatSeconds),
                              nameof(DisplayStatTokensPerSecond), nameof(HasStats))]
    private double _generationSeconds;

    /// <summary>True = show prefill stats; false = show decode stats (default).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DisplayStatTokens), nameof(DisplayStatSeconds),
                              nameof(DisplayStatTokensPerSecond), nameof(StatsModeLabel),
                              nameof(StatsModeIconKey))]
    private bool _showPrefillStats;

    /// <summary>Profile + GGUF filename snapshot taken when this reply was generated.
    /// Rendered as a chip in the stats row so the user can tell which model wrote it
    /// even after loading a different profile.</summary>
    [ObservableProperty] private string? _modelLabel;

    /// <summary>
    /// Why the streaming generation for this message ended. Drives
    /// <see cref="CanContinue"/> — if the model stopped itself with an EOG
    /// token, extending would just push past its intended stop point, so
    /// the Continue button is hidden. Other reasons (MaxTokens, cancelled,
    /// grammar-satisfied, unknown) leave Continue available.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanContinue))]
    private LlamaStopReason _stopReason = LlamaStopReason.None;

    /// <summary>
    /// Set by <see cref="ConversationViewModel"/> when the active-path tail
    /// changes; only the message at the end of the active path is eligible
    /// for Continue (older messages don't correspond to the current KV state).
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanContinue))]
    private bool _isLastInActivePath;

    /// <summary>
    /// True when this message is the anchor for the session's current KV
    /// cache — i.e. extending it via <c>StreamContinuationAsync</c> is
    /// semantically valid. Set on the just-finished assistant at Done;
    /// cleared on session load / unload / ClearKv. Survives through
    /// conversation save/reload as false, which is what we want: reloading
    /// the app leaves the KV empty, so Continue must be unavailable until
    /// the user runs a fresh turn.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanContinue))]
    private bool _isKvContinuable;

    /// <summary>
    /// Continue is offered only when:
    ///   1. this is an assistant message,
    ///   2. it's not currently streaming,
    ///   3. it sits at the active-path tail,
    ///   4. the stop reason wasn't a model-defined EOG token, and
    ///   5. the session's KV cache still matches this message (no app
    ///      restart, session reload, or intervening ClearKv since it was
    ///      generated).
    /// </summary>
    public bool CanContinue =>
        IsAssistant
        && !IsStreaming
        && IsLastInActivePath
        && IsKvContinuable
        && StopReason != LlamaStopReason.EndOfGeneration;

    public bool HasStats =>
        CompletionTokens > 0 || (ShowPrefillStats && PromptTokens > 0);

    public int DisplayStatTokens => ShowPrefillStats ? PromptTokens : CompletionTokens;
    public double DisplayStatSeconds => ShowPrefillStats ? PromptSeconds : GenerationSeconds;
    public double DisplayStatTokensPerSecond =>
        DisplayStatSeconds > 0 ? DisplayStatTokens / DisplayStatSeconds : 0;

    public string StatsModeLabel => ShowPrefillStats ? "Prefill" : "Generation";
    /// <summary>Lucide icon key for the mode-toggle button.</summary>
    public string StatsModeIconKey => ShowPrefillStats ? "IconEye" : "IconPlay";

    [RelayCommand]
    private void ToggleStatsMode() => ShowPrefillStats = !ShowPrefillStats;

    /// <summary>True while the inline-edit textarea is shown for this bubble.</summary>
    [ObservableProperty] private bool _isEditing;

    /// <summary>
    /// Working copy of <see cref="Content"/> while editing. Committed or
    /// discarded by the command that ends the edit — the VM never writes
    /// through to <see cref="Content"/> directly from the textarea binding.
    /// </summary>
    [ObservableProperty] private string _editDraft = string.Empty;

    /// <summary>
    /// Images (and later, audio) attached to this turn. Observable so the
    /// bubble template refreshes when attachments are pruned during an edit.
    /// </summary>
    public ObservableCollection<Attachment> Attachments { get; } = new();

    public bool HasAttachments => Attachments.Count > 0;

    public MessageViewModel()
    {
        Attachments.CollectionChanged += (_, _) => OnPropertyChanged(nameof(HasAttachments));
    }

    public bool HasReasoning => !string.IsNullOrEmpty(Reasoning);
    public bool HasAsrLanguage => !string.IsNullOrEmpty(AsrLanguage);

    public bool IsUser => Role == "user";
    public bool IsAssistant => Role == "assistant";
    public bool IsTool => Role == "tool";

    /// <summary>
    /// Name of the tool whose result this message holds. Only meaningful
    /// when <see cref="IsTool"/>. Used as the header of the tool bubble.
    /// </summary>
    [ObservableProperty] private string? _toolName;

    /// <summary>
    /// True when the tool call failed (exception, timeout, server-side
    /// <c>isError</c> flag, or no MCP server advertised the tool). Drives a
    /// destructive-coloured border on the tool bubble so the user can see at
    /// a glance that the assistant's next turn is reasoning about an error,
    /// not a successful result.
    /// </summary>
    [ObservableProperty] private bool _isToolError;

    // Threshold for auto-collapsing a tool result. Short replies (a single
    // number, a one-liner status) read naturally inline; long ones
    // (dumped API payloads, multi-paragraph summaries) get tucked behind
    // an expander so they don't dominate the transcript flow. Matches the
    // feel of the Reasoning panel — "inline when short, collapsible when
    // long."
    private const int ToolCollapseLineThreshold = 3;
    private const int ToolCollapseCharThreshold = 240;

    /// <summary>
    /// True when the tool result is long enough to benefit from collapsing.
    /// Driven by both line and character thresholds so a single very-long
    /// line still collapses, and three short lines do too.
    /// </summary>
    public bool ShouldCollapseTool
    {
        get
        {
            if (!IsTool) return false;
            var c = Content ?? string.Empty;
            if (c.Length > ToolCollapseCharThreshold) return true;
            int nl = 0;
            for (int i = 0; i < c.Length; i++)
            {
                if (c[i] == '\n' && ++nl >= ToolCollapseLineThreshold) return true;
            }
            return false;
        }
    }

    /// <summary>
    /// Current expanded state of the tool-result <c>Expander</c>. Lazily
    /// initialised to <c>!ShouldCollapseTool</c> on first read — short
    /// results open by default, long ones stay collapsed. After the user
    /// clicks the chevron the override sticks (they toggled for a reason).
    /// The always-present Expander around every tool bubble is what gives
    /// the user a clear "this collapses" affordance; this property just
    /// picks the initial position.
    /// </summary>
    private bool _isToolExpandedInitialised;
    private bool _isToolExpanded;
    public bool IsToolExpanded
    {
        get
        {
            if (!_isToolExpandedInitialised)
            {
                _isToolExpanded = !ShouldCollapseTool;
                _isToolExpandedInitialised = true;
            }
            return _isToolExpanded;
        }
        set
        {
            _isToolExpandedInitialised = true;
            SetProperty(ref _isToolExpanded, value);
        }
    }

    /// <summary>
    /// True when the bubble has any non-tool-call text worth rendering. Used
    /// by the assistant bubble to hide the MarkdownView entirely when the
    /// stripped content is empty — avoids a zero-height ghost element when
    /// the model's reply was purely a tool call.
    /// </summary>
    public bool HasDisplayContent => !string.IsNullOrEmpty(DisplayContent);

    /// <summary>
    /// Peek text shown in the collapsed tool-bubble header: first line of
    /// the result truncated to ~80 chars, plus a "+N more lines" suffix
    /// when the body has more. Most MCP tools put a status/summary on
    /// line 1, so this is usually enough to skip a click.
    /// </summary>
    public string ToolPeek
    {
        get
        {
            if (!IsTool) return string.Empty;
            var c = Content ?? string.Empty;
            if (c.Length == 0) return string.Empty;

            int nl = c.IndexOf('\n');
            var firstLine = nl < 0 ? c : c[..nl];
            if (firstLine.Length > 80) firstLine = firstLine[..77] + "…";

            int extraLines = 0;
            for (int i = 0; i < c.Length; i++) if (c[i] == '\n') extraLines++;
            if (extraLines > 0)
            {
                var label = extraLines == 1 ? "line" : "lines";
                return $"{firstLine}  (+{extraLines} more {label})";
            }
            return firstLine;
        }
    }

    /// <summary>
    /// Content with tool-call markup stripped, for the markdown view. The
    /// raw <see cref="Content"/> is preserved intact so that re-prompting
    /// the model on the next tool-loop round sees exactly what the model
    /// emitted — the template's tool-use branch needs those tags to
    /// reconstruct the turn.
    /// </summary>
    public string DisplayContent =>
        IsAssistant ? ToolCallDisplay.StripMarkup(Content) : Content;

    /// <summary>Compact chips for each tool call in this assistant reply.</summary>
    public IReadOnlyList<ToolCallDisplay.Chip> ToolCallChips =>
        IsAssistant ? ToolCallDisplay.ExtractChips(Content) : System.Array.Empty<ToolCallDisplay.Chip>();

    public bool HasToolCalls => ToolCallChips.Count > 0;

    partial void OnReasoningChanged(string? value) => OnPropertyChanged(nameof(HasReasoning));
    partial void OnAsrLanguageChanged(string? value) => OnPropertyChanged(nameof(HasAsrLanguage));

    partial void OnContentChanged(string value)
    {
        OnPropertyChanged(nameof(DisplayContent));
        OnPropertyChanged(nameof(HasDisplayContent));
        OnPropertyChanged(nameof(ToolCallChips));
        OnPropertyChanged(nameof(HasToolCalls));
        OnPropertyChanged(nameof(ShouldCollapseTool));
        OnPropertyChanged(nameof(ToolPeek));
    }

    partial void OnRoleChanged(string value)
    {
        OnPropertyChanged(nameof(IsUser));
        OnPropertyChanged(nameof(IsAssistant));
        OnPropertyChanged(nameof(IsTool));
        OnPropertyChanged(nameof(DisplayContent));
        OnPropertyChanged(nameof(HasDisplayContent));
        OnPropertyChanged(nameof(ToolCallChips));
        OnPropertyChanged(nameof(HasToolCalls));
        OnPropertyChanged(nameof(ShouldCollapseTool));
        OnPropertyChanged(nameof(ToolPeek));
    }

    public static MessageViewModel FromTurn(ChatTurn t)
    {
        var vm = new MessageViewModel
        {
            Id = t.Id,
            ParentId = t.ParentId,
            CreatedAt = t.CreatedAt,
            Role = t.Role.ToString().ToLowerInvariant(),
            Content = t.Content,
            Reasoning = t.Reasoning,
            IsStreaming = t.State == TurnState.Streaming,
            StatsSummary = t.Stats is { } s
                ? $"{s.CompletionTokens} tok · {s.TokensPerSecond:F1} tok/s"
                : null,
        };
        if (t.Attachments is { Count: > 0 })
        {
            foreach (var a in t.Attachments) vm.Attachments.Add(a);
        }
        return vm;
    }
}
