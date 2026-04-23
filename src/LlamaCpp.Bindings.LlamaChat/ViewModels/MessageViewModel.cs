using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.ViewModels;

public partial class MessageViewModel : ObservableObject
{
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
    [ObservableProperty] private bool _isStreaming = false;
    [ObservableProperty] private string? _statsSummary;

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

    partial void OnReasoningChanged(string? value) => OnPropertyChanged(nameof(HasReasoning));
    partial void OnAsrLanguageChanged(string? value) => OnPropertyChanged(nameof(HasAsrLanguage));

    partial void OnRoleChanged(string value)
    {
        OnPropertyChanged(nameof(IsUser));
        OnPropertyChanged(nameof(IsAssistant));
    }

    public static MessageViewModel FromTurn(ChatTurn t)
    {
        var vm = new MessageViewModel
        {
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
