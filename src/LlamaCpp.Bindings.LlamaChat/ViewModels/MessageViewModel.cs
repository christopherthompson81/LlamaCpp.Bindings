using CommunityToolkit.Mvvm.ComponentModel;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.ViewModels;

public partial class MessageViewModel : ObservableObject
{
    [ObservableProperty] private string _role = string.Empty;
    [ObservableProperty] private string _content = string.Empty;
    [ObservableProperty] private string? _reasoning;
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

    public bool HasReasoning => !string.IsNullOrEmpty(Reasoning);

    public bool IsUser => Role == "user";
    public bool IsAssistant => Role == "assistant";

    partial void OnReasoningChanged(string? value) => OnPropertyChanged(nameof(HasReasoning));

    partial void OnRoleChanged(string value)
    {
        OnPropertyChanged(nameof(IsUser));
        OnPropertyChanged(nameof(IsAssistant));
    }

    public static MessageViewModel FromTurn(ChatTurn t) => new()
    {
        Role = t.Role.ToString().ToLowerInvariant(),
        Content = t.Content,
        Reasoning = t.Reasoning,
        IsStreaming = t.State == TurnState.Streaming,
        StatsSummary = t.Stats is { } s
            ? $"{s.CompletionTokens} tok · {s.TokensPerSecond:F1} tok/s"
            : null,
    };
}
