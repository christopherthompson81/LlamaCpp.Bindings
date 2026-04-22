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

    public bool HasReasoning => !string.IsNullOrEmpty(Reasoning);

    partial void OnReasoningChanged(string? value) => OnPropertyChanged(nameof(HasReasoning));

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
