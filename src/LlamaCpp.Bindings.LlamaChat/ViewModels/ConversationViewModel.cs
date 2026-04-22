using System;
using System.Collections.ObjectModel;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.ViewModels;

/// <summary>
/// Observable wrapper around a <see cref="Conversation"/>. Owns its own
/// <see cref="Messages"/> collection (observable so the chat view updates
/// live), plus inline-rename state for the sidebar.
/// </summary>
public partial class ConversationViewModel : ObservableObject
{
    public Guid Id { get; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DisplayTitle))]
    private string _title;

    public DateTimeOffset CreatedAt { get; }

    [ObservableProperty] private DateTimeOffset _updatedAt;

    /// <summary>Inline-rename mode in the sidebar.</summary>
    [ObservableProperty] private bool _isEditing;

    public ObservableCollection<MessageViewModel> Messages { get; } = new();

    public string DisplayTitle => string.IsNullOrWhiteSpace(Title) ? "(untitled)" : Title;

    /// <summary>
    /// First few chars of the first user message, for the sidebar subtitle.
    /// Empty if no user message yet.
    /// </summary>
    public string Preview
    {
        get
        {
            var first = Messages.FirstOrDefault(m => m.IsUser);
            if (first is null) return string.Empty;
            var s = first.Content.Replace('\n', ' ').Trim();
            return s.Length > 80 ? s[..80] + "…" : s;
        }
    }

    public ConversationViewModel(Conversation model)
    {
        Id = model.Id;
        _title = model.Title;
        CreatedAt = model.CreatedAt;
        _updatedAt = model.UpdatedAt;

        foreach (var t in model.Turns)
        {
            Messages.Add(MessageViewModel.FromTurn(t));
        }

        Messages.CollectionChanged += (_, _) => OnPropertyChanged(nameof(Preview));
    }

    public static ConversationViewModel NewEmpty() => new(new Conversation());

    /// <summary>
    /// Project back to a persistable <see cref="Conversation"/> record.
    /// Messages are turned back into <see cref="ChatTurn"/>s; view-only
    /// fields (IsStreaming, IsReasoningExpanded) are dropped.
    /// </summary>
    public Conversation ToModel() => new()
    {
        Id = Id,
        Title = Title,
        CreatedAt = CreatedAt,
        UpdatedAt = UpdatedAt,
        Turns = Messages.Select(m => new ChatTurn(
            Id: Guid.NewGuid(),
            Role: m.IsUser ? TurnRole.User : TurnRole.Assistant,
            Content: m.Content,
            State: TurnState.Complete,
            CreatedAt: UpdatedAt,
            Reasoning: m.Reasoning,
            Stats: null)).ToList(),
    };
}
