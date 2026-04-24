using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.ViewModels;

/// <summary>
/// Observable wrapper around a <see cref="Conversation"/>. Maintains the
/// turn tree as a flat <see cref="AllMessages"/> list keyed by
/// <see cref="MessageViewModel.Id"/>, plus a computed
/// <see cref="Messages"/> collection that the chat view binds to — the
/// path from the root turn to the current <see cref="ActiveLeafId"/>.
///
/// Branches are created implicitly by <see cref="AddSibling"/> (retry,
/// edit-as-fork); the sibling-nav control in each bubble's header lets
/// the user switch which branch is active, which re-computes Messages
/// and triggers a KV-cache clear on the next turn.
/// </summary>
public partial class ConversationViewModel : ObservableObject
{
    public Guid Id { get; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DisplayTitle))]
    private string _title;

    public DateTimeOffset CreatedAt { get; }

    [ObservableProperty] private DateTimeOffset _updatedAt;

    /// <summary>Pinned conversations sort to the top of the sidebar.</summary>
    [ObservableProperty] private bool _pinned;

    /// <summary>Inline-rename mode in the sidebar.</summary>
    [ObservableProperty] private bool _isEditing;

    /// <summary>
    /// Every turn in the tree. Order of insertion, not display order —
    /// display order is determined by walking <see cref="Messages"/>.
    /// Consumers who want to mutate the tree go through the explicit
    /// tree-op methods (<see cref="AppendToActivePath"/>, <see cref="AddSibling"/>,
    /// <see cref="RemoveSubtree"/>, <see cref="ClearAll"/>), not this list.
    /// </summary>
    public ObservableCollection<MessageViewModel> AllMessages { get; } = new();

    /// <summary>
    /// The currently-active path, root → leaf. Rebuilt whenever
    /// <see cref="ActiveLeafId"/> changes or the tree is mutated. The chat
    /// view binds here; this is the "transcript the user sees."
    /// </summary>
    public ObservableCollection<MessageViewModel> Messages { get; } = new();

    /// <summary>
    /// Leaf of the active branch. The root→leaf walk via
    /// <see cref="MessageViewModel.ParentId"/> produces
    /// <see cref="Messages"/>. Null for an empty conversation.
    /// </summary>
    [ObservableProperty] private Guid? _activeLeafId;

    public string DisplayTitle => string.IsNullOrWhiteSpace(Title) ? "(untitled)" : Title;

    /// <summary>
    /// True when the active path ends with a user turn, meaning nothing
    /// replied to it yet. Typical ways to arrive here: the user deleted
    /// the last assistant reply with RemoveSubtree, cancelled generation
    /// before any tokens streamed, or switched to a sibling branch whose
    /// own leaf is a user turn. The UI surfaces a small "Generate reply"
    /// remedy card above the compose box when this fires.
    /// </summary>
    public bool NeedsAssistantReply
    {
        get
        {
            var last = Messages.Count > 0 ? Messages[Messages.Count - 1] : null;
            return last?.IsUser == true;
        }
    }

    /// <summary>
    /// First few chars of the first user message in the active path, for
    /// the sidebar subtitle. Empty if no user message yet.
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
        _pinned = model.Pinned;

        // Load every turn into AllMessages. Legacy files (written before
        // tree support) have no ParentId set on their turns; infer a
        // linear chain from the insertion order so the load is lossless.
        Guid? prevId = null;
        foreach (var t in model.Turns)
        {
            var vm = MessageViewModel.FromTurn(t);
            if (vm.ParentId is null && prevId is not null)
            {
                vm.ParentId = prevId;
            }
            vm.Owner = this;
            AllMessages.Add(vm);
            prevId = vm.Id;
        }

        // ActiveLeafId defaults to the last-added message (matches legacy
        // behaviour for flat transcripts) unless the model explicitly set it.
        _activeLeafId = model.ActiveLeafId ?? AllMessages.LastOrDefault()?.Id;

        RebuildActivePath();
        Messages.CollectionChanged += (_, _) =>
        {
            OnPropertyChanged(nameof(Preview));
            OnPropertyChanged(nameof(NeedsAssistantReply));
        };
    }

    public static ConversationViewModel NewEmpty() => new(new Conversation());

    // ========================================================
    // Tree operations — the allowed mutations on AllMessages.
    // ========================================================

    /// <summary>
    /// Extend the active path by one turn: <paramref name="msg"/> becomes
    /// the new leaf, parented to whatever leaf was previously active.
    /// This is the happy-path call from Send, Generate, Continue.
    /// </summary>
    public void AppendToActivePath(MessageViewModel msg)
    {
        msg.ParentId = ActiveLeafId;
        msg.Owner = this;
        AllMessages.Add(msg);
        ActiveLeafId = msg.Id;
        RebuildActivePath();
        NotifySiblingsFor(msg.ParentId);
    }

    /// <summary>
    /// Insert <paramref name="newMsg"/> as a child of
    /// <paramref name="parentId"/>. If that parent already has children
    /// this creates a new branch — those prior children become siblings
    /// of <paramref name="newMsg"/>. Used by Regenerate when the parent
    /// is a specific user turn that may or may not already have an
    /// assistant reply hanging off it.
    /// </summary>
    public void AddChildOf(Guid? parentId, MessageViewModel newMsg)
    {
        newMsg.ParentId = parentId;
        newMsg.Owner = this;
        AllMessages.Add(newMsg);
        ActiveLeafId = newMsg.Id;
        RebuildActivePath();
        NotifySiblingsFor(parentId);
    }

    /// <summary>
    /// Insert <paramref name="newMsg"/> as a sibling of
    /// <paramref name="existingId"/> — same <see cref="ParentId"/>, a new
    /// branch. Used by retry and edit-user flows where we want to preserve
    /// the original alongside the alternative. The new message becomes the
    /// active leaf.
    /// </summary>
    public void AddSibling(Guid existingId, MessageViewModel newMsg)
    {
        var existing = FindById(existingId);
        if (existing is null)
        {
            // No such anchor — fall back to a plain append so callers
            // don't need to defensive-check the tree state.
            AppendToActivePath(newMsg);
            return;
        }
        newMsg.ParentId = existing.ParentId;
        newMsg.Owner = this;
        AllMessages.Add(newMsg);
        ActiveLeafId = newMsg.Id;
        RebuildActivePath();
        // Every sibling of the new node (including the existing anchor)
        // just changed its SiblingCount — tell the bindings.
        NotifySiblingsFor(newMsg.ParentId);
    }

    /// <summary>
    /// Delete <paramref name="nodeId"/> and every descendant. When the
    /// active leaf was inside the deleted subtree we prefer to land on a
    /// surviving sibling's deepest leaf rather than fall back to the
    /// parent — "delete this version" implies the user wants to see the
    /// remaining version, not be stranded at a dangling user turn. Falls
    /// back to the parent only if no sibling survives the delete.
    /// No-op if the id isn't found.
    /// </summary>
    public void RemoveSubtree(Guid nodeId)
    {
        var node = FindById(nodeId);
        if (node is null) return;

        // Snapshot siblings BEFORE removal so we know what will survive.
        var survivingSiblings = new List<MessageViewModel>();
        foreach (var m in AllMessages)
        {
            if (m.ParentId == node.ParentId && m.Id != nodeId)
            {
                survivingSiblings.Add(m);
            }
        }

        // Collect nodeId + every transitive descendant.
        var toRemove = new HashSet<Guid> { nodeId };
        bool grew;
        do
        {
            grew = false;
            foreach (var m in AllMessages)
            {
                if (m.ParentId is { } p && toRemove.Contains(p) && !toRemove.Contains(m.Id))
                {
                    toRemove.Add(m.Id);
                    grew = true;
                }
            }
        }
        while (grew);

        // Remove all at once; mutate a snapshot because we're iterating.
        foreach (var m in AllMessages.Where(m => toRemove.Contains(m.Id)).ToList())
        {
            AllMessages.Remove(m);
        }

        // Pick a new active leaf if the current one got evicted.
        if (ActiveLeafId is { } leaf && toRemove.Contains(leaf))
        {
            if (survivingSiblings.Count > 0)
            {
                // Switch to the last-added surviving sibling (most recent
                // branch the user cared about) and descend to its leaf so
                // the transcript shows a real assistant reply, not just a
                // dangling parent turn with a remedy card.
                ActiveLeafId = DeepestLeafUnder(survivingSiblings[^1].Id);
            }
            else
            {
                ActiveLeafId = node.ParentId;
            }
        }
        RebuildActivePath();
        NotifySiblingsFor(node.ParentId);
    }

    /// <summary>
    /// Walk down from <paramref name="rootId"/> picking the most-recently
    /// added child at each branch until there are no more children. Used
    /// by <see cref="SwitchToSibling"/> and <see cref="RemoveSubtree"/>
    /// to restore or land on a full sub-path rather than a single turn.
    /// </summary>
    private Guid DeepestLeafUnder(Guid rootId)
    {
        var cursor = FindById(rootId);
        if (cursor is null) return rootId;
        while (true)
        {
            MessageViewModel? lastChild = null;
            foreach (var m in AllMessages)
            {
                if (m.ParentId == cursor.Id) lastChild = m;
            }
            if (lastChild is null) return cursor.Id;
            cursor = lastChild;
        }
    }

    /// <summary>Wipe the conversation. Leaves the conversation object itself intact.</summary>
    public void ClearAll()
    {
        AllMessages.Clear();
        ActiveLeafId = null;
        RebuildActivePath();
    }

    /// <summary>
    /// Switch the active branch to one whose root is <paramref name="siblingId"/>.
    /// If the sibling has children we walk down picking the most recently
    /// added child at each step, so switching to a branch restores its
    /// full sub-path rather than stopping at the sibling node itself.
    /// </summary>
    public void SwitchToSibling(Guid siblingId)
    {
        if (FindById(siblingId) is null) return;
        ActiveLeafId = DeepestLeafUnder(siblingId);
        RebuildActivePath();
    }

    // ========================================================
    // Sibling queries — used by MessageViewModel's computed info.
    // ========================================================

    public int GetSiblingCount(Guid messageId)
    {
        var node = FindById(messageId);
        if (node is null) return 1;
        var pid = node.ParentId;
        int count = 0;
        foreach (var m in AllMessages)
        {
            if (m.ParentId == pid) count++;
        }
        return count;
    }

    public int GetSiblingIndex(Guid messageId)
    {
        var node = FindById(messageId);
        if (node is null) return 1;
        var pid = node.ParentId;
        int idx = 0;
        foreach (var m in AllMessages)
        {
            if (m.ParentId == pid)
            {
                idx++;
                if (m.Id == messageId) return idx;
            }
        }
        return 1;
    }

    /// <summary>Previous sibling (cyclic); null if this message has no siblings.</summary>
    public MessageViewModel? PrevSibling(Guid messageId)
    {
        var node = FindById(messageId);
        if (node is null) return null;
        var sibs = AllMessages.Where(m => m.ParentId == node.ParentId).ToList();
        if (sibs.Count <= 1) return null;
        var idx = sibs.FindIndex(m => m.Id == messageId);
        if (idx < 0) return null;
        return sibs[(idx - 1 + sibs.Count) % sibs.Count];
    }

    /// <summary>Next sibling (cyclic); null if this message has no siblings.</summary>
    public MessageViewModel? NextSibling(Guid messageId)
    {
        var node = FindById(messageId);
        if (node is null) return null;
        var sibs = AllMessages.Where(m => m.ParentId == node.ParentId).ToList();
        if (sibs.Count <= 1) return null;
        var idx = sibs.FindIndex(m => m.Id == messageId);
        if (idx < 0) return null;
        return sibs[(idx + 1) % sibs.Count];
    }

    // ========================================================
    // Helpers
    // ========================================================

    public MessageViewModel? FindById(Guid id)
    {
        foreach (var m in AllMessages)
        {
            if (m.Id == id) return m;
        }
        return null;
    }

    /// <summary>
    /// Rebuild <see cref="Messages"/> from the current active leaf walking
    /// up via ParentId to the root, then reversed. Called after any tree
    /// or active-leaf mutation. Works by mutating the existing collection
    /// in place so bindings see a single CollectionChanged batch.
    /// </summary>
    private void RebuildActivePath()
    {
        var path = new List<MessageViewModel>();
        var cursor = ActiveLeafId is { } id ? FindById(id) : null;
        while (cursor is not null)
        {
            path.Add(cursor);
            cursor = cursor.ParentId is { } pid ? FindById(pid) : null;
        }
        path.Reverse();

        // Diff against current Messages to minimise binding churn.
        // For simplicity we just Clear + re-Add — the chat ScrollViewer
        // copes fine and paths are short (a few dozen at most).
        Messages.Clear();
        foreach (var m in path) Messages.Add(m);
        // Notify sibling-info for every visible turn — SiblingCount may have
        // changed because of a tree mutation on a non-visible branch.
        // Flag the tail message so the Continue button knows it's the only
        // eligible candidate (older messages don't correspond to the current
        // KV state and can't be extended in place).
        for (var i = 0; i < Messages.Count; i++)
        {
            Messages[i].NotifySiblingInfoChanged();
            Messages[i].IsLastInActivePath = (i == Messages.Count - 1);
        }
    }

    private void NotifySiblingsFor(Guid? parentId)
    {
        foreach (var m in AllMessages)
        {
            if (m.ParentId == parentId) m.NotifySiblingInfoChanged();
        }
    }

    partial void OnActiveLeafIdChanged(Guid? value) { /* RebuildActivePath is called explicitly by mutators. */ }

    /// <summary>
    /// Project back to a persistable <see cref="Conversation"/> record.
    /// Every node in <see cref="AllMessages"/> is serialised, preserving
    /// Id + ParentId so branches survive a save/load round-trip. View-only
    /// fields (IsStreaming, IsReasoningExpanded) are dropped.
    /// </summary>
    public Conversation ToModel() => new()
    {
        Id = Id,
        Title = Title,
        CreatedAt = CreatedAt,
        UpdatedAt = UpdatedAt,
        Pinned = Pinned,
        ActiveLeafId = ActiveLeafId,
        Turns = AllMessages.Select(m => new ChatTurn(
            Id: m.Id,
            Role: m.Role switch
            {
                "user" => TurnRole.User,
                "tool" => TurnRole.Tool,
                "assistant" => TurnRole.Assistant,
                _ => TurnRole.Assistant,
            },
            Content: m.Content,
            State: TurnState.Complete,
            CreatedAt: UpdatedAt,
            Reasoning: m.Reasoning,
            Stats: null,
            Attachments: m.Attachments.Count > 0
                ? new List<Attachment>(m.Attachments)
                : null,
            ParentId: m.ParentId)).ToList(),
    };
}
