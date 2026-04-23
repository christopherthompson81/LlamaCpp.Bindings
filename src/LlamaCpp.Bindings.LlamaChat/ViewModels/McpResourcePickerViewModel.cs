using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.Services;

namespace LlamaCpp.Bindings.LlamaChat.ViewModels;

/// <summary>
/// Backs the resource browser dialog. Flattens resources from every ready
/// server, supports a substring filter, and produces either (a) a "preview"
/// result (plain text shown in a modal) or (b) an "attach" result that the
/// compose box inserts as a resource reference.
/// </summary>
public partial class McpResourcePickerViewModel : ObservableObject
{
    public ObservableCollection<ResourceOption> All { get; } = new();
    public ObservableCollection<ResourceOption> Filtered { get; } = new();

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PreviewCommand), nameof(AttachCommand))]
    private ResourceOption? _selected;

    [ObservableProperty] private string _searchText = string.Empty;

    [ObservableProperty] private string _status = string.Empty;
    [ObservableProperty] private string _previewText = string.Empty;

    /// <summary>When the user picks Attach, this is what the caller inserts.</summary>
    public string? AttachedContent { get; private set; }
    public string? AttachedUri { get; private set; }
    public event EventHandler? AttachRequested;

    public McpResourcePickerViewModel()
    {
        foreach (var s in McpClientService.Instance.Servers)
        {
            if (s.State != McpConnectionState.Ready) continue;
            foreach (var r in s.Resources)
            {
                All.Add(new ResourceOption(s, r));
            }
        }
        RebuildFiltered();
        Selected = Filtered.FirstOrDefault();
    }

    partial void OnSearchTextChanged(string value) => RebuildFiltered();

    private void RebuildFiltered()
    {
        var saved = Selected;
        Filtered.Clear();
        IEnumerable<ResourceOption> source = All;
        if (!string.IsNullOrWhiteSpace(SearchText))
        {
            var q = SearchText.Trim();
            source = All.Where(o =>
                (o.Resource.Name ?? string.Empty).Contains(q, StringComparison.OrdinalIgnoreCase) ||
                o.Resource.Uri.Contains(q, StringComparison.OrdinalIgnoreCase) ||
                (o.Resource.Description ?? string.Empty).Contains(q, StringComparison.OrdinalIgnoreCase));
        }
        foreach (var o in source) Filtered.Add(o);
        if (saved is not null && Filtered.Contains(saved)) Selected = saved;
    }

    [RelayCommand(CanExecute = nameof(HasSelection))]
    private async Task PreviewAsync()
    {
        if (Selected is null) return;
        try
        {
            Status = "Reading…";
            PreviewText = await McpClientService.Instance.ReadResourceAsync(
                Selected.Server, Selected.Resource.Uri);
            Status = $"{PreviewText.Length} char(s) loaded.";
        }
        catch (Exception ex)
        {
            Status = $"Error: {ex.Message}";
            PreviewText = string.Empty;
        }
    }

    [RelayCommand(CanExecute = nameof(HasSelection))]
    private async Task AttachAsync()
    {
        if (Selected is null) return;
        try
        {
            Status = "Reading…";
            AttachedContent = await McpClientService.Instance.ReadResourceAsync(
                Selected.Server, Selected.Resource.Uri);
            AttachedUri = Selected.Resource.Uri;
            Status = "Attached.";
            AttachRequested?.Invoke(this, EventArgs.Empty);
        }
        catch (Exception ex)
        {
            Status = $"Error: {ex.Message}";
        }
    }

    private bool HasSelection() => Selected is not null;

    public sealed record ResourceOption(McpServerEntry Server, McpResourceInfo Resource)
    {
        public string Label => string.IsNullOrEmpty(Resource.Name)
            ? Resource.Uri
            : $"{Resource.Name} — {Resource.Uri}";
        public string ServerLabel => Server.Config.Name;
    }
}
