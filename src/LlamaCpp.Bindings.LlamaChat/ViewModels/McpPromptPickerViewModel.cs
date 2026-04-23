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
/// Flattens every ready MCP server's prompts into a single picker. Selecting
/// a prompt expands its argument form; clicking Insert fetches the rendered
/// prompt and surfaces the text through <see cref="PromptText"/>. The caller
/// (MainWindow) reads that field and inserts it into the compose box.
/// </summary>
public partial class McpPromptPickerViewModel : ObservableObject
{
    public ObservableCollection<PromptOption> Options { get; } = new();

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(InsertCommand))]
    [NotifyPropertyChangedFor(nameof(HasArguments))]
    private PromptOption? _selected;

    public ObservableCollection<ArgumentField> ArgumentFields { get; } = new();
    public bool HasArguments => ArgumentFields.Count > 0;

    [ObservableProperty] private string _status = string.Empty;

    /// <summary>Set when Insert completes — caller reads and applies.</summary>
    public string? PromptText { get; private set; }

    public McpPromptPickerViewModel()
    {
        foreach (var s in McpClientService.Instance.Servers)
        {
            if (s.State != McpConnectionState.Ready) continue;
            foreach (var p in s.Prompts)
            {
                Options.Add(new PromptOption(s, p));
            }
        }
        Selected = Options.FirstOrDefault();
    }

    partial void OnSelectedChanged(PromptOption? value)
    {
        ArgumentFields.Clear();
        if (value is null) return;
        foreach (var a in value.Prompt.Arguments)
        {
            ArgumentFields.Add(new ArgumentField(a));
        }
        OnPropertyChanged(nameof(HasArguments));
    }

    [RelayCommand(CanExecute = nameof(CanInsert))]
    private async Task InsertAsync()
    {
        if (Selected is null) return;
        var args = ArgumentFields
            .Where(f => !string.IsNullOrEmpty(f.Value))
            .ToDictionary(f => f.Argument.Name, f => f.Value);

        try
        {
            Status = "Fetching prompt…";
            PromptText = await McpClientService.Instance.GetPromptAsync(
                Selected.Server, Selected.Prompt.Name, args);
            Status = "Ready.";
            InsertCompleted?.Invoke(this, EventArgs.Empty);
        }
        catch (Exception ex)
        {
            Status = $"Error: {ex.Message}";
        }
    }

    public event EventHandler? InsertCompleted;

    private bool CanInsert() => Selected is not null;

    public sealed record PromptOption(McpServerEntry Server, McpPromptInfo Prompt)
    {
        public string Label => $"{Server.Config.Name} / {Prompt.Name}";
    }

    public partial class ArgumentField : ObservableObject
    {
        public McpPromptArgument Argument { get; }

        [ObservableProperty] private string _value = string.Empty;

        public ArgumentField(McpPromptArgument arg) { Argument = arg; }

        public string Label => Argument.Required ? Argument.Name + " *" : Argument.Name;
    }
}
