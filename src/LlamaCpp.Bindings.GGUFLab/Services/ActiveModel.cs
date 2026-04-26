using System;
using System.IO;
using CommunityToolkit.Mvvm.ComponentModel;

namespace LlamaCpp.Bindings.GGUFLab.Services;

/// <summary>
/// One model "in focus" across the whole app. The HF Browser, Local
/// Models tool, and any picker can <see cref="Set"/> it; tool view
/// models can read <see cref="Path"/> on activation to prefill their
/// input field. Implemented as a single shared instance owned by the
/// main window — kept on the surface as an observable so the active
/// strip rebinds without ceremony.
/// </summary>
public sealed partial class ActiveModel : ObservableObject
{
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasModel))]
    [NotifyPropertyChangedFor(nameof(DisplayName))]
    private string? _path;

    public bool HasModel => !string.IsNullOrEmpty(Path);

    /// <summary>File name without directory; "" when no model is active.</summary>
    public string DisplayName
    {
        get
        {
            if (string.IsNullOrEmpty(Path)) return "";
            try { return System.IO.Path.GetFileName(Path) ?? Path; }
            catch { return Path; }
        }
    }

    public void Set(string? path) => Path = string.IsNullOrWhiteSpace(path) ? null : path;
    public void Clear() => Path = null;
}
