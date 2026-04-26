using CommunityToolkit.Mvvm.ComponentModel;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Base class for everything that occupies the main content area. The sidebar
/// holds one of these per tool; <see cref="MainWindowViewModel.SelectedTool"/>
/// drives which one the window shows. Subclasses provide their own view via
/// the data-template hookup in <c>MainWindow.axaml</c>.
/// </summary>
public abstract partial class ToolPageViewModel : ObservableObject
{
    /// <summary>Sidebar label.</summary>
    public abstract string Title { get; }

    /// <summary>One-line subtitle shown under the page header.</summary>
    public abstract string Description { get; }

    /// <summary>
    /// Called by the main window when the active model changes, or when
    /// this tool becomes the selected page. Default is a no-op; tools
    /// that take a GGUF path override and copy <paramref name="path"/>
    /// into their input field — but only when the field is currently
    /// empty so a typed-in path is never silently overwritten.
    /// </summary>
    public virtual void ApplyActiveModel(string? path) { }
}
