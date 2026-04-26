namespace LlamaCpp.Bindings.GGUFSuite.ViewModels;

/// <summary>
/// Placeholder page for tools that aren't implemented yet. Carries the
/// rationale for the deferral (what binding work is missing or what the
/// architectural tradeoff is) so the user understands the roadmap without
/// digging through code.
/// </summary>
public sealed class RoadmapViewModel : ToolPageViewModel
{
    public override string Title { get; }
    public override string Description { get; }

    /// <summary>Why the tool is deferred — shown in the page body.</summary>
    public string Rationale { get; }

    /// <summary>Short list of the next blocking pieces of work.</summary>
    public string NextSteps { get; }

    public RoadmapViewModel(string title, string description, string rationale, string nextSteps)
    {
        Title = title;
        Description = description;
        Rationale = rationale;
        NextSteps = nextSteps;
    }
}
