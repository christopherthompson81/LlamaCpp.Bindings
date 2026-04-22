namespace LlamaCpp.Bindings.LlamaChat.Models;

/// <summary>
/// A named, persistable bundle of load + sampling + generation settings.
/// Users pick a profile to load; the profile's sampler settings stay live
/// for that session and can be edited from the Settings window.
/// </summary>
public sealed record ModelProfile
{
    public string Name { get; init; } = "New profile";

    /// <summary>
    /// Prepended as a system-role turn at the start of every transcript sent
    /// to the model under this profile. Empty = no system message.
    /// </summary>
    public string SystemPrompt { get; init; } = string.Empty;

    public ModelLoadSettings Load { get; init; } = new();
    public SamplerSettings Sampler { get; init; } = new();
    public GenerationSettings Generation { get; init; } = new();
}
