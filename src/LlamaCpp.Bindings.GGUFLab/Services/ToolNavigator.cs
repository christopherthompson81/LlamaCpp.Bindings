using System;

namespace LlamaCpp.Bindings.GGUFLab.Services;

/// <summary>
/// Lets a tool view model switch the shell over to a different tool,
/// optionally configuring it before activation. Drives "remedy" buttons
/// — when a tool can't proceed because it needs work done elsewhere
/// (e.g. Quantize needs a GGUF but only safetensors are available),
/// the button uses this to jump the user to the tool that can do it
/// and prefill its input.
/// </summary>
public sealed class ToolNavigator
{
    private Func<Type, object?>? _resolver;
    private Action<object>? _selector;

    /// <summary>
    /// Wire the navigator into the shell. <paramref name="resolver"/>
    /// returns the tool view model instance for a given type;
    /// <paramref name="selector"/> activates it.
    /// </summary>
    public void Bind(Func<Type, object?> resolver, Action<object> selector)
    {
        _resolver = resolver;
        _selector = selector;
    }

    /// <summary>
    /// Activate the tool of type <typeparamref name="T"/>, applying
    /// <paramref name="configure"/> beforehand so the destination is
    /// already populated when the user lands on it. Returns false if
    /// the navigator hasn't been bound yet or the tool isn't registered.
    /// </summary>
    public bool NavigateTo<T>(Action<T>? configure = null) where T : class
    {
        if (_resolver?.Invoke(typeof(T)) is not T tool) return false;
        configure?.Invoke(tool);
        _selector?.Invoke(tool);
        return true;
    }
}
