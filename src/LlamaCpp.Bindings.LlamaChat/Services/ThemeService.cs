using Avalonia;
using Avalonia.Styling;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Applies the user's <see cref="AppThemeMode"/> preference by flipping
/// <c>Application.RequestedThemeVariant</c>. Avalonia's ThemeVariant system
/// drives the <c>ResourceDictionary.ThemeDictionaries</c> lookup in
/// <c>Theme/Tokens.axaml</c>, so every <c>DynamicResource</c> brush in the
/// tree updates live — no view rebuild needed.
/// </summary>
internal static class ThemeService
{
    public static void Apply(AppThemeMode mode)
    {
        if (Application.Current is null) return;
        Application.Current.RequestedThemeVariant = mode switch
        {
            AppThemeMode.Light => ThemeVariant.Light,
            AppThemeMode.Dark => ThemeVariant.Dark,
            _ => ThemeVariant.Default,  // follow system
        };
    }
}
