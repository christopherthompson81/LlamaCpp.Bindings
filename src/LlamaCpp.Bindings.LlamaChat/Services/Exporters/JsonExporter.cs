using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services.Exporters;

/// <summary>
/// Structured export of the <i>entire</i> conversation — not just the
/// active branch — so the result is round-trippable back into the app via
/// <see cref="ConversationStore.ImportFromFile"/>. Every turn in every
/// branch survives, which makes this format the right choice for backups
/// and for piping conversation data into external scripts.
/// </summary>
public sealed class JsonExporter : IConversationExporter
{
    public string FormatId      => "json";
    public string DisplayName   => "JSON (.json)";
    public string FileExtension => "json";

    private static readonly JsonSerializerOptions _options = new()
    {
        WriteIndented = true,
    };

    public async Task ExportAsync(
        Conversation conversation,
        Stream output,
        ExportOptions options,
        CancellationToken cancellationToken = default)
    {
        // ExportOptions toggles don't apply to JSON — round-trip integrity
        // is the whole point of this format; stripping fields would defeat
        // it. Document that in the options summary but just serialise as-is.
        _ = options;

        await JsonSerializer.SerializeAsync(
            output, conversation, _options, cancellationToken);
        await output.FlushAsync(cancellationToken);
    }
}
