using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services.Exporters;

/// <summary>
/// Exports the active branch as plain text with role-separated blocks.
/// Markdown syntax in the content passes through as literal characters —
/// callers who want rendered formatting should prefer
/// <see cref="MarkdownExporter"/> or <see cref="HtmlExporter"/>.
/// </summary>
public sealed class TxtExporter : IConversationExporter
{
    public string FormatId      => "txt";
    public string DisplayName   => "Plain text (.txt)";
    public string FileExtension => "txt";

    private const string Separator = "----------------------------------------";

    public async Task ExportAsync(
        Conversation conversation,
        Stream output,
        ExportOptions options,
        CancellationToken cancellationToken = default)
    {
        var sb = new StringBuilder();
        sb.AppendLine(conversation.Title);
        if (options.IncludeTimestamps)
        {
            sb.Append("Created: ").Append(conversation.CreatedAt.ToLocalTime().ToString("yyyy-MM-dd HH:mm"))
              .Append(" | Updated: ").AppendLine(conversation.UpdatedAt.ToLocalTime().ToString("yyyy-MM-dd HH:mm"));
        }
        sb.AppendLine(Separator);
        sb.AppendLine();

        foreach (var turn in ConversationBranch.ActivePath(conversation))
        {
            if (turn.Role == TurnRole.System && !options.IncludeSystemPrompt) continue;

            sb.Append("[").Append(ConversationBranch.RoleLabel(turn.Role).ToUpperInvariant()).Append("]");
            if (options.IncludeTimestamps)
                sb.Append("  ").Append(turn.CreatedAt.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss"));
            sb.AppendLine();

            if (options.IncludeReasoning && !string.IsNullOrWhiteSpace(turn.Reasoning))
            {
                sb.AppendLine("(Reasoning)");
                sb.AppendLine(turn.Reasoning);
                sb.AppendLine();
            }

            var cleanContent = ContentSanitizer.StripReasoningSpans(turn.Content);
            if (!string.IsNullOrEmpty(cleanContent))
                sb.AppendLine(cleanContent);

            if (options.IncludeAttachmentList && turn.Attachments is { Count: > 0 })
            {
                sb.Append("Attachments:");
                foreach (var a in turn.Attachments)
                    sb.Append(' ').Append(a.FileName ?? "(unnamed)");
                sb.AppendLine();
            }

            sb.AppendLine();
            sb.AppendLine(Separator);
            sb.AppendLine();
        }

        await using var writer = new StreamWriter(output, new UTF8Encoding(false), leaveOpen: true);
        await writer.WriteAsync(sb.ToString()).WaitAsync(cancellationToken);
        await writer.FlushAsync(cancellationToken);
    }
}
