using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services.Exporters;

/// <summary>
/// Exports the active branch as Markdown. Content is emitted verbatim —
/// model replies that already contain markdown (fenced code blocks,
/// tables, lists) survive intact. Reasoning blocks are rendered as
/// block-quotes so they stand apart visually without requiring a
/// markdown extension.
/// </summary>
public sealed class MarkdownExporter : IConversationExporter
{
    public string FormatId      => "markdown";
    public string DisplayName   => "Markdown (.md)";
    public string FileExtension => "md";

    public async Task ExportAsync(
        Conversation conversation,
        Stream output,
        ExportOptions options,
        CancellationToken cancellationToken = default)
    {
        var sb = new StringBuilder();
        sb.Append("# ").AppendLine(conversation.Title);
        if (options.IncludeTimestamps)
        {
            sb.Append("_Created: ").Append(conversation.CreatedAt.ToLocalTime().ToString("yyyy-MM-dd HH:mm"))
              .Append(" · Updated: ").Append(conversation.UpdatedAt.ToLocalTime().ToString("yyyy-MM-dd HH:mm"))
              .AppendLine("_");
        }
        sb.AppendLine();

        foreach (var turn in ConversationBranch.ActivePath(conversation))
        {
            if (turn.Role == TurnRole.System && !options.IncludeSystemPrompt) continue;

            sb.Append("## ").Append(ConversationBranch.RoleLabel(turn.Role));
            if (options.IncludeTimestamps)
                sb.Append(" — ").Append(turn.CreatedAt.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss"));
            sb.AppendLine();
            sb.AppendLine();

            if (options.IncludeReasoning && !string.IsNullOrWhiteSpace(turn.Reasoning))
            {
                sb.AppendLine("> **Reasoning**");
                foreach (var line in turn.Reasoning!.Split('\n'))
                    sb.Append("> ").AppendLine(line.TrimEnd('\r'));
                sb.AppendLine();
            }

            var cleanContent = ContentSanitizer.StripReasoningSpans(turn.Content);
            if (!string.IsNullOrEmpty(cleanContent))
            {
                sb.AppendLine(cleanContent);
                sb.AppendLine();
            }

            if (options.IncludeAttachmentList && turn.Attachments is { Count: > 0 })
            {
                sb.Append("_Attachments: ");
                for (int i = 0; i < turn.Attachments.Count; i++)
                {
                    if (i > 0) sb.Append(", ");
                    sb.Append(turn.Attachments[i].FileName ?? "(unnamed)");
                }
                sb.AppendLine("_");
                sb.AppendLine();
            }

            if (options.IncludeStats && turn.Stats is { } stats)
            {
                sb.Append("_Stats: ").Append(stats.CompletionTokens).Append(" tok, ")
                  .Append(stats.TokensPerSecond.ToString("F1")).AppendLine(" tok/s_");
                sb.AppendLine();
            }
        }

        await using var writer = new StreamWriter(output, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false), leaveOpen: true);
        await writer.WriteAsync(sb.ToString()).WaitAsync(cancellationToken);
        await writer.FlushAsync(cancellationToken);
    }
}
