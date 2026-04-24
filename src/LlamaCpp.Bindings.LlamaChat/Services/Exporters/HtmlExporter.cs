using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using LlamaCpp.Bindings.LlamaChat.Models;
using Markdig;

namespace LlamaCpp.Bindings.LlamaChat.Services.Exporters;

/// <summary>
/// Exports the active branch as a self-contained HTML document with
/// inline CSS. The result opens in any browser with readable styling and
/// role-distinguished message cards. Content is HTML-escaped; markdown
/// is not rendered here — the goal is a portable snapshot that doesn't
/// depend on a markdown engine at display time. Users who want rendered
/// markdown can export to MD and view it through their own viewer.
/// </summary>
public sealed class HtmlExporter : IConversationExporter
{
    public string FormatId      => "html";
    public string DisplayName   => "HTML (.html)";
    public string FileExtension => "html";

    // Markdig pipeline covering the markdown the model actually emits —
    // pipe tables, fenced code blocks, task lists, autolinks. No unsafe
    // HTML passthrough; untrusted content gets escaped by Markdig's
    // normalization. Mirrors the in-app renderer's pipeline so the export
    // looks like the chat view.
    private static readonly MarkdownPipeline _pipeline = new MarkdownPipelineBuilder()
        .UsePipeTables()
        .UseAdvancedExtensions()
        .DisableHtml()
        .Build();

    public async Task ExportAsync(
        Conversation conversation,
        Stream output,
        ExportOptions options,
        CancellationToken cancellationToken = default)
    {
        var sb = new StringBuilder();
        sb.AppendLine("<!DOCTYPE html>");
        sb.AppendLine("<html lang=\"en\">");
        sb.AppendLine("<head>");
        sb.AppendLine("<meta charset=\"utf-8\">");
        sb.Append("<title>").Append(WebUtility.HtmlEncode(conversation.Title)).AppendLine("</title>");
        sb.AppendLine("<style>");
        sb.AppendLine(@"
body { font: 15px/1.5 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 860px; margin: 2rem auto; padding: 0 1rem; color: #222; background: #fafafa; }
h1 { border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }
.meta { color: #777; font-size: 13px; margin-bottom: 2rem; }
.turn { padding: 0.8rem 1rem; border-radius: 8px; margin: 0.9rem 0; border: 1px solid #e0e0e0; background: #fff; }
.turn.system    { background: #fff8e1; border-color: #ffe082; }
.turn.user      { background: #e3f2fd; border-color: #bbdefb; }
.turn.assistant { background: #f5f5f5; border-color: #e0e0e0; }
.turn.tool      { background: #f3e5f5; border-color: #ce93d8; }
.role { font-weight: 600; font-size: 13px; color: #555; text-transform: uppercase; letter-spacing: 0.04em; }
.ts { color: #999; font-size: 12px; margin-left: 0.6rem; }
.reasoning { border-left: 3px solid #bbb; padding: 0.3rem 0.8rem; color: #555; font-style: italic; margin: 0.5rem 0; white-space: pre-wrap; }
.content { margin-top: 0.5rem; }
.content p { margin: 0.5rem 0; }
.content h1, .content h2, .content h3, .content h4 { margin: 0.9rem 0 0.3rem; }
.content code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.92em; }
.content pre { background: #f0f0f0; padding: 0.7rem; border-radius: 5px; overflow-x: auto; }
.content pre code { background: transparent; padding: 0; }
.content table { border-collapse: collapse; margin: 0.7rem 0; }
.content th, .content td { border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: left; }
.content th { background: #eee; }
.content blockquote { border-left: 3px solid #bbb; margin: 0.5rem 0; padding: 0.1rem 0.8rem; color: #555; }
.content ul, .content ol { margin: 0.3rem 0; padding-left: 1.6rem; }
.attachments { font-size: 13px; color: #777; margin-top: 0.5rem; }
.attachments strong { color: #555; }
");
        sb.AppendLine("</style>");
        sb.AppendLine("</head>");
        sb.AppendLine("<body>");
        sb.Append("<h1>").Append(WebUtility.HtmlEncode(conversation.Title)).AppendLine("</h1>");

        if (options.IncludeTimestamps)
        {
            sb.Append("<div class=\"meta\">Created ")
              .Append(conversation.CreatedAt.ToLocalTime().ToString("yyyy-MM-dd HH:mm"))
              .Append(" · Updated ")
              .Append(conversation.UpdatedAt.ToLocalTime().ToString("yyyy-MM-dd HH:mm"))
              .AppendLine("</div>");
        }

        foreach (var turn in ConversationBranch.ActivePath(conversation))
        {
            if (turn.Role == TurnRole.System && !options.IncludeSystemPrompt) continue;

            var roleClass = turn.Role.ToString().ToLowerInvariant();
            sb.Append("<div class=\"turn ").Append(roleClass).AppendLine("\">");

            sb.Append("<div><span class=\"role\">")
              .Append(WebUtility.HtmlEncode(ConversationBranch.RoleLabel(turn.Role)))
              .Append("</span>");
            if (options.IncludeTimestamps)
            {
                sb.Append("<span class=\"ts\">")
                  .Append(turn.CreatedAt.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss"))
                  .Append("</span>");
            }
            sb.AppendLine("</div>");

            if (options.IncludeReasoning && !string.IsNullOrWhiteSpace(turn.Reasoning))
            {
                sb.Append("<div class=\"reasoning\">")
                  .Append(WebUtility.HtmlEncode(turn.Reasoning))
                  .AppendLine("</div>");
            }

            var cleanContent = ContentSanitizer.StripReasoningSpans(turn.Content);
            if (!string.IsNullOrEmpty(cleanContent))
            {
                // Render markdown → HTML so bold/tables/code/lists in the
                // model's reply survive into the exported document. User
                // turns and system prompts are written by the app/user and
                // typically don't contain markdown, but rendering them
                // through the same pipeline is safe and uniform.
                sb.Append("<div class=\"content\">")
                  .Append(Markdown.ToHtml(cleanContent, _pipeline))
                  .AppendLine("</div>");
            }

            if (options.IncludeAttachmentList && turn.Attachments is { Count: > 0 })
            {
                sb.Append("<div class=\"attachments\"><strong>Attachments:</strong> ");
                for (int i = 0; i < turn.Attachments.Count; i++)
                {
                    if (i > 0) sb.Append(", ");
                    sb.Append(WebUtility.HtmlEncode(turn.Attachments[i].FileName ?? "(unnamed)"));
                }
                sb.AppendLine("</div>");
            }

            sb.AppendLine("</div>");
        }

        sb.AppendLine("</body>");
        sb.AppendLine("</html>");

        await using var writer = new StreamWriter(output, new UTF8Encoding(false), leaveOpen: true);
        await writer.WriteAsync(sb.ToString()).WaitAsync(cancellationToken);
        await writer.FlushAsync(cancellationToken);
    }
}
