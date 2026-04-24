using System;
using System.IO;
using System.Text;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Dedicated debug log for the auto-title / regenerate-title flow.
/// Each call overwrites the file with a fully-annotated trace of the most
/// recent title generation: the rendered prompt, the raw model output,
/// what the reasoning extractor split out as reasoning vs. content, and
/// the final cleaned title. Diagnostic-only — turning off auto-title stops
/// writes entirely. File lives next to <see cref="ErrorLog.LogPath"/>.
/// </summary>
public static class TitleGenLog
{
    public static string LogPath { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "LlamaChat",
        "title-gen.log");

    public static void Write(
        string prompt,
        string rawOutput,
        string reasoning,
        string content,
        string? finalTitle,
        bool promptEndsInThink,
        Exception? exception = null)
    {
        try
        {
            var dir = Path.GetDirectoryName(LogPath);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

            var sb = new StringBuilder();
            sb.Append("title-gen @ ").AppendLine(DateTime.Now.ToString("o"));
            sb.Append("promptEndsInThink (pre-close) = ").Append(promptEndsInThink).AppendLine();
            sb.AppendLine();
            sb.AppendLine("---- PROMPT (full rendered template) ----");
            sb.AppendLine(prompt);
            sb.AppendLine();
            sb.AppendLine("---- RAW OUTPUT (everything the model emitted) ----");
            sb.AppendLine(rawOutput);
            sb.AppendLine();
            sb.AppendLine("---- REASONING (routed to em.Reasoning by the extractor) ----");
            sb.AppendLine(reasoning);
            sb.AppendLine();
            sb.AppendLine("---- CONTENT (routed to em.Content by the extractor — what we summarise from) ----");
            sb.AppendLine(content);
            sb.AppendLine();
            sb.Append("---- FINAL TITLE: ").AppendLine(finalTitle ?? "(null)");
            if (exception is not null)
            {
                sb.AppendLine();
                sb.AppendLine("---- EXCEPTION ----");
                sb.AppendLine(exception.ToString());
            }

            File.WriteAllText(LogPath, sb.ToString());
        }
        catch
        {
            // Diagnostic log — swallow write failures.
        }
    }
}
