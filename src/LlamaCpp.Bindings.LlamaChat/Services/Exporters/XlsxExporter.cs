using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ClosedXML.Excel;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services.Exporters;

/// <summary>
/// Exports the active branch as an Excel worksheet — one row per turn.
/// Columns: <c>#</c>, <c>Role</c>, <c>Timestamp</c>, <c>Content</c>,
/// <c>Reasoning</c>, <c>Attachments</c>, plus <c>Tokens</c> and <c>Tok/s</c>
/// if <see cref="ExportOptions.IncludeStats"/> is on. Content is placed
/// verbatim (cell wrap enabled) — markdown becomes literal text in the
/// cell, but the row organisation makes copy-per-turn trivial.
/// </summary>
public sealed class XlsxExporter : IConversationExporter
{
    public string FormatId      => "xlsx";
    public string DisplayName   => "Excel (.xlsx)";
    public string FileExtension => "xlsx";

    public Task ExportAsync(
        Conversation conversation,
        Stream output,
        ExportOptions options,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => WriteXlsx(conversation, output, options), cancellationToken);
    }

    private static void WriteXlsx(Conversation conversation, Stream output, ExportOptions options)
    {
        using var workbook = new XLWorkbook();
        var sheetName = SafeSheetName(conversation.Title);
        var sheet = workbook.Worksheets.Add(sheetName);

        // Header row
        var headers = new System.Collections.Generic.List<string> { "#", "Role" };
        if (options.IncludeTimestamps)     headers.Add("Timestamp");
                                            headers.Add("Content");
        if (options.IncludeReasoning)       headers.Add("Reasoning");
        if (options.IncludeAttachmentList)  headers.Add("Attachments");
        if (options.IncludeStats)           { headers.Add("Tokens"); headers.Add("Tok/s"); }

        for (int i = 0; i < headers.Count; i++)
        {
            var cell = sheet.Cell(1, i + 1);
            cell.Value = headers[i];
            cell.Style.Font.Bold = true;
            cell.Style.Fill.BackgroundColor = XLColor.LightGray;
        }

        int row = 2;
        int index = 1;
        foreach (var turn in ConversationBranch.ActivePath(conversation))
        {
            if (turn.Role == TurnRole.System && !options.IncludeSystemPrompt) continue;

            int col = 1;
            sheet.Cell(row, col++).Value = index++;
            var roleCell = sheet.Cell(row, col++);
            roleCell.Value = ConversationBranch.RoleLabel(turn.Role);
            roleCell.Style.Fill.BackgroundColor = RoleColor(turn.Role);
            roleCell.Style.Font.Bold = true;

            if (options.IncludeTimestamps)
                sheet.Cell(row, col++).Value = turn.CreatedAt.ToLocalTime().DateTime;

            var contentCell = sheet.Cell(row, col++);
            contentCell.Value = ContentSanitizer.StripReasoningSpans(turn.Content);
            contentCell.Style.Alignment.WrapText = true;
            contentCell.Style.Alignment.Vertical = XLAlignmentVerticalValues.Top;

            if (options.IncludeReasoning)
            {
                var reasoningCell = sheet.Cell(row, col++);
                reasoningCell.Value = turn.Reasoning ?? string.Empty;
                reasoningCell.Style.Alignment.WrapText = true;
                reasoningCell.Style.Alignment.Vertical = XLAlignmentVerticalValues.Top;
                reasoningCell.Style.Font.Italic = true;
            }

            if (options.IncludeAttachmentList)
            {
                var names = turn.Attachments is { Count: > 0 }
                    ? string.Join(", ", turn.Attachments.Select(a => a.FileName ?? "(unnamed)"))
                    : string.Empty;
                sheet.Cell(row, col++).Value = names;
            }

            if (options.IncludeStats && turn.Stats is { } s)
            {
                sheet.Cell(row, col++).Value = s.CompletionTokens;
                sheet.Cell(row, col++).Value = s.TokensPerSecond;
            }

            row++;
        }

        // Column widths: narrow index, auto role + timestamp, wide content.
        sheet.Column(1).Width = 4;   // #
        sheet.Column(2).Width = 12;  // Role
        int contentCol = options.IncludeTimestamps ? 4 : 3;
        sheet.Column(contentCol).Width = 80;
        if (options.IncludeTimestamps) sheet.Column(3).Width = 20;
        if (options.IncludeReasoning)  sheet.Column(contentCol + 1).Width = 60;

        sheet.SheetView.FreezeRows(1); // keep header visible while scrolling

        workbook.SaveAs(output);
    }

    private static XLColor RoleColor(TurnRole role) => role switch
    {
        TurnRole.System    => XLColor.LightYellow,
        TurnRole.User      => XLColor.LightCyan,
        TurnRole.Assistant => XLColor.White,
        TurnRole.Tool      => XLColor.Lavender,
        _                  => XLColor.White,
    };

    /// <summary>
    /// Excel sheet names: ≤ 31 chars, no <c>: \ / ? * [ ]</c>. We sanitise
    /// the conversation title into a safe prefix and fall back to
    /// "Conversation" if the title is empty.
    /// </summary>
    private static string SafeSheetName(string title)
    {
        var illegal = new[] { ':', '\\', '/', '?', '*', '[', ']' };
        var clean = new string((title ?? string.Empty).Where(c => !illegal.Contains(c)).ToArray()).Trim();
        if (string.IsNullOrEmpty(clean)) clean = "Conversation";
        return clean.Length > 31 ? clean[..31] : clean;
    }
}
