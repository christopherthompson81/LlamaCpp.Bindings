using System.Collections.Generic;
using System.Linq;

namespace LlamaCpp.Bindings.LlamaChat.Services.Exporters;

/// <summary>
/// Central list of known <see cref="IConversationExporter"/>s. Menus and
/// the command layer look up exporters by <see cref="IConversationExporter.FormatId"/>;
/// adding a new format means adding one entry here and one
/// <c>MenuItem</c> in each of the File menu and context menu.
/// </summary>
public static class ConversationExporterRegistry
{
    private static readonly IReadOnlyList<IConversationExporter> _all = new IConversationExporter[]
    {
        new MarkdownExporter(),
        new HtmlExporter(),
        new JsonExporter(),
        new TxtExporter(),
        new DocxExporter(),
        new PdfExporter(),
        new XlsxExporter(),
    };

    public static IReadOnlyList<IConversationExporter> All => _all;

    public static IConversationExporter? ByFormatId(string formatId) =>
        _all.FirstOrDefault(e => e.FormatId == formatId);
}
