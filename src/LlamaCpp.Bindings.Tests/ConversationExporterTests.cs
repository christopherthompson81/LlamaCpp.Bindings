using System.Text;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.Services.Exporters;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Shape-level tests for every exporter. We don't re-parse the produced
/// DOCX/PDF/XLSX to verify bytes — that would duplicate the libraries
/// we're trusting. Instead: produce bytes, confirm they're non-empty,
/// confirm textual formats contain the conversation's key strings, and
/// confirm the binary formats start with their expected magic header
/// so we catch accidental truncation or wrong-format plumbing.
/// </summary>
public class ConversationExporterTests
{
    private static Conversation BuildSampleConversation()
    {
        var root = new ChatTurn(
            Guid.NewGuid(), TurnRole.System, "You are a helpful assistant.",
            TurnState.Complete, DateTimeOffset.UtcNow.AddMinutes(-10));

        var userTurn = new ChatTurn(
            Guid.NewGuid(), TurnRole.User, "What's 2 + 2?",
            TurnState.Complete, DateTimeOffset.UtcNow.AddMinutes(-9),
            ParentId: root.Id,
            Attachments: new List<Attachment>
            {
                new(Data: new byte[] { 0x89, 0x50, 0x4E, 0x47 }, MimeType: "image/png", FileName: "diagram.png"),
            });

        var asstTurn = new ChatTurn(
            Guid.NewGuid(), TurnRole.Assistant,
            "The answer is **4**.",
            TurnState.Complete, DateTimeOffset.UtcNow.AddMinutes(-8),
            Reasoning: "User asked a simple arithmetic question.",
            Stats: new TurnStats(
                PromptTokens: 12, CompletionTokens: 8,
                PromptTime: TimeSpan.FromMilliseconds(120),
                GenerationTime: TimeSpan.FromMilliseconds(400)),
            ParentId: userTurn.Id);

        return new Conversation
        {
            Title = "Arithmetic chat",
            CreatedAt = DateTimeOffset.UtcNow.AddMinutes(-11),
            UpdatedAt = DateTimeOffset.UtcNow.AddMinutes(-7),
            Turns = new List<ChatTurn> { root, userTurn, asstTurn },
            ActiveLeafId = asstTurn.Id,
        };
    }

    [Fact]
    public async Task Markdown_Contains_Title_Roles_And_Content()
    {
        var conv = BuildSampleConversation();
        using var ms = new MemoryStream();
        await new MarkdownExporter().ExportAsync(conv, ms, ExportOptions.Default);
        var text = Encoding.UTF8.GetString(ms.ToArray());

        Assert.Contains("# Arithmetic chat", text);
        Assert.Contains("## User",     text);
        Assert.Contains("## Assistant", text);
        Assert.Contains("What's 2 + 2?", text);
        Assert.Contains("The answer is **4**.", text);
        Assert.Contains("diagram.png", text);
        // Reasoning is OFF by default — exports are for sharing.
        Assert.DoesNotContain("User asked a simple arithmetic question", text);
    }

    [Fact]
    public async Task Markdown_Emits_Reasoning_When_Opted_In()
    {
        var conv = BuildSampleConversation();
        using var ms = new MemoryStream();
        await new MarkdownExporter().ExportAsync(conv, ms,
            ExportOptions.Default with { IncludeReasoning = true });
        var text = Encoding.UTF8.GetString(ms.ToArray());

        Assert.Contains("Reasoning", text);
        Assert.Contains("User asked a simple arithmetic question", text);
    }

    [Fact]
    public async Task Markdown_Respects_IncludeSystemPrompt_Toggle()
    {
        var conv = BuildSampleConversation();
        using var withSys = new MemoryStream();
        await new MarkdownExporter().ExportAsync(conv, withSys, ExportOptions.Default);

        using var withoutSys = new MemoryStream();
        await new MarkdownExporter().ExportAsync(conv, withoutSys,
            ExportOptions.Default with { IncludeSystemPrompt = false });

        var withText    = Encoding.UTF8.GetString(withSys.ToArray());
        var withoutText = Encoding.UTF8.GetString(withoutSys.ToArray());

        Assert.Contains("helpful assistant", withText);
        Assert.DoesNotContain("helpful assistant", withoutText);
    }

    [Fact]
    public async Task Txt_Uses_Role_Markers_And_Separators()
    {
        var conv = BuildSampleConversation();
        using var ms = new MemoryStream();
        await new TxtExporter().ExportAsync(conv, ms, ExportOptions.Default);
        var text = Encoding.UTF8.GetString(ms.ToArray());

        Assert.Contains("[USER]", text);
        Assert.Contains("[ASSISTANT]", text);
        Assert.Contains("The answer is **4**.", text);
    }

    [Fact]
    public async Task Html_Produces_Valid_Document_With_Role_Classes()
    {
        var conv = BuildSampleConversation();
        using var ms = new MemoryStream();
        await new HtmlExporter().ExportAsync(conv, ms, ExportOptions.Default);
        var html = Encoding.UTF8.GetString(ms.ToArray());

        Assert.StartsWith("<!DOCTYPE html>", html);
        Assert.Contains("<title>Arithmetic chat</title>", html);
        Assert.Contains("class=\"turn user\"", html);
        Assert.Contains("class=\"turn assistant\"", html);
        // Content is rendered through Markdig — "**4**" becomes <strong>4</strong>.
        Assert.Contains("The answer is <strong>4</strong>", html);
        Assert.DoesNotContain("The answer is **4**.", html);
    }

    [Fact]
    public async Task Json_Roundtrips_Through_Persistence_Shape()
    {
        var conv = BuildSampleConversation();
        using var ms = new MemoryStream();
        await new JsonExporter().ExportAsync(conv, ms, ExportOptions.Default);

        // Deserialise as Conversation; every turn should survive.
        ms.Position = 0;
        var roundtripped = System.Text.Json.JsonSerializer.Deserialize<Conversation>(ms);
        Assert.NotNull(roundtripped);
        Assert.Equal(conv.Title, roundtripped!.Title);
        Assert.Equal(conv.Turns.Count, roundtripped.Turns.Count);
        Assert.Equal(conv.ActiveLeafId, roundtripped.ActiveLeafId);
    }

    [Fact]
    public async Task Docx_Produces_Valid_Zip_With_Word_Part()
    {
        var conv = BuildSampleConversation();
        using var ms = new MemoryStream();
        await new DocxExporter().ExportAsync(conv, ms, ExportOptions.Default);

        Assert.True(ms.Length > 0, "DOCX output should not be empty");
        // .docx is a ZIP; header is 'PK\x03\x04'.
        var bytes = ms.ToArray();
        Assert.Equal(0x50, bytes[0]); // P
        Assert.Equal(0x4B, bytes[1]); // K

        // Re-open as a WordprocessingDocument and confirm the main body
        // contains our content — catches any accidental corruption.
        ms.Position = 0;
        using var doc = DocumentFormat.OpenXml.Packaging.WordprocessingDocument.Open(ms, false);
        var bodyXml = doc.MainDocumentPart!.Document.Body!.InnerText;
        Assert.Contains("Arithmetic chat", bodyXml);
        Assert.Contains("What's 2 + 2?", bodyXml);
        // Markdown is rendered — "**4**" arrives as a bold run; the
        // asterisks themselves are consumed. Visible text: "The answer is 4."
        Assert.Contains("The answer is 4.", bodyXml);
    }

    [Fact]
    public async Task Pdf_Produces_Valid_Pdf_Header()
    {
        var conv = BuildSampleConversation();
        using var ms = new MemoryStream();
        await new PdfExporter().ExportAsync(conv, ms, ExportOptions.Default);

        Assert.True(ms.Length > 0, "PDF output should not be empty");
        var bytes = ms.ToArray();
        // %PDF- magic
        Assert.Equal((byte)'%', bytes[0]);
        Assert.Equal((byte)'P', bytes[1]);
        Assert.Equal((byte)'D', bytes[2]);
        Assert.Equal((byte)'F', bytes[3]);
    }

    [Fact]
    public async Task Xlsx_Produces_Valid_Zip_And_Readable_Cells()
    {
        var conv = BuildSampleConversation();
        using var ms = new MemoryStream();
        await new XlsxExporter().ExportAsync(conv, ms, ExportOptions.Default);

        Assert.True(ms.Length > 0);
        var bytes = ms.ToArray();
        Assert.Equal(0x50, bytes[0]); // P (ZIP)
        Assert.Equal(0x4B, bytes[1]); // K

        ms.Position = 0;
        using var wb = new ClosedXML.Excel.XLWorkbook(ms);
        var sheet = wb.Worksheets.First();
        // Header + at least one row per turn (minus system if filtered —
        // default includes it, so 3 turns = 4 rows total).
        Assert.True(sheet.LastRowUsed()!.RowNumber() >= 2);

        // Find "Role" column and confirm at least one "User" cell.
        var headerRow = sheet.Row(1).CellsUsed().Select(c => c.GetString()).ToList();
        Assert.Contains("Role", headerRow);
        Assert.Contains("Content", headerRow);

        // Confirm content made it into a cell somewhere.
        bool foundContent = false;
        foreach (var cell in sheet.CellsUsed())
        {
            if (cell.GetString().Contains("The answer is **4**."))
            {
                foundContent = true;
                break;
            }
        }
        Assert.True(foundContent, "Expected assistant content in an xlsx cell");
    }

    [Fact]
    public void Registry_Lists_All_Known_Formats_And_Looks_Up_By_Id()
    {
        var ids = ConversationExporterRegistry.All.Select(e => e.FormatId).ToArray();
        Assert.Contains("markdown", ids);
        Assert.Contains("html",     ids);
        Assert.Contains("json",     ids);
        Assert.Contains("txt",      ids);
        Assert.Contains("docx",     ids);
        Assert.Contains("pdf",      ids);
        Assert.Contains("xlsx",     ids);

        Assert.NotNull(ConversationExporterRegistry.ByFormatId("markdown"));
        Assert.Null(ConversationExporterRegistry.ByFormatId("does-not-exist"));
    }

    [Fact]
    public async Task Content_Think_Tags_Are_Stripped_From_All_Text_Formats()
    {
        // A turn whose Content still carries <think>…</think> — can happen
        // when generation ran with ExtractReasoning=false, or when the
        // streaming extractor missed a malformed tag. Exports must hide it
        // regardless.
        var id = Guid.NewGuid();
        var conv = new Conversation
        {
            Title = "Leaky think",
            Turns = new List<ChatTurn>
            {
                new(id, TurnRole.Assistant,
                    "Hello.\n<think>I should be quick here.</think>\n\nAnswer: 42.",
                    TurnState.Complete, DateTimeOffset.UtcNow),
            },
            ActiveLeafId = id,
        };

        async Task<string> RunAsync(IConversationExporter exp)
        {
            using var ms = new MemoryStream();
            await exp.ExportAsync(conv, ms, ExportOptions.Default);
            return Encoding.UTF8.GetString(ms.ToArray());
        }

        foreach (var exp in new IConversationExporter[]
        {
            new MarkdownExporter(), new TxtExporter(),
            new HtmlExporter(),
        })
        {
            var text = await RunAsync(exp);
            Assert.Contains("Hello.", text);
            Assert.Contains("Answer: 42.", text);
            Assert.DoesNotContain("<think>", text);
            Assert.DoesNotContain("I should be quick", text);
        }

        // DOCX: re-open and scan InnerText
        using (var ms = new MemoryStream())
        {
            await new DocxExporter().ExportAsync(conv, ms, ExportOptions.Default);
            ms.Position = 0;
            using var doc = DocumentFormat.OpenXml.Packaging.WordprocessingDocument.Open(ms, false);
            var body = doc.MainDocumentPart!.Document.Body!.InnerText;
            Assert.Contains("Hello.", body);
            Assert.Contains("Answer: 42.", body);
            Assert.DoesNotContain("I should be quick", body);
        }

        // XLSX: scan cells
        using (var ms = new MemoryStream())
        {
            await new XlsxExporter().ExportAsync(conv, ms, ExportOptions.Default);
            ms.Position = 0;
            using var wb = new ClosedXML.Excel.XLWorkbook(ms);
            var sheet = wb.Worksheets.First();
            var allText = string.Join("\n", sheet.CellsUsed().Select(c => c.GetString()));
            Assert.Contains("Answer: 42.", allText);
            Assert.DoesNotContain("I should be quick", allText);
        }
    }

    [Fact]
    public async Task Html_Renders_Markdown_Tables_Bold_And_Code()
    {
        var id = Guid.NewGuid();
        var conv = new Conversation
        {
            Title = "Markdown rendering",
            Turns = new List<ChatTurn>
            {
                new(id, TurnRole.Assistant,
                    "### Section\n\n**bold** and *italic* and `code`.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n",
                    TurnState.Complete, DateTimeOffset.UtcNow),
            },
            ActiveLeafId = id,
        };

        using var ms = new MemoryStream();
        await new HtmlExporter().ExportAsync(conv, ms, ExportOptions.Default);
        var html = Encoding.UTF8.GetString(ms.ToArray());

        Assert.Contains("<h3", html);
        Assert.Contains("<strong>bold</strong>", html);
        Assert.Contains("<em>italic</em>", html);
        Assert.Contains("<code>code</code>", html);
        Assert.Contains("<table>", html);
        Assert.Contains("<th>A</th>", html);
        Assert.Contains("<td>1</td>", html);
    }

    [Fact]
    public async Task Docx_Renders_Markdown_Tables_And_Bold_Runs()
    {
        var id = Guid.NewGuid();
        var conv = new Conversation
        {
            Title = "Markdown rendering",
            Turns = new List<ChatTurn>
            {
                new(id, TurnRole.Assistant,
                    "# Heading\n\n**bold** text and `inline code`.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n",
                    TurnState.Complete, DateTimeOffset.UtcNow),
            },
            ActiveLeafId = id,
        };

        using var ms = new MemoryStream();
        await new DocxExporter().ExportAsync(conv, ms, ExportOptions.Default);
        ms.Position = 0;
        using var doc = DocumentFormat.OpenXml.Packaging.WordprocessingDocument.Open(ms, false);
        var body = doc.MainDocumentPart!.Document.Body!;

        // Plain text check — "**bold**" should appear as "bold" (asterisks stripped).
        var text = body.InnerText;
        Assert.Contains("Heading", text);
        Assert.Contains("bold", text);
        Assert.DoesNotContain("**bold**", text);
        // Table structure should exist.
        Assert.True(body.Descendants<DocumentFormat.OpenXml.Wordprocessing.Table>().Any(),
            "Expected at least one Table element in the body");
        // Bold should be emitted as a RunProperties.Bold somewhere.
        Assert.True(body.Descendants<DocumentFormat.OpenXml.Wordprocessing.Bold>().Any(),
            "Expected at least one Bold run");
    }

    [Fact]
    public async Task Pdf_Render_Does_Not_Fail_On_Complex_Markdown()
    {
        // No re-parse of PDF structure — just confirm generation
        // completes without throwing on the block+inline combinations we
        // expect in real chat output.
        var id = Guid.NewGuid();
        var conv = new Conversation
        {
            Title = "PDF markdown smoke",
            Turns = new List<ChatTurn>
            {
                new(id, TurnRole.Assistant,
                    "# H1\n## H2\n\n" +
                    "Paragraph with **bold**, *italic*, `inline`, and a [link](https://example.com).\n\n" +
                    "```csharp\nvar x = 42;\n```\n\n" +
                    "- bullet one\n- bullet two\n\n" +
                    "1. first\n2. second\n\n" +
                    "> A quote.\n\n" +
                    "| A | B |\n|---|---|\n| 1 | 2 |\n\n" +
                    "---\n\nTrailing paragraph.",
                    TurnState.Complete, DateTimeOffset.UtcNow),
            },
            ActiveLeafId = id,
        };

        using var ms = new MemoryStream();
        await new PdfExporter().ExportAsync(conv, ms, ExportOptions.Default);
        Assert.True(ms.Length > 1000, "PDF should have substantial content for this much markdown");
        Assert.Equal((byte)'%', ms.ToArray()[0]);
    }

    [Fact]
    public void Sanitizer_Strips_Multiple_And_Dangling_Think_Tags()
    {
        var result = ContentSanitizer.StripReasoningSpans(
            "<think>first</think>visible<thinking>second</thinking> more");
        Assert.Equal("visible more", result);

        // Un-closed (generation cut off mid-block): drop to end.
        var dangling = ContentSanitizer.StripReasoningSpans(
            "kept text\n<think>never closed");
        Assert.Equal("kept text", dangling);

        // Null / empty passes through.
        Assert.Equal(string.Empty, ContentSanitizer.StripReasoningSpans(null));
        Assert.Equal(string.Empty, ContentSanitizer.StripReasoningSpans(""));

        // Case-insensitive + attribute-tolerant.
        Assert.Equal("ok",
            ContentSanitizer.StripReasoningSpans("<Think id=\"a\">x</Think>ok"));
    }

    [Fact]
    public async Task Exporters_Only_Serialise_Active_Branch_Not_Siblings()
    {
        // Build a conversation with a sibling the user abandoned — it must
        // not appear in anything except the JSON export (which is a full
        // tree dump by design).
        var conv = BuildSampleConversation();
        var orphan = new ChatTurn(
            Guid.NewGuid(), TurnRole.Assistant,
            "This sibling reply should NEVER appear in exports.",
            TurnState.Complete, DateTimeOffset.UtcNow,
            ParentId: conv.Turns[1].Id); // sibling of the "The answer is **4**." turn
        var convWithSibling = conv with { Turns = new List<ChatTurn>(conv.Turns) { orphan } };

        using var md = new MemoryStream();
        await new MarkdownExporter().ExportAsync(convWithSibling, md, ExportOptions.Default);
        Assert.DoesNotContain("NEVER", Encoding.UTF8.GetString(md.ToArray()));

        // JSON is the round-trip format, it SHOULD include the sibling.
        using var json = new MemoryStream();
        await new JsonExporter().ExportAsync(convWithSibling, json, ExportOptions.Default);
        Assert.Contains("NEVER", Encoding.UTF8.GetString(json.ToArray()));
    }
}
