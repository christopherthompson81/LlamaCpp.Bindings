# llama-server webui — feature + theming burn-down

Reference snapshot of llama.cpp/tools/server/webui as of commit `1d6d4cf7a5361046f778414c5b1f5ecbc07eeb77`, to match in LlamaCpp.Bindings.LlamaChat.

**State legend**

- `- [x]` **done** — implemented and wired up.
- `- [ ]` **TODO** — actionable now. No precursor needed.
- `- [~]` **deferred** — needs a precursor first (multimodal bindings, MCP client, tree-shaped transcript, etc.). The precursor is called out in the item's annotation.
- `- [-]` **N/A** — doesn't translate to a desktop Avalonia app (hash routing, mobile breakpoints, server-side knobs, in-browser-only concerns).

## Source layout

```
src/
├── app.css                          # Root CSS with design tokens (OKLCH color vars, semantic tokens)
├── routes/
│   ├── +layout.svelte               # Global layout with sidebar, routing, keyboard shortcuts (Ctrl+K search, Ctrl+Shift+O new chat, Ctrl+Shift+E edit)
│   ├── +page.svelte                 # Home/landing page (new chat entry point)
│   ├── chat/[id]/
│   │   ├── +page.svelte             # Chat conversation view (main UI)
│   │   └── +page.ts                 # Load server data, messages, model info
│   └── +error.svelte                # Error boundary page
├── lib/
│   ├── components/
│   │   ├── app/                     # Feature components
│   │   │   ├── chat/                # Chat UI modules
│   │   │   ├── mcp/                 # MCP server configuration
│   │   │   ├── models/              # Model selection
│   │   │   ├── actions/             # Message action buttons
│   │   │   ├── badges/              # Status badges (modality, statistics)
│   │   │   ├── content/             # Markdown rendering, code blocks
│   │   │   ├── dialogs/             # Modal dialogs
│   │   │   ├── forms/               # Form inputs
│   │   │   ├── navigation/          # Dropdowns, menus
│   │   │   ├── server/              # Loading/error splash screens
│   │   │   └── misc/                # Utilities, overlays
│   │   └── ui/                      # shadcn-svelte components (button, input, dialog, etc.)
│   ├── stores/                      # Svelte reactive state (chat, conversations, settings, mcp, models)
│   ├── services/                    # API communication, parameter sync, MCP orchestration
│   ├── markdown/                    # Remark/Rehype pipeline (KaTeX, syntax highlighting, links)
│   ├── styles/                      # Custom CSS (katex-custom.scss)
│   ├── utils/                       # Helpers (clipboard, file handling, markdown processing)
│   ├── enums/                       # TypeScript enums (roles, field types, keyboard keys, etc.)
│   ├── contexts/                    # Svelte context providers
│   ├── hooks/                       # Reactive hooks (mobile detection, auto-scroll)
│   ├── constants/                   # Config (settings keys, icons, file types, color modes, etc.)
│   └── types/                       # TypeScript interfaces
└── static/
    └── favicon.svg                  # Favicon (embedded as base64 data URL in build)
```

## Features

### 1. App shell

- [-] **Hash-based routing** — N/A (desktop single-window app; no URL routing). Deep-links to specific conversations could be simulated later if we add IPC.
- [x] **Global keyboard shortcuts** — `Window.KeyBindings` in `MainWindow.axaml` + `OnKeyDown` override in code-behind. Ctrl+N / Ctrl+Shift+O new chat, Ctrl+K focuses search, Ctrl+B toggles sidebar, Ctrl+L loads, Ctrl+, settings. Ctrl+Shift+E (edit title) deferred — use context menu "Rename" or F2 instead.
- [x] **Sidebar layout with collapse** — left-pinned Border bound to `IsSidebarVisible`; Ctrl+B toggles. Fixed 280px width (no responsive breakpoints in desktop).
- [-] **Mobile detection hook** — N/A.
- [-] **Responsive breakpoint** — N/A.
- [x] **Header region** — toolbar row below menu shows Profile combo + Load/Unload/Settings + live `ModelSummary` (filename · ctx · layers · template). Serves as the `ChatScreenHeader` equivalent.
- [x] **Sidebar header/footer slots** — RowDefinitions="Auto,Auto,*" lays out New button + Search + list; footer slot free for future (model-status pill, etc.).
- [x] **Main content inset** — main area uses `Margin="16,12,16,12"` for consistent gutter against the sidebar.
- [~] **Tooltip provider context** — deferred. Waits on a tooltip component pass; no visible tooltips in the app yet.
- [x] **Error boundary** — `Services/ErrorBoundary.cs` hooks the three exception funnels at `App.OnFrameworkInitializationCompleted`: `Dispatcher.UIThread.UnhandledException` (UI-thread handlers, layout, rendering), `TaskScheduler.UnobservedTaskException` (forgotten async tasks), and `AppDomain.CurrentDomain.UnhandledException` (last-chance fatals from any thread). UI-thread exceptions are marked `Handled=true` and routed to the error splash; unobserved-task exceptions are treated as non-fatal (log + toast). VMs can also call `ErrorBoundary.ReportFatal`/`ReportNonFatal` to funnel caught exceptions through the same path. `Services/ErrorLog.cs` is the single writer for `last-error.log` (replaces the inline writer that used to live in `MainWindowViewModel.GenerateAssistantReplyAsync`).

### 2. Conversation list

- [x] **List all conversations** — `MainWindowViewModel.Conversations`, persisted to `conversations.json` by `ConversationStore`. Sorted by `UpdatedAt` descending in `FilteredConversations`.
- [x] **Create new conversation** — `NewConversationCommand` (Ctrl+N / Ctrl+Shift+O, File menu item). Auto-selects the new one.
- [x] **Rename conversation** — inline edit in sidebar (right-click → Rename or F2-style — currently via context menu). Commit on Enter/LostFocus via `EndRenameCommand`; Escape cancels.
- [x] **Delete conversation** — right-click → Delete, or Chat menu. No confirmation yet — deferred.
- [x] **Conversation tree/forking** — `ChatTurn` gained `ParentId` and `Conversation` gained `ActiveLeafId`. `ConversationViewModel` keeps `AllMessages` as the full tree and recomputes `Messages` as the root → active-leaf path. Tree mutations go through `AppendToActivePath` / `AddSibling` / `AddChildOf` / `RemoveSubtree` / `SwitchToSibling`. Legacy files with flat `Turns` load as a linear chain (each turn parented to the previous one's Id). Forking happens implicitly on Retry and user-message edits.
- [x] **Search conversations** — case-insensitive substring on Title + Preview, live-filtered as user types. Ctrl+K focuses the search box.
- [x] **Active conversation highlight** — `ListBox.SelectedItem` bound to `SelectedConversation`; styled via existing `ListBoxItem:selected` accent from Theme/Controls.
- [x] **Conversation preview text** — `ConversationViewModel.Preview` — first user message truncated to 80 chars, rendered below the title in the sidebar.
- [x] **Pinned/recent grouping** — `Conversation.Pinned` bool + `ConversationViewModel.Pinned` + `TogglePinnedCommand` (context menu "Pin / unpin") + `RebuildFilteredConversations` sorts by `Pinned` desc then `UpdatedAt` desc. A 📌 indicator appears next to pinned items' titles in the sidebar.
- [x] **Export/import conversations** — `ConversationStore.ExportToFile` / `ImportFromFile`, `DialogService.PickExportFileAsync` / `PickImportFileAsync` via `StorageProvider`. File menu gains Export / Import items. Import de-dupes by conversation Id.

### 3. Message rendering

- [x] **Markdown pipeline** — Markdig → Avalonia control tree, no HTML intermediate (no WebView on Avalonia). Implemented in `Services/MarkdownRenderer.cs`. Pipeline uses:
  - `UseEmphasisExtras()` — strikethrough via `~~...~~`, sub/sup
  - `UseAutoLinks()` — bare URLs render as styled links
  - `UsePipeTables()` — GFM tables via `Markdig.Extensions.Tables`
  - `UseTaskLists()` — `[ ]` / `[x]` rendered with `☐`/`☑` glyph markers
- [x] **Block-level coverage** — paragraphs, ATX headings (h1-h3 distinct sizes, h4+ body-weight), bullet/ordered lists with configurable start index, blockquotes (3px left border + 85% opacity), fenced + indented code blocks, thematic breaks, pipe tables (bordered Grid).
- [x] **Inline-level coverage** — Literal, CodeInline (monospace run w/ CodeBackground), EmphasisInline (bold/italic), strikethrough via `TextDecorations`, LinkInline + AutolinkInline (coloured Ring + underline — inert in v1), LineBreak, HtmlInline (raw tag shown as text = no HTML passthrough), HtmlEntityInline.
- [x] **KaTeX math rendering** — Markdig's `UseMathematics()` emits `MathInline` / `MathBlock`; `Services/MathRenderer.cs` wraps `CSharpMath.SkiaSharp 1.0.0-pre.1` to render LaTeX → `SKBitmap` → PNG stream → `Avalonia.Media.Imaging.Bitmap`. Block math renders as a centred `Image` at display-style size 20; inline math as an `InlineUIContainer(Image)` at text-style size 16 with centred baseline. Foreground colour baked from the theme's `Foreground` brush at render time; cache keyed on `(latex, displayStyle, fontSize, argb)` so light/dark flips regenerate cleanly. AvaloniaMath considered but hard-constrains to `Avalonia <12`; SkiaSharp path is version-agnostic.
- [x] **Syntax highlighting** — `Services/CodeHighlighter.cs` wraps `ColorCode.Core`'s `LanguageParser` into a `(Text, Scope)` token stream; `MarkdownRenderer.BuildCodeBlock` emits a `Run` per token with Foreground bound to a `Syntax*` theme brush (`SyntaxKeyword` / `String` / `Comment` / `Number` / `Type` / `Operator` / `Preprocessor` / `Tag`). GitHub Light/Dark palettes in `Theme/Tokens.axaml`. Languages covered: C#, C++, CSS, F#, HTML, Java, JS/TS/JSON, Python, PHP, PowerShell, SQL, XML/XAML, Markdown, Haskell, MATLAB, VB.NET, Fortran — everything ColorCode.Core ships. Rust/Go/Ruby/YAML/bash fall through to plain text.
- [x] **Code-block copy button** — ghost+sm `Copy` button in the code-block header row. Uses `DialogService.CopyToClipboardAsync`.
- [x] **Code-block preview/expand dialog** — `Views/CodePreviewDialog.cs` — modal window (1000×700, centre-of-owner) that re-renders the same highlighted block at full size, with a footer Copy + Close and Escape-to-close. "Expand" button in each code block opens it.
- [x] **Incomplete code block / mid-stream robustness** — `Markdown.Parse` is wrapped in `try`; if parsing fails (e.g. unclosed fence mid-stream) we fall back to showing the raw text in a `TextBlock` instead of leaving the bubble blank. Markdig actually tolerates most mid-stream cases — the try is belt-and-braces.
- [x] **Image display from attachments** — attached images render as a `WrapPanel` thumbnail row above the bubble content in the user bubble template, plus a matching 72×72 strip above the compose text box. `byte[] → Avalonia.Bitmap` via `Services/AttachmentThumbnailConverter.cs` (registered as `AttachmentThumbnail` resource in `App.axaml`). Conversations persist attachments inline — `System.Text.Json` serializes `byte[]` as base64 automatically, so a round-trip through `conversations.json` preserves the payload.
- [x] **Image error fallback** — `AttachmentThumbnailConverter` returns null on decode exception; the `Image` control renders as an empty 72×72 rectangle (with the `Muted` background still visible) without taking down the bubble.
- [x] **GFM tables** — rendered as a bordered Grid with Auto columns. Header row detected via Markdig's `TableRow.IsHeader`, emitted SemiBold.
- [~] **Footnotes** — deferred. Not in our `MarkdownPipelineBuilder` extensions; webui doesn't support them either. Low priority.
- [x] **Mermaid diagrams** — flowchart v1 lands in `Services/Mermaid/`. `FlowchartParser` covers `graph / flowchart TD/TB/BT/LR/RL` headers, rectangle / rounded / stadium / circle / rhombus node shapes, `-->` / `---` / `-.->` / `==>` edges, pipe-form edge labels, chains, inline node defs, comments. `FlowchartLayout` runs a small Sugiyama (DFS back-edge cycle removal, longest-path layering, barycenter crossing reduction, evenly-spaced coords per-layer-centred). `FlowchartRenderer` walks the laid-out graph into an Avalonia `Canvas` with `Rectangle`/`Ellipse`/`Path` shapes + cubic Bezier edges whose tangent axis matches the flowchart direction + tangent-aligned arrowheads. Theme-aware brushes throughout. Parser + layout are Avalonia-free (36 unit tests in the bindings test project). Deferred: subgraphs, `A & B --> C` shorthand, obstacle-aware edge routing (edges can cross unrelated nodes when siblings converge — [#11](https://github.com/christopherthompson81/LlamaCpp.Bindings/issues/11)), and the other graph/linear/chart diagram types — the renderer is structured to accept those as node-template + edge-decorator registries once prioritised.
- [x] **Streaming cursor/indicator** — `MarkdownView` exposes an `IsStreaming` StyledProperty; bubble template binds it from `MessageViewModel.IsStreaming`. When true, an `InlineUIContainer(TextBlock.cursor)` carrying `▌` is appended to the live tail. Blink animation (1s cycle, opacity 1 → 0.2 → 1) lives on the `TextBlock.cursor` style in `Theme/Controls.axaml`.
- [x] **HTML sanitisation** — inert by construction: `HtmlInline` renders the raw tag string as literal text, so injected `<script>` etc. never becomes a control.
- [x] **Streaming-safe re-render throttling** — `MarkdownView` coalesces property changes through a 40ms `DispatcherTimer` debounce to avoid thrashing the layout pass on every decoded token (~8ms intervals at 120 tok/s → ~4-5 tokens per render).

### 4. Message actions

- [x] **Copy message** — `CopyMessageCommand` → `DialogService.CopyToClipboardAsync` (wraps `Avalonia.Input.Platform.ClipboardExtensions.SetTextAsync`). Button bound in the bubble template.
- [x] **Edit message** — inline edit surface (TextBox + Save / Cancel) replaces the read-only bubble body while `MessageViewModel.IsEditing`. On Save: user message → truncate downstream + regenerate; assistant message → overwrite in place. `EditDraft` is the buffer; `Content` only commits on Save.
- [x] **Regenerate response** — `RegenerateMessageCommand` truncates the transcript starting at the target message (or right after, if the target was a user turn), `ClearKv()`s the session, and re-enters `GenerateAssistantReplyAsync`. Extracted from `SendAsync` so both share the same streaming path.
- [x] **Continue generation** — `ContinueMessageCommand` on `MainWindowViewModel`, Continue button in the assistant bubble's action bar. Routes to `ChatSession.StreamContinuationAsync` which trims the last cached token from the KV and re-decodes it via the back-off-by-one path in `LlamaGenerator` so sampler priming happens against the full cached history. Only the *last* assistant message can be extended — clicking it on an older bubble surfaces a toast; extending an older message would require re-decoding the intervening transcript which the v1 flow doesn't cover.
- [x] **Delete message** — `DeleteMessageCommand` removes the single clicked message (not downstream) and clears the KV cache. No confirmation dialog yet.
- [x] **Branch navigation** — compact `‹ N/M ›` pill in each bubble header when `HasSiblings` is true. `SwitchPrevSibling` / `SwitchNextSibling` commands on `MainWindowViewModel` cycle through the siblings; `ConversationViewModel.SwitchToSibling` walks down to the most-recently-added leaf of the chosen branch so switching restores the sub-path through that branch, not just the single turn. Clears the KV cache on switch — next turn re-prefills through the prefix-cache path.
- [x] **Fork conversation** — effectively covered by the tree + sibling-nav pair. Retry on an assistant creates a sibling reply under the same user turn; editing a user message adds a sibling user turn + regenerates under it. Both preserve the original branch. A dedicated "duplicate up to here into a new conversation" command is still a nice-to-have but not load-bearing.
- [x] **Message deletion dialog** — `DialogService.ConfirmAsync` + `ConfirmDialog` (multi-choice). `DeleteMessageAsync` skips the prompt when there's nothing downstream; otherwise offers Cancel / Just this / This + N after.

### 5. Compose

- [x] **Textarea input** — the compose TextBox in `MainWindow.axaml` with `MinHeight=56 MaxHeight=200 TextWrapping=Wrap AcceptsReturn=False` + custom Enter/Shift+Enter handling. Webui's auto-height sizing via `field-sizing: content` would be a polish pass; the fixed min/max is serviceable.
- [x] **File attachment picker** — `AttachImagesCommand` on `MainWindowViewModel` → `DialogService.PickImageFilesAsync` (`OpenFilePickerAsync`, `AllowMultiple=true`, filters `*.jpg;*.png;*.bmp;*.gif;*.webp`). The 📎 button in the compose bar is gated on `CanAttachImages` — disabled when the loaded profile has no mmproj.
- [x] **Attachment preview** — the same 72×72 thumbnail Border used in user bubbles sits above the compose `TextBox`, one per `PendingAttachments` entry, each with a × remove button (`RemovePendingAttachmentCommand`). `HasPendingAttachments` collapses the row when empty.
- [~] **Attachment list modal** — deferred. The inline compose-bar strip covers the typical few-images-per-message case; a dedicated gallery modal is a polish pass that only matters once users attach many items at once.
- [~] **Audio recording** — partially landed. Audio **file attachment** works end-to-end via the same compose paperclip + drag-drop + `ChatSession` multimodal path as images; `MtmdBitmap.FromAudioSamples` is the public wrapper over `mtmd_bitmap_init_from_audio` for future callers with raw PCM-F32 buffers. What's still deferred is **mic capture** — the compose bar doesn't have a record button yet, because that needs a platform-specific audio-capture pipeline (NAudio on Windows, PortAudio/ALSA cross-platform).
- [x] **Drag-and-drop file upload** — `DragDrop.AllowDrop="True"` on the main `Window`; `OnComposeDragOver` in `MainWindow.axaml.cs` accepts the drop only when the payload contains `DataFormat.File` **and** `CanAttachImages`; `OnComposeDrop` walks `IDataTransfer.TryGetFiles()` and feeds local paths to `MainWindowViewModel.TryAddPendingImage`.
- [x] **Paste handling (files)** — Ctrl+V in the compose `TextBox` is intercepted by `OnComposeKeyDown`: if the clipboard has a bitmap payload (`IClipboard.TryGetBitmapAsync`), the bitmap is re-encoded to PNG bytes and queued as an attachment, and the default text-paste is suppressed. Non-image clipboard content (plain text, mixed) falls through to the `TextBox`'s native paste.
- [x] **MCP prompt picker** — `Views/McpPromptPickerDialog.cs` — modal with a server/prompt ComboBox, dynamic argument fields (name/required/description), Insert button fetches `prompts/get` and inserts the rendered text into the compose box.
- [x] **MCP resource picker** — `Views/McpResourcePickerDialog.cs` — split-pane resource browser with search, Preview reads the content inline via `resources/read`, Attach inserts a `<!-- resource: uri -->\nbody` block into the compose box.
- [x] **Slash command support** — `SendAsync` intercepts leading `/` and dispatches: `/clear` + `/reset` wipe the current conversation, `/new` creates one, `/settings` opens prefs, `/help` / `/?` show the shortcuts overlay, `/copy` copies the last assistant message. Unknown one-word commands surface a warning toast listing the set; anything with a space falls through to the model.
- [x] **Token count display** — `UserInputTokenCount` (debounced 150 ms via `DispatcherTimer`) tokenises the compose text with `Session.Model.Vocab.Tokenize`. Shown under the compose TextBox as `N tok`, hidden when no model is loaded.
- [x] **Send button state** — disabled/enabled bound to `CanSend`.
- [x] **Stop generation button** — Send and Stop share one column in the compose grid; `IsVisible` swaps on `IsGenerating`. Stop uses the `destructive` button class.

### 6. Chat settings

- [x] **Settings sidebar panel** — `SettingsWindow.axaml` — modal `ShowDialog(owner)`, left-placed tab strip with Profiles + Display. Header with "Settings" title, footer with Save/Close + status line.
- [~] **General tab** — N/A shape. Theme/API-key/paste-threshold/PDF-as-image are web-only concerns; System Message promoted to its own section inside ProfileEditorView (per-profile instead of global). "Continue button" deferred with continue-generation itself.
- [x] **Display tab** — three toggles, all wired: `AutoScroll`, `ShowMessageStats`, `ShowReasoningInProgress`. Code-block theme and copy-as-plain-text deferred until we add the code-block toolbar. `AppSettings` record + `AppSettingsStore` persist to `app-settings.json` alongside profiles.
- [x] **Sampling tab** — already implemented in `Views/SamplerPanelView.axaml` (embedded inside ProfileEditorView): temperature + dynatemp, top-k/p, min-p, typical, top-n-σ, XTC, DRY, repetition/frequency/presence penalties.
- [x] **Advanced tab** — merged with Sampling: seed, dynamic-temp range/exponent, mirostat v1/v2 + tau/eta, grammar (GBNF). "Tokens to keep" + "ignore EOS" deferred (both require LlamaGenerator API extensions).
- [x] **Tools/MCP tab** — third tab in `SettingsWindow`. `Views/McpSettingsView.axaml` renders server list + editor with URL/Headers/enable controls, live connection status, and an expandable per-tool view showing each tool's JSON input schema.
- [x] **System prompt** — per-profile multi-line TextBox at the top of the profile editor. Prepended to every transcript as a `TurnRole.System` turn when that profile is loaded.
- [x] **Response format tab** — new "Response format" tab in `SettingsWindow`. Mode dropdown with four options (`Off`, `Json`, `JsonSchema`, `Gbnf`) picks how the free-text editor is interpreted; the editor hides entirely for `Off` / `Json` where no free-text is needed. A live `CompiledGbnfPreview` pane on the right shows the grammar that will feed the sampler — parses + recompiles on every keystroke, with compile errors surfaced inline. JSON Schema compilation is a pure-C# port of `llama.cpp/common/json-schema-to-grammar.cpp` in `src/LlamaCpp.Bindings/JsonSchemaToGbnf.cs` + `JsonSchemaToGbnf.Converter.cs` — full coverage of the reference including `$ref` chains, `oneOf`/`anyOf`/`allOf`, integer ranges, string formats (date / time / date-time / uuid), anchored `pattern` regex → GBNF, and `additionalProperties` via trie-based `_not_strings`. 16 smoke tests in `JsonSchemaToGbnfTests.cs`. `SamplerFactory.BuildResponseFormatGrammar` dispatches on the mode; legacy profiles that set the pre-refactor `GbnfGrammar` slot get migrated transparently to `Gbnf` mode.
- [-] **Parameter sync source indicator** — N/A. Settings are local-only; there's no server-default/session-override hierarchy to visualise.
- [x] **Reset to defaults** — `ResetSamplerDefaults` command on the Profiles tab footer restores `SamplerSettings.Default` + fresh `GenerationSettings` on the current profile. Load settings and system prompt are preserved.
- [x] **Settings persistence** — three stores: `ProfileStore` (profiles.json), `AppSettingsStore` (app-settings.json), `ConversationStore` (conversations.json) — all under `$XDG_CONFIG_HOME/LlamaChat/`. Auto-save on dialog close; explicit Save button covers the in-dialog case.
- [-] **User overrides tracking** — N/A. Not meaningful without a server-default baseline to diff against.

### 7. Tool calling / MCP

Streamable HTTP transport only (most common for HTTP-facing MCP servers; stdio servers can be front-ended by a local HTTP shim). Hand-rolled JSON-RPC client in `Services/McpClient.cs` — no external SDK dependency. Tool-calling uses the Hermes-style `<tool_call>{...}</tool_call>` wrapper, which is what modern Qwen/DeepSeek/Llama tool-use Jinja templates produce.

- [x] **MCP server add** — `Add` button in the settings tab inserts a disabled-by-default stub; user fills in URL + headers and clicks Save.
- [x] **MCP server list** — `Views/McpSettingsView.axaml` left pane is a ListBox of `McpServerEntry` rows (name + URL + state pill). Backed by `McpClientService.Instance.Servers`.
- [x] **MCP server enable/disable** — `Connect on startup` CheckBox in the editor toggles `Config.Enabled`, which disconnects/reconnects through `McpClientService.ToggleEnabledAsync`.
- [x] **MCP server delete** — destructive `Delete` button; no confirmation dialog in v1 (deferred — same as conversation delete).
- [x] **MCP server edit** — URL + headers + name edit in place on the selected server; `Save` reconnects with the new config. Headers entered as `Key: value` one per line.
- [x] **MCP connection status indicator** — `StateLabel` pill on each list item + the editor status panel shows `Idle / Connecting… / Ready / Error / Disabled` plus the last error text.
- [x] **MCP tool list** — per-server Expander list of tools; each shows the tool's `description` + a pretty-printed `inputSchema` JSON in a code block. Pretty-printing via `Services/JsonPrettyPrinter`.
- [x] **MCP resource browser** — `Views/McpResourcePickerDialog.cs` — flat list of every ready server's resources, search box, server label per entry.
- [x] **MCP resource preview** — `Preview` button in the browser calls `resources/read` and renders the concatenated text content in the right pane (monospace, scrollable).
- [x] **MCP prompt picker** — `Views/McpPromptPickerDialog.cs` — ComboBox of `server / prompt` entries, dynamic argument form per prompt, Insert fetches `prompts/get` and stuffs the rendered text into the compose box.
- [x] **MCP prompt with arguments** — argument form built from each prompt's `arguments[]` schema; required args show a trailing `*` in the label. Empty-value fields are omitted from the call.
- [x] **MCP resource attachment** — `Attach` button on the browser reads the resource and inserts a `<!-- resource: uri -->\nbody` block into the compose textbox.
- [x] **MCP execution logs** — `Views/McpExecutionLogDialog.cs` — ring buffer (500 entries) of request/response JSON blobs, shown newest-first with timestamps and server names. Opened from `Settings → MCP execution log`.
- [x] **MCP capabilities badges** — tools/prompts/resources pills in the editor status panel, driven by the server's declared `capabilities` in the `initialize` response.
- [x] **MCP active servers avatars** — compact circular initial badges in the toolbar, bound to `MainWindowViewModel.ActiveMcpServers` (only servers in `Ready` state appear).

**Tool-calling loop.** When `MainWindowViewModel.GenerateAssistantReplyAsync` completes, `MaybeExecuteToolCallsAsync` scans the reply for `<tool_call>` blocks via `Services/ToolCallParser`. Each call is routed to the right server via the `serverName__toolName` prefix convention; the result JSON is appended as a `tool`-role message and the loop re-enters generation (max 6 rounds, cap tunable via `ToolCallMaxRounds`). Tools are injected into the Jinja template through a new `tools:` parameter on `LlamaChatTemplate.Apply` — the template's existing tool-use branches render without further changes.

### 8. Multi-model

This whole section is shaped by webui's server-side model-list model. We use per-profile model loads instead, so most items map differently — the *profile picker* covers selection, filtering is trivial on a small profile list, and there's no remote "available/offline" state to display. Vision/audio badges come with multimodal.

- [-] **Model selector dropdown** — N/A. The profile ComboBox in the toolbar already does this (plus load settings and sampler bundled with each profile).
- [-] **Model search/filter** — N/A. Few-profile case; a search box is overkill.
- [-] **Grouped model list** — N/A.
- [-] **Model option** — N/A.
- [~] **Vision modality badge** — deferred. `ChatSession.SupportsImages` / `MtmdContext.SupportsVision` already expose the capability flag; a 👁 badge next to the profile name in the toolbar is a quick UI pass still to do. The paperclip button's enabled state already implicitly signals vision capability.
- [~] **Audio modality badge** — deferred. `ChatSession.SupportsAudio` exposes the capability and the compose paperclip is now gated on `CanAttachMedia` (images OR audio), so the enable-state signals audio-capable models implicitly. A dedicated 🎵 badge next to the profile name is a polish pass still to do.
- [x] **Model info dialog** — `Views/ModelInfoDialog.cs`. File → Model info… (disabled when no model is loaded). Shows a summary block (profile / filename / description / parameter count / file size / training context / layers / embedding dim / capabilities / vocab size / template presence) plus the full GGUF key/value bag in a scrollable lower table. Long values clipped to 400 chars so the embedded Jinja template doesn't blow up the dialog.
- [-] **Model not available dialog** — N/A. We read from a file path; "not available" = file missing, which we already report in the status bar.
- [-] **Router mode** — N/A.
- [-] **Single model display** — N/A. Profile name + filename shown in the toolbar.
- [-] **Model change handler** — N/A.

### 9. Miscellaneous

- [x] **Theme toggle** — `AppSettings.ThemeMode` (Auto/Light/Dark) + Settings → Display ComboBox. `Services/ThemeService.Apply` maps to `Application.RequestedThemeVariant`; applied at startup and live on each change via `AppSettingsViewModel.OnThemeModeChanged`. `App.axaml`'s default variant flipped to `Default` so the user setting wins.
- [-] **Dark mode class strategy** — N/A. Avalonia's `ThemeVariant` is the equivalent and is already wired.
- [-] **Mode-watcher integration** — N/A. Avalonia equivalent (system-theme detection) is built in via `ThemeVariant.Default`.
- [-] **Language/locale selector** — N/A. No i18n framework; app is English-only.
- [x] **About dialog / keyboard shortcut overlay** — `Views/ShortcutsDialog.cs`. Help → Keyboard shortcuts… lists Ctrl+N / Ctrl+Shift+O / Ctrl+K / Ctrl+B / Ctrl+L / Ctrl+, and Enter / Shift+Enter / Escape.
- [x] **Error toast messages** — `Services/ToastService` + `Views/ToastHost`. Bottom-right overlay; error path wired in generation failure + model load failure; Destructive border colour.
- [x] **Success/info toasts** — same host; success (green border) wired for Copy, model loaded, export/import, pin; info for /clear; warning for unknown slash command.
- [-] **Empty state — no conversations** — N/A. Ctor guarantees at least one conversation exists (auto-creates a blank one if the store is empty), so the sidebar is never empty in practice.
- [x] **Empty state — empty conversation** — centred "Start the conversation" hint shown in the chat area when `SelectedConversation.Messages.Count == 0`; sits behind the ScrollViewer.
- [x] **Loading splash screen** — full-window overlay on MainWindow, visible while `IsBusy`. Shows profile name + indeterminate progress bar + current status text. Blocks interaction during `ChatSession.Load`.
- [x] **Error splash screen** — `Views/ErrorSplashDialog.cs`, opened by `Services/ErrorBoundary` on any UI-thread unhandled exception (or by an explicit `ReportFatal` call). Heading + one-line exception summary + optional context + scrollable full trace in a monospace TextBox. Actions: Copy details (full trace + timestamp to clipboard), Try to continue (dismiss — may or may not leave a usable app, the splash warns about this), Close application (clean `Shutdown()`). Escape = close. Reentrancy-guarded so a cascade of errors shows only the first splash.
- [-] **Onboarding / feature tour** — N/A. Probably out of scope for a developer-oriented desktop app.

## Visual theming

### Framework + design system

These items describe what the webui uses, not TODOs for us — we replaced Tailwind + shadcn-svelte with Avalonia's styling system + hand-rolled control styles in `Theme/`.

- [-] **Tailwind version** — N/A (we use Avalonia styles + Fluent theme).
- [-] **Tailwind plugins** — N/A.
- [-] **Component library** — N/A (custom controls; shadcn-equivalent variants hand-rolled in `Theme/Controls.axaml`).
- [-] **Design system generator** — N/A.
- [-] **Tailwind config path** — N/A.

### Color system

- [x] **OKLCH color space** — translated to sRGB hex anchored on Tailwind v4's official neutral palette (see `Theme/Tokens.axaml`). OKLCH not natively supported by Avalonia brushes.
- [x] **Semantic color tokens (light mode)** — mirrored in `Theme/Tokens.axaml` Light dictionary (Background/Foreground/Card/Primary/Secondary/Muted/Accent/Destructive/Border/Input/Ring/Sidebar/CodeBackground/CodeForeground + chat-specific UserBubble/AssistantBubble):
  - `--background: oklch(1 0 0)` — white
  - `--foreground: oklch(0.145 0 0)` — near-black text
  - `--card: oklch(1 0 0)` — white card background
  - `--card-foreground: oklch(0.145 0 0)` — dark card text
  - `--primary: oklch(0.205 0 0)` — very dark navy
  - `--primary-foreground: oklch(0.985 0 0)` — near-white on primary
  - `--secondary: oklch(0.95 0 0)` — light gray
  - `--secondary-foreground: oklch(0.205 0 0)` — dark text on secondary
  - `--muted: oklch(0.97 0 0)` — very light gray
  - `--muted-foreground: oklch(0.556 0 0)` — mid-tone text
  - `--accent: oklch(0.95 0 0)` — light accent (same as secondary)
  - `--accent-foreground: oklch(0.205 0 0)` — dark on accent
  - `--destructive: oklch(0.577 0.245 27.325)` — red-orange hue 27°, medium saturation
  - `--border: oklch(0.875 0 0)` — light gray border
  - `--input: oklch(0.92 0 0)` — very light input background
  - `--ring: oklch(0.708 0 0)` — medium gray for focus rings
- [~] **Chart colors** — `--chart-1` through `--chart-5` deferred; no charts yet.
- [x] **Sidebar tokens** — `Sidebar`, `SidebarForeground`, `SidebarBorder` in both dictionaries (simpler subset — no separate primary/ring variants needed for our sidebar usage)
- [x] **Code block colors** — `CodeBackground`/`CodeForeground` in both variants (ready for markdown rendering)
- [x] **Dark mode override** — separate `Dark` theme dictionary in `Theme/Tokens.axaml` overrides all tokens:
  - `--background: oklch(0.16 0 0)` — very dark gray
  - `--foreground: oklch(0.985 0 0)` — near-white text
  - `--card: oklch(0.205 0 0)` — dark card background
  - `--primary: oklch(0.922 0 0)` — very light (nearly white) primary
  - `--secondary: oklch(0.29 0 0)` — dark secondary
  - `--muted: oklch(0.269 0 0)` — dark muted
  - `--muted-foreground: oklch(0.708 0 0)` — light text on dark
  - `--code-background: oklch(0.225 0 0)` — very dark code bg
  - `--code-foreground: oklch(0.875 0 0)` — light code text
  - `--chart-1` through `--chart-5` (deferred — no charts yet)
- [x] **Border opacity in dark** — `Border`/`Input`/`SidebarBorder` brushes set `Color="#FFFFFF" Opacity="0.18"` / `0.10` (shadcn uses 30% for border; 18% composites closer to the visual we want on `#121212`)
- [x] **CSS variable plumbing** — Avalonia equivalent: `ResourceDictionary.ThemeDictionaries`. `{DynamicResource Background}` picks the right variant at runtime.
- [x] **Radius scale** — `CornerRadius` resources:
  - `--radius: 0.625rem` — default radius (10px)
  - `--radius-sm: calc(var(--radius) - 4px)` — 6px
  - `--radius-md: calc(var(--radius) - 2px)` — 8px
  - `--radius-lg: var(--radius)` — 10px
  - `--radius-xl: calc(var(--radius) + 4px)` — 14px
- [-] **Z-index tokens** — N/A. Avalonia manages popup z-order; no explicit layer tokens needed.
- [x] **Dark/light mode switch** — Avalonia `ThemeVariant` mechanism. Currently hard-coded to `Dark` in `App.axaml`; a user toggle + persisted preference is tracked as a follow-up.

### Typography

- [x] **Font stack** — system default inherited from FluentTheme (Segoe UI Variable / system on Windows, system-ui on Linux). No custom override; matches webui's no-explicit-family approach.
- [x] **Base font size** — 13px desktop body (`FontSizeBase` token). Webui uses 16px for the web context; desktop apps conventionally run 2-3pt smaller.
- [x] **Heading scale** — `TextBlock.h1`/`h2`/`h3` classes in `Theme/Controls.axaml` mapped to `FontSize2xl`/`Xl`/`Lg` tokens.
- [x] **Code font** — `CodeFontFamily` resource (`Consolas, Menlo, DejaVu Sans Mono, monospace`). Used by markdown code blocks when we add them.
- [-] **Line height** — Avalonia defaults. Would only revisit if message density feels off.
- [-] **Letter spacing** — Avalonia defaults. Would only revisit if needed.
- [x] **Button text** — `FontSize="FontSizeSm"` + `FontWeight="Medium"` baked into base Button style.
- [x] **Label text** — default `TextBlock` picks up `FontSizeBase`; field labels in forms use `FontSizeSm` via Grid layout.
- [x] **Helper text** — `TextBlock` classes `muted` + `xs` combine to the shadcn `text-xs text-muted-foreground` pattern.
- [-] **Markdown prose** — N/A. Markdig renders to Avalonia controls directly; there's no `.prose` equivalent to wrap. Obsolete annotation.

### Spacing + radius scale

- [x] **Spacing scale** — `SpacingXs/Sm/Md/Lg/Xl` = 4/8/12/16/24 + `PaddingXs..Lg` Thickness resources in `Theme/Tokens.axaml`.
- [x] **Padding presets** — Button `Padding="12,6"`, TextBox `Padding="10,6"`, Card `Padding="16"` — set in control styles.
- [x] **Gaps** — consistent `Spacing="6"` or `"8"` in StackPanels; 16px between major grid cells.
- [x] **Border radius**:
  - Buttons: `RadiusMd` (8px)
  - Inputs: `RadiusMd` (8px)
  - Cards: `RadiusLg` (10px)
  - Code blocks: will use `RadiusMd` when added
  - Message bubbles: `RadiusLg` (10px)
- [x] **Radius for icon buttons** — icon-only buttons inherit the base Button's 8px radius (`RadiusMd`); no bespoke pill-round variant needed for the sizes we use.
- [x] **Sidebar spacing** — `RadiusLg` container, 6px inner list margin, 10px button row padding in `SettingsWindow.axaml`.
- [x] **Message padding** — bubble `Padding="14,10"`, `Margin="0,4"` between bubbles via `Border.bubble` style.

### Iconography

- [x] **Icon library** — embedded lucide SVG paths as `StreamGeometry` resources in `Theme/Icons.axaml` (Send, Square, X, Plus, Search, Copy, Pencil, RotateCw, Play, Trash, Paperclip, Zap, Folder, Settings, Wrench, Pin, Eye, Volume, Globe, ChevronDown, Menu, Info, Alert, Unplug, Network). Rendered via `Path` with stroke matching the 24-unit viewBox × 2px stroke convention. Chosen over `Projektanker.Icons.Avalonia.MaterialDesign` because lucide matches the shadcn/webui aesthetic the rest of the app emulates, and hand-embedding keeps the dependency footprint small.
- [x] **Common icon usage** — swapped in across compose bar (Paperclip / Zap / Folder / Send / Square), message action bar (Copy / Pencil / RotateCw / Play / Trash), toolbar (Play / Unplug / Settings), sidebar (Plus for New chat), profile + MCP settings CRUD (Plus / Copy / Trash), conversation pin indicator (Pin), audio attachment chips (Volume), ASR language chip (Globe), tool-call chips + tool-bubble Expander headers (Wrench), and MCP capability badges (Wrench / Zap / Folder).
- [x] **Icon sizing** — tokenised via `Path.icon` base style (16×16), plus `.xs` (12×12), `.sm` (14×14), `.lg` (18×18) class variants. `Button.icon` gives a square 28×28 slot (with `.sm` → 24×24, `.lg` → 36×36) so icon-only buttons line up with existing ghost/outline sizes.
- [x] **Icon colors** — `Path.icon` strokes from `{DynamicResource Foreground}`; a nested selector (`Button Path.icon`) rebinds the stroke to the parent Button's `Foreground` so icons on primary/destructive/accent backgrounds pick up correct contrast automatically.
- [~] **Custom SVGs** — app icon is the committed `Assets/icon-256.png`; a proper monochrome SVG set for in-app ornamentation (splash, empty states) is deferred.
- [x] **Icon-only buttons** — `Button.icon` / `Button.icon.sm` / `Button.icon.lg` variants; used by the action bar (Copy/Edit/Retry/Continue/Delete), compose-bar attach/prompt/resource trio, and the attachment strip's × remove. Each has a `ToolTip.Tip` for label-free discoverability.

### Component patterns

- [x] **Button variants** — in `Theme/Controls.axaml`:
  - base (no class) = shadcn `default` — primary bg/fg, 90% opacity hover
  - `destructive` — Destructive bg
  - `outline` — Background bg + Border brush; hover swaps to Accent
  - `secondary` — Secondary bg
  - `ghost` — transparent; hover Accent
  - `link` — inline text with underline-on-hover
- [x] **Button sizes** — base (32px), `sm` (28px), `lg` (36px). No icon variants yet.
- [x] **Input styling** — `Background="Input" BorderBrush="Border" CornerRadius="RadiusMd" Padding="10,6"`. Focus swaps `BorderBrush` to `Ring`.
- [x] **Textarea styling** — inherits TextBox style; compose bar uses `MinHeight="56" MaxHeight="200"` with `TextWrapping="Wrap"`.
- [x] **Select/dropdown** — ComboBox picks up Input/Border/Ring tokens. No custom item-template styling yet.
- [x] **Label + field layout** — `ProfileEditorView` / `SamplerPanelView` use 130,*-column Grid rows with labels on the left, helpers below as `xs muted`.
- [~] **Form validation** — deferred. No inputs with validation yet.
- [x] **Checkbox** — basic styling (Foreground + FontSize). Toggle/switch variant deferred.
- [~] **Switch/toggle** — deferred. `CheckBox` is a stand-in for bool toggles; a proper animated switch is polish.
- [x] **Cards/panels** — `Border.card` and `Border.panel` classes in `Theme/Controls.axaml`. SettingsWindow uses `card`; MainWindow toolbar/status use `panel`/`statusbar`.
- [x] **Message bubbles** — `Border.bubble.user` right-aligned Primary bg + max-width 640; `Border.bubble.assistant` left/stretch, Card bg + border. Role selection via bound bools `IsUser`/`IsAssistant`.
- [x] **Settings form** — grouped `section` headers, vertical stack of rows with label column, footer with Save/Close.
- [x] **Sidebar** — `SettingsWindow` left pane uses Sidebar/SidebarBorder tokens; list items highlight Accent on hover/selected via `ListBoxItem` styles.
- [x] **Code block** — fenced/indented code rendered as `Border` with `CodeBackground`, monospace `TextBlock` inside a `ScrollViewer`, optional language label in a two-row grid.
- [x] **Dialog/modal** — `SettingsWindow` shown via `ShowDialog(owner)` with `WindowStartupLocation="CenterOwner"`.
- [x] **Toast/notification** — see Miscellaneous. `ToastService` + `ToastHost` + severity-variant `Border.toast` styles.
- [~] **Tooltip** — deferred.

### Motion + accessibility

- [~] **Transitions** — deferred. Avalonia `Transitions` on properties; add fade-in for new messages and slide for Expander in a polish pass.
- [x] **Hover effects** — `:pointerover /template/ ContentPresenter#PART_ContentPresenter` setters on every Button variant.
- [x] **Focus rings** — `TextBox:focus` thickens BorderBrush to `Ring`. Avalonia's built-in focus adorner is inherited from FluentTheme.
- [x] **Disabled state** — `Button:disabled` and `TextBox:disabled` drop Opacity to 0.5; Button also flips Cursor to Arrow.
- [x] **Streaming cursor animation** — see Message rendering. Avalonia `Animation` on `TextBlock.cursor` in `Theme/Controls.axaml`.
- [x] **Auto-scroll on new messages** — implemented. Subscribes to `SelectedConversation.Messages.CollectionChanged` for new bubbles + 100 ms poll during `IsGenerating`, both gated on `AppSettings.AutoScroll`.
- [~] **Reduced motion support** — deferred. Avalonia has no built-in `prefers-reduced-motion`; would need to poll platform APIs or add a user setting.
- [x] **Focus trap in dialogs** — Avalonia's `ShowDialog` traps focus natively.
- [x] **Keyboard navigation** — Avalonia defaults handle Tab/arrows/Escape. Our `Enter/Shift+Enter` handler is in `MainWindow.axaml.cs:12-22`. Ctrl+L (Load) and Ctrl+, (Preferences) wired via `InputGesture` on MenuItems.
- [x] **ARIA labels** — Avalonia's analog is the `Avalonia.Automation.AutomationProperties` attached properties, which drive UIA on Windows / AT-SPI on Linux / NSAccessibility on macOS. Every icon-only button in `MainWindow.axaml` carries an explicit `AutomationProperties.Name` (sidebar toggle, load / unload / settings, branch nav, per-bubble action bar, compose bar attach / MCP prompt / resource, attachment remove, send / stop) plus a `HelpText` on the two buttons whose tooltip carried extra detail (Continue reply, Attach file). Chat bubbles get `LiveSetting=Polite` + a role-prefixed `Name` so new messages announce. Toasts get `LiveSetting=Polite` by default, escalated to `Assertive` via a style selector on `.warning` and `.error` so they interrupt. MCP toolbar avatars get a `Name` binding with the full server name. Dialog titles already serve as accessible names (verified across `ShortcutsDialog`, `ModelInfoDialog`, `CodePreviewDialog`, `McpExecutionLogDialog`, `McpPromptPickerDialog`, `McpResourcePickerDialog`, `AboutDialog`, `ConfirmDialog`, `SettingsWindow`). Buttons in other views (`SettingsWindow`, `McpSettingsView`) carry visible text content — the default `ButtonAutomationPeer` picks that up without further annotation. New `AppSettings.HighAccessibilityMode` toggle adds a root-window `Classes.a11y` binding; when on, an override style in `Theme/Controls.axaml` forces the message action bar always visible (bypassing hover-to-reveal) so every action is reachable without a pointer.
- [x] **Contrast** — sRGB hex values chosen from Tailwind neutral palette (WCAG AA compliant) + shadcn destructive red.

### Styling techniques

- [x] **Utility-first via Style Selectors** — `Classes="outline sm"` composes like Tailwind classes. No CSS-in-C# library.
- [x] **CSS variables analog** — `ResourceDictionary.ThemeDictionaries` + `DynamicResource` binding.
- [~] **Backdrop blur** — deferred. Avalonia `ExperimentalAcrylicBorder` covers this; add if we want shadcn's frosted ghost hover.
- [~] **Shadow scale** — deferred. No shadows applied; Avalonia uses `BoxShadows` property on Border.
- [~] **Scrollbar styling** — deferred.
- [x] **Global styles** — base `Window` and `TextBlock` selectors in `Theme/Controls.axaml` set Background/Foreground/FontSize.
- [-] **Custom SCSS for external content** — N/A.

## Parity strategy notes

### High-confidence, cheap wins — landed
- [x] Basic UI shell (layout, menu + toolbar, sidebar)
- [x] Settings panels (Profiles + Display tabs)
- [x] Message list UI (MVVM binding against `SelectedConversation.Messages`)
- [x] Button / input component library (hand-rolled variants under `Theme/Controls.axaml`)
- [x] Dark mode (ThemeVariant + tokens); theme toggle UI still TODO
- [x] Keyboard shortcuts (Ctrl+N / Ctrl+Shift+O / Ctrl+K / Ctrl+B / Ctrl+L / Ctrl+,)
- [x] Conversation list with search (JSON-persisted, case-insensitive title + preview filter)

### Medium effort, well-scoped — status
- [x] Markdown rendering with remark/rehype analog — Markdig → Avalonia control tree (see Message rendering section for the detailed breakdown).
- [x] File attachments & drag-drop — images only for v1 (compose paperclip + drag-drop + clipboard paste + user-bubble thumbnails). Audio input still deferred.
- [~] Audio recording — file attachment works (compose paperclip, drag-drop, audio chip in bubble). Mic capture still deferred — needs platform-specific capture pipeline (NAudio / PortAudio).
- [x] MCP protocol client — hand-rolled JSON-RPC over Streamable HTTP in `Services/McpClient.cs`. No external SDK — matches the project's thin-wrapper philosophy and avoids SDK version drift. Supports the full surface the UI needs: `initialize`, `tools/list`, `tools/call`, `prompts/list`, `prompts/get`, `resources/list`, `resources/read`. Responses parsed from both direct JSON bodies and SSE streams.
- [-] Model selector with search — N/A (profile-based UI instead).

### Known blockers or server-specific stubs
- [-] **Multi-model routing** — N/A. Single model per profile-load; no "model per conversation" concept.
- [x] **MCP resource browser** — see §7.
- [x] **Tool calling / prompt picker** — see §7. Hermes-style `<tool_call>` format; compatible with Qwen, DeepSeek, Hermes-family templates.
- [x] **Continue generation** — Continue button in the assistant bubble's action bar; `StreamContinuationAsync` extends the last reply from the current KV state without re-rendering.
- [x] **Token count estimation** — `UserInputTokenCount` on the main VM, debounced 150 ms, shown under the compose box.

### Visual implementation notes
- [x] **OKLCH colors** — translated to sRGB hex anchored on Tailwind v4 neutral palette.
- [x] **Radius scale** — `CornerRadius` resources in `Theme/Tokens.axaml`.
- [x] **Fonts** — inherited system stack.
- [x] **Spacing scale** — `SpacingXs/Sm/Md/Lg/Xl` + `PaddingXs/Sm/Md/Lg` resources.
- [x] **Icons** — embedded lucide SVG paths as `StreamGeometry` resources in `Theme/Icons.axaml`; see Iconography section above for the full inventory.
- [~] **Animations** — deferred.

---

**Total feature items:** 150+ checkboxes  
**Total theming detail lines:** 80+ bullets  

Track implementation progress by checking items off. Use this checklist to ensure no UX blind spots as desktop client development proceeds. Cross-reference component files above when implementing each feature.
