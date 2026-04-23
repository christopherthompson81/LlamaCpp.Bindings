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
- [~] **Error boundary** — deferred. Generation/load errors are caught in-VM and surfaced to the status bar + `last-error.log`; a proper boundary wrapping the whole app would catch view-tree exceptions too.

### 2. Conversation list

- [x] **List all conversations** — `MainWindowViewModel.Conversations`, persisted to `conversations.json` by `ConversationStore`. Sorted by `UpdatedAt` descending in `FilteredConversations`.
- [x] **Create new conversation** — `NewConversationCommand` (Ctrl+N / Ctrl+Shift+O, File menu item). Auto-selects the new one.
- [x] **Rename conversation** — inline edit in sidebar (right-click → Rename or F2-style — currently via context menu). Commit on Enter/LostFocus via `EndRenameCommand`; Escape cancels.
- [x] **Delete conversation** — right-click → Delete, or Chat menu. No confirmation yet — deferred.
- [~] **Conversation tree/forking** — deferred. Needs `Conversation` refactored from a flat `Turns` list to a tree with `parentId` on each turn, plus sibling-nav UI.
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
- [~] **KaTeX math rendering** — deferred. Needs a LaTeX renderer; strategy: LaTeX→SkiaSharp or LaTeX→bitmap, then replace `$...$` spans with an `InlineUIContainer` holding the bitmap.
- [x] **Syntax highlighting** — `Services/CodeHighlighter.cs` wraps `ColorCode.Core`'s `LanguageParser` into a `(Text, Scope)` token stream; `MarkdownRenderer.BuildCodeBlock` emits a `Run` per token with Foreground bound to a `Syntax*` theme brush (`SyntaxKeyword` / `String` / `Comment` / `Number` / `Type` / `Operator` / `Preprocessor` / `Tag`). GitHub Light/Dark palettes in `Theme/Tokens.axaml`. Languages covered: C#, C++, CSS, F#, HTML, Java, JS/TS/JSON, Python, PHP, PowerShell, SQL, XML/XAML, Markdown, Haskell, MATLAB, VB.NET, Fortran — everything ColorCode.Core ships. Rust/Go/Ruby/YAML/bash fall through to plain text.
- [x] **Code-block copy button** — ghost+sm `Copy` button in the code-block header row. Uses `DialogService.CopyToClipboardAsync`.
- [x] **Code-block preview/expand dialog** — `Views/CodePreviewDialog.cs` — modal window (1000×700, centre-of-owner) that re-renders the same highlighted block at full size, with a footer Copy + Close and Escape-to-close. "Expand" button in each code block opens it.
- [x] **Incomplete code block / mid-stream robustness** — `Markdown.Parse` is wrapped in `try`; if parsing fails (e.g. unclosed fence mid-stream) we fall back to showing the raw text in a `TextBlock` instead of leaving the bubble blank. Markdig actually tolerates most mid-stream cases — the try is belt-and-braces.
- [x] **Image display from attachments** — attached images render as a `WrapPanel` thumbnail row above the bubble content in the user bubble template, plus a matching 72×72 strip above the compose text box. `byte[] → Avalonia.Bitmap` via `Services/AttachmentThumbnailConverter.cs` (registered as `AttachmentThumbnail` resource in `App.axaml`). Conversations persist attachments inline — `System.Text.Json` serializes `byte[]` as base64 automatically, so a round-trip through `conversations.json` preserves the payload.
- [x] **Image error fallback** — `AttachmentThumbnailConverter` returns null on decode exception; the `Image` control renders as an empty 72×72 rectangle (with the `Muted` background still visible) without taking down the bubble.
- [x] **GFM tables** — rendered as a bordered Grid with Auto columns. Header row detected via Markdig's `TableRow.IsHeader`, emitted SemiBold.
- [~] **Footnotes** — deferred. Not in our `MarkdownPipelineBuilder` extensions; webui doesn't support them either. Low priority.
- [~] **Mermaid diagrams** — deferred. Would need a SkiaSharp implementation of Mermaid's graph layout engine; nontrivial.
- [x] **Streaming cursor/indicator** — `MarkdownView` exposes an `IsStreaming` StyledProperty; bubble template binds it from `MessageViewModel.IsStreaming`. When true, an `InlineUIContainer(TextBlock.cursor)` carrying `▌` is appended to the live tail. Blink animation (1s cycle, opacity 1 → 0.2 → 1) lives on the `TextBlock.cursor` style in `Theme/Controls.axaml`.
- [x] **HTML sanitisation** — inert by construction: `HtmlInline` renders the raw tag string as literal text, so injected `<script>` etc. never becomes a control.
- [x] **Streaming-safe re-render throttling** — `MarkdownView` coalesces property changes through a 40ms `DispatcherTimer` debounce to avoid thrashing the layout pass on every decoded token (~8ms intervals at 120 tok/s → ~4-5 tokens per render).

### 4. Message actions

- [x] **Copy message** — `CopyMessageCommand` → `DialogService.CopyToClipboardAsync` (wraps `Avalonia.Input.Platform.ClipboardExtensions.SetTextAsync`). Button bound in the bubble template.
- [x] **Edit message** — inline edit surface (TextBox + Save / Cancel) replaces the read-only bubble body while `MessageViewModel.IsEditing`. On Save: user message → truncate downstream + regenerate; assistant message → overwrite in place. `EditDraft` is the buffer; `Content` only commits on Save.
- [x] **Regenerate response** — `RegenerateMessageCommand` truncates the transcript starting at the target message (or right after, if the target was a user turn), `ClearKv()`s the session, and re-enters `GenerateAssistantReplyAsync`. Extracted from `SendAsync` so both share the same streaming path.
- [~] **Continue generation** — deferred. The binding-level precursor (prefix-cache reuse) landed in `docs/webui_parity_investigation.md` Run 2, and `LlamaGenerator.GenerateAsync(tokens, firstNewIndex=tokens.Count)` + the back-off-by-one path already "continue from current KV"; what's still missing is the UI plumbing — a "Continue" button on the last assistant bubble that re-enters `GenerateAssistantReplyAsync` with the partial reply treated as the assistant prefix.
- [x] **Delete message** — `DeleteMessageCommand` removes the single clicked message (not downstream) and clears the KV cache. No confirmation dialog yet.
- [~] **Branch navigation** — deferred. Precursor: tree-shaped transcript with `parentId` on each turn.
- [~] **Fork conversation** — deferred. Trivial wrapper once branching or a "duplicate-up-to-here" command lands.
- [x] **Message deletion dialog** — `DialogService.ConfirmAsync` + `ConfirmDialog` (multi-choice). `DeleteMessageAsync` skips the prompt when there's nothing downstream; otherwise offers Cancel / Just this / This + N after.

### 5. Compose

- [x] **Textarea input** — the compose TextBox in `MainWindow.axaml` with `MinHeight=56 MaxHeight=200 TextWrapping=Wrap AcceptsReturn=False` + custom Enter/Shift+Enter handling. Webui's auto-height sizing via `field-sizing: content` would be a polish pass; the fixed min/max is serviceable.
- [x] **File attachment picker** — `AttachImagesCommand` on `MainWindowViewModel` → `DialogService.PickImageFilesAsync` (`OpenFilePickerAsync`, `AllowMultiple=true`, filters `*.jpg;*.png;*.bmp;*.gif;*.webp`). The 📎 button in the compose bar is gated on `CanAttachImages` — disabled when the loaded profile has no mmproj.
- [x] **Attachment preview** — the same 72×72 thumbnail Border used in user bubbles sits above the compose `TextBox`, one per `PendingAttachments` entry, each with a × remove button (`RemovePendingAttachmentCommand`). `HasPendingAttachments` collapses the row when empty.
- [~] **Attachment list modal** — deferred. The inline compose-bar strip covers the typical few-images-per-message case; a dedicated gallery modal is a polish pass that only matters once users attach many items at once.
- [~] **Audio recording** — deferred. The `mtmd_bitmap_init_from_audio` P/Invoke + `MtmdContext.SupportsAudio` / `AudioSampleRate` properties are already in place; missing pieces are the `MtmdBitmap.FromAudio*` public wrappers and a capture/encode pipeline (NAudio on Windows / PortAudio cross-platform) driving the compose bar's mic button.
- [x] **Drag-and-drop file upload** — `DragDrop.AllowDrop="True"` on the main `Window`; `OnComposeDragOver` in `MainWindow.axaml.cs` accepts the drop only when the payload contains `DataFormat.File` **and** `CanAttachImages`; `OnComposeDrop` walks `IDataTransfer.TryGetFiles()` and feeds local paths to `MainWindowViewModel.TryAddPendingImage`.
- [x] **Paste handling (files)** — Ctrl+V in the compose `TextBox` is intercepted by `OnComposeKeyDown`: if the clipboard has a bitmap payload (`IClipboard.TryGetBitmapAsync`), the bitmap is re-encoded to PNG bytes and queued as an attachment, and the default text-paste is suppressed. Non-image clipboard content (plain text, mixed) falls through to the `TextBox`'s native paste.
- [~] **MCP prompt picker** — deferred. Precursor: MCP client.
- [~] **MCP resource picker** — deferred (with MCP).
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
- [~] **Tools/MCP tab** — deferred. Precursor: MCP client support (phase-2 item in `docs/webui_parity_investigation.md`).
- [x] **System prompt** — per-profile multi-line TextBox at the top of the profile editor. Prepended to every transcript as a `TurnRole.System` turn when that profile is loaded.
- [~] **Response format tab** — deferred. Precursor: JSON-schema → GBNF converter + preset templates. Raw GBNF slot already exists on the profile.
- [-] **Parameter sync source indicator** — N/A. Settings are local-only; there's no server-default/session-override hierarchy to visualise.
- [x] **Reset to defaults** — `ResetSamplerDefaults` command on the Profiles tab footer restores `SamplerSettings.Default` + fresh `GenerationSettings` on the current profile. Load settings and system prompt are preserved.
- [x] **Settings persistence** — three stores: `ProfileStore` (profiles.json), `AppSettingsStore` (app-settings.json), `ConversationStore` (conversations.json) — all under `$XDG_CONFIG_HOME/LlamaChat/`. Auto-save on dialog close; explicit Save button covers the in-dialog case.
- [-] **User overrides tracking** — N/A. Not meaningful without a server-default baseline to diff against.

### 7. Tool calling / MCP

All items deferred — **precursor for every one is an MCP client + tool-calling wiring**. See `docs/webui_parity_investigation.md` gap #2 (full Jinja — landed) and the in-flight MCP work which is phase-2.

- [~] **MCP server add** — form: URL input, optional headers textarea, add button, validation.
- [~] **MCP server list** — card list sorted by recency, loading skeletons during health check.
- [~] **MCP server enable/disable** — per-server toggle switch, persists to conversation config.
- [~] **MCP server delete** — delete button, confirmation modal.
- [~] **MCP server edit** — edit URL/headers inline, save/cancel.
- [~] **MCP connection status indicator** — loading/success/error health-check badge.
- [~] **MCP tool list** — tool names, descriptions, parameter schema in a collapsed JSON viewer.
- [~] **MCP resource browser** — hierarchical per-server list, search, preview.
- [~] **MCP resource preview** — modal showing content, copy, full-text display.
- [~] **MCP prompt picker** — dropdown of server prompts + argument form + inserter.
- [~] **MCP prompt with arguments** — argument-form fields + validation.
- [~] **MCP resource attachment** — attach resource URI + content to a message.
- [~] **MCP execution logs** — debug panel of request/response, parse/exec errors.
- [~] **MCP capabilities badges** — resource/prompt/tool capability flags per server.
- [~] **MCP active servers avatars** — compact icons in header showing active MCP servers.

### 8. Multi-model

This whole section is shaped by webui's server-side model-list model. We use per-profile model loads instead, so most items map differently — the *profile picker* covers selection, filtering is trivial on a small profile list, and there's no remote "available/offline" state to display. Vision/audio badges come with multimodal.

- [-] **Model selector dropdown** — N/A. The profile ComboBox in the toolbar already does this (plus load settings and sampler bundled with each profile).
- [-] **Model search/filter** — N/A. Few-profile case; a search box is overkill.
- [-] **Grouped model list** — N/A.
- [-] **Model option** — N/A.
- [~] **Vision modality badge** — deferred. `ChatSession.SupportsImages` / `MtmdContext.SupportsVision` already expose the capability flag; a 👁 badge next to the profile name in the toolbar is a quick UI pass still to do. The paperclip button's enabled state already implicitly signals vision capability.
- [~] **Audio modality badge** — deferred. `MtmdContext.SupportsAudio` is ready; will land alongside the audio-input UI pass.
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
- [~] **Error splash screen** — deferred. Precursor: formal app-wide error boundary. For now load/generation errors go to `last-error.log` + status bar.
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
- [~] **Radius for icon buttons** — deferred with icons.
- [x] **Sidebar spacing** — `RadiusLg` container, 6px inner list margin, 10px button row padding in `SettingsWindow.axaml`.
- [x] **Message padding** — bubble `Padding="14,10"`, `Margin="0,4"` between bubbles via `Border.bubble` style.

### Iconography

All deferred — the app currently uses text labels (Copy / Edit / Delete etc.). An icon pass is its own focused task; candidates are `Projektanker.Icons.Avalonia.MaterialDesign` or embedding lucide SVG path data as `Geometry` resources.

- [~] **Icon library** — deferred.
- [~] **Common icon usage**:
  - Edit: `Edit`
  - Copy: `Copy`
  - Delete: `Trash2`
  - Settings: `Settings`
  - Menu: `Menu`, `ChevronDown`, `MoreVertical`
  - Search: `Search`
  - Refresh: `RefreshCw`
  - Send: `Send` (inferred in form submit)
  - Stop: `Square` or `Pause` (inferred)
  - Plus: `Plus`
  - X/close: `X`
  - Chevron: `ChevronLeft`, `ChevronRight`, `ChevronUp`, `ChevronDown`
  - Alert: `AlertTriangle`
  - Info: `Info`
  - Loading: `Loader2` (animated spinner)
  - Vision: `Eye`
  - Audio: `Volume2`
  - Git: `GitBranch` (for fork)
  - External: `ExternalLink`
  - Code: `Code`
- [~] **Icon sizing** — deferred.
- [~] **Icon colors** — deferred.
- [~] **Custom SVGs** — deferred. App icon is also TBD.
- [~] **Icon-only buttons** — deferred.

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
- [~] **ARIA labels** — deferred. Avalonia's analog is `AutomationProperties.Name` etc.; needed for screen-reader support.
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
- [~] Audio recording — deferred. Multimodal C API bound; wrappers + capture pipeline (NAudio/CoreAudio) still TODO.
- [~] MCP protocol client — deferred (phase 2; `ModelContextProtocol` NuGet is an option when we take this on).
- [-] Model selector with search — N/A (profile-based UI instead).

### Known blockers or server-specific stubs
- [-] **Multi-model routing** — N/A. Single model per profile-load; no "model per conversation" concept.
- [~] **MCP resource browser** — deferred (phase 2 with MCP client).
- [~] **Tool calling / prompt picker** — deferred (phase 2 with MCP).
- [~] **Continue generation** — deferred. `LlamaGenerator` resume-from-offset support is in (landed with prefix-cache reuse in webui_parity_investigation.md Run 2); remaining work is a "Continue" button wiring into `GenerateAssistantReplyAsync`.
- [x] **Token count estimation** — `UserInputTokenCount` on the main VM, debounced 150 ms, shown under the compose box.

### Visual implementation notes
- [x] **OKLCH colors** — translated to sRGB hex anchored on Tailwind v4 neutral palette.
- [x] **Radius scale** — `CornerRadius` resources in `Theme/Tokens.axaml`.
- [x] **Fonts** — inherited system stack.
- [x] **Spacing scale** — `SpacingXs/Sm/Md/Lg/Xl` + `PaddingXs/Sm/Md/Lg` resources.
- [~] **Icons** — deferred. Candidates: `Projektanker.Icons.Avalonia.MaterialDesign` for a Material set, or embed lucide SVG path data as `Geometry` resources.
- [~] **Animations** — deferred.

---

**Total feature items:** 150+ checkboxes  
**Total theming detail lines:** 80+ bullets  

Track implementation progress by checking items off. Use this checklist to ensure no UX blind spots as desktop client development proceeds. Cross-reference component files above when implementing each feature.
