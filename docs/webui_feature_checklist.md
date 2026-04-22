# llama-server webui — feature + theming burn-down

Reference snapshot of llama.cpp/tools/server/webui as of commit `1d6d4cf7a5361046f778414c5b1f5ecbc07eeb77`, to match in LlamaCpp.Bindings.LlamaChat. Each item is a checkbox; check off as we implement.

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

- [ ] **Hash-based routing** — N/A (desktop single-window app; no URL routing). Deep-links to specific conversations could be simulated later if we add IPC.
- [x] **Global keyboard shortcuts** — `Window.KeyBindings` in `MainWindow.axaml` + `OnKeyDown` override in code-behind. Ctrl+N / Ctrl+Shift+O new chat, Ctrl+K focuses search, Ctrl+B toggles sidebar, Ctrl+L loads, Ctrl+, settings. Ctrl+Shift+E (edit title) deferred — use context menu "Rename" or F2 instead.
- [x] **Sidebar layout with collapse** — left-pinned Border bound to `IsSidebarVisible`; Ctrl+B toggles. Fixed 280px width (no responsive breakpoints in desktop).
- [ ] **Mobile detection hook** — N/A.
- [ ] **Responsive breakpoint** — N/A.
- [x] **Header region** — toolbar row below menu shows Profile combo + Load/Unload/Settings + live `ModelSummary` (filename · ctx · layers · template). Serves as the `ChatScreenHeader` equivalent.
- [x] **Sidebar header/footer slots** — RowDefinitions="Auto,Auto,*" lays out New button + Search + list; footer slot free for future (model-status pill, etc.).
- [x] **Main content inset** — main area uses `Margin="16,12,16,12"` for consistent gutter against the sidebar.
- [ ] **Tooltip provider context** — deferred along with tooltips themselves.
- [ ] **Error boundary** — deferred. Avalonia propagates to `Dispatcher.UnhandledException`; for now generation/load errors are caught in-VM and shown in the status bar.

### 2. Conversation list

- [x] **List all conversations** — `MainWindowViewModel.Conversations`, persisted to `conversations.json` by `ConversationStore`. Sorted by `UpdatedAt` descending in `FilteredConversations`.
- [x] **Create new conversation** — `NewConversationCommand` (Ctrl+N / Ctrl+Shift+O, File menu item). Auto-selects the new one.
- [x] **Rename conversation** — inline edit in sidebar (right-click → Rename or F2-style — currently via context menu). Commit on Enter/LostFocus via `EndRenameCommand`; Escape cancels.
- [x] **Delete conversation** — right-click → Delete, or Chat menu. No confirmation yet — deferred.
- [ ] **Conversation tree/forking** — deferred. Model supports adding a `ForkedFrom` field later; UI would need tree rendering.
- [x] **Search conversations** — case-insensitive substring on Title + Preview, live-filtered as user types. Ctrl+K focuses the search box.
- [x] **Active conversation highlight** — `ListBox.SelectedItem` bound to `SelectedConversation`; styled via existing `ListBoxItem:selected` accent from Theme/Controls.
- [x] **Conversation preview text** — `ConversationViewModel.Preview` — first user message truncated to 80 chars, rendered below the title in the sidebar.
- [ ] **Pinned/recent grouping** — deferred. Current sort is pure recency; pins would need a bool field + a second list section.
- [ ] **Export/import conversations** — deferred. JSON is already the on-disk format so import/export is basically a file-copy dialog.

### 3. Message rendering

- [x] **Markdown pipeline** — Markdig → Avalonia control tree, no HTML intermediate (no WebView on Avalonia). Implemented in `Services/MarkdownRenderer.cs`. Pipeline uses:
  - `UseEmphasisExtras()` — strikethrough via `~~...~~`, sub/sup
  - `UseAutoLinks()` — bare URLs render as styled links
  - `UsePipeTables()` — GFM tables via `Markdig.Extensions.Tables`
  - `UseTaskLists()` — `[ ]` / `[x]` rendered with `☐`/`☑` glyph markers
- [x] **Block-level coverage** — paragraphs, ATX headings (h1-h3 distinct sizes, h4+ body-weight), bullet/ordered lists with configurable start index, blockquotes (3px left border + 85% opacity), fenced + indented code blocks, thematic breaks, pipe tables (bordered Grid).
- [x] **Inline-level coverage** — Literal, CodeInline (monospace run w/ CodeBackground), EmphasisInline (bold/italic), strikethrough via `TextDecorations`, LinkInline + AutolinkInline (coloured Ring + underline — inert in v1), LineBreak, HtmlInline (raw tag shown as text = no HTML passthrough), HtmlEntityInline.
- [ ] **KaTeX math rendering** — deferred. Strategy: keep a LaTeX→SkiaSharp or LaTeX→bitmap converter and replace `$...$` spans with `InlineUIContainer` holding the bitmap.
- [ ] **Syntax highlighting** — deferred. Candidates: `ColorCode.Universal` (NuGet, language-aware tokeniser) or `TextMateSharp` (grammar-based, matches VS Code). Code blocks currently render plain monospace on `CodeBackground`.
- [ ] **Code-block copy button** — deferred. Place a ghost-variant icon button in the code-block header (language row); uses `TopLevel.Clipboard`.
- [ ] **Code-block preview/expand dialog** — deferred.
- [x] **Incomplete code block / mid-stream robustness** — `Markdown.Parse` is wrapped in `try`; if parsing fails (e.g. unclosed fence mid-stream) we fall back to showing the raw text in a `TextBlock` instead of leaving the bubble blank. Markdig actually tolerates most mid-stream cases — the try is belt-and-braces.
- [ ] **Image display from attachments** — deferred with multimodal input.
- [ ] **Image error fallback** — deferred with the above.
- [x] **GFM tables** — rendered as a bordered Grid with Auto columns. Header row detected via Markdig's `TableRow.IsHeader`, emitted SemiBold.
- [ ] **Footnotes** — deferred (not in v1 `MarkdownPipelineBuilder` extensions; webui doesn't support them either).
- [ ] **Mermaid diagrams** — deferred (no equivalent native renderer; would need a SkiaSharp implementation).
- [ ] **Streaming cursor/indicator** — deferred (no blinking caret yet; `IsStreaming` flag is available in the VM for when we add one).
- [x] **HTML sanitisation** — inert by construction: `HtmlInline` renders the raw tag string as literal text, so injected `<script>` etc. never becomes a control.
- [x] **Streaming-safe re-render throttling** — `MarkdownView` coalesces property changes through a 40ms `DispatcherTimer` debounce to avoid thrashing the layout pass on every decoded token (~8ms intervals at 120 tok/s → ~4-5 tokens per render).

### 4. Message actions

- [x] **Copy message** — `CopyMessageCommand` → `DialogService.CopyToClipboardAsync` (wraps `Avalonia.Input.Platform.ClipboardExtensions.SetTextAsync`). Button bound in the bubble template.
- [x] **Edit message** — inline edit surface (TextBox + Save / Cancel) replaces the read-only bubble body while `MessageViewModel.IsEditing`. On Save: user message → truncate downstream + regenerate; assistant message → overwrite in place. `EditDraft` is the buffer; `Content` only commits on Save.
- [x] **Regenerate response** — `RegenerateMessageCommand` truncates the transcript starting at the target message (or right after, if the target was a user turn), `ClearKv()`s the session, and re-enters `GenerateAssistantReplyAsync`. Extracted from `SendAsync` so both share the same streaming path.
- [ ] **Continue generation** — deferred. Needs `LlamaGenerator` support for resuming from a position in an already-decoded transcript (we re-prefill each turn today — see `docs/webui_parity_investigation.md` Run 1 item 6 "Prefix-cache reuse").
- [x] **Delete message** — `DeleteMessageCommand` removes the single clicked message (not downstream) and clears the KV cache. No confirmation dialog yet.
- [ ] **Branch navigation** — deferred. Conversation is currently a flat list; branching needs a tree-shaped transcript with `parentId` on each turn plus sibling nav controls.
- [ ] **Fork conversation** — deferred (straightforward wrapper once branching or a simple "duplicate up to here" command lands).
- [ ] **Message deletion dialog** — deferred (we delete immediately; add confirmation once there's a "delete downstream" variant to choose).

### 5. Compose

- [ ] **Textarea input** — `ChatFormTextarea.svelte` — `field-sizing-content` for auto-height, min-height 16, Tailwind classes, backdrop blur
- [ ] **File attachment picker** — `ChatFormFileInputInvisible.svelte`, `ChatFormActionAttachmentsDropdown.svelte` — file input `accept="*/*"`, accepts images, audio, PDF, text files
- [ ] **Attachment preview** — `ChatAttachmentsList.svelte`, `ChatAttachmentThumbnailImage.svelte`, `ChatAttachmentThumbnailFile.svelte` — shows image thumbnails or file icons, remove button per attachment
- [ ] **Attachment list modal** — `DialogChatAttachmentsViewAll.svelte` — full list of uploaded files with preview modals
- [ ] **Audio recording** — `ChatFormActionRecord.svelte` — uses Web Audio API, `AudioRecorder` + `convertToWav()` utilities, streams to uploaded files
- [ ] **Drag-and-drop file upload** — `ChatScreenDragOverlay.svelte`, `ChatScreen.svelte` — overlay during drag, drop handler validates file types and modalities
- [ ] **Paste handling** — `ChatForm.svelte` — clipboard paste triggers file upload or text insertion, uses `parseClipboardContent()` utility
- [ ] **MCP prompt picker** — `ChatFormPromptPicker.svelte`, `ChatFormPromptPickerArgumentForm.svelte` — dropdown to select MCP prompts, shows arguments form with inputs, inserts prompt text
- [ ] **MCP resource picker** — `ChatFormResourcePicker.svelte`, `ChatFormResourcePicker/ChatFormResourcePickerArgumentForm.svelte` — browse/select MCP resources from server, inserts resource reference
- [ ] **Slash command support** — Not explicitly found; prompt picker prefixed with `/prompt:` may serve as command-like interface
- [ ] **Token count display** — `ChatScreenProcessingInfo.svelte` — shows estimated token count during typing, updated in header
- [ ] **Send button state** — `ChatFormActionSubmit.svelte` — enabled/disabled based on text length and loading state, shows spinner during submission
- [ ] **Stop generation button** — `ChatFormActionSubmit.svelte` — changes to "Stop" during streaming, calls `chatStore.stopMessage()`

### 6. Chat settings

- [x] **Settings sidebar panel** — `SettingsWindow.axaml` — modal `ShowDialog(owner)`, left-placed tab strip with Profiles + Display. Header with "Settings" title, footer with Save/Close + status line.
- [~] **General tab** — N/A shape. Theme/API-key/paste-threshold/PDF-as-image are web-only concerns; System Message promoted to its own section inside ProfileEditorView (per-profile instead of global). "Continue button" deferred with continue-generation itself.
- [x] **Display tab** — three toggles, all wired: `AutoScroll`, `ShowMessageStats`, `ShowReasoningInProgress`. Code-block theme and copy-as-plain-text deferred until we add the code-block toolbar. `AppSettings` record + `AppSettingsStore` persist to `app-settings.json` alongside profiles.
- [x] **Sampling tab** — already implemented in `Views/SamplerPanelView.axaml` (embedded inside ProfileEditorView): temperature + dynatemp, top-k/p, min-p, typical, top-n-σ, XTC, DRY, repetition/frequency/presence penalties.
- [x] **Advanced tab** — merged with Sampling: seed, dynamic-temp range/exponent, mirostat v1/v2 + tau/eta, grammar (GBNF). "Tokens to keep" + "ignore EOS" deferred (both require LlamaGenerator API extensions).
- [ ] **Tools/MCP tab** — deferred. MCP client support is a phase-2 item in `docs/webui_parity_investigation.md`.
- [x] **System prompt** — per-profile multi-line TextBox at the top of the profile editor. Prepended to every transcript as a `TurnRole.System` turn when that profile is loaded.
- [ ] **Response format tab** — deferred. GBNF slot exists on the profile; JSON-schema → GBNF conversion + preset templates is a separate task.
- [ ] **Parameter sync source indicator** — N/A for the desktop model. Settings are local-only; there's no server-default/session-override hierarchy to visualise.
- [x] **Reset to defaults** — `ResetSamplerDefaults` command on the Profiles tab footer restores `SamplerSettings.Default` + fresh `GenerationSettings` on the current profile. Load settings and system prompt are preserved.
- [x] **Settings persistence** — three stores: `ProfileStore` (profiles.json), `AppSettingsStore` (app-settings.json), `ConversationStore` (conversations.json) — all under `$XDG_CONFIG_HOME/LlamaChat/`. Auto-save on dialog close; explicit Save button covers the in-dialog case.
- [ ] **User overrides tracking** — deferred. Not meaningful without a server-default baseline to diff against.

### 7. Tool calling / MCP

- [ ] **MCP server add** — `McpServersSettings.svelte`, `McpServerForm.svelte` — form: URL input, optional headers textarea, add button, validation
- [ ] **MCP server list** — `McpServersSettings.svelte` — displays all servers as cards, sorted by recency, loading skeletons during health check
- [ ] **MCP server enable/disable** — `McpServerCard.svelte`, `McpServerCardHeader.svelte` — toggle switch per server, persists to conversation config
- [ ] **MCP server delete** — `McpServerCard.svelte`, `McpServerCardDeleteDialog.svelte` — delete button, confirmation modal with description
- [ ] **MCP server edit** — `McpServerCard.svelte`, `McpServerCardEditForm.svelte` — edit URL/headers inline, save/cancel buttons
- [ ] **MCP connection status indicator** — `McpServerCard.svelte`, `McpServerCardHeader.svelte` — health check status badge (loading/success/error), spinner during health check
- [ ] **MCP tool list** — `McpServerCardToolsList.svelte` — shows tool names, descriptions, parameters schema (collapsed JSON viewer)
- [ ] **MCP resource browser** — `McpResourceBrowser.svelte`, `McpResourceBrowserServerItem.svelte` — hierarchical resource list per server, search, preview button
- [ ] **MCP resource preview** — `DialogMcpResourcePreview.svelte` — modal showing resource content (text/JSON), copy button, full-text display
- [ ] **MCP prompt picker** — `ChatFormPromptPicker.svelte` — dropdown to select from server prompts, shows arguments form, inserts prompt
- [ ] **MCP prompt with arguments** — `ChatFormPromptPickerArgumentForm.svelte` — renders form fields for prompt arguments, validation on submit
- [ ] **MCP resource attachment** — `ChatAttachmentMcpResources.svelte`, `ChatFormResourcePicker.svelte` — attaches resource URI + content to message, shown in attachments list
- [ ] **MCP execution logs** — `McpConnectionLogs.svelte` — debug panel showing tool call requests/responses, parsing/execution errors
- [ ] **MCP capabilities badges** — `McpCapabilitiesBadges.svelte` — shows resource/prompt/tool capability flags if server advertises them
- [ ] **MCP active servers avatars** — `McpActiveServersAvatars.svelte` — compact icons in header showing which MCP servers are active for conversation

### 8. Multi-model

- [ ] **Model selector dropdown** — `ModelsSelector.svelte` — searchable dropdown with model name, capability badges, grouped by favorite/available/offline
- [ ] **Model search/filter** — `ModelsSelector.svelte` — input field filters options by name, groups results by category
- [ ] **Grouped model list** — `ModelsSelector.svelte`, `filterModelOptions()`, `groupModelOptions()` utilities — groups by favorite, then available, then offline
- [ ] **Model option** — `ModelsSelectorOption.svelte` — displays model name, description truncated, capability badges (vision/audio)
- [ ] **Vision modality badge** — `BadgeModality.svelte`, icon from `MODALITY_ICONS[ModelModality.VISION]` (Eye icon) — shows if model supports vision
- [ ] **Audio modality badge** — `BadgeModality.svelte`, icon from `MODALITY_ICONS[ModelModality.AUDIO]` (Volume icon) — shows if model supports audio
- [ ] **Model info dialog** — `DialogModelInformation.svelte` — full model name, description, capabilities, parameters, context window
- [ ] **Model not available dialog** — `DialogModelNotAvailable.svelte` — error state when selected model is offline/unavailable
- [ ] **Router mode** — `isRouterMode` derived from `serverStore` — if true, model selector changes to show model per-conversation, not global
- [ ] **Single model display** — `singleModelName` store state — if not in router mode, displays server's single model name (read-only)
- [ ] **Model change handler** — `ModelsSelector.svelte` — calls `onModelChange()` callback on selection, updates `selectedModelId` store and conversation config

### 9. Miscellaneous

- [ ] **Theme toggle** — `ChatSettings.svelte` general tab — SELECT with options: Auto (system), Light, Dark; calls `setMode()` from mode-watcher, persists theme to localStorage
- [ ] **Dark mode class strategy** — `src/app.css` — uses `.dark` class on `<html>` element, CSS variables change via `.dark { --color-X: ...}` media query
- [ ] **Mode-watcher integration** — `src/routes/+layout.svelte`, `ModeWatcherDecorator.svelte` — provides reactive `mode` store for theme detection
- [ ] **Language/locale selector** — Not found in current codebase; no i18n framework visible
- [ ] **About dialog / keyboard shortcut overlay** — `KeyboardShortcutInfo.svelte` — modal listing all keyboard shortcuts (Ctrl+K, Ctrl+Shift+O, etc.)
- [ ] **Error toast messages** — `svelte-sonner` Toaster in `+layout.svelte` — displays error notifications from `chatStore` errors
- [ ] **Success/info toasts** — `svelte-sonner` Toaster — displays operation confirmations (copy to clipboard, message deleted, etc.)
- [ ] **Empty state — no conversations** — home page or sidebar shows "Start new chat" prompt
- [ ] **Empty state — empty conversation** — chat view shows centered prompt "Ask anything..." with example questions
- [ ] **Loading splash screen** — `ServerLoadingSplash.svelte` — fullscreen spinner while server is initializing
- [ ] **Error splash screen** — `ServerErrorSplash.svelte` — fullscreen error message if server connection fails
- [ ] **Onboarding / feature tour** — Not explicitly found; may be deferred to desktop client

## Visual theming

### Framework + design system

- [ ] **Tailwind version** — `@tailwindcss/vite: ^4.0.0`, `tailwindcss: ^4.0.0` (`package.json`) — modern Vite-integrated build
- [ ] **Tailwind plugins** — `@tailwindcss/forms: ^0.5.9`, `@tailwindcss/typography: ^0.5.15` (for `.prose` classes on markdown)
- [ ] **Component library** — shadcn-svelte via `components.json` registry, manually curated components in `src/lib/components/ui/`
- [ ] **Design system generator** — `components.json` alias `ui` → `$lib/components/ui`, baseColor `neutral`, Tailwind CSS path `src/app.css`
- [ ] **Tailwind config path** — No `tailwind.config.ts`; uses Tailwind v4 inline `@theme` config in `src/app.css` (see below)

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
- [ ] **Chart colors** — `--chart-1` through `--chart-5` (deferred — no charts yet)
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
- [ ] **Z-index tokens** — N/A (Avalonia manages popup z-order; no explicit layer tokens needed)
- [x] **Dark/light mode switch** — Avalonia `ThemeVariant` mechanism. Currently hard-coded to `Dark` in `App.axaml`; a user toggle + persisted preference is tracked as a follow-up.

### Typography

- [x] **Font stack** — system default inherited from FluentTheme (Segoe UI Variable / system on Windows, system-ui on Linux). No custom override; matches webui's no-explicit-family approach.
- [x] **Base font size** — 13px desktop body (`FontSizeBase` token). Webui uses 16px for the web context; desktop apps conventionally run 2-3pt smaller.
- [x] **Heading scale** — `TextBlock.h1`/`h2`/`h3` classes in `Theme/Controls.axaml` mapped to `FontSize2xl`/`Xl`/`Lg` tokens.
- [x] **Code font** — `CodeFontFamily` resource (`Consolas, Menlo, DejaVu Sans Mono, monospace`). Used by markdown code blocks when we add them.
- [ ] **Line height** — using Avalonia defaults. Revisit if message density feels off.
- [ ] **Letter spacing** — defaults (no tracking overrides; revisit if needed).
- [x] **Button text** — `FontSize="FontSizeSm"` + `FontWeight="Medium"` baked into base Button style.
- [x] **Label text** — default `TextBlock` picks up `FontSizeBase`; field labels in forms use `FontSizeSm` via Grid layout.
- [x] **Helper text** — `TextBlock` classes `muted` + `xs` combine to the shadcn `text-xs text-muted-foreground` pattern.
- [ ] **Markdown prose** — deferred (no markdown rendering yet; Markdig integration is its own task).

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
- [ ] **Radius for icon buttons** — N/A until we add an icon library.
- [x] **Sidebar spacing** — `RadiusLg` container, 6px inner list margin, 10px button row padding in `SettingsWindow.axaml`.
- [x] **Message padding** — bubble `Padding="14,10"`, `Margin="0,4"` between bubbles via `Border.bubble` style.

### Iconography

- [ ] **Icon library** — `@lucide/svelte: ^0.515.0` (`package.json`)
- [ ] **Common icon usage**:
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
- [ ] **Icon sizing** — `size-4` (16px) default in buttons, `size-3` (12px) in badges, `size-5`/`size-6` for standalone icons
- [ ] **Icon colors** — inherit from button/text color context, or explicit `text-muted-foreground` in secondary contexts
- [ ] **Custom SVGs** — favicon inlined as base64 data URL in build (from `static/favicon.svg`)
- [ ] **Icon-only buttons** — variant `icon`, `icon-sm`, `icon-lg` sizes (9/5/10 units respectively)

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
- [ ] **Form validation** — deferred (no inputs with validation yet).
- [x] **Checkbox** — basic styling (Foreground + FontSize). Toggle/switch variant deferred.
- [ ] **Switch/toggle** — deferred.
- [x] **Cards/panels** — `Border.card` and `Border.panel` classes in `Theme/Controls.axaml`. SettingsWindow uses `card`; MainWindow toolbar/status use `panel`/`statusbar`.
- [x] **Message bubbles** — `Border.bubble.user` right-aligned Primary bg + max-width 640; `Border.bubble.assistant` left/stretch, Card bg + border. Role selection via bound bools `IsUser`/`IsAssistant`.
- [x] **Settings form** — grouped `section` headers, vertical stack of rows with label column, footer with Save/Close.
- [x] **Sidebar** — `SettingsWindow` left pane uses Sidebar/SidebarBorder tokens; list items highlight Accent on hover/selected via `ListBoxItem` styles.
- [ ] **Code block** — deferred until markdown rendering lands.
- [x] **Dialog/modal** — `SettingsWindow` shown via `ShowDialog(owner)` with `WindowStartupLocation="CenterOwner"`.
- [ ] **Toast/notification** — deferred.
- [ ] **Tooltip** — deferred.

### Motion + accessibility

- [ ] **Transitions** — deferred (Avalonia `Transitions` on properties; add fade-in for new messages and slide for Expander when polish pass happens).
- [x] **Hover effects** — `:pointerover /template/ ContentPresenter#PART_ContentPresenter` setters on every Button variant.
- [x] **Focus rings** — `TextBox:focus` thickens BorderBrush to `Ring`. Avalonia's built-in focus adorner is inherited from FluentTheme.
- [x] **Disabled state** — `Button:disabled` and `TextBox:disabled` drop Opacity to 0.5; Button also flips Cursor to Arrow.
- [ ] **Streaming cursor animation** — deferred.
- [ ] **Auto-scroll on new messages** — deferred.
- [ ] **Reduced motion support** — N/A until we add animations; Avalonia has no built-in `prefers-reduced-motion` query, need to poll or wire manually.
- [x] **Focus trap in dialogs** — Avalonia's `ShowDialog` traps focus natively.
- [x] **Keyboard navigation** — Avalonia defaults handle Tab/arrows/Escape. Our `Enter/Shift+Enter` handler is in `MainWindow.axaml.cs:12-22`. Ctrl+L (Load) and Ctrl+, (Preferences) wired via `InputGesture` on MenuItems.
- [ ] **ARIA labels** — deferred. Avalonia's analog is `AutomationProperties.Name` etc.
- [x] **Contrast** — sRGB hex values chosen from Tailwind neutral palette (WCAG AA compliant) + shadcn destructive red.

### Styling techniques

- [x] **Utility-first via Style Selectors** — `Classes="outline sm"` composes like Tailwind classes. No CSS-in-C# library.
- [x] **CSS variables analog** — `ResourceDictionary.ThemeDictionaries` + `DynamicResource` binding.
- [ ] **Backdrop blur** — deferred (Avalonia `ExperimentalAcrylicBorder` covers this; add if we want shadcn's frosted ghost hover).
- [ ] **Shadow scale** — deferred (no shadows applied; Avalonia uses `BoxShadows` property on Border).
- [ ] **Scrollbar styling** — deferred.
- [x] **Global styles** — base `Window` and `TextBlock` selectors in `Theme/Controls.axaml` set Background/Foreground/FontSize.
- [ ] **Custom SCSS for external content** — N/A.

## Parity strategy notes

### High-confidence, cheap wins
- [ ] Basic UI shell (layout, routing, sidebar) — straightforward Avalonia port, no API dependencies
- [ ] Settings panels — form generation is mechanical, can template per section
- [ ] Message list UI — straightforward MVVM binding to chat history
- [ ] Button/input component library — shadcn-svelte → shadcn-avalonia or custom, all variants documented above
- [ ] Dark mode — Avalonia supports `ThemeVariant.Dark/Light`, CSS variables → static resource brushes
- [ ] Keyboard shortcuts — Avalonia keybinding system, register same Ctrl+K, Ctrl+Shift+O, Ctrl+Shift+E
- [ ] Conversation list with search — IndexedDB → local SQLite or memory cache, search trivial

### Medium effort, well-scoped
- [ ] Markdown rendering with remark/rehype — C# markdown libraries available (Markdig), syntax highlighting via Prism/Pygments, KaTeX rendering via HTML → WPF `FlowDocument` or custom XAML parser
- [ ] File attachments & drag-drop — Avalonia has file dialogs and drag-drop APIs
- [ ] Audio recording — NAudio or CoreAudio bindings available
- [ ] MCP protocol client — already have `@modelcontextprotocol/sdk` equivalent in dotnet (MCP.NET); integrate server communication
- [ ] Model selector with search — bind to model list, filter on TextChanged, group by capability

### Known blockers or server-specific stubs
- [ ] **Multi-model routing** — if server only runs one model, disable model-per-conversation UI and just show selected model in header
- [ ] **MCP resource browser** — MCP protocol support required; if not needed, stub with "MCP not available" state
- [ ] **Tool calling / prompt picker** — tied to MCP; stub if MCP not used
- [ ] **Continue generation** — needs server-side streaming support; verify llama.cpp supports continuing from offset
- [ ] **Token count estimation** — requires tokenizer (tiktoken or llama.cpp vocab); may need to request from server or embed tokenizer

### Visual implementation notes
- [x] **OKLCH colors** — translated to sRGB hex anchored on Tailwind v4 neutral palette.
- [x] **Radius scale** — `CornerRadius` resources in `Theme/Tokens.axaml`.
- [x] **Fonts** — inherited system stack.
- [x] **Spacing scale** — `SpacingXs/Sm/Md/Lg/Xl` + `PaddingXs/Sm/Md/Lg` resources.
- [ ] **Icons** — deferred (candidates: `Projektanker.Icons.Avalonia.MaterialDesign` for a Material set, or embed lucide SVG path data as `Geometry` resources).
- [ ] **Animations** — deferred.

---

**Total feature items:** 150+ checkboxes  
**Total theming detail lines:** 80+ bullets  

Track implementation progress by checking items off. Use this checklist to ensure no UX blind spots as desktop client development proceeds. Cross-reference component files above when implementing each feature.
