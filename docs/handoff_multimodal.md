# Multimodal handoff

Picks up from a context-full session. The app is feature-complete against
the current native surface — 109 done / 0 actionable / 65 deferred / 30 N/A
in [webui_feature_checklist.md](webui_feature_checklist.md). 39 of the 65
deferred items unblock when multimodal lands.

## Goal for the new session

Let a user attach images to messages and have a vision-capable model reply.
Starts with **images only**; audio is a follow-up phase (same architecture,
different native path).

Acceptance criteria:

1. Tests can load a small vision model + mmproj and decode a turn that
   contains an image.
2. A profile can point at both a model GGUF and an mmproj GGUF.
3. Compose bar accepts images via file picker, drag-drop, and clipboard
   paste. Thumbnails with a remove button sit above the text box.
4. Attached images render in the user bubble.
5. The model answers correctly when asked about an attached image
   (smoke test: a tiny JPEG + "what's in this image?").

## Landscape — what's in llama.cpp

Look at `llama.cpp/tools/mtmd/`:

- Header: `mtmd.h` (public C API)
- Also `mtmd-helper.h` (convenience helpers we'll want to bind —
  bitmap-from-file / -from-buf, `eval_chunks`)
- `tools/mtmd/README.md` is brief but covers the concept

Multimodal in llama.cpp is a sibling library to libllama, exposed as a
C API. Core types and functions we need to bind:

```
// Opaque handles
struct mtmd_context;
struct mtmd_bitmap;
struct mtmd_input_chunks;

// Lifecycle
mtmd_context_params mtmd_context_params_default(void);
mtmd_context*       mtmd_init_from_file(const char* mmproj_path,
                                        const llama_model* text_model,
                                        mtmd_context_params params);
void                mtmd_free(mtmd_context*);

// Bitmap — pixel buffer the native side consumes
mtmd_bitmap*        mtmd_bitmap_init(size_t nx, size_t ny, const unsigned char* data);
void                mtmd_bitmap_free(mtmd_bitmap*);
// Helpers (in mtmd-helper.h):
int32_t             mtmd_helper_bitmap_init_from_file(mtmd_context*,
                                                      const char* path,
                                                      mtmd_bitmap** out);
int32_t             mtmd_helper_bitmap_init_from_buf(mtmd_context*,
                                                      const unsigned char*, size_t,
                                                      mtmd_bitmap** out);

// Tokenize a prompt that contains image markers + bitmaps into chunks
mtmd_input_chunks*  mtmd_input_chunks_init(void);
void                mtmd_input_chunks_free(mtmd_input_chunks*);
int32_t             mtmd_tokenize(mtmd_context*, mtmd_input_chunks* out,
                                   const mtmd_input_text* text,
                                   const mtmd_bitmap** bitmaps, size_t n_bitmaps);

// Decode chunks into llama_context (writes to KV cache)
int32_t             mtmd_helper_eval_chunks(mtmd_context*,
                                             llama_context*, mtmd_input_chunks*,
                                             llama_pos n_past, llama_seq_id seq_id,
                                             int32_t n_batch, bool logits_last,
                                             llama_pos* out_n_past);
```

Models with vision ship an mmproj file in addition to the base text GGUF.
Qwen2.5-VL, Gemma-3 vision, SmolVLM, LLaVA, MoonDream, etc. A vision model
with no mmproj does nothing useful; a text-only model with an mmproj errors
at load.

## Work plan

### Phase A — native bindings

Scope: ~400–600 LOC across these new files. Mirror the shapes of
`LlamaModel.cs` / `LlamaContext.cs` exactly.

1. `src/LlamaCpp.Bindings/Native/NativeMethods.cs` — add `mtmd_*` +
   `mtmd_helper_*` declarations. Keep C names verbatim.
2. `src/LlamaCpp.Bindings/Native/NativeStructs.cs` — `mtmd_context_params`,
   `mtmd_input_text`, etc. with `[StructLayout(LayoutKind.Sequential)]` +
   `SizeAssertion()` called at module init.
3. `src/LlamaCpp.Bindings/Native/SafeHandles/`
   - `SafeMtmdContextHandle.cs` → `mtmd_free`
   - `SafeMtmdBitmapHandle.cs` → `mtmd_bitmap_free`
   - `SafeMtmdInputChunksHandle.cs` → `mtmd_input_chunks_free`
4. `src/LlamaCpp.Bindings/MtmdContext.cs` — public wrapper. Ctor takes
   `LlamaModel` + mmproj path, implements `IDisposable`.
5. `src/LlamaCpp.Bindings/MtmdBitmap.cs` — public wrapper. Three factories:
   `FromFile(MtmdContext, string)`, `FromBytes(MtmdContext, ReadOnlySpan<byte>)`,
   `FromPixels(int w, int h, ReadOnlySpan<byte> rgb)`. `IDisposable`.
6. Method on `MtmdContext`: `EvalPromptAsync(LlamaContext, string prompt, IReadOnlyList<MtmdBitmap>, int nPast, int seqId, int nBatch, bool logitsLast, CancellationToken)`
   returning the new `nPast`. Internally: tokenize → eval_chunks → free chunks.
7. Tests: `src/LlamaCpp.Bindings.Tests/MtmdTests.cs` via the existing
   `ModelFixture` pattern. Uses a tiny vision model (SmolVLM-256M-Instruct
   or similar — choose by CI time budget; keep under 500MB ideally).

### Phase B — app integration

Scope: ~300–500 LOC.

1. **ModelProfile**: add `MmprojPath` (optional). Load MtmdContext in
   `ChatSession.Load` if present.
2. **ChatTurn / MessageViewModel**: add `List<Attachment>` where
   `Attachment = (byte[] Data, string MimeType, string? FileName)`.
   Serialises to JSON as base64. Image-only for v1.
3. **ChatSession**:
   - When MtmdContext is set, prefill via `MtmdContext.EvalPromptAsync`
     instead of `LlamaGenerator`'s plain tokenize+decode.
   - Compose the prompt so image placeholder tokens appear where the user
     attached images. Qwen2.5-VL and most VL models embed `<|vision_start|>
     <|image_pad|> <|vision_end|>` in chat templates automatically — the
     Jinja renderer already exercises this branch in `render_content` when
     content is a list of parts. Check the template dump (see caveats).
   - Keep the v1 KV policy: clear + re-decode every turn. Image prefill is
     expensive but correct; prefix-cache reuse stays deferred.
4. **DialogService**: `PickImageFilesAsync` (OpenFilePickerAsync with image
   MIME filter).
5. **Compose**:
   - Attachment button + hidden file input.
   - Drag-drop: handle `DragDrop.Drop` on the compose grid; read files from
     `DataObject`. Show overlay while dragging.
   - Clipboard paste: listen for `PastingFromClipboard`, check for image
     data on `DataObject.Contains(DataFormats.Bitmap)` or similar.
   - `ObservableCollection<Attachment> PendingAttachments` on
     MainWindowViewModel; cleared after send.
   - Thumbnail strip above the TextBox (small Image controls bound to
     `Data` via a bitmap converter, with × remove button).
6. **Bubble template**: render `Attachments` as a thumbnail row above the
   content. Click opens a full-size preview dialog (reuse the
   `CodePreviewDialog` pattern).

## Files to read first (priorities)

1. This doc + `docs/webui_parity_investigation.md` (Run 1, gap #1)
2. `CLAUDE.md` (project conventions — non-negotiable)
3. `src/LlamaCpp.Bindings/LlamaModel.cs` + `LlamaContext.cs` — the canonical
   shape for a bindings wrapper; copy this structure.
4. `src/LlamaCpp.Bindings/Native/NativeMethods.cs` — P/Invoke conventions
5. `src/LlamaCpp.Bindings/Native/SafeHandles/` — one per opaque pointer
6. `llama.cpp/tools/mtmd/mtmd.h` + `mtmd-helper.h` — the API to bind
7. `src/LlamaCpp.Bindings.LlamaChat/Services/ChatSession.cs` — where the
   app currently does prefill + decode; this grows a multimodal branch
8. `src/LlamaCpp.Bindings.LlamaChat/ViewModels/MessageViewModel.cs` — where
   attachments will live in the VM layer

## Conventions (from CLAUDE.md)

- `[LibraryImport("llama", StringMarshalling = StringMarshalling.Utf8)]`
  — never `DllImport`.
- C# method names mirror C exactly (`mtmd_init_from_file`, not
  `MtmdInitFromFile`). Makes header diffs mechanical to apply when
  llama.cpp updates.
- Every opaque native pointer gets a dedicated `SafeHandle` subclass under
  `Native/SafeHandles/`. Public API takes/returns `SafeXxxHandle`, never
  `IntPtr`.
- `[StructLayout(LayoutKind.Sequential)]` on every mirror struct. Field
  order matches the header *exactly*. `[MarshalAs(UnmanagedType.I1)] bool`
  for C `_Bool`. Every struct gets a `SizeAssertion()` method called at
  module init.
- Native functions returning status `int`: nonzero → throw
  `LlamaException(functionName, code)`.
- Native functions returning pointers: null → throw
  `LlamaException(functionName)`.
- Public classes are `IDisposable`; `Dispose()` disposes the
  SafeHandle(s).
- Any native log lines go through the `LlamaBackend` log callback that's
  already wired — don't re-implement stderr capture.
- Every path that can take meaningful time accepts a `CancellationToken`.

## Caveats / gotchas this codebase has

- **Avalonia 12 clipboard**: `IClipboard.SetTextAsync` is an **extension
  method** in `Avalonia.Input.Platform.ClipboardExtensions`, not on
  IClipboard. `DialogService.CopyToClipboardAsync` is the working pattern.
- **AXAML quirks**: XML comments can't contain `--`. `xmlns:` declarations
  must be on the root element, not nested. Both bit us early.
- **Jinja renderer**: we wrote our own subset in `src/LlamaCpp.Bindings/Jinja/`
  because `llama_chat_apply_template` doesn't understand Qwen3-family
  templates properly. The `render_content` macro handles string content
  today; it will also handle list-of-parts (text + image_url) via `is
  iterable` / `is not mapping` branches when multimodal lands. Make sure
  we emit the right image placeholder tokens (`<|vision_start|>
  <|image_pad|> <|vision_end|>` for Qwen, different for others) when
  rendering.
- **Template dump**: `ChatSession.Load` writes the model's embedded Jinja
  template to `~/.config/LlamaChat/last-template.jinja`. Always look at
  that first when adapting to a new model family.
- **Thinking-mode prompts**: Qwen3 templates pre-open `<think>`;
  `ReasoningExtractor` primes accordingly when the prompt ends that way.
  Qwen2.5-VL probably has the same structure. Watch for it.
- **KV policy is v1** (clear + re-prefill every turn). Don't regress to
  accidentally share across turns when adding multimodal — image embeds
  are expensive to recompute but correct. Prefix-cache reuse is a
  separate deferred item.
- **One-line test-data trick**: we stage the real Qwen3.6 template under
  `src/LlamaCpp.Bindings.Tests/TestData/qwen36-template.jinja`, the
  csproj has `<None Update="TestData\**\*"><CopyToOutputDirectory>
  PreserveNewest</CopyToOutputDirectory></None>`. Drop a tiny vision
  model's mmproj there if we end up shipping a fixture.

## Build / run / test cheatsheet

```bash
# Whole solution
dotnet build LlamaCpp.Bindings.slnx

# Just the app
dotnet build src/LlamaCpp.Bindings.LlamaChat/LlamaCpp.Bindings.LlamaChat.csproj

# Run the app
dotnet run --project src/LlamaCpp.Bindings.LlamaChat/LlamaCpp.Bindings.LlamaChat.csproj

# Tests
dotnet test src/LlamaCpp.Bindings.Tests/LlamaCpp.Bindings.Tests.csproj

# Jinja tests only (model-free)
dotnet test src/LlamaCpp.Bindings.Tests/LlamaCpp.Bindings.Tests.csproj --filter FullyQualifiedName~JinjaTests
```

Native binaries are in `runtimes/<rid>/native/`. Pinned version at
`third_party/llama.cpp/VERSION`. To pick up mtmd library files you may
need to re-run `tools/fetch-binaries.py` after bumping the pinned version
if mtmd was added or changed recently — see `docs/updating-llama-cpp.md`.

## Commit style + scope

Match what's already in the log:

```
218f015 LlamaChat: remaining actionable TODOs — 0 left
9c8b22e LlamaChat: theme toggle, merged Send/Stop, toasts, shortcuts + model info dialogs, splash
6125019 LlamaChat: syntax highlighting + code-block copy/expand + streaming cursor
006f219 Chat settings, Jinja template engine, streaming architecture
3b64c8f LlamaChat: per-message actions (copy / edit / regenerate / delete)
dfaf519 LlamaChat: visual theming + app shell (sidebar, conversations)
5c04c86 New LlamaChat app scaffold targeting llama-server webui parity
```

Two commits is likely right: one for Phase A (native bindings + tests),
one for Phase B (app integration). Keeps blame-bisect clean.

## Starting prompt for the new session

Suggested opening (paste this into the new conversation):

> Continuing LlamaCpp.Bindings work from a context-full session. Read
> `docs/handoff_multimodal.md` first — it's the whole brief. We're starting
> Phase A (native mtmd bindings). Begin by inspecting
> `llama.cpp/tools/mtmd/mtmd.h` + `mtmd-helper.h` and propose the P/Invoke
> surface. Don't code yet; show me the plan first.
