# Updating the pinned llama.cpp version

The binding pins to a specific llama.cpp commit. The pin lives in two places:

- `third_party/llama.cpp/VERSION` — the human-readable record (git describe, SHA, date, and *why* this specific commit was chosen)
- `third_party/llama.cpp/include/llama.h.pinned` — a byte-exact copy of `llama.h` from that commit
- `third_party/llama.cpp/api.pinned.json` — a structured snapshot of the ABI surface extracted via libclang; consumed by the diff pipeline

This page describes the maintenance workflow for bumping that pin. The design premise is that updates should be guided edits against a machine-generated change report, not manual diffs against a 1500-line C header.

## The one-command path

```bash
tools/check-for-updates.sh            # compares pin -> GitHub's latest release
# or
tools/check-for-updates.sh b9050      # compares pin -> a specific tag
```

Exit codes:
- `0` — no API changes between the pin and the target; you can bump the pin without binding code changes.
- `3` — API changes detected; review `UPDATE_REPORT.md` at the repo root.
- `1`/`2` — setup failure (missing venv, network, etc.).

The script:
1. Bootstraps `tools/.venv` with `libclang` if it doesn't exist (one-time).
2. Resolves the target tag (from GitHub's "latest release" API or your argument).
3. Downloads the target tag's `include/llama.h` plus the transitively-included `ggml/include/*.h` files.
4. Parses the target header with libclang → JSON (`tools/extract-api.py`).
5. Diffs against `third_party/llama.cpp/api.pinned.json` → markdown (`tools/diff-api.py`).
6. Cross-references each changed symbol against the C# sources under `src/` (`tools/xref-bindings.py`).
7. Concatenates everything into `UPDATE_REPORT.md`.

`UPDATE_REPORT.md` is gitignored — it's generated on demand, not kept in version control.

## Reading `UPDATE_REPORT.md`

The report has three sections in a single file.

**Header** — pin vs. target metadata.

**Change report** — what moved in `llama.h`, grouped by risk:
- `Functions removed` / `Functions added` — self-explanatory. Removals are hazards: code we call that no longer exists.
- `Function signatures changed` — most common mode is parameter type drift (`int32_t` → `int`, pointer-vs-value) or new parameters at the end.
- `Function deprecation changes` — not urgent, but worth noting if we depend on a newly-deprecated function.
- `Structs added/removed` — low-risk unless we mirror one of them.
- **`Struct layout changes`** — the highest-risk category. A changed field order, added/removed field, or size mismatch silently corrupts memory on the next P/Invoke call unless our C# mirror is updated. The report flags these explicitly with "re-run struct-size probe" guidance.
- `Enum value changes` — usually safe, but if a value we reference has a new numeric value, our managed enum needs the same update.
- `Typedef changes` — often cosmetic (e.g., `int32_t` → an alias), but worth a glance.

**Binding xref** — for every changed symbol, shows exactly which C# files and lines reference it. Uses plain text word-boundary matching, which works because our convention is to mirror C names 1:1 (`llama_decode`, not `LlamaDecode`). A "Symbols not referenced in C#" section at the end lists changes that don't touch our bindings — either they're in the unbound surface (expected) or the name-mirror convention has slipped somewhere.

## Applying a change: concrete example

Suppose the report shows:

```
### Function signatures changed
- `llama_decode`
    - old: `int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch)`
    - new: `int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch, int32_t flags)`
```

And xref shows:

```
## `llama_decode` (function, signature changed)
- src/LlamaCpp.Bindings/Native/NativeMethods.cs
    - L98: `internal static partial int llama_decode(IntPtr ctx, llama_batch batch);`
- src/LlamaCpp.Bindings/LlamaGenerator.cs
    - L172: `var rc = NativeMethods.llama_decode(_context.Handle.DangerousHandle, batch);`
```

Steps to apply:

1. Update the P/Invoke declaration in `NativeMethods.cs` to match the new signature (add the `int flags` parameter).
2. Update every call site (`LlamaGenerator.cs:172`) to pass a sensible default.
3. Run the tests: `dotnet test`. The generation tests will exercise the change end-to-end against the new native binary.
4. Once green, move on to the next change.

## Applying a struct layout change

This is the dangerous category — the binding's correctness depends on byte-for-byte match.

1. **Regenerate ground truth.** Run `tools/dump-struct-sizes.sh <path-to-new-llama.cpp-checkout>` and commit the new output to `tools/struct-sizes.json`. This gives you the authoritative byte sizes and offsets from the new native build.
2. **Update the mirror** in `src/LlamaCpp.Bindings/Native/NativeStructs.cs`. Preserve field order exactly as declared in the new `llama.h` — do NOT reorder for readability. Add/remove fields to match. Update the struct's `ExpectedSize` constant.
3. **Update the assertions** in `src/LlamaCpp.Bindings.Tests/StructLayoutTests.cs` — size and every field offset. The ground-truth JSON from step 1 has every number you need.
4. `dotnet test` — `NativeLayout.Verify()` runs at backend init; if sizes don't match, it throws rather than corrupting memory. Field offset tests will fail loudly if any field is off.

**If a struct becomes opaque or gains an inner union,** stop and think about whether we actually need to mirror it — opaque types are just handed around as pointers. Sometimes the right move is to stop mirroring and treat it as `IntPtr`.

## Promoting the new pin

Once tests are green on the new target:

1. Copy the target header: `cp <new-llama.cpp>/include/llama.h third_party/llama.cpp/include/llama.h.pinned`
2. Regenerate the JSON snapshot: `tools/.venv/bin/python tools/extract-api.py third_party/llama.cpp/include/llama.h.pinned -I <new-llama.cpp>/include -I <new-llama.cpp>/ggml/include -o third_party/llama.cpp/api.pinned.json`
3. Edit `third_party/llama.cpp/VERSION` to record the new git-describe, commit SHA, SOVERSION, header date, and (crucially) **why** we're at this commit. If we're deliberately not at `latest`, say so.
4. Refetch the native library against the new pin: `tools/fetch-binaries.py --tag <new-tag> --platform <rid> --backend <backend>`
5. Commit. Suggested message format:
   ```
   Bump pinned llama.cpp to <new-describe>
   
   Changes applied:
   - <what moved in llama.h>
   - <what we had to change in NativeMethods/NativeStructs/etc.>
   
   Tests: 85/85 passing.
   ```

## Automation

`check-for-updates.sh` is cron-friendly. A weekly CI job along the lines of:

```yaml
- name: Weekly llama.cpp drift check
  run: |
    tools/setup-venv.sh
    tools/check-for-updates.sh || true
    # Post UPDATE_REPORT.md to wherever (issue, chat, email) if non-empty.
```

...keeps the pin from silently going stale. The report is cheap to generate (seconds) and costs nothing until you decide to act on it.

## When the pipeline itself breaks

Possible failures in the tooling:

- **libclang can't find `stdbool.h`.** Happens if no system `clang` is installed — `extract-api.py` auto-detects the clang resource dir by shelling out to `clang -print-resource-dir`. Install `clang` (or `clang-18` / `clang-20`) and retry.
- **Header parse has errors.** Usually a missing `-I` for the ggml include tree. The script searches for `ggml/include` relative to `llama.h`'s directory; if that guess is wrong, pass `-I` explicitly.
- **Symbol in the diff isn't in xref's "not referenced" list but also not shown in any C# file.** Something's broken our name-mirror convention (the binding is exposing `Foo` for something C calls `foo_thing`). Audit the new binding code — the convention is in [CLAUDE.md](../CLAUDE.md).
