# Native binaries

LlamaCpp.Bindings doesn't ship the native library; it expects one to be present at `runtimes/<rid>/native/`, where `<rid>` is a .NET [runtime identifier](https://learn.microsoft.com/dotnet/core/rid-catalog) like `win-x64`, `linux-x64`, or `osx-arm64`. The [`NativeLibraryResolver`](../src/LlamaCpp.Bindings/Native/NativeLibraryResolver.cs) hooks into .NET's P/Invoke resolution and probes that location first.

This page describes how to get that library in place on each platform, and the trade-offs between the options.

## tl;dr

| Platform | What upstream ships | Recommended default |
|---|---|---|
| Windows x64 | CPU, CUDA 12.4, CUDA 13.1, Vulkan, HIP (AMD), SYCL (Intel) | **CUDA 12.4** for NVIDIA users (forward-compatible with 12.x drivers); Vulkan as a universal fallback |
| macOS arm64 | Metal | **Metal** — it's the only sensible option |
| macOS x64 | Generic CPU | **CPU** |
| Linux x64 | CPU, Vulkan, ROCm (AMD), OpenVINO (Intel) — **no CUDA prebuilt** | **Vulkan** for portable default; build your own **CUDA** for max NVIDIA performance |
| Linux arm64 | CPU, Vulkan | **Vulkan** or **CPU** depending on GPU |

## The `fetch-binaries.py` script

`tools/fetch-binaries.py` downloads prebuilt archives from [ggml-org/llama.cpp's GitHub releases](https://github.com/ggml-org/llama.cpp/releases), extracts the shared libraries (not the CLI tools), and writes them under `runtimes/<rid>/native/`.

### Release mode (prebuilt)

```bash
python tools/fetch-binaries.py \
    --tag b8875 \
    --platform <rid> \
    --backend <backend> \
    [--with-cudart <version>]    # Windows CUDA only; see below
```

Examples:

```bash
# Windows CUDA 12.4: grab the binding + the separate CUDA runtime bundle in one go
python tools/fetch-binaries.py --tag b8875 --platform win-x64 \
    --backend cuda-12.4 --with-cudart 12.4

# Linux x64 Vulkan (NVIDIA-compatible, AMD-compatible, Intel-compatible)
python tools/fetch-binaries.py --tag b8875 --platform linux-x64 --backend vulkan

# macOS Apple Silicon Metal
python tools/fetch-binaries.py --tag b8875 --platform osx-arm64 --backend metal
```

Downloads are cached under `tools/.cache/` so repeated invocations are idempotent. Pass `--force` to re-download.

### Local-build mode (for when no upstream prebuilt exists)

Common case: **Linux + NVIDIA CUDA**, for which upstream has never published a prebuilt (we scanned release tags back to `b2000` — it has always been build-your-own or use Vulkan). If you built llama.cpp yourself:

```bash
python tools/fetch-binaries.py \
    --from-local ~/Programming/llama.cpp/build_cuda/bin \
    --platform linux-x64
```

This walks the un-versioned SONAME entry points (`libllama.so`, `libggml*.so`, `libmtmd.so`) and copies each symlink chain into `runtimes/linux-x64/native/` — preserving symlinks so SOVERSION resolution still works at load time. Stale siblings from older builds (e.g. a lingering `libllama.so.0.0.8578` next to the current `libllama.so.0.0.8620`) are ignored.

#### RUNPATH gotcha on Linux local builds

Locally-built `.so` files often have an `RUNPATH` entry pointing at the original build directory — `readelf -d libllama.so.0.0.XXX | grep RUNPATH`. That means at load time, sibling libraries are resolved from your build tree, not from `runtimes/`. The script prints a warning when it detects this.

**Consequences:**
- Works on your dev box while the build tree stays put.
- Breaks silently the moment the build tree is moved or deleted.
- Not redistributable — don't ship a tarball of `runtimes/linux-x64/` from a local CUDA build without fixing the RUNPATH first.

**Fix for redistribution:** either use `patchelf --set-rpath '$ORIGIN' <lib>` on each file, or rebuild llama.cpp with `cmake -DCMAKE_INSTALL_RPATH='$ORIGIN'` and install.

## Per-platform details

### Windows x64

Upstream ships several backend variants. Pick by GPU vendor:

| GPU vendor | Backend arg | Notes |
|---|---|---|
| NVIDIA, any 12.x driver | `cuda-12.4` | Forward-compatible within CUDA major; works on 12.8 drivers. |
| NVIDIA, 13.x driver | `cuda-13.1` | Use if you're on a recent driver and want the freshest kernels. |
| NVIDIA (no CUDA toolkit) | `vulkan` | Works without the CUDA runtime; usually 10-20% slower than CUDA. |
| AMD Radeon | `hip-radeon` | |
| Intel | `sycl` or `vulkan` | |
| None / CPU | `cpu` | |

**Don't forget `--with-cudart <ver>` on CUDA builds.** The `cudart*.dll` files ship as a separate asset (`cudart-llama-bin-win-cuda-<ver>-x64.zip`). Without them, `llama.dll` fails to load with an obscure DLL-not-found error at startup unless the user already has the CUDA Toolkit installed system-wide.

### macOS

Apple Silicon: use Metal. Intel Mac: there's only a CPU build; Metal support for Intel Macs was dropped upstream.

### Linux x64

**No CUDA prebuilt** is shipped by upstream — confirmed by scanning every release tag from `b2000` through `b8875`. Options:

- **Vulkan prebuilt (recommended default).** Works on NVIDIA, AMD, and Intel GPUs. Single binary, redistributable, no runtime toolkit dependency. ~80-90% of CUDA performance on LLM workloads.
- **Build CUDA yourself.** For max NVIDIA performance and access to CUDA-only features. One-time build, then use `--from-local` to stage the `.so` files. See the RUNPATH note above before redistributing.
- **ROCm prebuilt** for AMD cards if you want the ROCm path instead of Vulkan.
- **CPU prebuilt** (the plain `ubuntu-x64` asset) — fine for small models, impractical for 30B+ parameter models.

### Linux arm64

Vulkan or CPU prebuilts. Same general story as x64.

## Build output layout

After `fetch-binaries.py`, the tree looks like:

```
runtimes/
  linux-x64/
    native/
      libllama.so
      libllama.so.0            → libllama.so.0.0.8620
      libllama.so.0.0.8620
      libggml.so
      libggml-base.so.0
      libggml-cuda.so.0
      ...
```

When you build your project (`dotnet build` / `dotnet publish`), MSBuild copies this tree into `bin/Debug/<tfm>/runtimes/<rid>/native/` alongside the managed DLL. The `NativeLibraryResolver` finds it there at runtime.

## Diagnosing native load failures

If the managed code throws `DllNotFoundException: Unable to load shared library 'llama'`:

1. **Check the probe list in the exception.** .NET lists every path it tried. Our resolver adds `runtimes/<rid>/native/<libname>` next to the managed DLL and at `AppContext.BaseDirectory`. If neither is shown, the resolver didn't register — make sure your code calls `LlamaBackend.Initialize()` at least once before any P/Invoke.
2. **Is the file actually there?** `ls bin/Debug/net*/runtimes/*/native/libllama.*`. If it's missing, the csproj's `<None Include="...runtimes\**">` glob didn't fire — make sure `runtimes/` exists before building.
3. **Does it load standalone?** `ldd bin/Debug/net*/runtimes/linux-x64/native/libllama.so` — every line should resolve. Red "not found" lines reveal missing dependencies (usually a `libcudart.so.12` that wasn't installed).
4. **On Windows, check for `cudart64_*.dll`** in the output directory. Missing cudart is by far the most common first-time Windows failure.

See also [troubleshooting.md](troubleshooting.md) for more specific failure modes.
