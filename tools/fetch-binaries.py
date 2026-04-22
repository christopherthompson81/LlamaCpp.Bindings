#!/usr/bin/env python3
"""Fetch llama.cpp native binaries for the managed binding.

Two modes:

  Release mode (default, for shippable/CI builds):
    python tools/fetch-binaries.py --tag b8875 --platform linux-x64 --backend vulkan
    python tools/fetch-binaries.py --tag b8875 --platform win-x64    --backend cuda-12.4
    python tools/fetch-binaries.py --tag b8875 --platform osx-arm64  --backend metal
      1. Resolves (platform, backend) -> GitHub release asset filename.
      2. Downloads to tools/.cache/ (idempotent; reuses cached archive).
      3. Extracts only native shared libraries into runtimes/<rid>/native/.

  Local-build mode (for dev boxes that build llama.cpp themselves, e.g. CUDA
  on Linux where no prebuilt is published):
    python tools/fetch-binaries.py --from-local ~/Programming/llama.cpp/build_cuda/bin \
                                   --platform linux-x64
      Copies lib{llama,ggml*,mtmd}.so* into runtimes/<platform>/native/,
      preserving symlinks so SONAME chains resolve at runtime.

Windows CUDA release builds need cudart DLLs that ship in a separate asset
(cudart-llama-bin-win-cuda-<ver>-x64.zip). Pass --with-cudart to grab
those too in one go.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = "ggml-org/llama.cpp"
RELEASE_URL = f"https://github.com/{REPO}/releases/download"

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "tools" / ".cache"
RUNTIMES_DIR = REPO_ROOT / "runtimes"

# .NET runtime identifiers we care about. Extend as needed.
SUPPORTED_RIDS = {
    "win-x64",
    "win-arm64",
    "linux-x64",
    "linux-arm64",
    "osx-x64",
    "osx-arm64",
}


@dataclasses.dataclass(frozen=True)
class AssetSpec:
    filename: str
    archive_kind: str  # "zip" or "tar.gz"


def resolve_asset(tag: str, platform: str, backend: str) -> AssetSpec:
    """Map (platform, backend) -> release asset filename.

    Naming scheme observed on llama.cpp releases:
      Windows:  llama-<tag>-bin-win-<backend>-<arch>.zip
                (backend is literal: cpu, cuda-12.4, cuda-13.1, vulkan,
                 hip-radeon, sycl, opencl-adreno)
      Linux:    llama-<tag>-bin-ubuntu[-<backend>]-<arch>.tar.gz
                (backend omitted == CPU; vulkan, rocm-7.2 known)
      macOS:    llama-<tag>-bin-macos-<arch>[-kleidiai].tar.gz
                (Metal is the default and has no suffix)
    """
    b = backend.lower()
    if platform not in SUPPORTED_RIDS:
        raise SystemExit(f"unsupported platform: {platform} (expected one of {sorted(SUPPORTED_RIDS)})")

    if platform.startswith("win-"):
        arch = platform.split("-", 1)[1]
        return AssetSpec(
            filename=f"llama-{tag}-bin-win-{b}-{arch}.zip",
            archive_kind="zip",
        )

    if platform.startswith("linux-"):
        arch = platform.split("-", 1)[1]
        # "cpu" is spelled by omitting the backend suffix
        backend_part = "" if b == "cpu" else f"-{b}"
        return AssetSpec(
            filename=f"llama-{tag}-bin-ubuntu{backend_part}-{arch}.tar.gz",
            archive_kind="tar.gz",
        )

    # macOS
    arch = platform.split("-", 1)[1]
    # metal == default (no suffix); kleidiai is an explicit variant on arm64
    if b in ("metal", "cpu", "default", ""):
        suffix = ""
    elif b == "kleidiai" and arch == "arm64":
        suffix = "-kleidiai"
    else:
        raise SystemExit(f"unknown macOS backend: {backend}")
    return AssetSpec(
        filename=f"llama-{tag}-bin-macos-{arch}{suffix}.tar.gz",
        archive_kind="tar.gz",
    )


def cudart_asset(tag: str, cuda_version: str) -> AssetSpec:
    """The separate cudart bundle shipped with Windows CUDA releases.

    Note: as of b8875 this asset is not prefixed with the llama.cpp tag;
    it lives at cudart-llama-bin-win-cuda-<ver>-x64.zip under the same
    release. The 'tag' param is kept for signature symmetry.
    """
    _ = tag
    return AssetSpec(
        filename=f"cudart-llama-bin-win-cuda-{cuda_version}-x64.zip",
        archive_kind="zip",
    )


def download(tag: str, asset: AssetSpec, force: bool) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{tag}__{asset.filename}"
    if dest.exists() and not force:
        print(f"  cached: {dest.relative_to(REPO_ROOT)}")
        return dest

    url = f"{RELEASE_URL}/{tag}/{asset.filename}"
    print(f"  download: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:  # noqa: S310 (trusted host)
            total = int(resp.headers.get("Content-Length", 0))
            bytes_read = 0
            while chunk := resp.read(1 << 16):
                f.write(chunk)
                bytes_read += len(chunk)
                if total:
                    pct = 100 * bytes_read / total
                    print(f"\r  {bytes_read / 1e6:7.1f} / {total / 1e6:7.1f} MB ({pct:5.1f}%)", end="", flush=True)
        print()
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"download failed: HTTP {e.code} for {url}") from e

    tmp.rename(dest)
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
    print(f"  sha256 (truncated): {sha}")
    return dest


# File extensions we consider shared libraries worth shipping.
LIB_SUFFIXES = (".dll", ".dylib")


def is_shared_lib(name: str) -> bool:
    low = name.lower()
    if low.endswith(LIB_SUFFIXES):
        return True
    # Linux .so and versioned variants (libllama.so, libllama.so.1, ...)
    if ".so" in low:
        base = low.split("/")[-1]
        parts = base.split(".")
        if "so" in parts:
            return True
    return False


def is_cli_tool(name: str) -> bool:
    """Best-effort filter for bin/ executables that aren't shared libs."""
    base = name.rsplit("/", 1)[-1].lower()
    # Windows: llama-cli.exe, llama-server.exe, etc.
    if base.endswith(".exe"):
        return True
    # Linux/macOS CLI tools frequently live as bare executables named
    # llama-*. We only want libs, so just require the shared-lib test.
    return False


def extract_libs(archive: Path, kind: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    def _take(name: str, reader) -> None:
        base = Path(name).name
        target = out_dir / base
        with open(target, "wb") as w:
            shutil.copyfileobj(reader, w)
        extracted.append(target)

    if kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if is_cli_tool(name):
                    continue
                if not is_shared_lib(name):
                    continue
                with zf.open(info) as r:
                    _take(name, r)
    elif kind == "tar.gz":
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                if is_cli_tool(name):
                    continue
                if not is_shared_lib(name):
                    continue
                r = tf.extractfile(member)
                if r is None:
                    continue
                _take(name, r)
                # preserve exec bit where relevant
                os.chmod(out_dir / Path(name).name, 0o755)
    else:
        raise SystemExit(f"unknown archive kind: {kind}")

    return extracted


def fetch_one(tag: str, platform: str, backend: str, force: bool) -> None:
    asset = resolve_asset(tag, platform, backend)
    print(f"[{platform} / {backend}] asset = {asset.filename}")
    archive = download(tag, asset, force)
    out_dir = RUNTIMES_DIR / platform / "native"
    # Fresh extract each time so removed files don't linger.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    libs = extract_libs(archive, asset.archive_kind, out_dir)
    if not libs:
        raise SystemExit(f"no shared libraries extracted from {archive} — aborting")
    for p in sorted(libs):
        print(f"  -> {p.relative_to(REPO_ROOT)}  ({p.stat().st_size / 1e6:.2f} MB)")
    print(f"  {len(libs)} file(s) written to {out_dir.relative_to(REPO_ROOT)}")


# Un-versioned SONAME entry points we care about. Everything reachable by
# following symlinks from these names will be brought along. This avoids
# picking up stale sibling files from previous builds (e.g. libllama.so.0.0.8578
# lingering next to the current libllama.so.0.0.8620).
_LOCAL_ENTRY_POINTS_LINUX = (
    "libllama.so",
    "libggml.so",
    "libggml-base.so",
    "libggml-cpu.so",
    "libggml-cuda.so",
    "libggml-vulkan.so",
    "libggml-metal.so",
    "libggml-rpc.so",
    "libggml-hip.so",
    "libggml-sycl.so",
    "libmtmd.so",
)
_LOCAL_ENTRY_POINTS_DARWIN = (
    "libllama.dylib",
    "libggml.dylib",
    "libggml-base.dylib",
    "libggml-cpu.dylib",
    "libggml-metal.dylib",
    "libmtmd.dylib",
)


def _chase_symlinks(src_dir: Path, entry: Path, seen: set[Path]) -> None:
    """Walk a symlink chain inside src_dir one hop at a time, adding every
    hop to `seen`. Stops at the first regular file or a missing target.
    Refuses to escape src_dir (absolute symlinks or .. traversal are ignored).
    """
    if entry in seen or not entry.exists():
        return
    seen.add(entry)
    if not entry.is_symlink():
        return
    link_target = os.readlink(entry)
    if os.path.isabs(link_target) or ".." in Path(link_target).parts:
        return
    # Important: take exactly one hop. Do NOT resolve() the whole chain —
    # we need to preserve every intermediate symlink so the dynamic loader
    # can follow SONAME (e.g. libllama.so -> libllama.so.0 -> libllama.so.0.0.8620).
    next_hop = entry.parent / link_target
    _chase_symlinks(src_dir, next_hop, seen)


def copy_from_local(src_dir: Path, platform: str) -> None:
    """Copy shared libs from a local llama.cpp build directory.

    Starts from the un-versioned SONAME entry points (libllama.so, libggml*.so,
    libmtmd.so, etc.) and follows each symlink chain, copying every hop while
    preserving symlinks so the dynamic loader can resolve at runtime.

    Stale sibling files from older builds (e.g. libllama.so.0.0.8578 hanging
    around next to a current libllama.so.0.0.8620) are ignored because nothing
    in the active chain points to them.
    """
    if not src_dir.is_dir():
        raise SystemExit(f"--from-local: not a directory: {src_dir}")

    entry_points = _LOCAL_ENTRY_POINTS_DARWIN if platform.startswith("osx-") else _LOCAL_ENTRY_POINTS_LINUX

    reachable: set[Path] = set()
    for name in entry_points:
        entry = src_dir / name
        if entry.exists():
            _chase_symlinks(src_dir, entry, reachable)

    if not reachable:
        raise SystemExit(
            f"no SONAME entry points found in {src_dir} — expected one of: {', '.join(entry_points)}"
        )

    out_dir = RUNTIMES_DIR / platform / "native"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for entry in sorted(reachable):
        target = out_dir / entry.name
        if entry.is_symlink():
            os.symlink(os.readlink(entry), target)
        else:
            shutil.copy2(entry, target)
            os.chmod(target, 0o755)
        copied.append(target)

    print(f"  copied from: {src_dir}")
    for p in sorted(copied):
        if p.is_symlink():
            tag = f"symlink -> {os.readlink(p)}"
        else:
            tag = f"{p.stat().st_size / 1e6:6.2f} MB"
        print(f"  -> {p.relative_to(REPO_ROOT)}  ({tag})")
    print(f"  {len(copied)} file(s) written to {out_dir.relative_to(REPO_ROOT)}")

    # Warn about embedded RUNPATH — locally-built libs often point back at the build tree.
    # If the build tree moves or is deleted, loading fails silently. We can't easily fix
    # this without patchelf; surface it as a warning.
    main_lib = out_dir / ("libllama.dylib" if platform.startswith("osx-") else "libllama.so")
    if main_lib.exists() and shutil.which("readelf"):
        import subprocess
        result = subprocess.run(
            ["readelf", "-d", str(main_lib.resolve())],
            capture_output=True, text=True, check=False,
        )
        for line in result.stdout.splitlines():
            if "RUNPATH" in line or "RPATH" in line:
                # Extract path in brackets
                if "[" in line and "]" in line:
                    rpath = line.split("[", 1)[1].rsplit("]", 1)[0]
                    if rpath and "$ORIGIN" not in rpath:
                        print(
                            f"  WARNING: {main_lib.name} has RUNPATH=[{rpath}]\n"
                            f"           These libs will load siblings from the build tree, not from runtimes/.\n"
                            f"           Safe while the build tree stays in place; breaks silently if it moves.\n"
                            f"           Fix: patchelf --set-rpath '$ORIGIN' <lib>  (install `patchelf`)."
                        )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Fetch llama.cpp native binaries (prebuilt release OR local build).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--platform", required=True, help=f"RID, one of: {', '.join(sorted(SUPPORTED_RIDS))}")

    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--tag", help="release mode: llama.cpp release tag, e.g. b8875")
    src_group.add_argument(
        "--from-local", metavar="DIR", type=Path,
        help="local-build mode: copy *.so/*.dylib from this build output directory",
    )

    ap.add_argument(
        "--backend",
        help="release mode: backend name (cpu, cuda-12.4, cuda-13.1, vulkan, "
             "hip-radeon, sycl, metal, rocm-7.2, kleidiai)",
    )
    ap.add_argument(
        "--with-cudart", metavar="VERSION",
        help="release mode, Windows CUDA only: also fetch "
             "cudart-llama-bin-win-cuda-<VERSION>-x64.zip",
    )
    ap.add_argument("--force", action="store_true", help="release mode: re-download even if cached")
    args = ap.parse_args(argv)

    if args.from_local:
        if args.backend or args.with_cudart or args.force:
            print("warning: --backend/--with-cudart/--force ignored in local-build mode", file=sys.stderr)
        copy_from_local(args.from_local, args.platform)
        return 0

    if not args.backend:
        ap.error("--backend is required in release mode (or pass --from-local)")

    fetch_one(args.tag, args.platform, args.backend, args.force)

    if args.with_cudart:
        if not args.platform.startswith("win-"):
            raise SystemExit("--with-cudart only applies to Windows platforms")
        print(f"[{args.platform} / cudart-{args.with_cudart}] fetching CUDA runtime bundle")
        asset = cudart_asset(args.tag, args.with_cudart)
        archive = download(args.tag, asset, args.force)
        out_dir = RUNTIMES_DIR / args.platform / "native"
        libs = extract_libs(archive, asset.archive_kind, out_dir)
        for p in sorted(libs):
            print(f"  -> {p.relative_to(REPO_ROOT)}  ({p.stat().st_size / 1e6:.2f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
