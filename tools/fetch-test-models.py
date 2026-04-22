#!/usr/bin/env python3
"""Fetch GGUF test models useful for exercising Tier-1 / Tier-2 code paths
that Qwen3.6 (the main golden model) doesn't hit.

Default target: /mnt/data/models/
Override with --dest.

Each model is identified by a short key you can pass to --only to download
just that model. Checksums are verified when provided. Downloads are
atomic (temp file + rename) and skip if the final file already exists.

Model set reflects what's needed for Tier-2 testing:
  bge-small-en-v1.5      100 MB   BGE embedding model — exercises pooled-
                                  embedding code paths Qwen3 doesn't hit.
  tinyllama-chat-v1.0    650 MB   Small LLaMA-family chat model — exercises
                                  BOS auto-prepend and plain-attention rope
                                  code paths that Qwen3's IMRope skips.
  qwen2.5-coder-3b       1.9 GB   Qwen coder model with first-class FIM
                                  tokens + grammar-friendly output.
  bge-reranker-v2-m3     370 MB   Cross-encoder reranker — classifier head
                                  with ClassifierOutputCount > 1.

Usage:
  python tools/fetch-test-models.py              # fetch the default set
  python tools/fetch-test-models.py --list       # show what's configured
  python tools/fetch-test-models.py --only bge-small-en-v1.5 tinyllama-chat-v1.0
  python tools/fetch-test-models.py --dest ~/models
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Model:
    key: str
    filename: str
    url: str
    approx_size_mb: int
    why: str
    # Optional SHA-256 for integrity verification. Left blank means we skip
    # the check — these are best-effort; populate as you verify them.
    sha256: str = ""


MODELS: list[Model] = [
    Model(
        key="bge-small-en-v1.5",
        filename="bge-small-en-v1.5-q8_0.gguf",
        url="https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf",
        approx_size_mb=37,
        why="BGE embedding model — real embedding code paths (pooling != None).",
    ),
    Model(
        key="tinyllama-chat-v1.0",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        url=(
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
            "/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        ),
        approx_size_mb=670,
        why="Small LLaMA-family model — BOS auto-prepend + standard RoPE path.",
    ),
    Model(
        key="qwen2.5-coder-3b",
        filename="Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf",
        url=(
            "https://huggingface.co/bartowski/Qwen2.5-Coder-3B-Instruct-GGUF"
            "/resolve/main/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"
        ),
        approx_size_mb=1960,
        why="Coder model with first-class FIM tokens + structured output.",
    ),
    Model(
        key="bge-reranker-v2-m3",
        filename="bge-reranker-v2-m3-Q4_K_M.gguf",
        url="https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q4_K_M.gguf",
        approx_size_mb=366,
        why="Cross-encoder reranker — classifier head (ClassifierOutputCount > 1).",
    ),
]

BY_KEY = {m.key: m for m in MODELS}
DEFAULT_SET = ("bge-small-en-v1.5", "tinyllama-chat-v1.0")
DEFAULT_DEST = Path("/mnt/data/models")


def download(url: str, dest: Path) -> None:
    """Atomic download with progress. Skips if dest already exists."""
    if dest.exists():
        print(f"  already present: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"  downloading: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:  # noqa: S310 (HF URLs are trusted)
            total = int(resp.headers.get("Content-Length", 0))
            bytes_read = 0
            while chunk := resp.read(1 << 20):  # 1 MB chunks
                f.write(chunk)
                bytes_read += len(chunk)
                if total:
                    pct = 100 * bytes_read / total
                    print(
                        f"\r  {bytes_read / 1e6:7.1f} / {total / 1e6:7.1f} MB ({pct:5.1f}%)",
                        end="", flush=True,
                    )
        print()
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.rename(dest)


def verify_sha256(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual.lower() != expected.lower():
        print(f"  sha256 MISMATCH: got {actual}, expected {expected}")
        return False
    print(f"  sha256 ok ({actual[:16]}...)")
    return True


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"destination directory (default: {DEFAULT_DEST})")
    ap.add_argument("--only", nargs="+", metavar="KEY",
                    help=f"download only these keys (default: {' '.join(DEFAULT_SET)})")
    ap.add_argument("--all", action="store_true",
                    help=f"download all {len(MODELS)} configured models")
    ap.add_argument("--list", action="store_true",
                    help="list available models and exit")
    args = ap.parse_args(argv)

    if args.list:
        total_mb = sum(m.approx_size_mb for m in MODELS)
        print(f"Configured models ({len(MODELS)}, ~{total_mb / 1000:.1f} GB total):\n")
        for m in MODELS:
            print(f"  {m.key:25s} ~{m.approx_size_mb:5d} MB   {m.why}")
            print(f"  {'':25s} {m.filename}")
        return 0

    if args.all:
        keys = [m.key for m in MODELS]
    elif args.only:
        keys = args.only
    else:
        keys = list(DEFAULT_SET)

    unknown = [k for k in keys if k not in BY_KEY]
    if unknown:
        print(f"error: unknown key(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"valid keys: {', '.join(BY_KEY)}", file=sys.stderr)
        return 2

    args.dest.mkdir(parents=True, exist_ok=True)
    total_to_fetch = sum(BY_KEY[k].approx_size_mb for k in keys)
    have, free, _ = shutil.disk_usage(args.dest)
    free_mb = free // (1024 * 1024)
    print(f"Destination: {args.dest} (free: {free_mb} MB)")
    print(f"Fetching {len(keys)} model(s), ~{total_to_fetch} MB total\n")

    if total_to_fetch * 1.1 > free_mb:
        print(
            f"error: not enough free space (need ~{int(total_to_fetch * 1.1)} MB, have {free_mb} MB)",
            file=sys.stderr,
        )
        return 2

    for key in keys:
        m = BY_KEY[key]
        print(f"[{key}] {m.why}")
        dest = args.dest / m.filename
        try:
            download(m.url, dest)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            # Continue with the rest so one broken URL doesn't block everything.
            continue

        if m.sha256:
            if not verify_sha256(dest, m.sha256):
                print(f"  removing corrupted file: {dest}")
                dest.unlink(missing_ok=True)

        print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
