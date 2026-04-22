#!/usr/bin/env python3
"""Cross-reference an API diff against the C# binding source tree.

Given the OLD and NEW API JSON snapshots (same inputs as diff-api.py), this
script finds every C# file that references a symbol that actually changed
between the two snapshots. The output is a markdown section appended to (or
printed alongside) the main diff report, so a reviewer sees both:

    - WHAT changed in llama.h
    - WHERE in our C# code each change lands

Usage:
    tools/xref-bindings.py OLD.json NEW.json [-o XREF.md] [--src src/]

The search is plain text: each changed symbol name is grepped against every
.cs file under `--src`. This works because our convention (CLAUDE.md rule) is
to mirror C names 1:1 in C# — `llama_decode` stays `llama_decode`, never
`LlamaDecode`. If that convention slips, this tool under-reports.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def index_by_name(items: list[dict]) -> dict[str, dict]:
    return {item["name"]: item for item in items if item.get("name")}


def affected_symbols(old: dict, new: dict) -> list[tuple[str, str, str]]:
    """Return (category, symbol_name, reason) for every symbol that changed.

    category is one of: function, struct, enum, enumerator, typedef.
    reason is a short human-readable description.
    """
    out: list[tuple[str, str, str]] = []

    o_fn, n_fn = index_by_name(old.get("functions", [])), index_by_name(new.get("functions", []))
    for name in sorted(set(o_fn) ^ set(n_fn)):
        reason = "removed" if name in o_fn else "added"
        out.append(("function", name, reason))
    for name in sorted(set(o_fn) & set(n_fn)):
        o, n = o_fn[name], n_fn[name]
        if o.get("return_type") != n.get("return_type") or o.get("params") != n.get("params"):
            out.append(("function", name, "signature changed"))
        if bool(o.get("is_deprecated")) != bool(n.get("is_deprecated")):
            out.append(("function", name, "deprecation changed"))

    o_st, n_st = index_by_name(old.get("structs", [])), index_by_name(new.get("structs", []))
    for name in sorted(set(o_st) ^ set(n_st)):
        reason = "removed" if name in o_st else "added"
        out.append(("struct", name, reason))
    for name in sorted(set(o_st) & set(n_st)):
        o, n = o_st[name], n_st[name]
        if (o.get("size") != n.get("size")
                or [(f["name"], f["type"]) for f in o.get("fields", [])]
                != [(f["name"], f["type"]) for f in n.get("fields", [])]):
            out.append(("struct", name, "layout changed"))

    o_en, n_en = index_by_name(old.get("enums", [])), index_by_name(new.get("enums", []))
    for name in sorted(set(o_en) ^ set(n_en)):
        reason = "removed" if name in o_en else "added"
        out.append(("enum", name, reason))
    for name in sorted(set(o_en) & set(n_en)):
        old_v = {v["name"]: v["value"] for v in o_en[name].get("values", [])}
        new_v = {v["name"]: v["value"] for v in n_en[name].get("values", [])}
        if old_v != new_v:
            out.append(("enum", name, "value set changed"))
            # Also surface individual enumerators that changed, so xref can
            # find the specific LLAMA_* name in C#.
            for vname in sorted(set(old_v) ^ set(new_v)):
                out.append(("enumerator", vname, "added" if vname in new_v else "removed"))
            for vname in sorted(set(old_v) & set(new_v)):
                if old_v[vname] != new_v[vname]:
                    out.append(("enumerator", vname, "value changed"))

    o_td, n_td = index_by_name(old.get("typedefs", [])), index_by_name(new.get("typedefs", []))
    for name in sorted(set(o_td) ^ set(n_td)):
        reason = "removed" if name in o_td else "added"
        out.append(("typedef", name, reason))
    for name in sorted(set(o_td) & set(n_td)):
        if o_td[name].get("underlying") != n_td[name].get("underlying"):
            out.append(("typedef", name, "underlying changed"))

    return out


def find_references(src_root: Path, symbol: str) -> list[tuple[Path, int, str]]:
    """Return (file, line_number, line_text) for every line in .cs files under
    src_root that contains `symbol` as a word-boundary match.
    """
    pattern = re.compile(r"\b" + re.escape(symbol) + r"\b")
    matches: list[tuple[Path, int, str]] = []
    for cs_file in src_root.rglob("*.cs"):
        try:
            with cs_file.open("r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, start=1):
                    if pattern.search(line):
                        matches.append((cs_file, i, line.rstrip()))
        except OSError:
            continue
    return matches


def build_report(old: dict, new: dict, src_root: Path) -> str:
    affected = affected_symbols(old, new)

    lines: list[str] = []
    lines.append("# Binding xref for API changes")
    lines.append("")
    lines.append(f"- src tree: `{src_root}`")
    lines.append(f"- affected symbols: {len(affected)}")
    lines.append("")

    if not affected:
        lines.append("No affected symbols detected.")
        lines.append("")
        return "\n".join(lines)

    unreferenced: list[tuple[str, str, str]] = []

    for category, symbol, reason in affected:
        refs = find_references(src_root, symbol)
        if not refs:
            unreferenced.append((category, symbol, reason))
            continue

        lines.append(f"## `{symbol}` ({category}, {reason})")
        lines.append("")
        # Group by file for readability.
        by_file: dict[Path, list[tuple[int, str]]] = {}
        for f, n, t in refs:
            by_file.setdefault(f, []).append((n, t))
        for f in sorted(by_file.keys()):
            rel = f.relative_to(src_root.parent) if src_root.parent in f.parents else f
            lines.append(f"- `{rel}`")
            for n, t in by_file[f]:
                lines.append(f"    - L{n}: `{t.strip()}`")
        lines.append("")

    if unreferenced:
        lines.append("## Symbols not referenced in C# (informational)")
        lines.append("")
        lines.append(
            "These symbols changed in llama.h but don't appear in the binding. "
            "Either they're part of the unbound surface (expected) or our "
            "convention broke and the name-mirror invariant doesn't hold here."
        )
        lines.append("")
        for category, symbol, reason in unreferenced:
            lines.append(f"- `{symbol}` ({category}, {reason})")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_json", type=Path)
    ap.add_argument("new_json", type=Path)
    ap.add_argument("--src", type=Path, default=Path("src"),
                    help="root of the C# source tree to search (default: src/)")
    ap.add_argument("-o", "--output", type=Path, help="write markdown here (default: stdout)")
    args = ap.parse_args(argv)

    if not args.src.is_dir():
        print(f"error: src tree not found: {args.src}", file=sys.stderr)
        return 2

    old = load(args.old_json)
    new = load(args.new_json)
    report = build_report(old, new, args.src)

    if args.output:
        args.output.write_text(report)
        print(f"wrote xref report -> {args.output}", file=sys.stderr)
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
