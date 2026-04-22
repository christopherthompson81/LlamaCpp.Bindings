#!/usr/bin/env python3
"""Diff two API JSON snapshots (from extract-api.py) and emit a markdown report.

The report is optimised for human review + Claude Code action: each section
lists changes as a bullet list with enough surrounding context to decide what
the binding update should look like. Struct layout changes are the
highest-risk category and get extra detail (old vs new size, field-by-field
deltas).

Usage:
    tools/diff-api.py OLD.json NEW.json [-o REPORT.md]

Exits 0 always (an empty report is valid — it means "no ABI-affecting
changes"). The orchestrator checks report length to decide whether manual
action is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def function_signature(fn: dict) -> str:
    params = ", ".join(f"{p['type']} {p['name']}".strip() for p in fn["params"])
    return f"{fn['return_type']} {fn['name']}({params})"


def index_by_name(items: list[dict]) -> dict[str, dict]:
    return {item["name"]: item for item in items if item.get("name")}


def diff_functions(old: dict, new: dict) -> list[str]:
    old_fns = index_by_name(old.get("functions", []))
    new_fns = index_by_name(new.get("functions", []))

    added   = sorted(set(new_fns) - set(old_fns))
    removed = sorted(set(old_fns) - set(new_fns))
    kept    = sorted(set(old_fns) & set(new_fns))

    lines: list[str] = []

    if removed:
        lines.append("### Functions removed")
        lines.append("")
        for name in removed:
            lines.append(f"- **`{name}`** — `{function_signature(old_fns[name])}`")
        lines.append("")

    if added:
        lines.append("### Functions added")
        lines.append("")
        for name in added:
            fn = new_fns[name]
            dep = " *(deprecated)*" if fn.get("is_deprecated") else ""
            lines.append(f"- **`{name}`**{dep} — `{function_signature(fn)}`")
        lines.append("")

    signature_changes: list[tuple[str, dict, dict]] = []
    deprecation_changes: list[tuple[str, bool, bool]] = []
    for name in kept:
        o, n = old_fns[name], new_fns[name]
        if function_signature(o) != function_signature(n):
            signature_changes.append((name, o, n))
        if bool(o.get("is_deprecated")) != bool(n.get("is_deprecated")):
            deprecation_changes.append((name, o.get("is_deprecated", False), n.get("is_deprecated", False)))

    if signature_changes:
        lines.append("### Function signatures changed")
        lines.append("")
        for name, o, n in signature_changes:
            lines.append(f"- **`{name}`**")
            lines.append(f"    - old: `{function_signature(o)}`")
            lines.append(f"    - new: `{function_signature(n)}`")
        lines.append("")

    if deprecation_changes:
        lines.append("### Function deprecation changes")
        lines.append("")
        for name, was, is_ in deprecation_changes:
            arrow = "→ deprecated" if is_ and not was else "→ un-deprecated"
            lines.append(f"- **`{name}`** {arrow}")
        lines.append("")

    return lines


def diff_structs(old: dict, new: dict) -> list[str]:
    old_structs = index_by_name(old.get("structs", []))
    new_structs = index_by_name(new.get("structs", []))

    added   = sorted(set(new_structs) - set(old_structs))
    removed = sorted(set(old_structs) - set(new_structs))
    kept    = sorted(set(old_structs) & set(new_structs))

    lines: list[str] = []

    if removed:
        lines.append("### Structs removed")
        lines.append("")
        for name in removed:
            lines.append(f"- **`{name}`**")
        lines.append("")

    if added:
        lines.append("### Structs added")
        lines.append("")
        for name in added:
            s = new_structs[name]
            note = " *(opaque)*" if s.get("is_opaque") else ""
            size = s.get("size")
            lines.append(f"- **`{name}`**{note} — size {size} bytes, {len(s['fields'])} fields")
        lines.append("")

    layout_changes: list[str] = []
    for name in kept:
        o, n = old_structs[name], new_structs[name]

        # Use both (field_name, field_type) and field order to detect drift.
        old_fields = [(f["name"], f["type"]) for f in o.get("fields", [])]
        new_fields = [(f["name"], f["type"]) for f in n.get("fields", [])]

        opaque_changed = bool(o.get("is_opaque")) != bool(n.get("is_opaque"))
        size_changed   = o.get("size") != n.get("size")
        layout_changed = old_fields != new_fields

        if not (opaque_changed or size_changed or layout_changed):
            continue

        layout_changes.append(f"- **`{name}`**")
        if size_changed:
            layout_changes.append(f"    - size: `{o.get('size')}` → `{n.get('size')}`")
        if opaque_changed:
            layout_changes.append(f"    - opaque: `{o.get('is_opaque')}` → `{n.get('is_opaque')}`")
        if layout_changed:
            # Show field-level diffs.
            old_names = [name_ for name_, _ in old_fields]
            new_names = [name_ for name_, _ in new_fields]
            added_fields   = [f for f in new_fields if f[0] not in old_names]
            removed_fields = [f for f in old_fields if f[0] not in new_names]
            type_changes = []
            new_by_name = {name_: typ for name_, typ in new_fields}
            for name_, typ in old_fields:
                if name_ in new_by_name and new_by_name[name_] != typ:
                    type_changes.append((name_, typ, new_by_name[name_]))

            # Detect pure reorders (same set of (name, type) but different order).
            if set(old_fields) == set(new_fields) and old_fields != new_fields:
                layout_changes.append("    - **field order changed** (CRITICAL — re-run struct-size probe)")
                layout_changes.append(f"        - old order: `{', '.join(old_names)}`")
                layout_changes.append(f"        - new order: `{', '.join(new_names)}`")
            else:
                for name_, typ in added_fields:
                    layout_changes.append(f"    - + `{typ} {name_}`")
                for name_, typ in removed_fields:
                    layout_changes.append(f"    - - `{typ} {name_}`")
                for name_, old_t, new_t in type_changes:
                    layout_changes.append(f"    - ~ `{name_}`: `{old_t}` → `{new_t}`")

    if layout_changes:
        lines.append("### Struct layout changes")
        lines.append("")
        lines.append("**These are the highest-risk category — one wrong byte offset corrupts memory.**")
        lines.append("After reconciling, re-run `tools/dump-struct-sizes.sh` and update")
        lines.append("`tools/struct-sizes.json` + `NativeStructs.cs` mirrors + `ExpectedSize` constants.")
        lines.append("")
        lines.extend(layout_changes)
        lines.append("")

    return lines


def diff_enums(old: dict, new: dict) -> list[str]:
    old_enums = index_by_name(old.get("enums", []))
    new_enums = index_by_name(new.get("enums", []))

    added   = sorted(set(new_enums) - set(old_enums))
    removed = sorted(set(old_enums) - set(new_enums))
    kept    = sorted(set(old_enums) & set(new_enums))

    lines: list[str] = []

    if removed:
        lines.append("### Enums removed")
        lines.append("")
        for name in removed:
            lines.append(f"- **`{name}`**")
        lines.append("")

    if added:
        lines.append("### Enums added")
        lines.append("")
        for name in added:
            e = new_enums[name]
            lines.append(f"- **`{name}`** ({len(e['values'])} values)")
        lines.append("")

    value_changes: list[str] = []
    for name in kept:
        o, n = old_enums[name], new_enums[name]
        old_v = {v["name"]: v["value"] for v in o.get("values", [])}
        new_v = {v["name"]: v["value"] for v in n.get("values", [])}
        if old_v == new_v:
            continue
        value_changes.append(f"- **`{name}`**")
        for vname in sorted(set(old_v) - set(new_v)):
            value_changes.append(f"    - removed: `{vname} = {old_v[vname]}`")
        for vname in sorted(set(new_v) - set(old_v)):
            value_changes.append(f"    - added: `{vname} = {new_v[vname]}`")
        for vname in sorted(set(old_v) & set(new_v)):
            if old_v[vname] != new_v[vname]:
                value_changes.append(f"    - changed: `{vname}` `{old_v[vname]}` → `{new_v[vname]}`")

    if value_changes:
        lines.append("### Enum value changes")
        lines.append("")
        lines.extend(value_changes)
        lines.append("")

    return lines


def diff_typedefs(old: dict, new: dict) -> list[str]:
    old_t = index_by_name(old.get("typedefs", []))
    new_t = index_by_name(new.get("typedefs", []))
    added   = sorted(set(new_t) - set(old_t))
    removed = sorted(set(old_t) - set(new_t))
    changed = [
        (name, old_t[name]["underlying"], new_t[name]["underlying"])
        for name in sorted(set(old_t) & set(new_t))
        if old_t[name]["underlying"] != new_t[name]["underlying"]
    ]
    lines: list[str] = []
    if added or removed or changed:
        lines.append("### Typedef changes")
        lines.append("")
        for name in added:
            lines.append(f"- + `typedef {new_t[name]['underlying']} {name}`")
        for name in removed:
            lines.append(f"- - `typedef {old_t[name]['underlying']} {name}`")
        for name, old_u, new_u in changed:
            lines.append(f"- ~ `{name}`: `{old_u}` → `{new_u}`")
        lines.append("")
    return lines


def build_report(old: dict, new: dict) -> str:
    lines: list[str] = []
    lines.append("# llama.cpp API change report")
    lines.append("")
    lines.append(f"- old source: `{old.get('source', '?')}`")
    lines.append(f"- new source: `{new.get('source', '?')}`")
    lines.append("")

    sections = [
        diff_functions(old, new),
        diff_structs(old, new),
        diff_enums(old, new),
        diff_typedefs(old, new),
    ]

    any_changes = any(s for s in sections)
    if not any_changes:
        lines.append("**No API changes.** Safe to bump the pinned version without binding work.")
        lines.append("")
    else:
        for s in sections:
            lines.extend(s)

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_json", type=Path)
    ap.add_argument("new_json", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="write markdown here (default: stdout)")
    args = ap.parse_args(argv)

    old = load(args.old_json)
    new = load(args.new_json)
    report = build_report(old, new)

    if args.output:
        args.output.write_text(report)
        print(f"wrote change report -> {args.output}", file=sys.stderr)
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
