#!/usr/bin/env python3
"""Parse llama.h with libclang and emit a structured JSON description.

The JSON captures what the binding mirrors need to track:

    {
      "source": "<path to llama.h>",
      "functions": [
        {"name": "...", "return_type": "...", "params": [{"name":"...","type":"..."}]},
        ...
      ],
      "structs": [
        {"name": "...", "fields": [{"name":"...","type":"..."}], "size": N|null},
        ...
      ],
      "enums": [
        {"name": "...", "values": [{"name":"...","value":N}]},
        ...
      ],
      "typedefs": [
        {"name": "...", "underlying": "..."},
        ...
      ]
    }

Only declarations whose source file IS the target header are kept. Transitively
included headers (ggml.h & friends) are not walked — we only need llama's API
surface. Types from those headers appear by name in function signatures and
field declarations, which is exactly what the diff pipeline wants.

Usage:
    tools/extract-api.py PATH_TO_LLAMA_H [-I INCLUDE_DIR ...] [-o OUTPUT_JSON]

Must be run via tools/.venv (see tools/setup-venv.sh). The orchestrator
check-for-updates.sh does this for you.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import clang.cindex as cindex
except ImportError:
    print(
        "error: clang.cindex not importable. Run tools/setup-venv.sh and invoke this "
        "script via tools/.venv/bin/python.",
        file=sys.stderr,
    )
    raise


# On Ubuntu, python libclang bundles libclang.so; nothing to configure.
# If a specific system libclang is preferred, uncomment:
# cindex.Config.set_library_file("/usr/lib/x86_64-linux-gnu/libclang-18.so.1")


def canonical_type(t: cindex.Type) -> str:
    """Best-effort canonical type string as it appears in C source."""
    s = t.spelling
    # libclang tends to use "struct llama_model *" vs "const struct llama_model *"
    # in specific ways — match what appears in the header for lossless diffing.
    return s


def extract_struct(cursor: cindex.Cursor) -> dict | None:
    """Extract struct info, or None if this cursor is just a forward decl."""
    fields: list[dict] = []
    has_definition = False

    for child in cursor.get_children():
        if child.kind == cindex.CursorKind.FIELD_DECL:
            has_definition = True
            fields.append({
                "name": child.spelling,
                "type": canonical_type(child.type),
                "offset_bits": cursor.type.get_offset(child.spelling),
            })
        elif child.kind in (cindex.CursorKind.STRUCT_DECL, cindex.CursorKind.UNION_DECL):
            # Anonymous union/struct inside — record the nested composition without
            # descending, since layout of anonymous unions is subtle and we only
            # mirror top-level structs in C#. Its fields still count as own fields
            # via libclang's flattening, so nothing more to do here.
            pass

    # Pure forward declarations (no fields) still get recorded — callers can see
    # them as "opaque type".
    try:
        size_bytes = cursor.type.get_size()
        if size_bytes < 0:
            size_bytes = None
    except cindex.TypeError:
        size_bytes = None

    return {
        "name": cursor.spelling or cursor.type.spelling,
        "is_opaque": not has_definition,
        "size": size_bytes,
        "fields": fields,
    }


def extract_enum(cursor: cindex.Cursor) -> dict:
    values: list[dict] = []
    for child in cursor.get_children():
        if child.kind == cindex.CursorKind.ENUM_CONSTANT_DECL:
            values.append({
                "name": child.spelling,
                "value": child.enum_value,
            })
    return {
        "name": cursor.spelling or cursor.type.spelling,
        "underlying": canonical_type(cursor.enum_type),
        "values": values,
    }


def extract_function(cursor: cindex.Cursor) -> dict:
    params = []
    for arg in cursor.get_arguments():
        params.append({
            "name": arg.spelling,
            "type": canonical_type(arg.type),
        })
    return {
        "name": cursor.spelling,
        "return_type": canonical_type(cursor.result_type),
        "params": params,
        "is_deprecated": any(
            c.kind == cindex.CursorKind.UNEXPOSED_ATTR and c.spelling == "deprecated"
            for c in cursor.get_children()
        ) or (cursor.availability != cindex.AvailabilityKind.AVAILABLE),
    }


def extract_typedef(cursor: cindex.Cursor) -> dict:
    return {
        "name": cursor.spelling,
        "underlying": canonical_type(cursor.underlying_typedef_type),
    }


def walk(tu: cindex.TranslationUnit, target_path: Path) -> dict:
    """Walk the TU, keeping only decls whose *defining* file is target_path.

    Uses canonical (symlink-resolved, absolute) path comparison — libclang
    hands back paths that may differ in form from what the caller passed.
    """
    target_canonical = target_path.resolve()

    functions: list[dict] = []
    structs: list[dict] = []
    enums: list[dict] = []
    typedefs: list[dict] = []

    # Track names already emitted: libclang visits forward decls + full decls
    # separately for the same struct. We keep the richer one.
    seen_struct_names: dict[str, int] = {}

    for cursor in tu.cursor.walk_preorder():
        loc = cursor.location
        if loc.file is None:
            continue
        try:
            if Path(loc.file.name).resolve() != target_canonical:
                continue
        except OSError:
            continue

        kind = cursor.kind

        if kind == cindex.CursorKind.FUNCTION_DECL:
            functions.append(extract_function(cursor))

        elif kind == cindex.CursorKind.STRUCT_DECL:
            data = extract_struct(cursor)
            if data is None:
                continue
            name = data["name"]
            # If we've seen a forward decl and now see the definition, replace.
            if name in seen_struct_names:
                prior_idx = seen_struct_names[name]
                prior = structs[prior_idx]
                if prior["is_opaque"] and not data["is_opaque"]:
                    structs[prior_idx] = data
            else:
                seen_struct_names[name] = len(structs)
                structs.append(data)

        elif kind == cindex.CursorKind.ENUM_DECL:
            enums.append(extract_enum(cursor))

        elif kind == cindex.CursorKind.TYPEDEF_DECL:
            # Anonymous struct typedefs (typedef struct {...} X) create BOTH a
            # STRUCT_DECL (anonymous) and a TYPEDEF_DECL. Keep the typedef name
            # in typedefs for completeness, but also rename the struct entry if
            # the struct name was empty.
            typedefs.append(extract_typedef(cursor))

    return {
        "functions": functions,
        "structs": structs,
        "enums": enums,
        "typedefs": typedefs,
    }


def _clang_resource_dir() -> str | None:
    """Locate clang's builtin header dir (stdbool.h, stdint.h, etc.).

    The `libclang` pip wheel bundles its own libclang.so but NOT the resource
    headers — those come from a system clang install. Without this, parsing
    any header that includes <stdbool.h> fails immediately.
    """
    import shutil, subprocess
    clang_bin = shutil.which("clang") or shutil.which("clang-18") or shutil.which("clang-17")
    if not clang_bin:
        return None
    try:
        out = subprocess.run([clang_bin, "-print-resource-dir"], capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def parse_header(header: Path, include_dirs: list[Path]) -> dict:
    args = ["-x", "c", "-std=c11"]

    resource_dir = _clang_resource_dir()
    if resource_dir:
        args.extend(["-resource-dir", resource_dir])

    for inc in include_dirs:
        args.extend(["-I", str(inc)])

    index = cindex.Index.create()
    tu = index.parse(
        str(header),
        args=args,
        options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
              | cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES,
    )

    # Surface hard parse errors loudly — a missing include dir will produce
    # hundreds of cascading errors that make the output meaningless.
    hard_errors = [d for d in tu.diagnostics if d.severity >= cindex.Diagnostic.Error]
    if hard_errors:
        msgs = "\n".join(
            f"  {d.location}: {d.spelling}" for d in hard_errors[:20]
        )
        print(
            "error: clang reported parse errors. Likely a missing -I for ggml/include.\n"
            f"first 20 of {len(hard_errors)}:\n{msgs}",
            file=sys.stderr,
        )
        sys.exit(2)

    api = walk(tu, header)
    api["source"] = str(header)
    return api


def default_include_dirs(header: Path) -> list[Path]:
    """Best-effort guess at the ggml include directory near llama.h.

    Supports both the tree layout of a llama.cpp checkout (`include/llama.h`
    + `ggml/include/`) and the pinned header living in a sibling dir.
    """
    candidates: list[Path] = [header.parent]
    parent = header.parent.parent
    for probe in (parent / "ggml" / "include", parent / "include" / "ggml"):
        if probe.is_dir():
            candidates.append(probe)
    return candidates


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("header", type=Path, help="path to llama.h (or llama.h.pinned)")
    ap.add_argument("-I", dest="includes", action="append", type=Path, default=[],
                    help="extra -I include dir (defaults guessed from header location)")
    ap.add_argument("-o", "--output", type=Path, help="write JSON here (default: stdout)")
    args = ap.parse_args(argv)

    if not args.header.is_file():
        print(f"error: header not found: {args.header}", file=sys.stderr)
        return 2

    includes = args.includes + default_include_dirs(args.header)
    # Dedupe preserving order.
    seen = set()
    deduped: list[Path] = []
    for p in includes:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    api = parse_header(args.header, deduped)

    out = json.dumps(api, indent=2, sort_keys=False)
    if args.output:
        args.output.write_text(out + "\n")
        print(f"wrote {len(api['functions'])} functions, {len(api['structs'])} structs, "
              f"{len(api['enums'])} enums, {len(api['typedefs'])} typedefs -> {args.output}",
              file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
