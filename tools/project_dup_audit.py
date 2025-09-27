#!/usr/bin/env python3
"""
project_dup_audit.py

One-shot redundancy/dup scan without external shell commands.

Finds:
  - Duplicate function names across files
  - Duplicate class names across files
  - Magic-number hotspots (1729, 2027, 1e-12/1e-9/1e-6)
  - EXACT duplicate function bodies (AST-equal)
  - NEAR duplicate function bodies (same name, difflib similarity >= threshold)

Usage:
  python project_dup_audit.py /path/to/project --near-threshold 0.90 --out dup_report.json
"""

import os, sys, re, ast, json, hashlib, difflib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

EXCLUDE_DIRS = {".git", "__pycache__", "venv", ".venv", "build", "dist", ".mypy_cache", ".pytest_cache"}
MAGIC_PATTERNS = [
    r"\b1729\b",
    r"\b2027\b",
    r"\b1e-12\b",
    r"\b1e-9\b",
    r"\b1e-6\b",
]

@dataclass
class FuncInfo:
    name: str
    path: str
    lineno: int
    text: str
    ast_hash: str

@dataclass
class ClassInfo:
    name: str
    path: str
    lineno: int

def py_files(root: str) -> List[str]:
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in EXCLUDE_DIRS]
        for fn in fns:
            if fn.endswith(".py"):
                out.append(os.path.join(dp, fn))
    return out

def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def norm_ast_dump(node: ast.AST) -> str:
    for n in ast.walk(node):
        for attr in ("lineno","col_offset","end_lineno","end_col_offset"):
            if hasattr(n, attr):
                setattr(n, attr, None)
    return ast.dump(node, include_attributes=False)

def slice_text(src: str, node: ast.AST) -> str:
    lines = src.splitlines(True)
    s = getattr(node, "lineno", None)
    e = getattr(node, "end_lineno", None)
    if s and e and 1 <= s <= len(lines) and 1 <= e <= len(lines) and e >= s:
        return "".join(lines[s-1:e])
    return ""

def collect_symbols(paths: List[str]) -> Tuple[List[FuncInfo], List[ClassInfo]]:
    funcs: List[FuncInfo] = []
    classes: List[ClassInfo] = []
    for p in paths:
        src = read_text(p)
        if not src:
            continue
        try:
            tree = ast.parse(src, filename=p)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                dump = norm_ast_dump(node)
                h = hashlib.sha256(dump.encode()).hexdigest()
                txt = slice_text(src, node)
                funcs.append(FuncInfo(node.name, p, getattr(node,"lineno",0), txt, h))
            elif isinstance(node, ast.ClassDef):
                classes.append(ClassInfo(node.name, p, getattr(node,"lineno",0)))
    return funcs, classes

def group_by_name(items: List[Tuple[str, str, int]]) -> Dict[str, List[Tuple[str,int]]]:
    grouped: Dict[str, List[Tuple[str,int]]] = {}
    for name, path, ln in items:
        grouped.setdefault(name, []).append((path, ln))
    return grouped

def find_duplicate_names(funcs: List[FuncInfo], classes: List[ClassInfo]):
    func_group = group_by_name([(f.name, f.path, f.lineno) for f in funcs])
    class_group = group_by_name([(c.name, c.path, c.lineno) for c in classes])

    dup_funcs = {k:v for k,v in func_group.items() if len(v) > 1}
    dup_classes = {k:v for k,v in class_group.items() if len(v) > 1}
    return dup_funcs, dup_classes

def scan_magic_numbers(paths: List[str]) -> Dict[str, List[Tuple[int, str]]]:
    patt = re.compile("|".join(MAGIC_PATTERNS))
    hits: Dict[str, List[Tuple[int, str]]] = {}
    for p in paths:
        src = read_text(p)
        if not src:
            continue
        out = []
        for i, line in enumerate(src.splitlines(), start=1):
            if patt.search(line):
                out.append((i, line.strip()))
        if out:
            hits[p] = out
    return hits

def exact_duplicate_funcs(funcs: List[FuncInfo]) -> List[List[FuncInfo]]:
    buckets: Dict[str, List[FuncInfo]] = {}
    for f in funcs:
        buckets.setdefault(f.ast_hash, []).append(f)
    groups = [lst for lst in buckets.values() if len(lst) > 1]
    # filter out trivial duplicates within same file/line (rare)
    cleaned = []
    for g in groups:
        keyset = {(x.path, x.lineno) for x in g}
        if len(keyset) > 1:
            cleaned.append(g)
    return cleaned

def near_duplicate_funcs(funcs: List[FuncInfo], similarity: float = 0.90) -> List[Tuple[str, FuncInfo, FuncInfo, float]]:
    # Compare only functions with the same name across different files
    by_name: Dict[str, List[FuncInfo]] = {}
    for f in funcs:
        by_name.setdefault(f.name, []).append(f)
    out: List[Tuple[str, FuncInfo, FuncInfo, float]] = []
    for name, lst in by_name.items():
        if len(lst) < 2:
            continue
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                a, b = lst[i], lst[j]
                if not a.text or not b.text:
                    continue
                if a.path == b.path and a.lineno == b.lineno:
                    continue
                ratio = difflib.SequenceMatcher(None, a.text, b.text).ratio()
                if ratio >= similarity:
                    out.append((name, a, b, ratio))
    # de-dupe symmetric pairs (path order)
    uniq = []
    seen = set()
    for name, a, b, r in out:
        key = tuple(sorted([(a.path, a.lineno), (b.path, b.lineno)]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, a, b, r))
    return uniq

def print_header(title: str):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Project duplicate/clone audit")
    ap.add_argument("root", nargs="?", default=".", help="Project root")
    ap.add_argument("--near-threshold", type=float, default=0.90, help="Similarity threshold for near-duplicates (0-1)")
    ap.add_argument("--out", type=str, default="", help="Optional path to write JSON report")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    paths = py_files(root)

    funcs, classes = collect_symbols(paths)
    dup_funcs, dup_classes = find_duplicate_names(funcs, classes)
    magic_hits = scan_magic_numbers(paths)
    exact_groups = exact_duplicate_funcs(funcs)
    near_dups = near_duplicate_funcs(funcs, similarity=args.near_threshold)

    # --- Print report ---
    print_header(f"Duplicate Name Scan — functions (>{1})")
    if dup_funcs:
        for name, locs in sorted(dup_funcs.items()):
            print(f"\n{name}")
            for p, ln in locs:
                print(f"  - {p}:{ln}")
    else:
        print("None")

    print_header(f"Duplicate Name Scan — classes (>{1})")
    if dup_classes:
        for name, locs in sorted(dup_classes.items()):
            print(f"\n{name}")
            for p, ln in locs:
                print(f"  - {p}:{ln}")
    else:
        print("None")

    print_header("Magic Number Hotspots")
    if magic_hits:
        for p, rows in magic_hits.items():
            print(f"\n{p}")
            for ln, txt in rows[:50]:  # cap print per file
                print(f"  {ln:5d}: {txt}")
            if len(rows) > 50:
                print(f"  ... {len(rows)-50} more")
    else:
        print("None")

    print_header("EXACT Duplicate Function Bodies (AST-equal)")
    if exact_groups:
        for g in exact_groups:
            h = g[0].ast_hash[:12]
            print(f"\nGroup {h} ({len(g)} funcs):")
            for fi in g:
                print(f"  {fi.name:30s} {fi.path}:{fi.lineno}")
    else:
        print("None")

    print_header(f"NEAR Duplicate Function Bodies (same name, similarity >= {args.near_threshold:.2f})")
    if near_dups:
        for name, a, b, r in sorted(near_dups, key=lambda t: (-t[3], t[0])):
            print(f"\n{name}  sim={r:.2f}")
            print(f"  - {a.path}:{a.lineno}")
            print(f"  - {b.path}:{b.lineno}")
    else:
        print("None")

    # --- Optional JSON out ---
    if args.out:
        output = {
            "root": root,
            "duplicate_function_names": {k: v for k, v in dup_funcs.items()},
            "duplicate_class_names": {k: v for k, v in dup_classes.items()},
            "magic_numbers": {p: rows for p, rows in magic_hits.items()},
            "exact_duplicate_groups": [
                [
                    {"name": fi.name, "path": fi.path, "lineno": fi.lineno, "ast_hash": fi.ast_hash}
                    for fi in g
                ] for g in exact_groups
            ],
            "near_duplicate_pairs": [
                {
                    "name": name,
                    "a": {"path": a.path, "lineno": a.lineno},
                    "b": {"path": b.path, "lineno": b.lineno},
                    "similarity": r,
                } for (name, a, b, r) in near_dups
            ],
            "near_threshold": args.near_threshold,
        }
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
            print(f"\n[OK] JSON report written to {args.out}")
        except Exception as e:
            print(f"\n[WARN] Could not write JSON report: {e}")

if __name__ == "__main__":
    main()

