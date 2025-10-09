#!/usr/bin/env python3
"""
I analyze Python sources with LibCST to flag patterns that need human review under
strict style rules: try/except blocks, raise statements, nested functions (with
recursion detection), and expression-form if/else (ternary) constructs. I collect
findings with positions, summarize per file, and can write a JSON report.

Key classes/functions: ReviewCollector — LibCST visitor that records constructs and
positions; analyze_file — run a full metadata-aware pass; gather_python_files — walk the
tree with filters; main — command-line entry.

Inputs: a project path and optional output file. Outputs: console summary plus optional
JSON with per-file lists of findings and totals.

Dependencies: libcst and its metadata providers; Python stdlib. Invariants: I treat
nested functions as safe only if self-recursive; I mark broad or bare exception handlers
as higher risk. Performance: metadata-aware traversal is fast enough for whole-repo
audits and integrates into pre-commit or CI.
"""


from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import libcst as cst
from libcst import matchers as m
from libcst.metadata import (
    PositionProvider,
    ScopeProvider,
    ParentNodeProvider,
    QualifiedNameProvider,
    QualifiedName,
    Scope,
    GlobalScope,
    FunctionScope,
    ClassScope,
    Assignment,
)

DEFAULT_PROJECT = "/Users/cenkalkan/Downloads/HUNL"

def is_target_file(path: str) -> bool:
    name = os.path.basename(path).lower()
    return name.endswith(".py") and ("test" not in name)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def gather_python_files(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "build", "dist", ".venv", "venv")]
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if is_target_file(fp):
                files.append(fp)
    return files

# ----------------- Analysis visitor -----------------

class ReviewCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        PositionProvider,
        ScopeProvider,
        ParentNodeProvider,
        QualifiedNameProvider,
    )

    def __init__(self) -> None:
        # results
        self.try_blocks: List[Dict] = []
        self.raises: List[Dict] = []
        self.nested_functions: List[Dict] = []
        self.ternary_ifexprs: List[Dict] = []
        # internal stacks
        self.func_stack: List[cst.FunctionDef] = []

    # ---- helpers ----

    def pos(self, node: cst.CSTNode) -> Dict[str, int]:
        p = self.get_metadata(PositionProvider, node)
        return {"line": p.start.line, "col": p.start.column}

    def code_snippet(self, module: cst.Module, node: cst.CSTNode, max_len: int = 160) -> str:
        # Cheap slice via the original code positions
        p = self.get_metadata(PositionProvider, node)
        text = module.code.splitlines()
        line = text[p.start.line - 1][p.start.column : ] if p.start.line == p.end.line else text[p.start.line - 1][p.start.column :]
        snippet = line.strip()
        return snippet if len(snippet) <= max_len else (snippet[: max_len - 3] + "...")

    def is_nested_function(self, node: cst.FunctionDef) -> bool:
        # Nested if parent chain includes another FunctionDef
        parent = self.get_metadata(ParentNodeProvider, node, None)
        while parent is not None:
            if isinstance(parent, cst.FunctionDef):
                return True
            parent = self.get_metadata(ParentNodeProvider, parent, None)
        return False

    def function_qualname(self, node: cst.FunctionDef) -> str:
        qnames = self.get_metadata(QualifiedNameProvider, node, set())
        # Choose the first qualified name if present, else fallback to .name.value
        for q in qnames:
            if q.source == QualifiedNameSource.LOCAL or True:
                return q.name
        # fallback
        return node.name.value

    # ---- try/except ----

    def visit_Try(self, node: cst.Try) -> Optional[bool]:
        # Collect exception names caught (if any)
        kinds: List[str] = []
        broad = False
        for h in node.handlers:
            if h.type is None:
                kinds.append("BareExcept")
                broad = True
            else:
                if m.matches(h.type, m.Name()):
                    kinds.append(m.findall(h.type, m.Name())[0].value)  # simple name
                    if kinds[-1] in {"Exception", "BaseException"}:
                        broad = True
                else:
                    kinds.append("ExprExcept")
        info = {
            "loc": self.pos(node),
            "caught": kinds,
            "has_else": node.orelse is not None,
            "has_finally": node.finalbody is not None,
            "broad": broad,
        }
        self.try_blocks.append(info)
        return True

    # ---- raise ----

    def visit_Raise(self, node: cst.Raise) -> Optional[bool]:
        # We flag ALL raises; the refactorer auto-fixes only a very narrow “final statement with simple string”.
        exc_kind = "unknown"
        if isinstance(node.exc, cst.Call):
            # try to render callee form
            if isinstance(node.exc.func, cst.Name):
                exc_kind = node.exc.func.value
            elif isinstance(node.exc.func, cst.Attribute):
                # dotted
                parts = []
                cur = node.exc.func
                while isinstance(cur, cst.Attribute):
                    parts.append(cur.attr.value)
                    if isinstance(cur.value, cst.Name):
                        parts.append(cur.value.value)
                        break
                    elif isinstance(cur.value, cst.Attribute):
                        cur = cur.value
                    else:
                        break
                exc_kind = ".".join(reversed(parts)) or "attr"
        info = {
            "loc": self.pos(node),
            "exception": exc_kind,
        }
        self.raises.append(info)
        return True

    # ---- IfExp (ternary) ----

    def visit_IfExp(self, node: cst.IfExp) -> Optional[bool]:
        self.ternary_ifexprs.append({"loc": self.pos(node)})
        return True

    # ---- nested functions & recursion / free vars ----

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        self.func_stack.append(node)
        # If this is nested, gather info
        if self.is_nested_function(node):
            # Determine recursion: does this function body reference its own name?
            name = node.name.value
            self_ref = False

            class SelfRefFinder(cst.CSTVisitor):
                def __init__(self) -> None:
                    self.found = False
                def visit_Name(self, n: cst.Name) -> Optional[bool]:
                    if n.value == name:
                        self.found = True
                    return True

            finder = SelfRefFinder()
            node.visit(finder)
            self_ref = finder.found

            # free variables (captured from enclosing scopes)
            scope: Scope = self.get_metadata(ScopeProvider, node)
            free_vars: Set[str] = set()
            if isinstance(scope, FunctionScope):
                for name_str, access in scope.accesses.items():
                    # if name is referenced but not assigned in this function, and it resolves to an outer scope
                    assigned_here = name_str in scope.assignments
                    if not assigned_here:
                        # look up resolution chain
                        assignments: Set[Assignment] = set()
                        for a in scope.lookup(name_str):
                            assignments.add(a)
                        # If any assignment is from an outer (non-global) scope, treat as free var
                        for a in assignments:
                            if not isinstance(a.scope, (GlobalScope, FunctionScope)) or (isinstance(a.scope, FunctionScope) and a.scope is not scope):
                                free_vars.add(name_str)
                                break

            info = {
                "loc": self.pos(node),
                "name": name,
                "is_recursive": bool(self_ref),
                "free_vars": sorted(free_vars),
            }
            self.nested_functions.append(info)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self.func_stack.pop()

# ----------------- Run one file -----------------

def analyze_file(path: str) -> Dict[str, List[Dict]]:
    src = read_text(path)
    module = cst.parse_module(src)
    wrapper = cst.MetadataWrapper(module)
    collector = ReviewCollector()
    wrapper.visit(collector)

    # Prepare result
    return {
        "try_blocks": collector.try_blocks,
        "raises": collector.raises,
        "nested_functions": collector.nested_functions,
        "ternary_ifexprs": collector.ternary_ifexprs,
    }

# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(description="Identify code that needs HUMAN REVIEW before full refactor.")
    ap.add_argument("--project", default=DEFAULT_PROJECT, help="Path to the project root.")
    ap.add_argument("--out", default="human_review_report.json", help="Path to write the JSON report.")
    args = ap.parse_args()

    if not os.path.isdir(args.project):
        print(f"Project not found: {args.project}", file=sys.stderr)
        sys.exit(2)

    files = gather_python_files(args.project)

    overall: Dict[str, Dict[str, List[Dict]]] = {}
    totals = {"try_blocks": 0, "raises": 0, "nested_functions": 0, "ternary_ifexprs": 0}

    for fp in files:
        try:
            result = analyze_file(fp)
        except Exception as e:
            print(f"<<Skipping {fp}; parse error: {e}>>", file=sys.stderr)
            continue
        # Only record files that have any hits
        if any(result[k] for k in result):
            overall[fp] = result
            for k, v in result.items():
                totals[k] += len(v)

    # write JSON report
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    # console summary
    print("\n=== HUMAN REVIEW REPORT SUMMARY ===")
    print(f"Project: {args.project}")
    print(f"Files scanned: {len(files)}")
    print(f"Files with findings: {len(overall)}")
    print(f" - try/except blocks: {totals['try_blocks']}")
    print(f" - raise statements:  {totals['raises']}")
    print(f" - nested functions:  {totals['nested_functions']}")
    print(f" - ternary if-exprs:  {totals['ternary_ifexprs']}")
    print(f"\nFull details written to: {args.out}")

    # print first few per file for quick glance
    for fp, data in overall.items():
        print(f"\n--- {fp} ---")
        if data["try_blocks"]:
            print(f"  try/except ({len(data['try_blocks'])})")
            for x in data["try_blocks"][:5]:
                print(f"    - at {x['loc']} caught={x['caught']} broad={x['broad']} else={x['has_else']} finally={x['has_finally']}")
        if data["raises"]:
            print(f"  raise ({len(data['raises'])})")
            for x in data["raises"][:5]:
                print(f"    - at {x['loc']} exception={x['exception']}")
        if data["nested_functions"]:
            print(f"  nested functions ({len(data['nested_functions'])})")
            for x in data["nested_functions"][:5]:
                print(f"    - at {x['loc']} name={x['name']} recursive={x['is_recursive']} free_vars={x['free_vars']}")
        if data["ternary_ifexprs"]:
            print(f"  ternary if-exprs ({len(data['ternary_ifexprs'])})")
            for x in data["ternary_ifexprs"][:5]:
                print(f"    - at {x['loc']}")

if __name__ == "__main__":
    main()

