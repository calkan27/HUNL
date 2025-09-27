#!/usr/bin/env python3
"""
apply_constants_refactor.py

AST-safe refactor to replace scattered seeds/epsilons with hunl.constants.
- Replaces numeric literals: 1729, 2027, 1e-6, 1e-12, 1e-9
- Replaces string literals "1729", "2027" with str(SEED_DEFAULT/SEED_RIVER)
- Adds `from hunl.constants import ...` with only the names used
- Skips test files and test directories

Usage:
  python3 tools/apply_constants_refactor.py --root . --dry-run
  python3 tools/apply_constants_refactor.py --root .

Requires Python 3.9+ (uses ast.unparse). If you have astor installed, the script will
fallback to astor.to_source() on older Pythons.
"""

import argparse
import os
import sys
import ast
from typing import Set, List, Tuple

TRY_ASTOR = False
try:
    import astor  # type: ignore
    TRY_ASTOR = True
except Exception:
    TRY_ASTOR = False

SKIP_DIR_NAMES = {'.git', '.venv', 'venv', 'build', 'dist', '__pycache__', '.mypy_cache', '.pytest_cache'}
TEST_DIR_TOKENS = {'/tests/', os.sep + 'tests' + os.sep, os.sep + 'test' + os.sep}
TEST_FILE_PREFIXES = ('test_',)
TEST_FILE_SUFFIXES = ('_test.py',)

NUMERIC_MAP = {
    SEED_DEFAULT: 'SEED_DEFAULT',
    SEED_RIVER: 'SEED_RIVER',
    EPS_ZS: 'EPS_ZS',
    EPS_MASS: 'EPS_MASS',
    EPS_SUM: 'EPS_SUM',
}

STRING_SEEDS = {
    str(SEED_DEFAULT): 'SEED_DEFAULT',
    str(SEED_RIVER): 'SEED_RIVER',
}

CONST_MODULE = 'hunl.constants'


def is_test_path(path: str) -> bool:
    low = path.replace('\\', '/')
    if any(tok in low for tok in TEST_DIR_TOKENS):
        return True
    base = os.path.basename(path)
    if base.startswith(TEST_FILE_PREFIXES) or base.endswith(TEST_FILE_SUFFIXES):
        return True
    parts = low.strip('/').split('/')
    if 'tests' in parts:
        return True
    return False


class ConstRefactor(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.used: Set[str] = set()
        self.made_change: bool = False

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        # numeric literals
        if isinstance(node.value, (int, float)):
            key = node.value
            if key in NUMERIC_MAP:
                name = NUMERIC_MAP[key]
                self.used.add(name)
                self.made_change = True
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)

        # string seeds -> str(SEED_DEFAULT/SEED_RIVER)
        if isinstance(node.value, str):
            if node.value in STRING_SEEDS:
                name = STRING_SEEDS[node.value]
                self.used.add(name)
                self.made_change = True
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id='str', ctx=ast.Load()),
                        args=[ast.Name(id=name, ctx=ast.Load())],
                        keywords=[],
                    ),
                    node,
                )
        return node

    # Handle f-strings like f"{1729}" are rare here; skipping on purpose to avoid over-rewrites.


def ensure_import(module: ast.Module, names: List[str]) -> ast.Module:
    """Insert `from hunl.constants import names...` if not present and names is non-empty."""
    if not names:
        return module

    # Check existing imports
    existing: Set[str] = set()
    for n in module.body:
        if isinstance(n, ast.ImportFrom) and n.module == CONST_MODULE:
            for alias in n.names:
                existing.add(alias.name)

    to_add = [n for n in names if n not in existing]
    if not to_add:
        return module

    # Build import node
    import_node = ast.ImportFrom(
        module=CONST_MODULE,
        names=[ast.alias(name=n, asname=None) for n in sorted(to_add)],
        level=0,
    )

    # Insert after any future imports / module docstring / shebang-style comments (AST keeps docstring)
    insert_at = 0
    # Keep docstring at index 0 if present
    if module.body and isinstance(module.body[0], ast.Expr) and isinstance(getattr(module.body[0], 'value', None), ast.Constant) and isinstance(module.body[0].value.value, str):
        insert_at = 1

    # Place after any __future__ imports
    while insert_at < len(module.body):
        node = module.body[insert_at]
        if isinstance(node, ast.ImportFrom) and node.module == '__future__':
            insert_at += 1
        else:
            break

    module.body.insert(insert_at, import_node)
    return module


def rewrite_file(path: str, dry_run: bool, verbose: bool) -> Tuple[bool, int]:
    src = open(path, 'r', encoding='utf-8').read()
    try:
        tree = ast.parse(src)
    except Exception:
        return (False, 0)

    transformer = ConstRefactor()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    if not transformer.made_change:
        return (False, 0)

    new_tree = ensure_import(new_tree, sorted(transformer.used))

    if hasattr(ast, 'unparse'):
        new_src = ast.unparse(new_tree)  # type: ignore
    elif TRY_ASTOR:
        new_src = astor.to_source(new_tree)  # type: ignore
    else:
        print(f"[WARN] Could not write {path}: need Python 3.9+ or astor installed.", file=sys.stderr)
        return (False, 0)

    if dry_run:
        if verbose:
            print(f"[DRY] would update {path}")
        return (True, len(new_src))
    else:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_src)
        if verbose:
            used = ','.join(sorted(transformer.used))
            print(f"[OK] {path}  (imports: {used})")
        return (True, len(new_src))


def iter_py(root: str):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIR_NAMES]
        for fn in fns:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(dp, fn)
            if is_test_path(path):
                continue
            yield path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='Project root to rewrite')
    ap.add_argument('--dry-run', action='store_true', help='Do not write files')
    ap.add_argument('--verbose', action='store_true', help='Log per-file changes')
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    changed = 0
    for path in iter_py(root):
        ok, _ = rewrite_file(path, dry_run=args.dry_run, verbose=args.verbose)
        if ok:
            changed += 1

    print(f"[DONE] Files changed: {changed}{' (dry-run)' if args.dry_run else ''}")


if __name__ == '__main__':
    main()

