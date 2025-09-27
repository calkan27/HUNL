#!/usr/bin/env python3
import ast, hashlib, os, sys, difflib

EXCLUDE_DIRS = {".git", "__pycache__", "venv", ".venv", "build", "dist"}

def py_files(root):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in EXCLUDE_DIRS]
        for fn in fns:
            if fn.endswith(".py"):
                yield os.path.join(dp, fn)

def norm_ast(node):
    for n in ast.walk(node):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(n, attr):
                setattr(n, attr, None)
    return ast.dump(node, include_attributes=False)

def body_text(src, node):
    lines = src.splitlines(True)
    s = getattr(node, "lineno", None); e = getattr(node, "end_lineno", None)
    if s and e:
        return "".join(lines[s-1:e])
    return ""

def main(root):
    buckets, near = {}, []
    for path in py_files(root):
        try:
            src = open(path, "r", encoding="utf-8").read()
            tree = ast.parse(src, filename=path)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                dump = norm_ast(node)
                h = hashlib.sha256(dump.encode()).hexdigest()
                txt = body_text(src, node)
                buckets.setdefault(h, []).append((node.name, path, getattr(node, "lineno", 0), txt))
                near.append((node.name, path, getattr(node, "lineno", 0), txt))

    print("=== EXACT DUPLICATE FUNCTION BODIES (AST-equal) ===")
    any_exact = False
    for h, items in buckets.items():
        if len(items) > 1:
            any_exact = True
            print(f"\n-- group hash {h[:12]} ({len(items)} funcs)")
            for name, path, ln, _ in items:
                print(f"  {name:30s}  {path}:{ln}")
    if not any_exact:
        print("None")

    print("\n=== NEAR DUPLICATES (same name, similarity >= 0.90) ===")
    by_name = {}
    for name, path, ln, txt in near:
        by_name.setdefault(name, []).append((path, ln, txt))
    any_near = False
    for name, items in by_name.items():
        if len(items) < 2: continue
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                p1, l1, t1 = items[i]; p2, l2, t2 = items[j]
                if not t1 or not t2: continue
                ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
                if ratio >= 0.90:
                    any_near = True
                    print(f"\n{name}: {ratio:.2f}")
                    print(f"  - {p1}:{l1}")
                    print(f"  - {p2}:{l2}")
    if not any_near:
        print("None")

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    main(root)

