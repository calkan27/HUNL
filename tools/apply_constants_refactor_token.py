#!/usr/bin/env python3
from hunl.constants import SEED_DEFAULT, SEED_RIVER
import argparse, os, io, sys, tokenize

SKIP_DIRS = {'.git', '.venv', 'venv', 'build', 'dist', '__pycache__', '.mypy_cache', '.pytest_cache'}
TEST_MARKERS = (f'{os.sep}tests{os.sep}', f'{os.sep}test{os.sep}', f'{os.sep}tests/', '/tests/')

CONST_MOD = 'hunl.constants'
NAME_MAP_NUM = {
    str(SEED_DEFAULT): 'SEED_DEFAULT',
    str(SEED_RIVER): 'SEED_RIVER',
    '1e-6': 'EPS_ZS',  '1E-6': 'EPS_ZS',  '1e-06': 'EPS_ZS',  '1E-06': 'EPS_ZS',
    '1e-12': 'EPS_MASS','1E-12':'EPS_MASS','1e-012':'EPS_MASS','1E-012':'EPS_MASS',
    '1e-9': 'EPS_SUM', '1E-9': 'EPS_SUM', '1e-09': 'EPS_SUM', '1E-09': 'EPS_SUM',
}
NAME_MAP_STR = {
    '"1729"': 'SEED_DEFAULT', "'1729'": 'SEED_DEFAULT',
    '"2027"': 'SEED_RIVER',   "'2027'": 'SEED_RIVER',
}

def is_test_path(path: str) -> bool:
    low = path.replace('\\','/')
    if any(m in low for m in TEST_MARKERS):
        return True
    base = os.path.basename(path)
    return base.startswith('test_') or base.endswith('_test.py')

def iter_py(root: str):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            if fn.endswith('.py'):
                path = os.path.join(dp, fn)
                if not is_test_path(path):
                    yield path

def has_import_line(lines: list, needed: set) -> bool:
    for ln in lines[:50]:
        if ln.strip().startswith('from ') and CONST_MOD in ln:
            # crude: if any import exists, we’ll let injector merge missing names later
            return True
    return False

def inject_import(lines: list, names: set) -> list:
    if not names:
        return lines[:]
    import_line = f"from {CONST_MOD} import {', '.join(sorted(names))}\n"
    out = []
    i = 0

    # keep shebang/encoding and module docstring position intact
    # detect module docstring by first statement heuristic (token-level insertion is overkill here)
    # place after any __future__ imports if present
    # strategy: scan until first non-empty line; if it’s a triple-quoted docstring, keep it at top
    # then scan following lines for __future__ imports

    # Step 1: copy shebang/encoding (first two lines sometimes)
    while i < len(lines) and (lines[i].startswith('#!') or 'coding:' in lines[i][:50]):
        out.append(lines[i]); i += 1

    # Step 2: optional module docstring block
    j = i
    while j < len(lines) and lines[j].strip() == '':
        out.append(lines[j]); j += 1
    did_doc = False
    if j < len(lines):
        s = lines[j].lstrip()
        if (s.startswith('"""') or s.startswith("'''")):
            # copy docstring block
            out.append(lines[j]); j += 1
            triple = '"""' if s.startswith('"""') else "'''"
            while j < len(lines):
                out.append(lines[j])
                if triple in lines[j]:
                    j += 1
                    did_doc = True
                    break
                j += 1
    i = j

    # Step 3: copy any __future__ imports
    while i < len(lines):
        st = lines[i].strip()
        if st.startswith('from __future__ import '):
            out.append(lines[i]); i += 1
        else:
            break

    # Step 4: if there is already an import from constants nearby, try to merge later; for now insert our line unconditionally
    out.append(import_line)

    # Step 5: append rest
    out.extend(lines[i:])
    return out

def merge_existing_imports(text: str, names: set) -> str:
    # simple, non-destructive merge: if an import from CONST_MOD exists, add missing aliases to that line; else insertion already handled
    lines = text.splitlines(keepends=True)
    idx = None
    for i, ln in enumerate(lines[:100]):
        if ln.strip().startswith('from ') and CONST_MOD in ln and ' import ' in ln:
            idx = i; break
    if idx is None:
        return text
    ln = lines[idx]
    head, tail = ln.split(' import ', 1)
    existing = [x.strip() for x in tail.strip().rstrip('\n').split(',')]
    existing_set = {x for x in existing if x}
    need = sorted(n for n in names if n not in existing_set)
    if not need:
        return text
    new_tail = ', '.join(list(existing_set) + need) + '\n'
    lines[idx] = head + ' import ' + new_tail
    return ''.join(lines)

def rewrite_file(path: str, dry: bool, verbose: bool) -> bool:
    src = open(path, 'rb').read()
    tokens = list(tokenize.tokenize(io.BytesIO(src).readline))
    out = []
    used = set()
    changed = False

    for tok in tokens:
        ttype, tstr, start, end, line = tok
        if ttype == tokenize.NUMBER and tstr in NAME_MAP_NUM:
            out.append(tokenize.TokenInfo(type=tokenize.NAME, string=NAME_MAP_NUM[tstr], start=start, end=end, line=line))
            used.add(NAME_MAP_NUM[tstr]); changed = True
        elif ttype == tokenize.STRING and tstr in NAME_MAP_STR:
            name = NAME_MAP_STR[tstr]
            # turn "1729" -> str(SEED_DEFAULT) without touching quotes elsewhere
            rep = f"str({name})"
            out.append(tokenize.TokenInfo(type=tokenize.NAME, string=rep, start=start, end=end, line=line))
            used.add(name); changed = True
        else:
            out.append(tok)

    if not changed:
        return False

    new_bytes = tokenize.untokenize(out)
    new_text = new_bytes.decode('utf-8')

    # add or merge import
    if f'from {CONST_MOD} import ' not in new_text:
        lines = new_text.splitlines(keepends=True)
        new_text = ''.join(inject_import(lines, used))
    else:
        new_text = merge_existing_imports(new_text, used)

    if verbose:
        print(f"[OK] {path} (used: {', '.join(sorted(used))}){' [dry]' if dry else ''}")

    if not dry:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_text)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='project root')
    ap.add_argument('--dry-run', action='store_true', help='do not write files')
    ap.add_argument('--verbose', action='store_true', help='log files as they change')
    args = ap.parse_args()

    changed = 0
    for path in iter_py(args.root):
        try:
            if rewrite_file(path, args.dry_run, args.verbose):
                changed += 1
        except Exception as e:
            print(f"[WARN] skipping {path}: {e}", file=sys.stderr)
            continue
    print(f"[DONE] changed files: {changed}{' (dry-run)' if args.dry_run else ''}")

if __name__ == '__main__':
    main()

