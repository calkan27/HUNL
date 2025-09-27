#!/usr/bin/env python3
import argparse, io, os, sys, tokenize
from typing import Dict, Tuple, List

# ---------- configuration ----------
# Global rename map (old_name -> new_name)
RENAME_MAP: Dict[str, str] = {
    # resolver_integration + friends
    "_prepare_inputs": "_build_resolve_context",
    "_menu_and_checks": "_validate_root_policy_and_invariants",
    "_update_opp_upper_monotone": "_tighten_cfv_upper_bounds",
    "_bet_fracs_from_mode": "_bet_fraction_schedule_for_mode",
    "_depth_and_bets": "_depth_and_bets_from_config",
    "_build_root": "_build_lookahead_tree",
    "_leaf_value_core": "_leaf_value_from_value_server",
    "_run_cfr_subgame": "_solve_lookahead_with_constraints",
    "_allowed_actions_from_fracs": "_allowed_actions_from_bet_fractions",
    "_our_cfv_scalar_map": "_scalarize_cfv_by_cluster",
    "_merge_zero_sum_stats": "_merge_value_server_zero_sum_stats",
    "_to_vec": "_ranges_to_simplex_vector",

    # cfr_solver
    "_prepare_clusters_for_run_cfr": "_prepare_clusters_for_state",
    "_normalize_ranges_for_run_cfr": "_normalize_ranges_for_state",
    "_do_iterations_for_run_cfr": "_iterate_cfr_at_root",
    "_finalize_and_choose_action_for_run_cfr": "_finalize_and_sample_action",

    # cfr_core
    "_terminate_value": "_opponent_cfv_upper_bound_value",

    # cfr_solver_utils
    "_reweight_range_by_available": "_renormalize_range_vs_board",
    "_push_no_card_abstraction_for_node": "_push_full_hand_expansion",
    "_pop_no_card_abstraction": "_pop_full_hand_expansion",

    # eval_cli_utils.py
    "_spark_menu": "_sparse_action_menu",
    "_mass_conservation_ok_ranges": "is_range_mass_conserved",
    "_zero_sum_residual_ok_from_solver": "is_zero_sum_residual_ok",
    "_no_negative_pot_delta": "is_nonnegative_pot_delta",
    "_prep_diag_node_with_ranges": "_make_node_with_ranges",
    "_diag_defaults": "_default_diag_spec",
    "_pack_diag_from_solver": "_pack_solver_diag",

    # play_cli.py
    "_r_uniform": "_uniform_cluster_range",
    "_resolve": "_resolve_once",
    "_safe_cards_for_diag": "_available_cards_for_diag",
    "_resolve_with_diag": "_resolve_once_with_diag",
    "_safe_json_line_to_obj": "_parse_json_line_safe",

    # eval_cli_lbr.py
    # note: _spark_menu also mapped above
    "_mass_conservation_ok_ranges": "is_range_mass_conserved",
    "_no_negative_pot_delta": "is_nonnegative_pot_delta",
    "_make_tmp_solver_from": "_clone_solver_template",

    # data_generator_utils.py
    "_push_leaf_override": "_push_leaf_solve_mode",
    "_pop_leaf_override": "_pop_leaf_solve_mode",
    "_recursive_R": "_recursive_range_split",
    "_mass_conservation_ok": "is_range_mass_conserved",
    "map_handstorcho_clusters": "map_hands_to_clusters_compat",
}

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "build", "dist",
    "__pycache__", ".mypy_cache", ".pytest_cache"
}
TEST_MARKERS = (
    f"{os.sep}tests{os.sep}", "/tests/", f"{os.sep}test{os.sep}"
)

# ---------- helpers ----------

def is_test_path(path: str) -> bool:
    low = path.replace("\\", "/")
    if any(m in low for m in TEST_MARKERS):
        return True
    base = os.path.basename(path)
    return base.startswith("test_") or base.endswith("_test.py")

def iter_py(root: str):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            if fn.endswith(".py"):
                path = os.path.join(dp, fn)
                if not is_test_path(path):
                    yield path

def next_significant(tokens: List[tokenize.TokenInfo], idx: int) -> Tuple[int, tokenize.TokenInfo]:
    j = idx + 1
    while j < len(tokens):
        t = tokens[j]
        if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT, tokenize.ENCODING):
            return j, t
        j += 1
    return idx, tokens[idx]

def prev_significant(tokens: List[tokenize.TokenInfo], idx: int) -> Tuple[int, tokenize.TokenInfo]:
    j = idx - 1
    while j >= 0:
        t = tokens[j]
        if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT, tokenize.ENCODING):
            return j, t
        j -= 1
    return idx, tokens[idx]

def is_name_context_to_rename(tokens: List[tokenize.TokenInfo], i: int, old: str) -> bool:
    """
    Rename NAME token 'old' only when it's very likely a function/method name:
      - function definition: 'def old'
      - attribute access: '.' old
      - bare call: old '('
      - 'from x import old' (imported symbol rename)
    Avoid touching strings/comments automatically via tokenize.
    """
    tok = tokens[i]
    if tok.type != tokenize.NAME or tok.string != old:
        return False

    # Look at previous significant token
    pi, pt = prev_significant(tokens, i)
    # Look at next significant token
    ni, nt = next_significant(tokens, i)

    # def old(...):
    if pt.type == tokenize.NAME and pt.string == "def":
        return True

    # attribute access: . old
    if pt.type == tokenize.OP and pt.string == ".":
        return True

    # bare call: old ( ...
    if nt.type == tokenize.OP and nt.string == "(":
        return True

    # from X import old
    # pattern: ... 'from' NAME* 'import' old[, ...]
    # Simplified scan backwards for 'import' before a newline, and 'from' before that.
    k = i - 1
    saw_import = False
    while k >= 0 and tokens[k].type not in (tokenize.NEWLINE, tokenize.NL):
        if tokens[k].type == tokenize.NAME and tokens[k].string == "import":
            saw_import = True
            # continue scanning for 'from'
            k2 = k - 1
            while k2 >= 0 and tokens[k2].type not in (tokenize.NEWLINE, tokenize.NL):
                if tokens[k2].type == tokenize.NAME and tokens[k2].string == "from":
                    return True
                k2 -= 1
            break
        k -= 1

    return False

# ---------- core ----------

def rewrite_file(path: str, dry: bool, verbose: bool) -> bool:
    with open(path, "rb") as f:
        src_bytes = f.read()

    try:
        tokens = list(tokenize.tokenize(io.BytesIO(src_bytes).readline))
    except tokenize.TokenError:
        return False

    changed = False
    out: List[tokenize.TokenInfo] = []

    for i, tok in enumerate(tokens):
        if tok.type == tokenize.NAME:
            new_name = RENAME_MAP.get(tok.string)
            if new_name and is_name_context_to_rename(tokens, i, tok.string):
                # replace only the token string; keep start/end/line to preserve formatting
                tok = tokenize.TokenInfo(tok.type, new_name, tok.start, tok.end, tok.line)
                changed = True
        out.append(tok)

    if not changed:
        return False

    new_src = tokenize.untokenize(out)
    if verbose:
        print(f"[OK] {path}{' [dry]' if dry else ''}")

    if not dry:
        with open(path, "wb") as f:
            f.write(new_src)

    return True

def main():
    ap = argparse.ArgumentParser(description="Token-level method renamer (formatting-safe).")
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files")
    ap.add_argument("--verbose", action="store_true", help="Log changed files")
    args = ap.parse_args()

    changed = 0
    for path in iter_py(args.root):
        try:
            if rewrite_file(path, args.dry_run, args.verbose):
                changed += 1
        except Exception as e:
            print(f"[WARN] skipping {path}: {e}", file=sys.stderr)

    print(f"[DONE] files changed: {changed}{' (dry-run)' if args.dry_run else ''}")

if __name__ == "__main__":
    main()

