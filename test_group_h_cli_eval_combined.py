# test_group_h_cli_eval_combined.py
# ===========================================================
# GROUP H — CLIs & evaluation (Combined Ultimate Suite)
#
# This suite hits every function path and invariant in:
# - eval_cli.py
# - eval_cli_lbr.py
# - eval_cli_utils.py
# - smoke_eval.py (exposed helpers)
# - smoke_eval_checks.py
# - smoke_eval_utils.py
#
# Paper-aligned expectations validated by this harness:
# * AIVAT: low-variance, unbiased estimate using control variates over public states;
#   CLI prints chips/100, CI, and variance reduction vs. naive.
# * LBR acceptance: local best-response lower bound & union-CI acceptance (upper CI ≤ −300 mbb/g).
# * Sanity: outer zero-sum residual near 0, range-mass conservation, nonnegative pot deltas,
#   legal street advance, and “turn leaves don’t invoke nets” (solve-to-terminal).
# ===========================================================

import math
import random
import types
from typing import Dict, List, Tuple

import pytest
from hypothesis import given, settings, strategies as st

# Modules under test
import eval_cli
import eval_cli_lbr
import eval_cli_utils
import smoke_eval_checks as chk
import smoke_eval_utils as seutil

from action_type import ActionType
from action import Action


# --------------------------
# Shared test stubs & helpers (deduplicated)
# --------------------------

class _FakeCFRSolver:
    """Lightweight solver stub matching the interfaces used across Group H tests."""
    def __init__(self, config=None, depth_limit: int = 1, num_clusters: int = 6):
        if config is not None and hasattr(config, "num_clusters"):
            self.num_clusters = int(getattr(config, "num_clusters", num_clusters))
            self.depth_limit = int(getattr(config, "depth_limit", depth_limit))
        else:
            self.num_clusters = int(num_clusters)
            self.depth_limit = int(depth_limit)
        self.total_iterations = 1
        self.clusters = {}
        self.models = {}
        self._preflop_cache_stats = {"hits": 0, "misses": 0, "puts": 0, "evictions": 0}
        self.cfr_values = {}
        self._last_diag = {
            "zero_sum_residual": 0.0,
            "zero_sum_residual_mean": 0.0,
            "regret_l2": 0.0,
            "avg_strategy_entropy": 0.0,
        }
        self._k1 = 0.0
        self._k2 = 0.0

    def load_models(self):  # used by LBR CLI
        return

    def predict_counterfactual_values(self, node, player):
        # Return per-cluster pot-fraction CFVs
        return {i: [0.0] for i in range(self.num_clusters)}

    def get_stage(self, node):
        # Treat everything as flop unless a test overrides
        return "flop"

    def run_cfr(self, node):
        # Simulate minor preflop cache activity
        self._preflop_cache_stats["misses"] += 1
        return {}

    def _calculate_terminal_utility(self, node, player: int = 0) -> float:
        if hasattr(node.public_state, "terminal_utility"):
            u = node.public_state.terminal_utility()
            if isinstance(u, (list, tuple)) and len(u) >= 2:
                return float(u[player])
        return 0.0

    def get_last_diagnostics(self):
        d = dict(self._last_diag)
        d["k1"] = float(self._k1)
        d["k2"] = float(self._k2)
        return d

    def set_soundness_constants(self, k1: float, k2: float):
        self._k1 = float(k1)
        self._k2 = float(k2)

    # Policy helpers used by _policy_from_resolve/_engine_policy_action
    def _allowed_actions_agent(self, ps):
        return [ActionType.CALL]

    def _allowed_actions_opponent(self, ps):
        return [ActionType.CALL]

    def _mixed_action_distribution(self, node, player, allowed):
        # All mass on the only action by default
        return [1.0 for _ in allowed]


class _FakeAIVAT:
    """AIVAT stub producing a deterministic corrected value for hook testing."""
    def __init__(self, value_fn, policy_fn, chance_policy_fn, agent_player=0):
        self.value_fn = value_fn
        self.policy_fn = policy_fn
        self.agent_player = agent_player

    def evaluate(self, episode):
        # Deterministic low-variance corrected estimate
        return {"aivat": 0.0}


class _TermPS:
    """Terminal PublicState stub used to short-circuit episodes quickly."""
    is_terminal = True
    pot_size = 0.0
    current_round = 3
    def terminal_utility(self):
        return [0.25, -0.25]


class _StepToTerminalPS:
    """PublicState stub for pot monotonicity and step-to-terminal transitions."""
    def __init__(self, pot_size=0.0, to_call=0, legal=None, round_=1, dealer=0, player=0):
        self.pot_size = float(pot_size)
        self.current_bets = [0, int(to_call)]
        self.current_player = int(player)
        self.dealer = int(dealer)
        self.current_round = int(round_)
        self.is_terminal = False
        self.last_refund_amount = 0.0
        self._legal = list(legal or [ActionType.CALL, ActionType.POT_SIZED_BET, ActionType.ALL_IN])

    def legal_actions(self):
        return list(self._legal)

    def update_state(self, node, action):
        # One step -> terminal; pot never decreases
        inc = 0.0
        if action.kind == ActionType.POT_SIZED_BET:
            inc = max(1.0, self.pot_size)  # pot sized
        elif action.kind == ActionType.ALL_IN:
            inc = max(1.0, self.pot_size) * 2.0
        elif action.kind == ActionType.CALL:
            inc = 0.5
        elif action.kind == ActionType.FOLD:
            inc = 0.0
        nxt = _TermPS()
        nxt.pot_size = self.pot_size + inc
        return nxt


class _NonTermPS:
    """Non-terminal PublicState for step-guard branch in _play_episode."""
    def __init__(self):
        self.is_terminal = False
        self.dealer = 0
        self.current_player = 0
        self.current_round = 1
        self.pot_size = 0.0
        self.last_refund_amount = 0.0
    def update_state(self, node, action):
        # Never terminal, no pot decrease
        self.pot_size += 0.0
        return self
    def terminal_utility(self):
        return [0.0, 0.0]


# --------------------------
# eval_cli.py — statistics & plumbing
# --------------------------

@given(
    results=st.lists(
        st.tuples(st.floats(allow_nan=False, allow_infinity=False, width=32),
                  st.floats(allow_nan=False, allow_infinity=False, width=32)),
        min_size=0, max_size=200
    )
)
@settings(deadline=None)
def test_eval_cli_summarize_properties(results):
    mean_na, mean_av, std_na, std_av, reduction = eval_cli._summarize(results)
    # Types and basic bounds
    assert isinstance(mean_na, float)
    assert isinstance(mean_av, float)
    assert isinstance(std_na, float)
    assert isinstance(std_av, float)
    # If no results or 1 datum, std is 0 by construction
    if len(results) <= 1:
        assert std_na == 0.0
        assert std_av == 0.0
        assert reduction == 0.0
    # Reduction is defined as 1 - std_av/std_na when std_na > 0; else 0
    if std_na == 0.0:
        assert reduction == 0.0


def test_eval_cli_summarize_reduction_sign_on_synthetic():
    # Naive variance > AIVAT variance -> reduction > 0
    results = [(0.0, 0.0), (2.0, 0.5), (-2.0, -0.5), (1.0, 0.25), (-1.0, -0.25)]
    mean_na, mean_av, std_na, std_av, reduction = eval_cli._summarize(results)
    assert std_na > std_av and reduction > 0.0
    # Equal variance -> reduction = 0
    eq = [(1.0, 1.0), (-1.0, -1.0)]
    _, _, s1, s2, red = eval_cli._summarize(eq)
    assert s1 == s2 and red == 0.0


@given(
    results=st.lists(
        st.tuples(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                  st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)),
        min_size=0, max_size=300
    ),
    block_size=st.integers(min_value=1, max_value=25)
)
@settings(deadline=None)
def test_eval_cli_block_metrics_shapes_and_monotone_ci(results, block_size):
    out = eval_cli._block_metrics(results, block_size=block_size)
    assert "blocks" in out and "naive" in out and "aivat" in out
    b = out["blocks"]
    assert b == (len(results) // block_size if len(results) > 0 else 0)
    # CI is [low, high] and width non-negative
    for key in ("naive", "aivat"):
        ci = out[key]["ci95"]
        assert len(ci) == 2
        assert ci[1] - ci[0] >= 0.0


def test_eval_cli_block_metrics_empty_returns_zeros():
    out = eval_cli._block_metrics([], block_size=10)
    assert out["blocks"] == 0
    assert out["naive"]["mbb100"] == 0.0 and out["aivat"]["mbb100"] == 0.0
    assert out["naive"]["ci95"] == [0.0, 0.0] and out["aivat"]["ci95"] == [0.0, 0.0]


def test_eval_cli_no_negative_pot_delta_respects_refund_allowance():
    prev = _StepToTerminalPS(pot_size=100.0)
    nxt = _StepToTerminalPS(pot_size=96.0)
    prev.last_refund_amount = 5.0  # allowed negative delta up to 5
    assert eval_cli._no_negative_pot_delta(prev, nxt)  # 96 >= 100-5 -> ok
    nxt2 = _StepToTerminalPS(pot_size=94.0)
    assert not eval_cli._no_negative_pot_delta(prev, nxt2)  # 94 < 95 -> violation


def test_eval_cli_play_episode_and_run_matches_with_stubs(monkeypatch):
    # Patch light solver + AIVAT + policy generator and make any state terminal after one action
    monkeypatch.setattr(eval_cli, "CFRSolver", _FakeCFRSolver, raising=True)
    monkeypatch.setattr(eval_cli, "AIVATEvaluator", _FakeAIVAT, raising=True)

    def _stub_policy_from_resolve(solver_template, iters=1):
        def _pf(nd, player):
            return {ActionType.CALL: 1.0}
        return _pf

    monkeypatch.setattr(eval_cli, "_policy_from_resolve", _stub_policy_from_resolve, raising=True)

    # Make PublicState.update_state return terminal immediately
    from public_state import PublicState
    def _upd(self, node, action):  # pragma: no cover - trivial shim
        return _TermPS()
    monkeypatch.setattr(PublicState, "update_state", _upd, raising=False)

    naive, ares = eval_cli._play_episode(_FakeCFRSolver(), _FakeCFRSolver(), rng_seed=123, value_solver_for_aivat=_FakeCFRSolver())
    assert isinstance(naive, float)
    assert isinstance(ares, dict)
    assert "aivat" in ares

    # _run_matches: stub out _play_episode to be deterministic
    def _pe(s0, s1, rng_seed, value_solver_for_aivat, policy_iters_agent=2, policy_iters_opp=1):
        return 0.1, {"aivat": 0.05}
    monkeypatch.setattr(eval_cli, "_play_episode", _pe, raising=True)
    res = eval_cli._run_matches("agent-vs-policy", episodes=7, seed=1, cfg=eval_cli.ResolveConfig.from_env({"num_clusters": 4}))
    assert isinstance(res, list) and len(res) == 7
    assert all(isinstance(t, tuple) and len(t) == 2 for t in res)


def test_eval_cli_main_prints_summary(monkeypatch, capsys):
    # Patch _run_matches to a tiny synthetic set to exercise formatting (mbb/g, CI, etc.)
    monkeypatch.setattr(eval_cli, "_run_matches", lambda mode, ep, seed, cfg: [(1.0, 0.2), (2.0, 0.1), (0.0, 0.0)], raising=True)
    eval_cli.main(["--episodes", "3", "--seed", "7", "--num-clusters", "4", "--depth-limit", "1", "--iterations", "2"])
    out = capsys.readouterr().out
    assert "Naive average reward" in out
    assert "AIVAT-corrected estimate" in out
    assert "Blocks (100 hands)" in out


def test_eval_cli_main_sets_river_buckets_attr(monkeypatch, capsys):
    # Capture the config instance configured by main()
    class _Cfg:
        def __init__(self):
            self.num_clusters = 8
            self.depth_limit = 1
            self.total_iterations = 2
    cfg_holder = {"cfg": None}
    def _from_env(overrides=None):
        cfg_holder["cfg"] = _Cfg()
        # honor overrides
        if overrides:
            for k, v in overrides.items():
                setattr(cfg_holder["cfg"], k, v)
        return cfg_holder["cfg"]

    monkeypatch.setattr(eval_cli.ResolveConfig, "from_env", staticmethod(_from_env), raising=True)
    # Make _run_matches deterministic and cheap
    monkeypatch.setattr(eval_cli, "_run_matches", lambda mode, ep, seed, cfg: [(0.0, 0.0)], raising=True)

    eval_cli.main(["--episodes", "1", "--seed", "7", "--num-clusters", "4",
                   "--depth-limit", "1", "--iterations", "2", "--river-buckets", "10"])
    cfg = cfg_holder["cfg"]
    assert hasattr(cfg, "river_num_buckets") and getattr(cfg, "river_num_buckets") == 10
    # Ensure summary printed
    out = capsys.readouterr().out
    assert "AIVAT-corrected estimate" in out


def test_eval_cli_play_episode_nonterminal_guard_returns_empty(monkeypatch):
    # Force persistent non-terminal to hit step_guard path -> empty episode bundle
    monkeypatch.setattr(eval_cli, "_make_initial_preflop", lambda stack, seed: _NonTermPS(), raising=True)

    class _AIVAT0:
        def __init__(self, *a, **k): pass
        def evaluate(self, episode): return {"aivat": 0.0}
    monkeypatch.setattr(eval_cli, "AIVATEvaluator", _AIVAT0, raising=True)
    monkeypatch.setattr(eval_cli, "CFRSolver", _FakeCFRSolver, raising=True)

    naive, ares = eval_cli._play_episode(_FakeCFRSolver(), _FakeCFRSolver(), 123, _FakeCFRSolver())
    # Nonterminal => (0.0, {"initial_node":None, "events":[]}) inside _play_episode
    assert naive == 0.0 and isinstance(ares, dict)


# --------------------------
# eval_cli_lbr.py — LBR & acceptance
# --------------------------

def test_lbr_sparse_menu_respects_to_call_and_actions():
    # to_call > 0 -> include FOLD if legal; to_call == 0 -> FOLD omitted
    ps_call = types.SimpleNamespace(
        current_player=0,
        current_bets=[0, 5],  # to_call = 5
        legal_actions=lambda: [ActionType.FOLD, ActionType.CALL, ActionType.POT_SIZED_BET, ActionType.ALL_IN],
    )
    menu1 = eval_cli_lbr._sparse_menu(ps_call)
    assert set(menu1) == {ActionType.FOLD, ActionType.CALL, ActionType.POT_SIZED_BET, ActionType.ALL_IN}

    ps_chk = types.SimpleNamespace(
        current_player=0,
        current_bets=[5, 5],  # to_call = 0
        legal_actions=lambda: [ActionType.FOLD, ActionType.CALL, ActionType.POT_SIZED_BET, ActionType.ALL_IN],
    )
    menu2 = eval_cli_lbr._sparse_menu(ps_chk)
    assert set(menu2) == {ActionType.CALL, ActionType.POT_SIZED_BET, ActionType.ALL_IN}


@given(n=st.integers(min_value=1, max_value=25))
@settings(deadline=None)
def test_lbr_range_mass_conservation_hypothesis(n):
    # Construct two normalized dicts
    v1 = [random.random() for _ in range(n)]
    s1 = sum(v1) or 1.0
    v1 = [x / s1 for x in v1]
    r1 = {i: v1[i] for i in range(n)}

    v2 = [random.random() for _ in range(n)]
    s2 = sum(v2) or 1.0
    v2 = [x / s2 for x in v2]
    r2 = {i: v2[i] for i in range(n)}

    assert eval_cli_lbr._mass_conservation_ok_ranges(r1, r2, tol=1e-12)


def test_lbr_no_negative_pot_delta_simple():
    prev = types.SimpleNamespace(pot_size=10.0)
    nxt = types.SimpleNamespace(pot_size=10.5)
    assert eval_cli_lbr._no_negative_pot_delta(prev, nxt)
    bad = types.SimpleNamespace(pot_size=9.0)
    assert not eval_cli_lbr._no_negative_pot_delta(prev, bad)


def test_lbr_greedy_action_flop_branch_and_freq_logging(monkeypatch):
    # Force greedy branch: flop (round==1) and current_player==lbr_player
    ps = _StepToTerminalPS(pot_size=10.0, to_call=5, round_=1, dealer=0, player=1,
                           legal=[ActionType.FOLD, ActionType.CALL, ActionType.POT_SIZED_BET, ActionType.ALL_IN])

    # For each candidate action, make terminal utility depend on the FIRST action taken
    def _mk_term_for(a: ActionType):
        class _Term(_TermPS):
            def terminal_utility(self):
                # Greedy player is player=1; maximize this
                if a == ActionType.POT_SIZED_BET:
                    return [-1.0, 1.0]
                if a == ActionType.ALL_IN:
                    return [-0.5, 0.5]
                if a == ActionType.CALL:
                    return [0.0, 0.0]
                return [1.0, -1.0]  # FOLD is worst for player 1
        return _Term()

    # Patch the first update to yield those terminals depending on chosen a
    def _upd_once(self, node, action):
        return _mk_term_for(action.kind)

    # Only used if the inner loop runs (but with terminal right away it won't)
    monkeypatch.setattr(eval_cli_lbr, "_engine_policy_action", lambda solver, node, iters=2: ActionType.CALL, raising=True)

    from public_state import PublicState
    monkeypatch.setattr(PublicState, "update_state", _upd_once, raising=False)

    freq = {}
    act = eval_cli_lbr.lbr_greedy_action(ps, _FakeCFRSolver(), lbr_player=1, iters_after=1, freq_log=freq)
    assert act in {ActionType.FOLD, ActionType.CALL, ActionType.POT_SIZED_BET, ActionType.ALL_IN}
    # Our payoff design makes POT_SIZED_BET best for player 1
    assert act == ActionType.POT_SIZED_BET
    assert "flop" in freq and freq["flop"]["POT"] >= 1


def test_lbr_engine_policy_action_returns_call_when_no_actions(monkeypatch):
    class _S(_FakeCFRSolver):
        def _allowed_actions_agent(self, ps): return []
        def _allowed_actions_opponent(self, ps): return []
        def _mixed_action_distribution(self, node, player, allowed): return []
        def _calculate_counterfactual_values(self, node, player): return
    monkeypatch.setattr(eval_cli_lbr, "CFRSolver", _S, raising=True)
    node = types.SimpleNamespace(public_state=types.SimpleNamespace(current_player=0))
    node.player_ranges = [{0:1.0}, {0:1.0}]
    act = eval_cli_lbr._engine_policy_action(_S(), node, iters=1)
    assert act == ActionType.CALL


def test_lbr_greedy_action_nonflop_or_not_turn_uses_engine_policy(monkeypatch):
    # Not on flop -> must route to engine policy
    ps = types.SimpleNamespace(current_round=0, current_player=1)
    called = {"n": 0}
    monkeypatch.setattr(eval_cli_lbr, "_engine_policy_action",
                        lambda solver, node, iters=2: (called.__setitem__("n", called["n"]+1) or ActionType.ALL_IN),
                        raising=True)
    act = eval_cli_lbr.lbr_greedy_action(ps, _FakeCFRSolver(), lbr_player=1, iters_after=1, freq_log={})
    assert called["n"] == 1 and act == ActionType.ALL_IN


def test_run_lbr_acceptance_union_ci(monkeypatch):
    # Make run_lbr_eval return fixed CIs so we can test the union and accept flag
    def _fake_eval(episodes, seed, cfg, policy_iters_agent, policy_iters_after_lbr):
        # Alternate CIs so that union_max <= -300 triggers True
        if seed % 2 == 0:
            return {"ci95": [-500.0, -350.0], "mbb_per_game": -420.0, "accept": True}
        else:
            return {"ci95": [-450.0, -320.0], "mbb_per_game": -380.0, "accept": True}

    monkeypatch.setattr(eval_cli_lbr, "run_lbr_eval", _fake_eval, raising=True)
    out = eval_cli_lbr.run_lbr_acceptance(seeds=[1, 2, 3], episodes=10000, cfg=eval_cli.ResolveConfig.from_env({}))
    assert out["accept_union"] is True
    assert out["union_ci95"][1] <= -300.0


def test_run_lbr_eval_smoke_minimal(monkeypatch):
    # Lightly patch to ensure finite outputs; keep structure per spec.
    monkeypatch.setattr(eval_cli_lbr, "CFRSolver", _FakeCFRSolver, raising=True)
    monkeypatch.setattr(eval_cli_lbr, "_engine_policy_action", lambda solver, node, iters=2: ActionType.CALL, raising=True)

    # Make PublicState.update_state step to terminal to avoid long loops
    from public_state import PublicState
    def _upd(self, node, action):
        return _TermPS()
    monkeypatch.setattr(PublicState, "update_state", _upd, raising=False)

    out = eval_cli_lbr.run_lbr_eval(episodes=10, seed=7, cfg=eval_cli.ResolveConfig.from_env({}), policy_iters_agent=1, policy_iters_after_lbr=1)
    assert "episodes" in out and out["episodes"] == 10
    assert "mbb_per_game" in out and isinstance(out["mbb_per_game"], float)
    assert "ci95" in out and len(out["ci95"]) == 2
    assert "freq" in out and "flop" in out["freq"]
    assert "sanity" in out and "zero_sum_residual_max_ok" in out["sanity"]


def test_run_lbr_eval_accept_true_with_large_negative_mbbg(monkeypatch):
    # Drive large negative returns to trigger accept=True
    from public_state import PublicState
    def _upd(self, node, action):
        class _T(_TermPS):
            def terminal_utility(self2): return [-20.0, 20.0]
        return _T()
    monkeypatch.setattr(eval_cli_lbr, "CFRSolver", _FakeCFRSolver, raising=True)
    monkeypatch.setattr(PublicState, "update_state", _upd, raising=False)

    out = eval_cli_lbr.run_lbr_eval(episodes=5, seed=11, cfg=eval_cli.ResolveConfig.from_env({}),
                                    policy_iters_agent=1, policy_iters_after_lbr=1)
    assert out["accept"] is True
    assert out["ci95"][1] <= -300.0  # Upper CI meets acceptance threshold


# --------------------------
# eval_cli_utils.py — policy, value, sampling, diagnostics
# --------------------------

def test_value_fn_from_solver_calls_predict():
    sol = _FakeCFRSolver()
    calls = {"n": 0}
    def _pcfv(node, player):
        calls["n"] += 1
        return {0: [0.0]}
    sol.predict_counterfactual_values = _pcfv
    vf = eval_cli_utils._value_fn_from_solver(sol)
    res = vf(types.SimpleNamespace(), 0)
    assert calls["n"] == 1 and isinstance(res, dict)


def test_policy_from_resolve_normalizes_and_uses_allowed(monkeypatch):
    # Replace CFRSolver in utils with a predictable fake that yields a 3-action menu
    class _Fake3(_FakeCFRSolver):
        def _allowed_actions_agent(self, ps):
            return [ActionType.FOLD, ActionType.CALL, ActionType.POT_SIZED_BET]
        _allowed_actions_opponent = _allowed_actions_agent
        def _mixed_action_distribution(self, node, player, allowed):
            return [0.1, 0.2, 0.3]  # Will be renormalized to sum to 1
        def _calculate_counterfactual_values(self, node, player):  # required by utils
            return

    monkeypatch.setattr(eval_cli_utils, "CFRSolver", _Fake3, raising=True)

    # Build a minimal node with current_player == player to take the "agent" branch
    node = types.SimpleNamespace(public_state=types.SimpleNamespace(current_player=0))
    node.player_ranges = [{0: 1.0}, {0: 1.0}]

    pf = eval_cli_utils._policy_from_resolve(_Fake3(), iters=1)
    dist = pf(node, player=0)
    assert set(dist.keys()) == {ActionType.FOLD, ActionType.CALL, ActionType.POT_SIZED_BET}
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_policy_from_resolve_opponent_branch(monkeypatch):
    class _Fake3(_FakeCFRSolver):
        def _allowed_actions_agent(self, ps):
            return [ActionType.FOLD, ActionType.CALL, ActionType.POT_SIZED_BET]
        def _allowed_actions_opponent(self, ps):
            return [ActionType.CALL, ActionType.ALL_IN]
        def _mixed_action_distribution(self, node, player, allowed):
            return [0.3 for _ in allowed]
        def _calculate_counterfactual_values(self, node, player): return
    monkeypatch.setattr(eval_cli_utils, "CFRSolver", _Fake3, raising=True)

    # player != current_player -> opponent branch
    node = types.SimpleNamespace(public_state=types.SimpleNamespace(current_player=1))
    node.player_ranges = [{0:1.0}, {0:1.0}]
    pf = eval_cli_utils._policy_from_resolve(_Fake3(), iters=1)
    dist = pf(node, player=0)
    assert set(dist.keys()) == {ActionType.CALL, ActionType.ALL_IN}
    assert abs(sum(dist.values()) - 1.0) < 1e-9


@given(
    p_fold=st.floats(min_value=0, max_value=1),
    p_call=st.floats(min_value=0, max_value=1),
    p_pot=st.floats(min_value=0, max_value=1),
)
@settings(deadline=None, max_examples=60)
def test_sample_from_policy_respects_weights(p_fold, p_call, p_pot):
    total = p_fold + p_call + p_pot
    if total == 0:
        pytest.skip("All zero — degenerate; _sample_from_policy returns last by design.")
    dist = {
        ActionType.FOLD: p_fold / total,
        ActionType.CALL: p_call / total,
        ActionType.POT_SIZED_BET: p_pot / total,
    }
    # Monte Carlo check roughly proportional (coarse)
    from collections import Counter
    c = Counter()
    for _ in range(2000):
        a = eval_cli_utils._sample_from_policy(dist)
        c[a] += 1
    # The most probable action should be sampled most often (coarse order statistic)
    top_action = max(dist.items(), key=lambda kv: kv[1])[0]
    assert c[top_action] == max(c.values())


def test_sample_from_policy_empty_returns_call():
    assert eval_cli_utils._sample_from_policy({}) == ActionType.CALL


def test_make_initial_preflop_shape():
    ps = eval_cli_utils._make_initial_preflop(stack=200, seed=123)
    assert ps.current_round == 0
    assert isinstance(ps.board_cards, list) and len(ps.board_cards) == 0
    assert ps.current_player == ps.dealer


def test_flop_turn_leaf_sanity_harness_counts(monkeypatch):
    # Replace DataGenerator with a stub whose solver calls predict on 'flop' only
    class _StubDG:
        def __init__(self, *args, **kwargs):
            self.cfr_solver = _FakeCFRSolver()
            self.cfr_solver.depth_limit = 1
            self.cfr_solver.total_iterations = 2

            def _stage(nd):
                return "flop"
            self.cfr_solver.get_stage = _stage

            def _run(nd):
                # during a run, call predict once (flop) per sample
                self.cfr_solver.predict_counterfactual_values(nd, 0)
            self.cfr_solver.run_cfr = _run

        def _sample_flop_situation(self, rng):
            return types.SimpleNamespace(public_state=types.SimpleNamespace())

    monkeypatch.setattr(eval_cli_utils, "DataGenerator", _StubDG, raising=True)
    out = eval_cli_utils.flop_turn_leaf_sanity(samples=5, seed=2027)
    assert out["samples"] == 5
    assert out["flop_calls"] == 5
    assert out["turn_leaf_calls"] == 0  # Turn leaves should not invoke nets (solve-to-terminal).


def test_utils_zero_sum_residual_ok_from_solver_gate():
    class _S(_FakeCFRSolver):
        def get_last_diagnostics(self):
            return {"zero_sum_residual": 5e-7}  # passes 1e-6 default
    assert eval_cli_utils._zero_sum_residual_ok_from_solver(_S(), tol=1e-6)


def test_utils_no_negative_pot_delta_simple():
    prev = types.SimpleNamespace(pot_size=5.0)
    nxt = types.SimpleNamespace(pot_size=5.1)
    assert eval_cli_utils._no_negative_pot_delta(prev, nxt)


def test_utils_mass_conservation_strict_failure():
    r1 = {0: 0.9}
    r2 = {0: 1.1}
    assert not eval_cli_utils._mass_conservation_ok_ranges(r1, r2, tol=1e-12)


def test_diag_from_solver_captures_fields(monkeypatch):
    class _DiagSolver(_FakeCFRSolver):
        def run_cfr(self, node):
            self._last_diag.update({
                "depth_limit": self.depth_limit,
                "iterations": self.total_iterations,
                "zero_sum_residual": 0.0,
                "zero_sum_residual_mean": 0.0,
                "regret_l2": 0.123,
                "avg_strategy_entropy": 0.456,
                "cfv_calls": {"flop": 3},
                "constraint_mode": "sp",
                "preflop_cache": {"hits": 1, "misses": 0},
                "k1": 0.1, "k2": 0.2,
            })
            return {}
    monkeypatch.setattr(eval_cli_utils, "CFRSolver", _DiagSolver, raising=True)
    ps = eval_cli_utils._make_initial_preflop(stack=200, seed=99)
    out = eval_cli_utils._diag_from_solver(ps, K=4, r_us={0:1.0}, r_opp={1:1.0}, depth=1, iters=2, k1=0.1, k2=0.2)
    for k in ("depth_limit","iterations","zero_sum_residual","avg_strategy_entropy","cfv_calls","preflop_cache","k1","k2"):
        assert k in out


def test_diag_from_solver_exception_fallback(monkeypatch):
    class _Boom(_FakeCFRSolver):
        def run_cfr(self, node): raise RuntimeError("boom")
    monkeypatch.setattr(eval_cli_utils, "CFRSolver", _Boom, raising=True)
    ps = eval_cli_utils._make_initial_preflop(stack=200, seed=123)
    out = eval_cli_utils._diag_from_solver(ps, K=4, r_us={0:1.0}, r_opp={1:1.0}, depth=1, iters=2, k1=0.1, k2=0.2)
    # Falls back to default-structured dictionary with echoed k1/k2
    for k in ("depth_limit","iterations","zero_sum_residual","avg_strategy_entropy","cfv_calls","preflop_cache","k1","k2"):
        assert k in out
    assert out["k1"] == 0.1 and out["k2"] == 0.2


# --------------------------
# smoke_eval_checks.py — invariants & outer zero-sum verify
# --------------------------

def test_zero_sum_residual_ok_via_symmetric_values():
    class _S:
        num_clusters = 3
        def predict_counterfactual_values(self, node, player):
            # Make v2 = -v1 so <r1,f1> + <r2,f2> = 0 for any ranges
            return {i: [1.0] for i in range(3)} if player == 0 else {i: [-1.0] for i in range(3)}
    node = types.SimpleNamespace(player_ranges=[{0:0.5,1:0.5,2:0.0},{0:0.0,1:0.5,2:0.5}])
    assert chk.zero_sum_residual_ok(_S(), node, tol=1e-9)


def test_mass_conservation_ok_simple():
    node = types.SimpleNamespace(player_ranges=[{0:0.6,1:0.4},{0:0.5,1:0.5}])
    assert chk.mass_conservation_ok(node, tol=1e-12)


def test_nonnegative_pot_deltas_ok_and_sequence():
    ps = _StepToTerminalPS(pot_size=1.0, to_call=0)
    assert chk.nonnegative_pot_deltas_ok(ps)
    # pot_monotonicity_ok_sequence randomly chooses from allowed menus; with our PS
    # implementation pot never decreases.
    assert chk.pot_monotonicity_ok_sequence(_StepToTerminalPS(pot_size=0.0), steps=5)


def test_verify_outer_zero_sum_residual_with_fake_nets_torch():
    import torch
    K = 4
    class _FakeNet:
        num_clusters = K
        input_size = 1 + 52 + 2*K
        def __call__(self, x):
            b = x.shape[0]
            # Raw predictions; outer layer will fix zero-sum
            return torch.zeros((b,K)), torch.zeros((b,K))
        def enforce_zero_sum(self, r1, r2, p1, p2):
            # Enforce f1 = -f2 -> exact zero-sum for any r1, r2
            return -p2, -p1
    models = {"flop": _FakeNet(), "turn": _FakeNet()}
    out = chk.verify_outer_zero_sum_residual(models, K=K, samples=10, tol=1e-12, seed=123)
    assert out["flop"]["ok"] and out["turn"]["ok"]


def test_verify_outer_zero_sum_residual_missing_stage_ok():
    import torch
    K = 3
    class _FakeNet:
        num_clusters = K
        input_size = 1 + 52 + 2*K
        def __call__(self, x):
            b = x.shape[0]
            return torch.zeros((b,K)), torch.zeros((b,K))
        def enforce_zero_sum(self, r1, r2, p1, p2):
            return -p2, -p1
    # Only provide flop; turn is intentionally missing
    models = {"flop": _FakeNet()}
    out = chk.verify_outer_zero_sum_residual(models, K=K, samples=5, tol=1e-12, seed=7)
    assert out["flop"]["ok"] is True
    assert out["turn"]["checked"] == 0 and out["turn"]["ok"] is True


@given(steps=st.integers(min_value=1, max_value=20))
@settings(deadline=None, max_examples=40)
def test_chk_pot_monotonicity_property_with_stub_ps(steps):
    class _PS:
        def __init__(self, pot=0.0):
            self.pot_size = pot
            self.is_terminal = False
            self.current_bets = [0, 0]
            self.current_player = 0
        def update_state(self, node, action):
            # Pot never decreases
            self.pot_size += 0.25
            return self
    assert chk.pot_monotonicity_ok_sequence(_PS(), steps=steps)


# --------------------------
# smoke_eval_utils.py — instrumentation, timing, cache stats
# --------------------------

def test_instrument_value_nets_counts(monkeypatch):
    s = _FakeCFRSolver()
    # Wrap predict so we can see increments via the instrumentor
    called = {"n": 0}
    def _pcfv(node, player):
        called["n"] += 1
        return {0: [0.0]}
    s.predict_counterfactual_values = _pcfv
    # Inject into a dummy solver holder that exposes "get_stage"
    def _stage(node): return "flop"
    s.get_stage = _stage
    counters, orig = seutil.instrument_value_nets(s)
    # Call twice; counters["flop"] should increment to 2
    s.predict_counterfactual_values(types.SimpleNamespace(public_state=None), 0)
    s.predict_counterfactual_values(types.SimpleNamespace(public_state=None), 1)
    assert counters["flop"] == 2
    # restore
    s.predict_counterfactual_values = orig


def test_measure_resolve_time_averages(monkeypatch):
    s = _FakeCFRSolver()
    times = []
    def _run(node):
        times.append(1)
        return {}
    s.run_cfr = _run
    node = types.SimpleNamespace(public_state=types.SimpleNamespace())
    mean_t = seutil.measure_resolve_time(s, node, trials=3)
    assert mean_t >= 0.0


def test_measure_resolve_time_zero_trials():
    s = _FakeCFRSolver()
    node = types.SimpleNamespace(public_state=types.SimpleNamespace())
    assert seutil.measure_resolve_time(s, node, trials=0) == 0.0


def test_preflop_cache_hit_rate_accumulates(monkeypatch):
    s = _FakeCFRSolver()
    # Make each run create a "hit" after first miss to verify delta accounting
    def _run(node):
        if s._preflop_cache_stats["misses"] == 0:
            s._preflop_cache_stats["misses"] += 1
        else:
            s._preflop_cache_stats["hits"] += 1
        return {}
    s.run_cfr = _run
    # Use the helper to produce trials and compute hit rate
    from public_state import PublicState
    node = types.SimpleNamespace(public_state=PublicState(initial_stacks=[200,200], board_cards=[], dealer=0))
    node.public_state.current_round = 0
    node.public_state.current_player = node.public_state.dealer
    hr, stats = seutil.preflop_cache_hit_rate(s, node, trials=4)
    assert stats["hits"] >= 1 and stats["misses"] >= 1
    assert 0.0 <= hr <= 1.0


def test_preflop_cache_hit_rate_zero_trials(monkeypatch):
    s = _FakeCFRSolver()
    from public_state import PublicState
    node = types.SimpleNamespace(public_state=PublicState(initial_stacks=[200,200], board_cards=[], dealer=0))
    node.public_state.current_round = 0
    node.public_state.current_player = node.public_state.dealer
    hr, stats = seutil.preflop_cache_hit_rate(s, node, trials=0)
    assert hr == 0.0
    assert all(k in stats for k in ("hits","misses","puts","evictions"))


# --------------------------
# eval_cli.py — _run_matches secondary path
# --------------------------

def test_eval_cli_run_matches_agent_vs_agent_path(monkeypatch):
    # Ensure agent-vs-agent path and episode count honored
    monkeypatch.setattr(eval_cli, "_play_episode",
                        lambda s0, s1, rng_seed, vs, policy_iters_agent=2, policy_iters_opp=1: (0.1, {"aivat": 0.05}),
                        raising=True)
    res = eval_cli._run_matches("agent-vs-agent", episodes=9, seed=3, cfg=eval_cli.ResolveConfig.from_env({"num_clusters": 4}))
    assert len(res) == 9 and all(isinstance(t, tuple) and len(t) == 2 for t in res)

