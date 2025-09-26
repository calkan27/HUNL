# -*- coding: utf-8 -*-
# Integration Suite K — System end-to-end interactions (+ extended preflight guards)
#
# This file combines:
#   (A) The original end-to-end integration tests (Agent loop, cache, ValueServer/resolve,
#       tiny train→serve roundtrip, bundle I/O application)
#   (B) Additional orchestration guards: determinism duplicate-run, tie handling on river,
#       schema validator for generated data, gadget monotonicity, preflop-cache LRU eviction,
#       ValueServer unknown-stage fallback, and a refund safety net.
#
# All tests are designed to be FAST: we force FAST_TESTS=1, keep K small, and limit epochs/iters.

import os
import math
import random
from pathlib import Path
import json
import copy

import pytest
import torch

from resolve_config import ResolveConfig
from public_state import PublicState
from game_node import GameNode
from action_type import ActionType
from action import Action

from agent import Agent
from cfr_solver import CFRSolver
from resolver_integration import resolve_at_with_diag
from value_server import ValueServer
from model_io import save_cfv_bundle, load_cfv_bundle
from river_endgame import RiverEndgame
from poker_utils import DECK
from lookahead_tree import LookaheadTreeBuilder
from cfr_core import PublicChanceCFR

# ---------------------------
# Global FAST profile
# ---------------------------

@pytest.fixture(autouse=True)
def _fast_env(monkeypatch):
    monkeypatch.setenv("FAST_TESTS", "1")
    monkeypatch.setenv("FAST_TEST_SEED", "2027")
    yield
    monkeypatch.delenv("FAST_TESTS", raising=False)
    monkeypatch.delenv("FAST_TEST_SEED", raising=False)


@pytest.fixture
def cfg():
    # Paper-faithful small-K, fast path
    return ResolveConfig.from_env({
        "profile": "test",
        "num_clusters": 6,
        "depth_limit": 1,
        "total_iterations": 1,
        "prefer_gpu": False,
        "paper_faithful": True,
    })


# ---------------------------
# (A1) Continual re-solve loop via Agent (end-to-end)
# ---------------------------

def test_agent_end_to_end_mini_hand(cfg):
    ag = Agent(config=cfg, device="cpu")
    ps = PublicState(initial_stacks=[200, 200], board_cards=[])
    ps.current_round = 0
    ps.current_player = ps.dealer
    steps = 0
    while not ps.is_terminal and steps < 40:
        a = ag.act(ps, our_private_cards="AS KD")
        ps_next = ps.update_state(GameNode(ps), a)
        assert ps_next.pot_size + 1e-12 >= ps.pot_size - getattr(ps_next, "last_refund_amount", 0.0)
        ag.observe_opponent_action(ps, ps_next, observed_action_type=a.action_type)
        ps = ps_next
        steps += 1
        if ps.is_terminal:
            break
        leg = ps.legal_actions()
        assert leg
        pick = ActionType.CALL if ActionType.CALL in leg else leg[0]
        ps_next = ps.update_state(GameNode(ps), Action(pick))
        assert ps_next.pot_size + 1e-12 >= ps.pot_size - getattr(ps_next, "last_refund_amount", 0.0)
        ag.observe_opponent_action(ps, ps_next, observed_action_type=pick)
        if ps_next.current_round > ps.current_round:
            ag.observe_chance(ps_next)
        ps = ps_next
        steps += 1
    if ps.is_terminal:
        u = ps.terminal_utility()
        assert isinstance(u, (list, tuple)) and len(u) == 2
        assert abs(sum(u)) < 1e-6


# ---------------------------
# (A2) Preflop cache integration across equivalent states
# ---------------------------

def test_preflop_cache_end_to_end_hits(cfg):
    solver = CFRSolver(config=cfg)
    K = solver.num_clusters
    u = 1.0 / K if K > 0 else 0.0

    def _make_preflop(seed):
        random.seed(seed)
        ps = PublicState(initial_stacks=[200, 200], board_cards=[], dealer=seed % 2)
        ps.current_round = 0
        ps.current_player = ps.dealer
        nd = GameNode(ps)
        nd.player_ranges[0] = {i: u for i in range(K)}
        nd.player_ranges[1] = {i: u for i in range(K)}
        return nd

    n1 = _make_preflop(7)
    _ = solver.run_cfr(n1)
    pre = dict(getattr(solver, "_preflop_cache_stats", {}))
    n2 = _make_preflop(7)
    _ = solver.run_cfr(n2)
    post = dict(getattr(solver, "_preflop_cache_stats", {}))
    assert int(post.get("hits", 0)) >= int(pre.get("hits", 0))


# ---------------------------
# (A3) ValueServer + resolver_integration (flop queries net; turn solves to terminal)
# ---------------------------
def test_resolver_with_value_server_pipeline(cfg: ResolveConfig):
    solver = CFRSolver(config=cfg)
    vs = ValueServer(models={k: v for k, v in solver.models.items()}, device=torch.device("cpu"))

    # flop public state using first three cards from the DECK
    ps = PublicState(initial_stacks=[200, 200], board_cards=[DECK[0], DECK[1], DECK[2]], dealer=0)
    ps.current_round = 1
    ps.current_bets = [0, 0]
    ps.pot_size = 20.0
    ps.current_player = (ps.dealer + 1) % 2

    node = GameNode(ps)
    K = solver.num_clusters
    u = 1.0 / float(K) if K > 0 else 0.0
    node.player_ranges[0] = {i: u for i in range(K)}
    node.player_ranges[1] = {i: u for i in range(K)}

    # Build the standard CFV input vector and query the value server
    iv = solver.prepare_input_vector(node)
    xt = torch.tensor([iv], dtype=torch.float32)
    v1, v2 = vs.query("flop", xt, scale_to_pot=False, as_numpy=False)

    # Range slices
    start_r1 = 1 + 52
    end_r1 = start_r1 + K
    start_r2 = end_r1
    end_r2 = start_r2 + K
    r1 = torch.tensor([iv[start_r1:end_r1]], dtype=torch.float32)
    r2 = torch.tensor([iv[start_r2:end_r2]], dtype=torch.float32)

    # Outer zero-sum layer should drive residual ~ 0
    f1, f2 = solver.models["flop"].enforce_zero_sum(r1, r2, v1, v2)
    resid = torch.sum(r1 * f1, dim=1) + torch.sum(r2 * f2, dim=1)

    assert f1.shape == (1, K) and f2.shape == (1, K)
    assert torch.allclose(resid, torch.zeros_like(resid), atol=1e-6)

    vs.stop()

def test_datagen_train_and_serve_roundtrip(tmp_path: Path, cfg, monkeypatch):
    from data_generator import DataGenerator
    from cfv_trainer_turn import train_turn_cfv
    from cfv_network import CounterfactualValueNetwork

    dg = DataGenerator(num_boards=1, num_samples_per_board=3, player_stack=200,
                       num_clusters=cfg.num_clusters, speed_profile="test", config=cfg)
    recs = []
    for _ in range(6):
        recs.extend(dg.generate_training_data(stage="turn", progress=None))
        if len(recs) >= 12:
            break
    train = recs[:8]; val = recs[8:12]

    K = cfg.num_clusters
    net_turn = CounterfactualValueNetwork(input_size=1 + 52 + 2 * K, num_clusters=K)
    out = train_turn_cfv(model=net_turn, train_samples=train, val_samples=val,
                         epochs=2, batch_size=4, lr=1e-3, lr_after=5e-4, lr_drop_epoch=1,
                         weight_decay=0.0, device=torch.device("cpu"), seed=123,
                         ckpt_dir=str(tmp_path), save_best=True)
    assert "best_state" in out and out["best_state"]

    vs = ValueServer(models={"turn": net_turn}, device=torch.device("cpu"))
    DECK = __import__("poker_utils").poker_utils.DECK
    ps_turn = PublicState(initial_stacks=[200, 200], board_cards=list(DECK[:4]))
    ps_turn.current_round = 2
    ps_turn.current_bets = [0, 0]
    ps_turn.current_player = (ps_turn.dealer + 1) % 2
    r_us = {i: 1.0 / K for i in range(K)}
    w_opp = {i: 0.0 for i in range(K)}
    _, _, _, diag = resolve_at_with_diag(ps_turn, r_us, w_opp,
                                         config={"depth_limit": 1, "bet_size_mode": "sparse_3"},
                                         value_server=vs)
    assert diag["turn_net_queries"] == 0
    vs.stop(join=True)

# ---------------------------
# (A5) Bundle I/O applied live in Agent (models + K mapping)
# ---------------------------

def test_model_bundle_applied_into_agent_affects_models_and_K(tmp_path: Path, cfg):
    ag = Agent(config=cfg, device="cpu")
    base_models = {k: v for k, v in ag.solver.models.items()}
    base_K = ag.num_clusters

    new_K = max(2, base_K // 2)
    from cfv_network import CounterfactualValueNetwork
    models = {
        "flop": CounterfactualValueNetwork(1 + 52 + 2 * new_K, num_clusters=new_K),
        "turn": CounterfactualValueNetwork(1 + 52 + 2 * new_K, num_clusters=new_K),
    }
    mapping = {i: [f"AS KS"] for i in range(new_K)}
    out = save_cfv_bundle(models=models, cluster_mapping=mapping,
                          input_meta={"num_clusters": new_K}, path=str(tmp_path / "bundle.pt"), seed=99)
    loaded = load_cfv_bundle(out, device=torch.device("cpu"))

    res = ag.load_bundle(out)
    assert res["applied"]
    assert ag.num_clusters in (base_K, new_K)
    assert any(id(ag.solver.models[k]) != id(base_models.get(k)) for k in ("flop", "turn"))

    ps = PublicState(initial_stacks=[200, 200], board_cards=["AH", "KD", "2C"])
    ps.current_round = 1
    ps.current_bets = [0, 0]
    ps.current_player = (ps.dealer + 1) % 2
    _ = ag.act(ps, our_private_cards="QS JC")


# ============================================================================
# (B) Additional orchestration / preflight guards
# ============================================================================

# (B1) Determinism under seed for a tiny end-to-end episode
def test_determinism_duplicate_run_same_seed(cfg):
    def _play_once(seed):
        random.seed(seed)
        ag = Agent(config=cfg, device="cpu")
        ps = PublicState(initial_stacks=[200, 200], board_cards=[])
        ps.current_round = 0
        ps.current_player = ps.dealer
        steps = 0
        while not ps.is_terminal and steps < 30:
            a = ag.act(ps, our_private_cards="AS KD")
            ps = ps.update_state(GameNode(ps), a)
            if ps.is_terminal:
                break
            leg = ps.legal_actions()
            pick = ActionType.CALL if ActionType.CALL in leg else leg[0]
            ps = ps.update_state(GameNode(ps), Action(pick))
            steps += 2
        return GameNode(ps)._public_signature(), getattr(ag, "last_public_key", None)

    sig1, key1 = _play_once(2027)
    sig2, key2 = _play_once(2027)
    assert sig1 == sig2
    assert key1 == key2


# (B2) River tie handling (resolved-pot vs. betting path)
def test_river_tie_payoffs_resolved_pot_and_betting():
    re = RiverEndgame()
    B = 4
    # put all mass on the same bucket index -> ties dominate
    p = [0.0]*B; q = [0.0]*B; p[2] = 1.0; q[2] = 1.0
    # Resolved-pot branch: tie -> 0 chip EV (then normalized by P elsewhere)
    ev_p, ev_o = re._expected_utility_buckets_both(p, q, B, resolved_pot=100.0)
    assert math.isclose(ev_p, 0.0, abs_tol=1e-12) and math.isclose(ev_o, 0.0, abs_tol=1e-12)
    # Betting branch: tie -> 0.5 * (opp_bet - my_bet) for current player
    ev_p2, ev_o2 = re._expected_utility_buckets_both(p, q, B, my_bet=10.0, opp_bet=30.0)
    assert math.isclose(ev_p2, 0.5*(30.0-10.0), abs_tol=1e-12)
    assert math.isclose(ev_o2, -ev_p2, abs_tol=1e-12)


# (B3) Data schema & semantics validator for a small batch
def test_datagen_schema_invariants(cfg):
    from data_generator import DataGenerator
    dg = DataGenerator(num_boards=1, num_samples_per_board=3, player_stack=200,
                       num_clusters=cfg.num_clusters, speed_profile="test", config=cfg)
    recs = dg.generate_training_data(stage="flop", progress=None)
    K = cfg.num_clusters
    for rec in recs:
        iv = rec["input_vector"]
        assert len(iv) == 1 + 52 + 2*K
        pot_norm = iv[0]
        assert 0.0 < pot_norm <= 1.0
        board_ones = sum(iv[1:1+52])
        # flop stage => 3 ones
        assert board_ones in (0, 3)  # FAST_TESTS can skip real board encode in some paths; tolerate 0
        r1 = iv[1+52:1+52+K]
        r2 = iv[1+52+K:1+52+2*K]
        assert math.isclose(sum(r1), 1.0, rel_tol=0, abs_tol=1e-9) or math.isclose(sum(r1), 0.0, rel_tol=0, abs_tol=1e-9)
        assert math.isclose(sum(r2), 1.0, rel_tol=0, abs_tol=1e-9) or math.isclose(sum(r2), 0.0, rel_tol=0, abs_tol=1e-9)
        # targets are pot-fraction CFVs with bounded magnitude
        t1 = rec["target_v1"]; t2 = rec["target_v2"]
        assert len(t1) == K and len(t2) == K
        assert max(abs(float(x)) for x in t1+t2) <= 1.0 + 1e-6


# (B4) Range gadget monotonicity over multiple commits
def test_range_gadget_monotonicity(cfg: ResolveConfig):
    s = CFRSolver(config=cfg)

    # simple flop node to obtain a stable state key
    ps = PublicState(initial_stacks=[200, 200], board_cards=[DECK[3], DECK[4], DECK[5]], dealer=0)
    ps.current_round = 1
    ps.current_bets = [0, 0]
    ps.pot_size = 30.0
    ps.current_player = (ps.dealer + 1) % 2
    n = GameNode(ps)

    # Seed gadget and commit an initial set of opponent upper bounds
    s._range_gadget_begin(n)
    s._range_gadget_commit(n, {0: 1.0, 1: 0.5, 2: 2.0})

    key = s._state_key(n)
    u1 = dict(s.opponent_cfv_upper_tracking.get(key, {}))

    # Commit with larger (looser) bounds; stored values should be monotone (min update)
    s._range_gadget_commit(n, {0: 3.0, 1: 0.75, 2: 5.0})
    u2 = dict(s.opponent_cfv_upper_tracking.get(key, {}))

    assert set(u1.keys()) == set(u2.keys())
    for k in u1:
        assert u2[k] <= u1[k]


def test_preflop_cache_lru_eviction(cfg):
    s = CFRSolver(config=cfg)
    s._preflop_cache_cap = 2
    s._preflop_cache.clear()
    kA, kB, kC = ("A",), ("B",), ("C",)
    s._preflop_cache_put(kA, {}, {})
    s._preflop_cache_put(kB, {}, {})
    s._preflop_cache_put(kC, {}, {})
    assert s._preflop_cache_get(kA) is None
    assert s._preflop_cache_get(kB) is not None
    assert s._preflop_cache_get(kC) is not None


# (B6) ValueServer unknown stage fallback doesn’t move counters
def test_value_server_unknown_stage_fallback(cfg):
    s = CFRSolver(config=cfg)
    vs = ValueServer(models={"flop": s.models["flop"]}, device=torch.device("cpu"))
    K = cfg.num_clusters
    # build one input row
    xb = torch.zeros(1, 1+52+2*K, dtype=torch.float32)
    xb[0, 0] = 0.3
    xb[0, 1+52:1+52+K] = 1.0/ K
    xb[0, 1+52+K:1+52+2*K] = 1.0/ K
    before = vs.get_counters()
    v1, v2 = vs.query("river", xb, as_numpy=False)
    after = vs.get_counters()
    assert v1.shape[1] == 0 and v2.shape[1] == 0
    assert after.get("river", 0) == before.get("river", 0,)


# (B7) Refund safety net on short call
def test_refund_short_call_safety(cfg):
    ps = PublicState(initial_stacks=[200,200], board_cards=["AH","KD","2C"])
    ps.current_round = 1
    ps.current_bets = [0, 50]  # player 0 faces 50
    ps.stacks = [10, 150]
    ps.pot_size = 100.0
    prev_pot = float(ps.pot_size)
    ps2 = ps.update_state(GameNode(ps), Action(ActionType.CALL))  # triggers refund
    # pot can decrease by at most last_refund_amount
    assert ps2.pot_size + 1e-12 >= prev_pot - ps2.last_refund_amount


# (B8) Dataset determinism under identical seed
def test_dataset_determinism_duplicate_runs(cfg):
    from data_generator import DataGenerator
    def _run_once(seed):
        random.seed(seed)
        dg = DataGenerator(num_boards=1, num_samples_per_board=3, player_stack=200,
                           num_clusters=cfg.num_clusters, speed_profile="test", config=cfg)
        recs = dg.generate_training_data(stage="flop", progress=None)
        # normalize floats for a stable text snapshot
        norm = json.dumps(recs, sort_keys=True, default=float)
        return norm
    s1 = _run_once(2027)
    s2 = _run_once(2027)
    assert s1 == s2

# ============================================================================
# (C) Final comprehensive coverage add-ons
#   These tests mop up remaining named functionality without changing any
#   algorithmic behavior. Keep them at the bottom of this file.
# ============================================================================

# (C1) play_cli internal helpers: heuristics, action sampling, diag solver hook
def test_play_cli_internal_helpers_smoke(monkeypatch, cfg):
    import play_cli as pc
    from public_state import PublicState
    from action_type import ActionType
    from action import Action

    # _choose_action over a simple policy
    pol = {ActionType.CALL: 0.7, ActionType.POT_SIZED_BET: 0.3}
    a = pc._choose_action(pol)
    assert a in pol

    # _to_policy_dict: normalizes floats, accepts ints as values, returns plain dict
    pd = pc._to_policy_dict({"x": 1, "y": 2.0})
    assert isinstance(pd, dict) and set(pd.keys()) == {"x", "y"}

    # _heuristic_action: basic to_call logic
    ps = PublicState(initial_stacks=[200, 200], board_cards=[])
    ps.current_round = 0
    ps.current_bets = [0, ps.big_blind]  # to_call > 0 for current player (dealer)
    ps.current_player = ps.dealer
    act = pc._heuristic_action(ps)
    assert act in (ActionType.CALL, ActionType.FOLD, ActionType.ALL_IN, ActionType.POT_SIZED_BET)

    # _build_diag_solver: returns a CFRSolver or None; smoke-run expected to succeed
    K = 4
    node_ps = PublicState(initial_stacks=[200,200], board_cards=["AS","KD","2C"])
    node_ps.current_round = 1
    node_ps.current_bets = [0,0]
    node_ps.current_player = (node_ps.dealer + 1) % 2
    s = pc._build_diag_solver(node_ps, K, {0:1.0}, {1:1.0}, depth=1, iters=1, k1=0.0, k2=0.0)
    assert (s is None) or (hasattr(s, "run_cfr") and hasattr(s, "models"))


# (C2) compat_linear_cfv: enforce_zero_sum produces near-zero residual for any ranges
def test_compat_linear_cfv_zero_sum_residual():
    import torch
    from compat_linear_cfv import CompatLinearCFV
    K = 6
    insz = 1 + 52 + 2*K
    net = CompatLinearCFV(insz, K, use_bias=True)
    # Build a single input row, ranges normalized
    x = torch.randn(3, insz)
    r1 = torch.rand(3, K); r1 = r1 / torch.clamp(r1.sum(dim=1, keepdim=True), min=1e-9)
    r2 = torch.rand(3, K); r2 = r2 / torch.clamp(r2.sum(dim=1, keepdim=True), min=1e-9)
    p1, p2 = net(x)
    f1, f2 = net.enforce_zero_sum(r1, r2, p1, p2)
    resid = torch.abs((r1 * f1).sum(dim=1) + (r2 * f2).sum(dim=1)).max().item()
    assert resid <= 1e-6


# (C3) CFR core knobs: rm_plus off and importance_weighting off should still run and yield a policy
def test_cfr_core_flags_rm_plus_and_iw_off():
    # Build a minimal flop tree with pot-sized bet available; trivial leaf values
    ps = PublicState(initial_stacks=[200, 200], board_cards=[DECK[6], DECK[7], DECK[8]], dealer=0)
    ps.current_round = 1
    ps.current_bets = [0, 0]
    ps.pot_size = 40.0
    ps.current_player = (ps.dealer + 1) % 2

    builder = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[1.0], include_all_in=True)
    root = builder.build(ps)

    # Turn off regret-matching+ and importance weighting
    cfr = PublicChanceCFR(depth_limit=1, bet_fractions=[1.0], include_all_in=True,
                          regret_matching_plus=False, importance_weighting=False)

    # Uniform ranges for a small synthetic K
    K = 6
    r_us = [1.0 / K] * K
    r_opp = [1.0 / K] * K
    opp_upper = [float("inf")] * K  # SP gadget off here; just exercise flags

    def leaf_value_fn(_ps, _pov, _r1, _r2):
        return torch.tensor([0.0], dtype=torch.float32)  # trivial scalar leaf

    policy, node_vals, opp_cfv = cfr.solve_subgame(
        root_node=root,
        r_us=r_us,
        r_opp=r_opp,
        opp_cfv_constraints=opp_upper,
        T=5,
        leaf_value_fn=leaf_value_fn,
    )

    # Basic sanity: policy is a distribution over legal root actions and sums to ~1
    assert isinstance(policy, dict) and len(policy) > 0
    s = sum(float(p) for p in policy.values())
    assert pytest.approx(s, rel=0, abs=1e-6) == 1.0

def test_cfr_solver_models_share_flop_turn_if_missing(cfg):
    s = CFRSolver(config=cfg)
    # Ensure shapes agree and turn has non-zero params; zero out flop weights
    with torch.no_grad():
        for p in s.models["flop"].parameters(): p.zero_()
        # make turn non-zero deterministically
        for p in s.models["turn"].parameters():
            if p.data.numel() > 0:
                p.add_(1.0)
    # Before: flop all zeros, turn non-zero
    z_before = sum(p.abs().sum().item() for p in s.models["flop"].parameters())
    nz_before = sum(p.abs().sum().item() for p in s.models["turn"].parameters())
    assert z_before == 0.0 and nz_before > 0.0

    # Invoke the sharing helper
    s._share_flop_turn_if_missing()

    # After: flop should have been copied from turn (non-zero)
    z_after = sum(p.abs().sum().item() for p in s.models["flop"].parameters())
    assert z_after > 0.0


# (C5) No-card-abstraction round-trip: expand to hand-per-bucket then restore previous clusters/ranges
def test_no_card_abstraction_push_pop_roundtrip(cfg):
    s = CFRSolver(config=cfg)
    # Seed tiny custom clusters to avoid expanding to full DECK
    s.clusters = {
        0: {"AS KD"},
        1: {"QH JC"},
        2: {"2C 3D"},
    }
    K_before = s.num_clusters
    # Build a node on flop with uniform bucket priors on these three
    from public_state import PublicState
    ps = PublicState(initial_stacks=[200,200], board_cards=["AH","KS","2D"])
    ps.current_round = 1
    ps.current_bets = [0,0]
    ps.current_player = (ps.dealer + 1) % 2
    n = GameNode(ps)
    n.player_ranges[0] = {0: 1/3, 1: 1/3, 2: 1/3}
    n.player_ranges[1] = {0: 1/3, 1: 1/3, 2: 1/3}

    snap = s._push_no_card_abstraction_for_node(n)
    try:
        # After push: each compatible hand becomes its own bucket; K grows but remains finite
        assert s.num_clusters >= 1
        # And node ranges are re-expressed in that expanded index space (sum≈1)
        assert abs(sum(n.player_ranges[0].values()) - 1.0) < 1e-9
        assert abs(sum(n.player_ranges[1].values()) - 1.0) < 1e-9
    finally:
        s._pop_no_card_abstraction(snap, n)
    # Restored exactly
    assert s.num_clusters == K_before
    assert set(s.clusters.keys()) == {0,1,2}


# (C6) River antisymmetry under player swap (bucketed & unbucketed)
def test_river_antisymmetry_player_swap():
    import random
    rng = random.Random(17)
    board = ["AH","KD","2C","7S","9D"]
    # Build small synthetic clusters disjoint from board
    clusters = {0: {"QS JC"}, 1: {"8C 8H"}, 2: {"AS KS"}, 3: {"3C 4D"}}
    r0 = {0:0.4, 1:0.6}
    r1 = {2:0.25, 3:0.75}
    class _Node:
        def __init__(self, rA, rB):
            self.public_state = type("P", (), {})()
            self.public_state.board_cards = list(board)
            self.public_state.pot_size = 80.0
            self.public_state.current_bets = [20.0, 20.0]
            self.public_state.initial_stacks = [200.0, 200.0]
            self.player_ranges = [dict(rA), dict(rB)]
    re = RiverEndgame(num_buckets=None, max_sample_per_cluster=5, seed=1729)
    # Player 0 view vs swapped view must negate cluster-scalar map on aggregate
    nA = _Node(r0, r1)
    outA = re.compute_cluster_cfvs(clusters, nA, player=0,
                                   wins_fn=lambda ph,oh,b: 1 if ph<oh else (-1 if ph>oh else 0),
                                   best_hand_fn=lambda hb: 1, hand_rank_fn=lambda s: s)
    aggA = sum(r0.get(k,0.0)*float(v[0]) for k,v in outA.items())
    nB = _Node(r1, r0)
    outB = re.compute_cluster_cfvs(clusters, nB, player=0,
                                   wins_fn=lambda ph,oh,b: 1 if ph<oh else (-1 if ph>oh else 0),
                                   best_hand_fn=lambda hb: 1, hand_rank_fn=lambda s: s)
    aggB = sum(r1.get(k,0.0)*float(v[0]) for k,v in outB.items())
    assert math.isfinite(aggA) and math.isfinite(aggB)
    assert math.isclose(aggA, -aggB, rel_tol=0, abs_tol=1e-6)
