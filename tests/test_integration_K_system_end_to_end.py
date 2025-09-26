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
    import numpy as np
    from cfr_core import PublicChanceCFR
    from lookahead_tree import LookaheadTreeBuilder
    from public_state import PublicState

    # flop root, 2 actions; trivial leaves
    deck = __import__("poker_utils").poker_utils.DECK
    board = list(deck[:3])
    ps = PublicState(initial_stacks=[200,200], board_cards=board)
    ps.current_round = 1
    ps.current_bets = [10, 0]
    ps.last_raiser = 0
    ps.current_player = 1
    root = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[1.0], include_all_in=True).build(ps)

    def _leaf(ps_, pov, r1, r2):
        return np.array([0.0], dtype=float)

    cfr = PublicChanceCFR(depth_limit=1, bet_fractions=[1.0], include_all_in=True,
                          regret_matching_plus=False, importance_weighting=False)
    pol, node_vals, opp = cfr.solve_subgame(root, r_us=[0.5,0.5], r_opp=[0.5,0.5],
                                            opp_cfv_constraints=[0.0,0.0], T=2, leaf_value_fn=_leaf)
    s = sum(pol.values()) if pol else 0.0
    assert pol and abs(s - 1.0) < 1e-9


# (C4) Model sharing fallback: when one stage is zero-initialized, share weights across flop/turn
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
    # Seed tiny custom clusters to avoid expanding to full deck
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

