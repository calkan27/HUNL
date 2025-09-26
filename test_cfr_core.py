import numpy as np
import pytest

from cfr_core import PublicChanceCFR
from lookahead_tree import LookaheadTreeBuilder
from public_state import PublicState
from poker_utils import DECK


def _leaf_const(val):
    def fn(ps, pov, r1, r2):
        # return scalar leaf (pot-fraction CFV); traversal will take it as-is
        return np.array([val], dtype=float)
    return fn


def test_public_chance_cfr_root_policy_uniform_when_no_learning_signal():
    # Build a trivial tree at flop with depth 1 and equal actions; regret starts at zeros -> uniform
    deck = list(DECK)
    board = deck[:3]
    ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
    ps.current_round = 1
    ps.current_bets = [10, 0]  # betting node
    ps.last_raiser = 0
    ps.current_player = 1
    builder = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=False)
    root = builder.build(ps)

    cfr = PublicChanceCFR(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=False,
                          regret_matching_plus=True, importance_weighting=True)
    pol, node_values, opp_cfv = cfr.solve_subgame(
        root_node=root,
        r_us=[0.5, 0.5],
        r_opp=[0.5, 0.5],
        opp_cfv_constraints=[0.0, 0.0],
        T=1,
        leaf_value_fn=_leaf_const(0.0)
    )
    # If no differential reward across actions, root policy uniform over offered actions
    s = sum(pol.values())
    assert pytest.approx(s, rel=1e-9) == 1.0
    u = list(pol.values())[0]
    assert all(pytest.approx(v, rel=1e-6) == u for v in pol.values())


def test_public_chance_cfr_root_gadget_terminate_applies_for_opp_root():
    # Make a state where root is opponent (current_player != dealer) and give very negative
    # termination (for hero) so solver will consider it. We only verify cfr runs and
    # respects given opp_cfv_constraints passthrough.
    deck = list(DECK)
    board = deck[:3]
    ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
    ps.current_round = 1
    ps.current_bets = [0, 0]
    ps.last_raiser = None
    ps.current_player = (ps.dealer + 1) % 2  # opp at root
    builder = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=True)
    root = builder.build(ps)

    K = 4
    cfr = PublicChanceCFR(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=True)
    w_constraints = [10.0] * K  # large positive opp CFV upper bounds -> strong terminate option for opp root gadget (value for hero = -sum(r_opp*w))
    pol, node_values, opp_cfv = cfr.solve_subgame(
        root_node=root,
        r_us=[1.0/K]*K,
        r_opp=[1.0/K]*K,
        opp_cfv_constraints=w_constraints,
        T=2,
        leaf_value_fn=_leaf_const(0.0)
    )
    # Solver returns opp_cfv vector echo; no crash and dictionary filled
    assert opp_cfv == {i: float(w_constraints[i]) for i in range(K)}


def test_public_chance_cfr_set_warm_start_no_crash():
    deck = list(DECK)
    board = deck[:3]
    ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
    ps.current_round = 1
    builder = LookaheadTreeBuilder(depth_limit=1)
    root = builder.build(ps)

    cfr = PublicChanceCFR(depth_limit=1)
    cfr.set_warm_start({("dummy",): [0.5, 0.5]})  # not used, but should not crash
    pol, node_values, opp_cfv = cfr.solve_subgame(
        root_node=root,
        r_us=[0.5, 0.5],
        r_opp=[0.5, 0.5],
        opp_cfv_constraints=[0.0, 0.0],
        T=1,
        leaf_value_fn=_leaf_const(0.0)
    )
    assert isinstance(pol, dict)

