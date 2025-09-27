"""PublicChanceCFR smoke tests (small, deterministic)."""

import hunl.engine.poker_utils as poker_utils
import numpy as np
import pytest

from hunl.solving.cfr_core import PublicChanceCFR
from hunl.solving.lookahead_tree import LookaheadTreeBuilder
from hunl.engine.public_state import PublicState
from hunl.engine.poker_utils import DECK



def _leaf_const(val):
	"""Return a leaf value fn for CFR that ignores inputs and always yields np.array([val], dtype=float)."""
	def fn(ps, pov, r1, r2):
		return np.array([val], dtype=float)
	return fn


def test_public_chance_cfr_root_policy_uniform_when_no_learning_signal():
	"""test_public_chance_cfr_root_policy_uniform_when_no_learning_signal:
Builds a one-step flop tree with symmetric ranges and a constant zero leaf.
With regret matching plus and importance weighting on, there is no learning
signal, so the root strategy should be a proper probability distribution and
uniform across the available actions."""
	deck = list(DECK)
	board = deck[:3]
	ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
	ps.current_round = 1
	ps.current_bets = [10, 0]  
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
	s = sum(pol.values())
	assert pytest.approx(s, rel=1e-9) == 1.0
	u = list(pol.values())[0]
	assert all(pytest.approx(v, rel=1e-6) == u for v in pol.values())


def test_public_chance_cfr_root_gadget_terminate_applies_for_opp_root():
	"""test_public_chance_cfr_root_gadget_terminate_applies_for_opp_root:
Uses a root where the opponent acts and supplies large opponent CFV upper
bounds. Verifies the “terminate” gadget (safety constraint) is applied and
the returned opponent CFV map reflects those upper bounds."""
	deck = list(DECK)
	board = deck[:3]
	ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
	ps.current_round = 1
	ps.current_bets = [0, 0]
	ps.last_raiser = None
	ps.current_player = (ps.dealer + 1) % 2  
	builder = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=True)
	root = builder.build(ps)

	K = 4
	cfr = PublicChanceCFR(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=True)
	w_constraints = [10.0] * K  
	pol, node_values, opp_cfv = cfr.solve_subgame(
		root_node=root,
		r_us=[1.0/K]*K,
		r_opp=[1.0/K]*K,
		opp_cfv_constraints=w_constraints,
		T=2,
		leaf_value_fn=_leaf_const(0.0)
	)
	assert opp_cfv == {i: float(w_constraints[i]) for i in range(K)}


def test_public_chance_cfr_set_warm_start_no_crash():
	"""test_public_chance_cfr_set_warm_start_no_crash:
Exercises set_warm_start with a benign payload and runs a tiny solve to
confirm the API tolerates warm starts and yields a policy dictionary."""
	deck = list(DECK)
	board = deck[:3]
	ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
	ps.current_round = 1
	builder = LookaheadTreeBuilder(depth_limit=1)
	root = builder.build(ps)

	cfr = PublicChanceCFR(depth_limit=1)
	cfr.set_warm_start({("dummy",): [0.5, 0.5]})  
	pol, node_values, opp_cfv = cfr.solve_subgame(
		root_node=root,
		r_us=[0.5, 0.5],
		r_opp=[0.5, 0.5],
		opp_cfv_constraints=[0.0, 0.0],
		T=1,
		leaf_value_fn=_leaf_const(0.0)
	)
	assert isinstance(pol, dict)

