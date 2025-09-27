"""
Test suite for CLI interoperability, CFR solver knobs, model sharing, no-card-abstraction roundtrip, 
	 river antisymmetry, and zero-sum compatibility checks across components.
"""

import math
import pytest
import torch

import hunl.cli.eval_cli as eval_cli
import hunl.nets.model_io as model_io
import hunl.solving.cfr_solver as cfr_solver
import hunl.engine.public_state as public_state
import hunl.engine.game_node as game_node
import hunl.engine.poker_utils as poker_utils
from hunl.endgame.river_endgame import RiverEndgame
from hunl.resolve_config import ResolveConfig
from hunl.solving.cfr_solver import CFRSolver
from hunl.engine.game_node import GameNode


@pytest.fixture
def cfg():
	"""
	Provide a fast ResolveConfig suitable for smoke-level integration tests by enabling 
	test profile with small cluster count and iteration limits.
	"""
	return ResolveConfig.from_env({
	 "profile": "test",
	 "num_clusters": 6,
	 "depth_limit": 1,
	 "total_iterations": 1,
	})


def test_play_cli_internal_helpers_smoke(monkeypatch, cfg):
	"""
	Smoke-test core helpers in play_cli by exercising action selection, policy normalization,
	heuristic action under to_call logic, and diagnostic solver construction on a minimal public state.
	"""
	import hunl.cli.play_cli as pc
	from hunl.engine.public_state import PublicState
	from hunl.engine.action_type import ActionType
	from hunl.engine.action import Action
	pol = {ActionType.CALL: 0.7, ActionType.POT_SIZED_BET: 0.3}
	a = pc._choose_action(pol)
	assert a in pol
	pd = pc._to_policy_dict({"x": 1, "y": 2.0})
	assert isinstance(pd, dict) and set(pd.keys()) == {"x", "y"}
	ps = PublicState(initial_stacks=[200, 200], board_cards=[])
	ps.current_round = 0
	ps.current_bets = [0, ps.big_blind]
	ps.current_player = ps.dealer
	act = pc._heuristic_action(ps)
	assert act in (ActionType.CALL, ActionType.FOLD, ActionType.ALL_IN, ActionType.POT_SIZED_BET)
	K = 4
	node_ps = PublicState(initial_stacks=[200,200], board_cards=["AS","KD","2C"])
	node_ps.current_round = 1
	node_ps.current_bets = [0,0]
	node_ps.current_player = (node_ps.dealer + 1) % 2
	s = pc._build_diag_solver(node_ps, K, {0:1.0}, {1:1.0}, depth=1, iters=1, k1=0.0, k2=0.0)
	assert (s is None) or (hasattr(s, "run_cfr") and hasattr(s, "models"))


def test_compat_linear_cfv_zero_sum_residual():
	"""
	Validate that CompatLinearCFV.enforce_zero_sum reduces range-weighted expectations to 
	numerically zero for arbitrary inputs and normalized ranges.
	"""
	import torch
	from hunl.nets.compat_linear_cfv import CompatLinearCFV
	K = 6
	insz = 1 + 52 + 2*K
	net = CompatLinearCFV(insz, K, use_bias=True)
	x = torch.randn(3, insz)
	r1 = torch.rand(3, K); r1 = r1 / torch.clamp(r1.sum(dim=1, keepdim=True), min=1e-9)
	r2 = torch.rand(3, K); r2 = r2 / torch.clamp(r2.sum(dim=1, keepdim=True), min=1e-9)
	p1, p2 = net(x)
	f1, f2 = net.enforce_zero_sum(r1, r2, p1, p2)
	resid = torch.abs((r1 * f1).sum(dim=1) + (r2 * f2).sum(dim=1)).max().item()
	assert resid <= 1e-6


def test_cfr_core_flags_rm_plus_and_iw_off():
	"""
	Ensure PublicChanceCFR runs and produces a valid root policy 
	distribution when regret-matching-plus and importance weighting are disabled.
	"""
	import numpy as np
	from hunl.solving.cfr_core import PublicChanceCFR
	from hunl.solving.lookahead_tree import LookaheadTreeBuilder
	from hunl.engine.public_state import PublicState
	from hunl.engine.poker_utils import DECK
	board = list(DECK[:3])
	ps = PublicState(initial_stacks=[200, 200], board_cards=board)
	ps.current_round = 1
	ps.current_bets = [10, 0]
	ps.last_raiser = 0
	ps.current_player = 1
	root = LookaheadTreeBuilder(
	 depth_limit=1, bet_fractions=[1.0], include_all_in=True
	).build(ps)
	def _leaf(ps_, pov, r1, r2):
		"""
		Provide deterministic zero leaf values to complete the subgame 
		definition for the CFR solver in this configuration test.
		"""
		return np.array([0.0], dtype=float)
	cfr = PublicChanceCFR(
	 depth_limit=1,
	 bet_fractions=[1.0],
	 include_all_in=True,
	 regret_matching_plus=False,
	 importance_weighting=False,
	)
	pol, node_vals, opp = cfr.solve_subgame(
	 root,
	 r_us=[0.5, 0.5],
	 r_opp=[0.5, 0.5],
	 opp_cfv_constraints=[0.0, 0.0],
	 T=2,
	 leaf_value_fn=_leaf,
	)
	s = sum(pol.values()) if pol else 0.0
	assert pol and abs(s - 1.0) < 1e-9


def test_cfr_solver_models_share_flop_turn_if_missing(cfg):
	"""
	Check that CFRSolver weight sharing populates missing or zeroed flop parameters 
	from nonzero turn parameters, resulting in nonzero flop weights after sharing.
	"""
	s = CFRSolver(config=cfg)
	with torch.no_grad():
		for p in s.models["flop"].parameters():
			p.zero_()
		for p in s.models["turn"].parameters():
			if p.data.numel() > 0:
				p.add_(1.0)
	z_before = sum(p.abs().sum().item() for p in s.models["flop"].parameters())
	nz_before = sum(p.abs().sum().item() for p in s.models["turn"].parameters())
	assert z_before == 0.0 and nz_before > 0.0
	s._share_flop_turn_if_missing()
	z_after = sum(p.abs().sum().item() for p in s.models["flop"].parameters())
	assert z_after > 0.0


def test_no_card_abstraction_push_pop_roundtrip(cfg):
	"""
	Verify that pushing no-card-abstraction expands clusters to individual
	hands and re-expresses node ranges, and that popping fully restores original cluster configuration and solver state.
	"""
	s = CFRSolver(config=cfg)
	s.clusters = {
	 0: {"AS KD"},
	 1: {"QH JC"},
	 2: {"2C 3D"},
	}
	K_before = s.num_clusters
	from hunl.engine.public_state import PublicState
	ps = PublicState(initial_stacks=[200,200], board_cards=["AH","KS","2D"])
	ps.current_round = 1
	ps.current_bets = [0,0]
	ps.current_player = (ps.dealer + 1) % 2
	n = GameNode(ps)
	n.player_ranges[0] = {0: 1/3, 1: 1/3, 2: 1/3}
	n.player_ranges[1] = {0: 1/3, 1: 1/3, 2: 1/3}
	snap = s._push_full_hand_expansion(n)
	try:
		assert s.num_clusters >= 1
		assert abs(sum(n.player_ranges[0].values()) - 1.0) < 1e-9
		assert abs(sum(n.player_ranges[1].values()) - 1.0) < 1e-9
	finally:
		s._pop_full_hand_expansion(snap, n)
	assert s.num_clusters == K_before
	assert set(s.clusters.keys()) == {0,1,2}


def test_river_antisymmetry_player_swap():
	"""
	Demonstrate river-endgame antisymmetry by showing that swapping 
	players and ranges on identical states negates the aggregate pot-fraction EV when evaluated from each player’s perspective.
	"""
	import random
	rng = random.Random(17)
	board = ["AH","KD","2C","7S","9D"]
	clusters = {0: {"QS JC"}, 1: {"8C 8H"}, 2: {"AS KS"}, 3: {"3C 4D"}}
	r0 = {0:0.4, 1:0.6}
	r1 = {2:0.25, 3:0.75}
	class _Node:
		"""
		Provide a minimal container with a public_state and player_ranges to drive
		RiverEndgame.compute_cluster_cfvs for antisymmetry evaluation.
		"""
		def __init__(self, rA, rB):
			"""
			Initialize public state fields and attach player range dictionaries for both players.
			"""
			self.public_state = type("P", (), {})()
			self.public_state.board_cards = list(board)
			self.public_state.pot_size = 80.0
			self.public_state.current_bets = [20.0, 20.0]
			self.public_state.initial_stacks = [200.0, 200.0]
			self.player_ranges = [dict(rA), dict(rB)]
	re = RiverEndgame(num_buckets=None, max_sample_per_cluster=5, seed=1729)
	nA = _Node(r0, r1)
	outA = re.compute_cluster_cfvs(
	 clusters,
	 nA,
	 player=0,
	 wins_fn=lambda ph,oh,b: 1 if ph<oh else (-1 if ph>oh else 0),
	 best_hand_fn=lambda hb: 1,
	 hand_rank_fn=lambda s: s
	)
	aggA = sum(r0.get(k,0.0)*float(v[0]) for k,v in outA.items())
	nB = _Node(r1, r0)
	outB = re.compute_cluster_cfvs(
	 clusters,
	 nB,
	 player=0,
	 wins_fn=lambda ph,oh,b: 1 if ph<oh else (-1 if ph>oh else 0),
	 best_hand_fn=lambda hb: 1,
	 hand_rank_fn=lambda s: s
	)
	aggB = sum(r1.get(k,0.0)*float(v[0]) for k,v in outB.items())
	assert math.isfinite(aggA) and math.isfinite(aggB)
	assert math.isclose(aggA, -aggB, rel_tol=0, abs_tol=1e-6)

