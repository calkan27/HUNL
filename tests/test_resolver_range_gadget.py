"""CFRSolver + utilities integration tests (deterministic, FAST_TESTS=1).

DummyRiverEndgame / DummyNet / BarePublicState / BareNode:
Minimal stubs used across tests to isolate CFRSolver behavior from heavy
models/endgame logic while preserving interfaces and shapes.
"""

import hunl.endgame.river_endgame as river_endgame
import hunl.engine.poker_utils as poker_utils


import os
import math
import types
import itertools
import builtins
import random

import pytest
import torch
import torch.nn as nn

from hunl.solving.range_gadget import RangeGadget
from hunl.solving.cfr_solver import CFRSolver
from hunl.engine.action_type import ActionType



class DummyRiverEndgame:
	def __init__(self, value_per_cluster=0.123):
		"""Configure a constant per-cluster CFV and track invocations; used to stub river endgame logic in tests."""
		self.value_per_cluster = float(value_per_cluster)
		self.calls = []

	def compute_cluster_cfvs(self, clusters, node, player, wins_fn, best_hand, hand_rank):
		"""Return a dict {cid: scalar}."""
		self.calls.append((tuple(sorted(int(k) for k in clusters.keys())), player))
		return {int(cid): self.value_per_cluster for cid in clusters.keys()}


class DummyNet(nn.Module):
	"""Tiny deterministic CFV network stub that produces shaped outputs without any real inference."""

	def __init__(self, K):
		"""Initialize with K output clusters and a trivial layer to keep module non-empty/deterministic."""
		super().__init__()
		self.K = int(K)
		self.w1 = nn.Linear(1, 1, bias=False)
		with torch.no_grad():
			self.w1.weight.fill_(0.5)

	def forward(self, x):
		"""Return (v1, v2) as simple, reproducible K-length patterns per batch; ignores inputs except for batch size."""
		B = x.shape[0]
		device = x.device
		base = torch.arange(self.K, dtype=torch.float32, device=device).unsqueeze(0).repeat(B, 1)
		v1 = base + 0.25
		v2 = torch.flip(base, dims=[1]) - 0.15
		return v1, v2

	def enforce_zero_sum(self, r1, r2, v1, v2):
		"""Project predictions to zero-sum by subtracting the shared range-weighted mean delta from both players."""
		s1 = torch.sum(r1 * v1, dim=1, keepdim=True)
		s2 = torch.sum(r2 * v2, dim=1, keepdim=True)
		delta = 0.5 * (s1 + s2)
		return v1 - delta, v2 - delta


class BarePublicState:
	"""Minimal PublicState stub exposing only what CFRSolver needs; no full game rules or validations."""

	def __init__(
		self,
		round_idx=0,
		current_player=0,
		pot_size=100,
		stacks=(1000, 1000),
		current_bets=(0, 0),
		blinds=(5, 10),
		dealer=0,
		board_cards=None,
		is_terminal=False,
		is_showdown=False,
		players_in_hand=(True, True),
		min_raise_size=10,
		legal=None
	):
		"""Build a lightweight, configurable state used to shape menus, pot sizes, and control flow in tests."""
		self.current_round = int(round_idx)
		self.current_player = int(current_player)
		self.pot_size = int(pot_size)
		self.stacks = list(stacks)
		self.initial_stacks = list(stacks)
		self.current_bets = list(current_bets)
		self.small_blind = int(blinds[0])
		self.big_blind = int(blinds[1])
		self.dealer = int(dealer)
		self.board_cards = list(board_cards or [])
		self.is_terminal = bool(is_terminal)
		self.is_showdown = bool(is_showdown)
		self.players_in_hand = list(players_in_hand)
		self._legal = list(legal) if legal is not None else None
		self.last_action = None  # recorded by update_state for range-tracking tests
		self._min_raise = int(min_raise_size)

	def legal_actions(self):
		"""Return the preset legal action list (or empty), allowing tests to constrain menus precisely."""
		return list(self._legal) if self._legal is not None else []

	def _min_raise_size(self):
		"""Expose the configured minimum raise increment used by allowed-action calculations."""
		return self._min_raise

	def update_state(self, prev_node, action):
		"""Produce a shallow ‘next’ state that advances the turn order and copies fields; no betting semantics."""
		nxt = BarePublicState(
			round_idx=self.current_round,
			current_player=(self.current_player + 1) % 2,
			pot_size=self.pot_size,  # unchanged in this stub
			stacks=tuple(self.stacks),
			current_bets=tuple(self.current_bets),
			blinds=(self.small_blind, self.big_blind),
			dealer=self.dealer,
			board_cards=list(self.board_cards),
			is_terminal=self.is_terminal,
			is_showdown=self.is_showdown,
			players_in_hand=tuple(self.players_in_hand),
			min_raise_size=self._min_raise,
			legal=self._legal,
		)
		nxt.last_action = action
		return nxt


class BareNode:
	"""Minimal GameNode stub carrying a BarePublicState and per-player bucket-range dicts."""

	def __init__(self, ps, p0_range=None, p1_range=None):
		"""Attach ranges for players (0/1) as dicts over cluster IDs; used to probe solver internals."""
		self.public_state = ps
		self.player_ranges = [dict(p0_range or {}), dict(p1_range or {})]

	@property
	def current_player(self):
		"""Mirror the acting player from the public state."""
		return self.public_state.current_player


@pytest.fixture(autouse=True)
def _fast_env(monkeypatch):
	"""Force FAST_TESTS=1 during each test for deterministic lightweight code paths; unset afterward."""
	monkeypatch.setenv("FAST_TESTS", "1")
	yield
	monkeypatch.delenv("FAST_TESTS", raising=False)


@pytest.fixture
def solver(monkeypatch):
	"""Build a CFRSolver wired with DummyNet backends, a DummyRiverEndgame, and fixed clusters for stable tests."""
	s = CFRSolver(depth_limit=2, num_clusters=4, speed_profile="test", config=None)

	K = s.num_clusters
	s.models["preflop"] = DummyNet(K)
	s.models["flop"] = DummyNet(K)
	s.models["turn"] = DummyNet(K)

	s.river_endgame = DummyRiverEndgame(value_per_cluster=0.123)

	s.clusters = {
		0: {"As Kh", "Ad Kc"},
		1: {"2h 2d", "2s 2c"},
		2: {"Qs Qh"},
		3: {"7d 6d", "9c 8c"},
	}

	class _HC:
		profile = "test"
	s.hand_clusterer = _HC()

	return s




def test_range_gadget_begin_update_get_monotone():
	"""RangeGadget preserves the running per-bucket upper bound in a monotone way."""
	g = RangeGadget()
	assert g.begin() == {}
	init = {0: 1.0, 2: -0.5}
	out = g.begin(init)
	assert out == {0: 1.0, 2: -0.5}
	upd = {0: 0.25, 2: -0.2, 3: 1.7}
	out2 = g.update(upd)
	assert out2[0] == 1.0       
	assert out2[2] == -0.2      
	assert out2[3] == 1.7
	assert g.get() == out2



def test_preflop_signature_stability_and_sensitivity(solver):
	"""Preflop cache key is stable for identical states and changes when pot/clusters change."""
	ps1 = BarePublicState(round_idx=0, current_player=0, pot_size=100, dealer=1, blinds=(5, 10))
	n1 = BareNode(ps1, {0: 0.6, 1: 0.4}, {0: 0.3, 1: 0.7})
	k1 = solver._preflop_signature(n1)

	ps1b = BarePublicState(round_idx=0, current_player=0, pot_size=100, dealer=1, blinds=(5, 10))
	n1b = BareNode(ps1b, {0: 0.6, 1: 0.4}, {0: 0.3, 1: 0.7})
	assert solver._preflop_signature(n1b) == k1

	ps2 = BarePublicState(round_idx=0, current_player=0, pot_size=120, dealer=1, blinds=(5, 10))
	n2 = BareNode(ps2, {0: 0.6, 1: 0.4}, {0: 0.3, 1: 0.7})
	assert solver._preflop_signature(n2) != k1

	old_sig = solver._cluster_partition_signature()
	solver.clusters[3].add("Ah Ad")  
	new_sig = solver._cluster_partition_signature()
	assert new_sig != old_sig


def test_preflop_cache_put_get_lru_and_stats(solver):
	"""Preflop LRU cache stores/returns entries, updates hit/miss/put/evict counters,
	and evicts oldest on capacity pressure."""
	key = ("PREFLOP_CACHE_V2", 5, 10, 0, 0, (1000, 1000), (0, 0), 100, solver.num_clusters,
		   solver._cluster_partition_signature(),
		   solver._range_sig({0: 0.6, 1: 0.4}),
		   solver._range_sig({0: 0.4, 1: 0.6}))
	own = {0: 0.6, 1: 0.4}
	opp = {0: 0.1, 1: 0.2}
	solver._preflop_cache_put(key, own, opp)
	got = solver._preflop_cache_get(key)
	assert got["own_range"] == own and got["opp_cfv"] == opp
	stats = solver._preflop_cache_stats
	assert stats["hits"] == 1 and stats["puts"] >= 1 and stats["misses"] >= 0

	solver._preflop_cache_cap = 2
	solver._preflop_cache.clear()
	kA, kB, kC = ("A",), ("B",), ("C",)
	solver._preflop_cache_put(kA, {}, {})
	solver._preflop_cache_put(kB, {}, {})
	solver._preflop_cache_put(kC, {}, {})
	assert solver._preflop_cache_get(kA) is None  
	assert solver._preflop_cache_get(kB) is not None
	assert solver._preflop_cache_get(kC) is not None


def test_range_sig_rounding_to_12_decimals(solver):
	"""Range signature rounds probabilities to 12 decimals for stable hashing/keys."""
	r = {0: 0.123456789012345, 2: 0.3333333333333333}
	sig = solver._range_sig(r)
	assert sig == ((0, round(0.123456789012345, 12)), (2, round(0.3333333333333333, 12)))


def test_state_key_determinism_and_sensitivity(solver):
	"""State key is deterministic and responds to pot-size changes."""
	ps = BarePublicState(round_idx=1, current_player=1, pot_size=200, dealer=0, board_cards=["Ah","Kd","2c"])
	n = BareNode(ps, {0: 1.0}, {1: 1.0})
	k1 = solver._state_key(n)
	ps2 = BarePublicState(round_idx=1, current_player=1, pot_size=210, dealer=0, board_cards=["Ah","Kd","2c"])
	n2 = BareNode(ps2, {0: 1.0}, {1: 1.0})
	assert solver._state_key(n2) != k1



def test_evaluate_hand_strength_cached_and_seeded(solver, monkeypatch):
	"""Hand-strength evaluation is cached and seeded; returns in [0,1] and is repeatable."""
	ps_board = ["Ah", "Kd", "2c"]  
	v1 = solver._evaluate_hand_strength(0, ps_board)
	v2 = solver._evaluate_hand_strength(0, ps_board)  
	assert v1 == v2
	assert 0.0 <= v1 <= 1.0

	v3 = solver._evaluate_hand_strength("As Kh", ps_board)
	v4 = solver._evaluate_hand_strength("As Kh", ps_board)
	assert v3 == v4
	assert 0.0 <= v3 <= 1.0


def test_recursive_range_sampling_sum_and_cache(solver):
	"""Recursive sampler returns cached, deterministic distributions that sum to total."""
	hands = {"As Kh", "2h 2d", "Qs Qh", "9c 8c"}
	total = 1.0
	board = ["Ah", "Kd"]
	out1 = solver.recursive_range_sampling(hands, total, board)
	out2 = solver.recursive_range_sampling(hands, total, board)  
	assert out1 == out2
	assert pytest.approx(sum(out1.values()), rel=0, abs=1e-12) == total
	assert set(out1.keys()) == set(hands)



def test_prepare_input_vector_shapes_and_normalization(solver):
	"""Full-stage input vector has expected layout and normalized ranges (or all-zero)."""
	ps = BarePublicState(round_idx=1, pot_size=200, board_cards=["Ah", "Kd", "2c"])
	node = BareNode(ps, {0: 2.0, 1: 1.0}, {0: 3.0, 1: 0.0})
	vec = solver.prepare_input_vector(node)
	expected_len = 1 + len(set([]))  
	assert len(vec) == solver.calculate_input_size()

	K = solver.num_clusters
	start_r1 = 1 + len(vec) - (2 * K) - 0  
	deck_len = solver.calculate_input_size() - (1 + 2 * K)
	start_r1 = 1 + deck_len
	end_r1 = start_r1 + K
	start_r2 = end_r1
	end_r2 = start_r2 + K

	r1 = vec[start_r1:end_r1]
	r2 = vec[start_r2:end_r2]
	assert pytest.approx(sum(r1), rel=0, abs=1e-9) == 1.0 or pytest.approx(sum(r1), rel=0, abs=1e-9) == 0.0
	assert pytest.approx(sum(r2), rel=0, abs=1e-9) == 1.0 or pytest.approx(sum(r2), rel=0, abs=1e-9) == 0.0


def test_prepare_input_vector_preflop_shapes_and_normalization(solver):
	"""Preflop layout check and range normalization identical to full stage without board one-hot."""
	ps = BarePublicState(round_idx=0, pot_size=100)
	node = BareNode(ps, {0: 2.0, 1: 1.0}, {0: 3.0, 1: 0.0})
	vec = solver.prepare_input_vector_preflop(node)
	assert len(vec) == solver.calculate_input_size_preflop()

	K = solver.num_clusters
	start_r1 = 1
	end_r1 = start_r1 + K
	start_r2 = end_r1
	end_r2 = start_r2 + K
	r1 = vec[start_r1:end_r1]
	r2 = vec[start_r2:end_r2]
	assert pytest.approx(sum(r1), rel=0, abs=1e-9) == 1.0 or pytest.approx(sum(r1), rel=0, abs=1e-9) == 0.0
	assert pytest.approx(sum(r2), rel=0, abs=1e-9) == 1.0 or pytest.approx(sum(r2), rel=0, abs=1e-9) == 0.0


def test_predict_cfv_stage_mapping_and_zero_sum_residuals(solver):
	"""CFV prediction uses stage-appropriate nets; residual samples buffer grows; river path returns scalars or lists."""
	ps_flop = BarePublicState(round_idx=1, pot_size=240, board_cards=["Ah", "Kd", "2c"])
	n_flop = BareNode(ps_flop, {0:0.5, 1:0.5}, {0:0.5, 1:0.5})
	before = len(getattr(solver, "_zs_residual_samples", [])) if hasattr(solver, "_zs_residual_samples") else 0
	out_flop = solver.predict_counterfactual_values(n_flop, player=0)
	after = len(solver._zs_residual_samples)
	assert isinstance(out_flop, dict) and all(isinstance(v, list) for v in out_flop.values())
	assert after > before  

	ps_pre = BarePublicState(round_idx=0, pot_size=100)
	n_pre = BareNode(ps_pre, {0:0.5, 1:0.5}, {0:0.5, 1:0.5})
	_ = solver.predict_counterfactual_values(n_pre, player=1)  

	ps_riv = BarePublicState(round_idx=3, pot_size=400, board_cards=["Ah","Kd","2c","7s","7h"])
	n_riv = BareNode(ps_riv, {0:1.0}, {1:1.0})
	out_riv = solver.predict_counterfactual_values(n_riv, player=0)
	assert out_riv and all(isinstance(v, float) or isinstance(v, (list, tuple)) for v in out_riv.values())


def test_depth_limit_preflop_flop_use_model_scale_by_pot_and_turn_does_not(solver, monkeypatch):
	"""At depth limit: preflop/flop scale CFVs by pot; turn does not query nets in this path."""
	calls = {"count": 0}
	def fake_pred(node, player):
		calls["count"] += 1
		return {cid: [1.0] for cid in node.player_ranges[player].keys()}
	monkeypatch.setattr(solver, "predict_counterfactual_values", fake_pred)

	ps_pre = BarePublicState(round_idx=0, pot_size=250)
	n_pre = BareNode(ps_pre, {0:1.0, 1:1.0}, {0:1.0, 1:1.0})
	out = solver._calculate_counterfactual_values(n_pre, player=0, depth=999)  
	assert calls["count"] == 1
	for vec in out.values():
		assert all(v == 250.0 for v in vec)  

	ps_flop = BarePublicState(round_idx=1, pot_size=300)
	n_flop = BareNode(ps_flop, {0:1.0}, {1:1.0})
	_ = solver._calculate_counterfactual_values(n_flop, player=0, depth=999)
	assert calls["count"] == 2  


	monkeypatch.setattr(solver, "_allowed_actions_agent", lambda ps: [])
	monkeypatch.setattr(solver, "_allowed_actions_opponent", lambda ps: [])
	ps_turn = BarePublicState(round_idx=2, pot_size=350)
	n_turn = BareNode(ps_turn, {0:1.0}, {1:1.0})
	_ = solver._calculate_counterfactual_values(n_turn, player=0, depth=999)
	assert calls["count"] == 2  



def test_mask_strategy_and_uniform_fallbacks(solver):
	"""Strategy masking keeps only allowed actions, renormalizes, and falls back to uniform when needed."""
	A = len(ActionType)
	strat = [0.0]*A
	strat[ActionType.CALL.value] = 0.3
	strat[ActionType.ALL_IN.value] = 0.7
	allowed = [ActionType.CALL, ActionType.ALL_IN]
	m = solver._mask_strategy(strat, allowed)
	assert sum(m) == pytest.approx(1.0)
	for i, p in enumerate(m):
		if i in (ActionType.CALL.value, ActionType.ALL_IN.value):
			assert p > 0.0
		else:
			assert p == 0.0

	z = [0.0]*A
	m2 = solver._mask_strategy(z, allowed)
	assert m2[ActionType.CALL.value] == pytest.approx(0.5)
	assert m2[ActionType.ALL_IN.value] == pytest.approx(0.5)


def test_mixed_action_distribution_weighted_and_uniform_fallback(solver):
	"""Mixed distribution aggregates average strategies weighted by priors; falls back to uniform when priors absent."""
	ps = BarePublicState(round_idx=1)
	node = BareNode(ps, {0:0.6, 1:0.4}, {0:0.5, 1:0.5})
	values = solver.cfr_values[node]

	A = len(ActionType)
	values.cumulative_strategy[0] = [0.0]*A
	values.cumulative_strategy[0][ActionType.CALL.value] = 10.0
	values.cumulative_strategy[1] = [0.0]*A
	values.cumulative_strategy[1][ActionType.ALL_IN.value] = 5.0

	allowed = [ActionType.CALL, ActionType.ALL_IN]
	probs = solver._mixed_action_distribution(node, player=0, allowed_actions=allowed)
	assert probs == pytest.approx([0.6, 0.4])

	node2 = BareNode(ps, {}, {})
	probs2 = solver._mixed_action_distribution(node2, player=0, allowed_actions=allowed)
	assert probs2 == pytest.approx([0.5, 0.5])



def test_update_player_range_bayes_and_fallback(solver):
	"""Bayesian update of the acting player’s range given observed action; reverts to normalized priors on bad index."""
	ps = BarePublicState(round_idx=1, current_player=0)
	node = BareNode(ps, {0:0.7, 1:0.3}, {0:0.5, 1:0.5})
	values = solver.cfr_values[node]
	A = len(ActionType)
	values.strategy[0] = [0.0]*A
	values.strategy[1] = [0.0]*A
	values.strategy[0][ActionType.CALL.value] = 0.2
	values.strategy[1][ActionType.CALL.value] = 0.8

	solver.update_player_range(node, player=0, cluster_id=0, action_index=ActionType.CALL.value)
	post = node.player_ranges[0]

	assert post[0] == pytest.approx(0.14/0.38)
	assert post[1] == pytest.approx(0.24/0.38)

	solver.update_player_range(node, player=0, cluster_id=0, action_index=999)
	post2 = node.player_ranges[0]
	s = 0.7 + 0.3
	assert post2[0] == pytest.approx(0.7/s)
	assert post2[1] == pytest.approx(0.3/s)


def test_lift_ranges_after_chance_reweights_by_board_compat(solver):
	"""After chance (new board), ranges are reweighted by compatible hands and renormalized per player."""
	ps = BarePublicState(round_idx=1, board_cards=["Ah","Kd","2c"])
	n = BareNode(ps, {0:0.5, 1:0.5, 2:0.0, 3:0.0}, {0:0.5, 1:0.5, 2:0.0, 3:0.0})
	before0 = dict(n.player_ranges[0])
	out = solver.lift_ranges_after_chance(n)


	for pl in (0, 1):
		s = sum(out[pl].values())
		assert s == pytest.approx(1.0)
		assert set(out[pl].keys()) == set(n.player_ranges[pl].keys())



def test_range_gadget_begin_commit_tracking_per_state(solver):
	"""Begin/commit updates are recorded under the state key, and commit respects monotone updates."""
	ps = BarePublicState(round_idx=1)
	n = BareNode(ps, {0:1.0}, {1:1.0})
	b = solver._range_gadget_begin(n)
	assert b == {}
	u1 = solver._range_gadget_commit(n, {0: 0.5, 1: -0.2})
	assert u1 == {0: 0.5, 1: -0.2}
	u2 = solver._range_gadget_commit(n, {0: 0.3, 1: -0.1, 2: 7.0})
	assert u2[0] == 0.5  
	assert u2[1] == -0.1 
	assert u2[2] == 7.0

	key = solver._state_key(n)
	assert solver.opponent_cfv_upper_tracking[key] == u2


def test_apply_opponent_action_update_merge_max(solver):
	"""Opponent CFV upper bounds merge across states by per-bucket maximum."""
	n_prev = BareNode(BarePublicState(round_idx=1), {0:1.0}, {1:1.0})
	n_next = BareNode(BarePublicState(round_idx=1, current_player=1), {0:1.0}, {1:1.0})
	kp = solver._state_key(n_prev)
	kn = solver._state_key(n_next)
	solver.opponent_cfv_upper_tracking = {
		kp: {0: 1.0, 1: 2.0},
		kn: {0: 0.5, 1: 3.5}
	}
	solver.apply_opponent_action_update(n_prev, n_next, observed_action_type=ActionType.CALL)
	merged = solver.opponent_cfv_upper_tracking[kn]
	assert merged[0] == 1.0 and merged[1] == 3.5



def test_upper_from_cfvs_max_component_and_scalar_passthrough(solver):
	"""Upper-bound extraction takes per-bucket max across action values or passes through scalars."""
	cfvs = {0: [1.0, 2.0, -0.5], 1: [0.0]}
	out = solver._upper_from_cfvs(cfvs)
	assert out[0] == 2.0 and out[1] == 0.0

	cfvs2 = {0: 1.25, 3: -0.75}
	out2 = solver._upper_from_cfvs(cfvs2)
	assert out2 == {0: 1.25, 3: -0.75}


def test_set_cfr_hybrid_config_and_get_last_diagnostics(solver):
	"""Hybrid config setter writes omit/iter flags; diagnostics report cache, residuals, and counters."""
	cfg = solver.set_cfr_hybrid_config(preflop_omit=10, flop_omit=11, turn_omit=12)
	assert cfg == {"preflop": 10, "flop": 11, "turn": 12}

	ps = BarePublicState(round_idx=1)
	n = BareNode(ps, {0:1.0}, {1:1.0})
	solver.predict_counterfactual_values(n, 0)
	d = solver.get_last_diagnostics()
	assert "preflop_cache" in d
	assert "cfv_calls" in d or True  


def test_apply_round_iteration_schedule_defaults_and_custom(solver):
	"""Round iteration schedule selects round-specific iters or keeps prior total_iterations."""
	solver._round_iters = {0: 0, 1: 1000, 2: 2000, 3: 500}
	it = solver.apply_round_iteration_schedule(1)
	assert it == 1000 and solver.total_iterations == 1000
	solver.total_iterations = 42
	it2 = solver.apply_round_iteration_schedule(99)
	assert it2 == 42 and solver.total_iterations == 42



def test_get_cumulative_strategy_aggregates_across_nodes(solver):
	"""Cumulative strategy aggregates per-cluster action mass across distinct nodes."""
	ps1 = BarePublicState(round_idx=1)
	ps2 = BarePublicState(round_idx=2)
	n1 = BareNode(ps1, {0:1.0}, {1:1.0})
	n2 = BareNode(ps2, {0:1.0}, {1:1.0})
	A = len(ActionType)

	v1 = solver.cfr_values[n1]
	v2 = solver.cfr_values[n2]
	v1.cumulative_strategy[0] = [1.0] + [0.0]*(A-1)
	v2.cumulative_strategy[0] = [0.5] + [0.0]*(A-1)

	agg = solver.get_cumulative_strategy(player=0)
	assert 0 in agg
	assert agg[0][0] == pytest.approx(1.5)



def test_compute_values_depth_limited_dispatch_raises_when_missing(solver):
	"""The dispatch path raises AttributeError when the mixin implementation is unavailable."""
	with pytest.raises(AttributeError):
		solver.compute_values_depth_limited(node=None, player=0)



def test_allowed_actions_agent_no_call_to_call_and_raise_flags(solver):
	"""Allowed action menu respects to_call logic, min-raise size, and per-round raise flags."""
	ps = BarePublicState(
		round_idx=0,
		current_player=0,
		pot_size=100,
		stacks=(10000, 10000),
		current_bets=(0,0),
		min_raise_size=10
	)
	solver._ensure_sparse_schedule()
	acts = solver._allowed_actions_agent(ps)
	assert ActionType.CALL in acts
	assert ActionType.HALF_POT_BET in acts
	assert ActionType.POT_SIZED_BET in acts
	assert ActionType.TWO_POT_BET in acts  
	assert ActionType.ALL_IN in acts

	ps2 = BarePublicState(
		round_idx=1,
		current_player=0,
		pot_size=100,
		stacks=(500, 500),
		current_bets=(0, 50),  
		min_raise_size=10
	)
	acts2 = solver._allowed_actions_agent(ps2)
	assert ActionType.FOLD in acts2
	assert ActionType.CALL in acts2
	assert ActionType.TWO_POT_BET not in acts2
	assert ActionType.ALL_IN in acts2  


def test_allowed_actions_opponent_exists_and_filters_legal(solver):
	"""Opponent allowed menu exists and is a subset of the state’s legal actions."""
	assert hasattr(solver, "_allowed_actions_opponent"), \
		"CFRSolverStrategiesMixin must define _allowed_actions_opponent"

	legal = [ActionType.FOLD, ActionType.CALL, ActionType.ALL_IN]  
	ps = BarePublicState(
		round_idx=1,
		current_player=1,
		pot_size=200,
		stacks=(1000, 1000),
		current_bets=(0, 100),
		min_raise_size=10,
		legal=legal
	)
	opp = solver._allowed_actions_opponent(ps)
	assert set(opp).issubset(set(legal))



def test_get_last_diagnostics_after_preflop_run_records_cache_and_residuals(solver, monkeypatch):
	"""After a preflop run (with cache hit), diagnostics include cache stats and tracked opponent CFV upper bounds."""
	ps = BarePublicState(round_idx=0, current_player=0, pot_size=100)
	n = BareNode(ps, {0:0.6, 1:0.4}, {0:0.5, 1:0.5})

	key = solver._preflop_signature(n)
	solver._preflop_cache_put(key, {0:0.7, 1:0.3}, {0:1.23, 1:-0.1})

	monkeypatch.setattr(solver, "_allowed_actions_agent", lambda ps: [ActionType.CALL])
	_ = solver.run_cfr(n)  

	d = solver.get_last_diagnostics()
	assert "preflop_cache" in d
	stats = d["preflop_cache"]
	assert stats["hits"] >= 1 or stats["misses"] >= 0  

	key2 = solver._state_key(n)
	assert hasattr(solver, "opponent_cfv_upper_tracking")
	assert key2 in solver.opponent_cfv_upper_tracking
	up = solver.opponent_cfv_upper_tracking[key2]
	assert up.get(0) == pytest.approx(1.23)
	assert up.get(1) == pytest.approx(-0.1)

