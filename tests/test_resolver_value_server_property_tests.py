"""
Test suite for HUNL continual re-solving with CFV networks: verifies zero-sum enforcement,
value-server batching, action menus for sparse lookahead, lookahead propagation and reach,
CFR-based root gadget behavior, helper utilities in resolver_integration, end-to-end resolve
diagnostics for flop/turn, and CFV bundle save/load I/O.
"""

import hunl.engine.poker_utils as poker_utils
import hunl.solving.resolver_integration as resolver_integration
import hunl.nets.model_io as model_io
import math
import threading
import time
import os
import tempfile
from typing import List, Dict, Any

import numpy as np
import pytest
import torch
from torch import nn
from hypothesis import given, settings, strategies as st

from hunl.nets.value_server import ValueServer
from hunl.utils.result_handle import ResultHandle
from hunl.solving.lookahead_tree import LookaheadTreeBuilder
from hunl.solving.cfr_core import PublicChanceCFR
from hunl.solving.resolver_integration import (
 resolve_at_with_diag,
 resolve_at,
 _stage_from_round,
 _ranges_to_simplex_vector,
 _ensure_value_server,
 _depth_and_bets_from_config,
 _bet_fraction_schedule_for_mode,
 _tighten_cfv_upper_bounds,
)

from hunl.nets.model_io import save_cfv_bundle, load_cfv_bundle
from hunl.nets.cfv_network import CounterfactualValueNetwork
from hunl.engine.action_type import ActionType
from hunl.engine.poker_utils import DECK, board_one_hot


def _norm(v: List[float]) -> List[float]:
	"""
	Normalize a vector of nonnegative values to sum to 1.0; return the original vector if the sum is zero.
	"""
	s = sum(v)
	if s > 0:
		return [x / s for x in v]
	return v


def make_input_batch(K: int, B: int, stage: str = "flop") -> torch.Tensor:
	"""
	Create a synthetic input batch for CFV models with pot normalization, board one-hot (52), and two K-length ranges.
	"""
	num_board = 3 if stage == "flop" else 4
	X = []
	for _ in range(B):
		pot_norm = np.clip(np.random.uniform(1e-3, 1.0), 1e-3, 1.0)
		bvec = [0] * 52
		idxs = np.random.choice(52, size=num_board, replace=False)
		for j in idxs:
			bvec[int(j)] = 1
		r1 = np.random.rand(K).tolist()
		r2 = np.random.rand(K).tolist()
		r1 = _norm(r1) if sum(r1) > 0 else [1.0 / K] * K
		r2 = _norm(r2) if sum(r2) > 0 else [1.0 / K] * K
		X.append([pot_norm] + bvec + r1 + r2)
	return torch.tensor(X, dtype=torch.float32)


class DummyZeroSumCFVNet(nn.Module):
	"""
	Torch module that outputs per-player K-length CFV vectors and supports zero-sum enforcement under given ranges.
	"""
	def __init__(self, input_size: int, num_clusters: int):
		"""
		Initialize with a linear head mapping input_size to 2*K outputs (first K for player 1, next K for player 2).
		"""
		super().__init__()
		self.input_size = int(input_size)
		self.num_clusters = int(num_clusters)
		self.lin = nn.Linear(self.input_size, 2 * self.num_clusters, bias=False)
		torch.manual_seed(123)
		nn.init.uniform_(self.lin.weight, -0.01, 0.01)

	def forward(self, x: torch.Tensor):
		"""
		Compute raw per-bucket CFV predictions for both players and return (p1, p2) tensors of shape [B, K].
		"""
		out = self.lin(x)
		B, _ = out.shape
		K = self.num_clusters
		p1 = out[:, :K]
		p2 = out[:, K:]
		return p1, p2

	@torch.no_grad()
	def enforce_zero_sum(self, r1: torch.Tensor, r2: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor):
		"""
		Shift predictions so that the range-weighted expectations are equal in magnitude and opposite in sign per sample.
		"""
		s1 = (r1 * p1).sum(dim=1, keepdim=True)
		s2 = (r2 * p2).sum(dim=1, keepdim=True)
		d = (s1 + s2) / 2.0
		return p1 - d, p2 - d


class _NodeWrap:
	"""
	Simple wrapper carrying a public_state attribute to mimic lookahead node containers in tests.
	"""
	def __init__(self, ps):
		"""
		Store the provided public-state-like object under .public_state.
		"""
		self.public_state = ps


class _FakePS:
	"""
	Minimal public-state stand-in for testing menus, dealing, rounds, and basic getters.
	"""
	def __init__(self, legal, board, hole0, hole1, cr=1, dealer=0, cur=0, terminal=False):
		"""
		Initialize with legal actions, board and hole cards, round index, dealer, current player, and terminal flag.
		"""
		self._legal = list(legal)
		self.board_cards = list(board)
		self.hole_cards = [list(hole0), list(hole1)]
		self.current_round = int(cr)
		self.dealer = int(dealer)
		self.current_player = int(cur)
		self.is_terminal = bool(terminal)
		self.current_bets = [0, 0]
		self.pot_size = 100.0

	def legal_actions(self):
		"""
		Return the list of currently legal actions for the acting player.
		"""
		return list(self._legal)



@settings(deadline=None, max_examples=60)
@given(
 K=st.integers(min_value=1, max_value=8),
 B=st.integers(min_value=1, max_value=16),
)
def test_value_server_enforces_zero_sum_and_slices_ranges_correctly_hypothesis(K, B):
	"""Verifies that ValueServer returns per-player K-length outputs, enforces per-sample zero-sum 
		under range weights, and updates internal query counters."""
	insz = 1 + 52 + 2 * K
	model = DummyZeroSumCFVNet(insz, K)
	vs = ValueServer(models={"flop": model}, device=torch.device("cpu"), max_batch_size=1024, max_wait_ms=2)
	xb = make_input_batch(K, B, stage="flop")
	v1, v2 = vs.query("flop", xb, as_numpy=False)
	assert v1.shape == (B, K)
	assert v2.shape == (B, K)
	start_r1 = 1 + 52
	end_r1 = start_r1 + K
	start_r2 = end_r1
	end_r2 = start_r2 + K
	r1 = xb[:, start_r1:end_r1]
	r2 = xb[:, start_r2:end_r2]
	s1 = (r1 * v1).sum(dim=1)
	s2 = (r2 * v2).sum(dim=1)
	residual = torch.abs(s1 + s2).max().item()
	assert residual < 1e-5
	ctr = vs.get_counters()
	assert int(ctr.get("flop", 0)) >= B
	vs.stop(join=True)


def test_value_server_missing_model_returns_zeros_and_no_counter_increment():
	"""Checks that querying a missing stage returns zero-sized outputs and does not increment counters 
		for that stage while preserving counts for available models."""
	K = 4
	insz = 1 + 52 + 2 * K
	model = DummyZeroSumCFVNet(insz, K)
	vs = ValueServer(models={"flop": model}, device=torch.device("cpu"))
	xb = make_input_batch(K, 3, stage="flop")
	v1, v2 = vs.query("flop", xb, as_numpy=False)
	assert v1.shape == (3, K) and v2.shape == (3, K)
	v1t, v2t = vs.query("turn", xb, as_numpy=False)
	assert v1t.shape[1] == 0 and v2t.shape[1] == 0
	ctr = vs.get_counters()
	assert ctr.get("turn", 0) == 0
	assert ctr.get("flop", 0) >= 3
	vs.stop(join=True)


def test_result_handle_blocking_and_types():
	"""Ensures ResultHandle blocks until value is set, returns numpy tensors on request, and 
	preserves tensor types and shapes."""
	h = ResultHandle()

	def _setter():
		time.sleep(0.05)
		v = (torch.ones(1, 2), torch.zeros(1, 2))
		h.set(v)

	t = threading.Thread(target=_setter)
	t.start()
	v1_np, v2_np = h.result(as_numpy=True)
	assert isinstance(v1_np, np.ndarray) and isinstance(v2_np, np.ndarray)
	assert v1_np.shape == (1, 2) and v2_np.shape == (1, 2)
	h2 = ResultHandle()
	h2.set((torch.full((1, 3), 7.0), torch.full((1, 3), -7.0)))
	v1_t, v2_t = h2.result(as_numpy=False)
	assert torch.allclose(v1_t, torch.full((1, 3), 7.0))
	assert torch.allclose(v2_t, torch.full((1, 3), -7.0))
	t.join(timeout=1.0)


def test_action_menu_respects_sparse_sizes_and_allin_flag():
	"""Validates LookaheadTreeBuilder action menu construction for sparse bet fractions
		and all-in inclusion, respecting max_actions_per_branch constraints."""
	legal = [
	 ActionType.FOLD,
	 ActionType.CALL,
	 ActionType.HALF_POT_BET,
	 ActionType.POT_SIZED_BET,
	 ActionType.TWO_POT_BET,
	 ActionType.ALL_IN,
	]
	ps = _FakePS(legal=legal, board=[], hole0=[], hole1=[], cr=1, dealer=0, cur=0)
	builder = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=True)
	menu = builder._action_menu(ps, for_player=True, pot_fracs=(0.5, 1.0), is_root=True)
	assert ActionType.FOLD in menu
	assert ActionType.CALL in menu
	assert ActionType.HALF_POT_BET in menu
	assert ActionType.POT_SIZED_BET in menu
	assert ActionType.ALL_IN in menu
	assert ActionType.TWO_POT_BET not in menu
	menu2 = builder._action_menu(ps, True, (0.5, 1.0, 2.0), True)
	assert ActionType.TWO_POT_BET in menu2
	builder2 = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=True, max_actions_per_branch=3)
	menu3 = builder2._action_menu(ps, True, (0.5, 1.0), True)
	assert len(menu3) == 3


def test_deal_next_card_counts():
	"""Checks that the chance expander returns correct 
		counts for next-board cards on flop (→45) and turn (→44) given used cards."""
	ps_flop = _FakePS(legal=[], board=["AS", "KD", "3c".upper()],
	  hole0=["7H", "2D"], hole1=["9C", "9D"], cr=1)
	builder = LookaheadTreeBuilder(depth_limit=0)
	nxt = builder._deal_next_card(ps_flop)
	assert len(nxt) == 45
	ps_turn = _FakePS(legal=[], board=["AS", "KD", "3C", "5S"], hole0=["7H", "2D"], hole1=["9C", "9D"], cr=2)
	nxt2 = builder._deal_next_card(ps_turn)
	assert len(nxt2) == 44


def test_propagate_calls_leaf_callback_and_propagates_reach():
	"""Ensures propagate invokes the registered leaf callback and correctly splits
		reach probabilities across equally weighted child actions."""
	K = 4
	builder = LookaheadTreeBuilder(depth_limit=1)

	class _LeafPS:
		def __init__(self):
			self.initial_stacks = [200, 200]
			self.pot_size = 100.0
			self.board_cards = []
			self.current_round = 1
			self.is_terminal = False

	root_ps = _LeafPS()
	tree = {
	 "nodes": [_NodeWrap(root_ps), _NodeWrap(_LeafPS()), _NodeWrap(_LeafPS())],
	 "parents": [-1, 0, 0],
	 "edges": [None, ActionType.CALL, ActionType.POT_SIZED_BET],
	 "kinds": ["our", "leaf", "leaf"],
	 "depth_actions": [0, 1, 1],
	 "menus": [[ActionType.CALL, ActionType.POT_SIZED_BET], [], []],
	 "stage_start": 1,
	}
	seen = []

	def leaf_cb(ps, pov, r1, r2):
		seen.append((r1, r2))
		return np.zeros((K,), dtype=float)

	builder.set_leaf_callback(leaf_cb)
	r_us = [1.0 / K] * K
	r_opp = [1.0 / K] * K
	out = builder.propagate(tree, r_us, r_opp, pov_player=0)
	assert len(seen) == 2
	for (ru, ro) in seen:
		assert np.allclose(ru, np.array(r_us) * 0.5)
		assert np.allclose(ro, np.array(r_opp) * 0.5)
	assert isinstance(out["values"][1], np.ndarray) and isinstance(out["values"][2], np.ndarray)


def test_public_chance_cfr_root_gadget_and_warm_start_setter(monkeypatch):
	"""Validates PublicChanceCFR behavior at an opponent-root: warm-start acceptance,
	terminate-vs-follow min selection, normalized root policy, and mirrored opponent CFV constraints."""
	K = 3

	class _LeafPS:
		def __init__(self):
			self.initial_stacks = [200, 200]
			self.pot_size = 100.0
			self.board_cards = []
			self.current_round = 1
			self.is_terminal = False

	r_us = [1.0 / K] * K
	good_v = np.full((K,), 0.6)

	def leaf_fn(ps, pov, ru, ro):
		return torch.tensor(good_v, dtype=torch.float32)

	tree = {
	 "nodes": [_NodeWrap(_LeafPS()), _NodeWrap(_LeafPS()), _NodeWrap(_LeafPS())],
	 "parents": [-1, 0, 0],
	 "edges": [None, ActionType.CALL, ActionType.POT_SIZED_BET],
	 "kinds": ["opp", "leaf", "leaf"],
	 "depth_actions": [0, 1, 1],
	 "menus": [[ActionType.CALL, ActionType.POT_SIZED_BET], [], []],
	 "stage_start": 1,
	}

	solver = PublicChanceCFR(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=True)
	solver.set_warm_start({"dummy": [0.2, 0.8]})
	r_opp = [1.0 / K] * K
	opp_upper = [0.1] * K

	root_policy, node_values, opp_cfv = solver.solve_subgame(
	 root_node=tree,
	 r_us=r_us,
	 r_opp=r_opp,
	 opp_cfv_constraints=opp_upper,
	 T=10,
	 leaf_value_fn=leaf_fn,
	)
	assert set(root_policy.keys()) == set(tree["menus"][0])
	assert abs(sum(root_policy.values()) - 1.0) < 1e-8
	for i in range(len(opp_upper)):
		assert pytest.approx(opp_cfv[i], rel=0, abs=1e-12) == opp_upper[i]


@settings(deadline=None, max_examples=80)
@given(
 K=st.integers(min_value=1, max_value=16),
 items=st.lists(
  st.tuples(
   st.integers(min_value=-5, max_value=25),
   st.floats(min_value=0.0, max_value=10.0),
  ),
  min_size=0,
  max_size=40,
 ),
)
def test__to_vec_normalizes_and_clips_indices(K, items):
	"""Checks _to_vec builds a length-K, nonnegative, L1-normalized 
		vector from a sparse dict while discarding out-of-range indices."""
	d: Dict[int, float] = {}
	for i, p in items:
		if p < 0:
			p = 0.0
		d[int(i)] = d.get(int(i), 0.0) + float(p)
	v = _ranges_to_simplex_vector(d, K)
	assert len(v) == K
	s = sum(v)
	if any((0 <= i < K and p > 0) for i, p in d.items()):
		assert abs(s - 1.0) < 1e-9
		assert all(x >= 0.0 for x in v)
	else:
		assert s == 0.0


def test__stage_from_round_mapping():
	"""Verifies the round-to-stage mapping used for selecting 
		CFV networks and depth-limit behavior."""
	assert _stage_from_round(0) == "flop"
	assert _stage_from_round(1) == "flop"
	assert _stage_from_round(2) == "turn"
	assert _stage_from_round(3) == "river" or _stage_from_round(3) == "flop"
	assert _stage_from_round(99) == "flop"


def test__bet_fracs_from_mode_and__depth_and_bets_turn_and_flop():
	"""Checks bet fraction presets per mode and default depth/bet settings returned by _depth_and_bets for flop and turn."""
	assert _bet_fraction_schedule_for_mode("sparse_2", "flop") == [0.5, 1.0]
	assert _bet_fraction_schedule_for_mode("sparse_3", "flop") == [0.5, 1.0, 2.0]
	assert _bet_fraction_schedule_for_mode("full", "flop") == [0.5, 1.0, 2.0]
	dl, bf, include_all, cm = _depth_and_bets_from_config("turn", 1, {})
	assert dl >= 90 and 2.0 in bf and include_all and cm == "sp"
	dl2, bf2, include_all2, cm2 = _depth_and_bets_from_config("flop", 2, {})
	assert dl2 == 2 and 2.0 not in bf2 and include_all2 and cm2 == "sp"


@settings(deadline=None, max_examples=60)
@given(
 prev=st.dictionaries(st.integers(0, 10), st.floats(min_value=-1e2, max_value=1e2)),
 prop=st.dictionaries(st.integers(0, 10), st.floats(min_value=-1e2, max_value=1e2)),
)
def test__update_opp_upper_monotone_is_coordinatewise_min(prev, prop):
	"""Confirms that proposed opponent upper bounds are merged by coordinatewise 
	minimum, ensuring monotone tightening."""
	out = _tighten_cfv_upper_bounds(prev, prop)
	for k in set(list(prev.keys()) + list(prop.keys())):
		a = float(prev.get(k, float("inf")))
		b = float(prop.get(k, float("inf")))
		assert out[k] == (a if a < b else b)


def test__ensure_value_server_from_models_and_bundle(tmp_path):
	"""Ensures _ensure_value_server builds a ValueServer from in-memory models and
		from a saved bundle path, and that created servers are operational."""
	K = 4
	insz = 1 + 52 + 2 * K
	net = CounterfactualValueNetwork(input_size=insz, num_clusters=K)
	models = {"flop": net}
	vs1 = _ensure_value_server({"models": models}, None)
	assert isinstance(vs1, ValueServer)
	vs1.stop(join=True)
	bundle_path = os.path.join(tmp_path, "cfv_bundle.pt")
	save_cfv_bundle(models=models, cluster_mapping={}, input_meta={"num_clusters": K}, path=bundle_path, seed=123)
	vs2 = _ensure_value_server({"bundle_path": bundle_path}, None)
	assert isinstance(vs2, ValueServer)
	vs2.stop(join=True)


def _fake_tree_for_resolve(root_round: int, menu: List[Any], leaf_round: int):
	class _PS:
		def __init__(self, cr):
			self.initial_stacks = [200, 200]
			self.pot_size = 100.0
			self.board_cards = [] if cr == 0 else (["AS", "KD", "3C"] if cr == 1 else ["AS", "KD", "3C", "2D"])
			self.current_round = cr
			self.current_player = 0
			self.dealer = 0
			self.is_terminal = False
			self.current_bets = [0, 0]

		def terminal_utility(self):
			return [0.0, 0.0]

	root_ps = _PS(root_round)
	leaf_ps = _PS(leaf_round)
	tree = {
	 "nodes": [_NodeWrap(root_ps), _NodeWrap(leaf_ps)],
	 "parents": [-1, 0],
	 "edges": [None, menu[0]],
	 "kinds": ["our", "leaf"],
	 "depth_actions": [0, 1],
	 "menus": [menu, []],
	 "stage_start": root_round,
	}
	return tree


def test_resolve_at_with_diag_flop_queries_net_and_acceptance(monkeypatch):
	"""Tests resolve_at_with_diag on a flop root: verifies ValueServer is queried
		for flop, acceptance diagnostics are true, and returned policy is a valid distribution."""
	K = 5
	insz = 1 + 52 + 2 * K
	flop_net = DummyZeroSumCFVNet(insz, K)
	vs = ValueServer(models={"flop": flop_net}, device=torch.device("cpu"))
	fake_menu = [ActionType.CALL]
	fake_tree = _fake_tree_for_resolve(root_round=1, menu=fake_menu, leaf_round=1)

	def _fake_build(self, public_state):
		return fake_tree

	monkeypatch.setattr(LookaheadTreeBuilder, "build", _fake_build)
	r_us = {i: 1.0 / K for i in range(K)}
	w_opp = {i: 0.0 for i in range(K)}
	ps = fake_tree["nodes"][0].public_state
	pol, w_next, our_cfv, diag = resolve_at_with_diag(
	 public_state=ps,
	 r_us=r_us,
	 w_opp=w_opp,
	 config={"depth_limit": 1, "bet_size_mode": "sparse_2"},
	 value_server=vs,
	)
	assert diag["stage"] == "flop"
	assert diag["range_mass_ok"] is True
	assert diag["policy_actions_ok"] is True
	assert diag["flop_net_queries"] >= 1
	assert diag["turn_net_queries"] == 0
	assert diag["turn_leaf_net_ok"] is True
	assert set(pol.keys()) == set(fake_menu)
	assert abs(sum(pol.values()) - 1.0) < 1e-9
	vs.stop(join=True)


def test_resolve_at_with_diag_turn_does_not_query_net_and_acceptance(monkeypatch):
	"""Tests resolve_at_with_diag on a turn root: ensures no ValueServer queries occur 
		for turn, acceptance diagnostics pass, and menu validity holds."""
	K = 6
	insz = 1 + 52 + 2 * K
	turn_net = DummyZeroSumCFVNet(insz, K)
	vs = ValueServer(models={"turn": turn_net}, device=torch.device("cpu"))
	fake_menu = [ActionType.CALL]
	fake_tree = _fake_tree_for_resolve(root_round=2, menu=fake_menu, leaf_round=2)

	def _fake_build(self, public_state):
		return fake_tree

	monkeypatch.setattr(LookaheadTreeBuilder, "build", _fake_build)
	r_us = {i: 1.0 / K for i in range(K)}
	w_opp = {i: 0.0 for i in range(K)}
	ps = fake_tree["nodes"][0].public_state
	pol, w_next, our_cfv, diag = resolve_at_with_diag(
	 public_state=ps,
	 r_us=r_us,
	 w_opp=w_opp,
	 config={"depth_limit": 1, "bet_size_mode": "sparse_3"},
	 value_server=vs,
	)
	assert diag["stage"] == "turn"
	assert diag["turn_net_queries"] == 0
	assert diag["turn_leaf_net_ok"] is True
	assert diag["policy_actions_ok"] is True
	vs.stop(join=True)


def test_resolve_at_constraint_modes_sp_and_br(monkeypatch):
	"""Checks that resolve_at returns solver-derived next constraints 
		in self-play mode and passes through provided opponent weights in best-response mode."""
	K = 4
	insz = 1 + 52 + 2 * K
	flop_net = DummyZeroSumCFVNet(insz, K)
	vs = ValueServer(models={"flop": flop_net}, device=torch.device("cpu"))
	fake_menu = [ActionType.CALL]
	fake_tree = _fake_tree_for_resolve(root_round=1, menu=fake_menu, leaf_round=1)
	monkeypatch.setattr(LookaheadTreeBuilder, "build", lambda self, ps: fake_tree)
	r_us = {i: 1.0 / K for i in range(K)}
	w_opp = {i: float(i) / K for i in range(K)}
	ps = fake_tree["nodes"][0].public_state
	pol_sp, w_next_sp, _, diag_sp = resolve_at_with_diag(ps, r_us, w_opp, config={"depth_limit": 1}, value_server=vs)
	pol_sp2, w_next_sp2, _ = resolve_at(ps, r_us, w_opp, config={"depth_limit": 1}, value_server=vs)
	assert set(w_next_sp.keys()) == set(range(K))
	assert w_next_sp == w_next_sp2
	pol_br, w_next_br, _ = resolve_at(
	 ps, r_us, w_opp, config={"depth_limit": 1, "constraint_mode": "br"}, value_server=vs
	)
	assert w_next_br == w_opp
	vs.stop(join=True)


def test_model_io_bundle_roundtrip(tmp_path):
	"""Ensures CFV bundle round-trip: saved file exists, loaded meta matches, model runs with correct shapes."""
	K = 3
	insz = 1 + 52 + 2 * K
	net = CounterfactualValueNetwork(input_size=insz, num_clusters=K)
	models = {"flop": net}
	cluster_mapping = {i: i for i in range(K)}
	input_meta = {"num_clusters": K, "board_one_hot_dim": 52, "uses_pot_norm": True}
	path = os.path.join(tmp_path, "bundle.pt")
	out_path = save_cfv_bundle(models=models, cluster_mapping=cluster_mapping, input_meta=input_meta, path=path, seed=42)
	assert os.path.isfile(out_path)
	loaded = load_cfv_bundle(out_path, device=torch.device("cpu"))
	assert "models" in loaded and "meta" in loaded
	lm = loaded["models"]
	assert "flop" in lm
	ln = lm["flop"]
	assert getattr(ln, "num_clusters", None) == K
	assert getattr(ln, "input_size", None) == insz
	meta = loaded["meta"]
	assert meta["input_meta"]["num_clusters"] == K
	assert meta["input_meta"]["board_one_hot_dim"] == 52
	xb = make_input_batch(K, 2, stage="flop")
	with torch.no_grad():
		p1, p2 = ln(xb)
	assert p1.shape == (2, K) and p2.shape == (2, K)

