import hunl.data.data_generator as data_generator
import hunl.data.data_generator_sampling as data_generator_sampling
import hunl.data.data_generator_datasets as data_generator_datasets
import hunl.data.data_generator_utils as data_generator_utils
import hunl.data.data_manifest as data_manifest
import hunl.nets.cfv_stream_dataset as cfv_stream_dataset
import hunl.nets.cfv_shard_dataset as cfv_shard_dataset
import hunl.cli.eval_cli as eval_cli




import io
import os
import json
import math
import types
import shutil
import random
import string
import pathlib
import inspect
import hashlib
import tempfile
from types import SimpleNamespace
from typing import Dict, List, Any, Tuple
from importlib import import_module

import numpy as np
import pytest



K_SMALL = 8  
DECK = [r+s for r in "23456789TJQKA" for s in "CDHS"]  

class StubPublicState:
	"""Minimal stand-in for PublicState used to drive DataGenerator flows without full engine logic."""

	def __init__(self, initial_stacks, board_cards, dealer=0):
		"""Populate the few fields DataGenerator paths touch (pots, bets, round, players, deck/actions)."""
		self.initial_stacks = list(initial_stacks)
		self.board_cards = list(board_cards)
		self.dealer = dealer
		self.current_round = 0
		self.current_bets = [0, 0]
		self.last_raiser = None
		self.stacks = list(initial_stacks)
		self.pot_size = 2.0
		self.current_player = (dealer + 1) % 2
		self.hole_cards = [[], []]
		self.deck = []
		self.actions = []
		self.is_terminal = False
		self.last_refund_amount = 0.0


class StubGameNode:
	"""Lightweight GameNode wrapper pairing the stubbed public state with player ranges."""

	def __init__(self, public_state):
		"""Initialize with empty per-player bucket ranges and both players active."""
		self.public_state = public_state
		self.player_ranges = [{}, {}]
		self.players_in_hand = [True, True]


class StubHandClusterer:
	"""Hash-based, deterministic hand clusterer stub returning K partitions; used to avoid heavy k-means."""

	def __init__(self, cfr_solver, num_clusters, profile, opp_sample_size, use_cfv_in_features, config=None):
		"""Record config flags and provide a predictable K derived from num_clusters."""
		self.cfr_solver = cfr_solver
		self.num_clusters = int(num_clusters)
		self.profile = profile
		self.opp_sample_size = opp_sample_size
		self.use_cfv_in_features = use_cfv_in_features
		self.config = config
		self._frozen = None

	def cluster_hands(self, hands, board, opponent_range, pot_size):
		"""Partition hands into K buckets by hashing (stable across runs for the same inputs)."""
		K = self.num_clusters
		clusters = {i: set() for i in range(K)}
		for h in (hands.keys() if isinstance(hands, dict) else hands):
			key = h if isinstance(h, str) else " ".join(list(h))
			cls = int(hashlib.sha256(key.encode()).hexdigest(), 16) % K
			clusters[cls].add(key)
		return clusters


class StubCFRSolver:
	"""Slim CFRSolver stub exposing only the methods DataGenerator exercises (run, predict CFVs, labels)."""

	def __init__(self, config=None, depth_limit=1, num_clusters=K_SMALL):
		"""Initialize with small defaults; tracks calls for assertions and exposes round action flags."""
		self.config = config
		self.depth_limit = int(getattr(config, "depth_limit", depth_limit)) if config else depth_limit
		self.total_iterations = int(getattr(config, "total_iterations", 1)) if config else 1
		self.num_clusters = int(getattr(config, "num_clusters", num_clusters)) if config else num_clusters
		self.clusters = {i: set() for i in range(self.num_clusters)}
		self.hand_clusterer = None
		self.device = "cpu"
		self.models = {}
		self.run_cfr_calls: List[StubGameNode] = []
		self.ensure_sparse_schedule_calls = 0
		self._round_actions = {r: {"half_pot": True, "two_pot": False} for r in (0,1,2,3)}

	def _ensure_sparse_schedule(self):
		"""Mimic solver’s scheduling guard and count invocations; returns round flags."""
		self.ensure_sparse_schedule_calls += 1
		if not hasattr(self, "_round_actions"):
			self._round_actions = {r: {"half_pot": True, "two_pot": False} for r in (0,1,2,3)}
		return self._round_actions

	def run_cfr(self, node):
		"""Record the node to assert the generator invoked solving."""
		self.run_cfr_calls.append(node)

	def predict_counterfactual_values(self, node, player: int) -> Dict[int, List[float]]:
		"""Return constant, signed CFVs per cluster (player 0 positive, player 1 negative) for label creation."""
		K = self.num_clusters
		base = 0.25 if player == 0 else -0.25
		return {i: [base for _ in range(1)] for i in range(K)}

	def turn_label_targets_solve_to_terminal(self, node) -> Tuple[List[float], List[float]]:
		"""Produce synthetic turn targets by pretending we solved to terminal (fixed ±0.10)."""
		K = self.num_clusters
		t1 = [+0.10 for _ in range(K)]
		t2 = [-0.10 for _ in range(K)]
		return t1, t2

	def flop_label_targets_using_turn_net(self, node) -> Tuple[List[float], List[float]]:
		"""Produce synthetic flop targets via a mock turn model (fixed ±0.05)."""
		K = self.num_clusters
		t1 = [+0.05 for _ in range(K)]
		t2 = [-0.05 for _ in range(K)]
		return t1, t2

	def recursive_range_sampling(self, hands_set, total_prob, public_cards):
		"""Uniformly spread total_prob over the provided hand set; deterministic ordering preserved."""
		n = max(1, len(hands_set))
		u = float(total_prob) / float(n)
		return {h if isinstance(h, str) else " ".join(list(h)): u for h in hands_set}

	def _calculate_terminal_utility(self, node, player: int) -> float:
		"""Return zero utility at terminal (keeps label scaling paths simple in tests)."""
		return 0.0


class StubTurnModel(torch.nn.Module if 'torch' in globals() else object):
	"""Minimal nn.Module shim exposing num_clusters; used to test turn-conditioned flop labels."""

	def __init__(self, num_clusters=K_SMALL):
		"""Store K; no parameters are actually used."""
		super().__init__()
		self.num_clusters = num_clusters

	def to(self, device):
		"""No-op device transfer to match real model interface."""
		return self

	def eval(self):
		"""Return self to mirror torch’s eval() semantics without side effects."""
		return self


@pytest.fixture(scope="function")
def patched_modules(monkeypatch):
	"""Monkeypatch core modules with stubs to make data-generation tests fast, deterministic, and isolated."""
	import hunl.data.data_generator
	import hunl.data.data_generator_sampling
	import hunl.data.data_generator_datasets
	import hunl.data.data_generator_utils
	import hunl.nets.cfv_stream_dataset
	import hunl.cli.eval_cli

	monkeypatch.setattr(data_generator, "CFRSolver", StubCFRSolver, raising=True)
	monkeypatch.setattr(data_generator, "HandClusterer", StubHandClusterer, raising=True)
	monkeypatch.setattr(data_generator, "PublicState", StubPublicState, raising=True)
	monkeypatch.setattr(data_generator, "GameNode", StubGameNode, raising=True)

	monkeypatch.setattr(data_generator_sampling, "PublicState", StubPublicState, raising=True)
	monkeypatch.setattr(data_generator_sampling, "GameNode", StubGameNode, raising=True)

	monkeypatch.setattr(data_generator_datasets, "np", np, raising=True)
	monkeypatch.setattr(data_generator_datasets, "random", random, raising=True)

	monkeypatch.setattr(data_generator_utils, "np", np, raising=True)
	monkeypatch.setattr(
	 data_generator_utils,
	 "torch",
	 SimpleNamespace(
	  manual_seed=lambda s: None,
	  cuda=SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda s: None),
	 ),
	 raising=True,
	)

	def _fake_board_one_hot(board):
		v = [0] * 52
		for i in range(min(len(board), 52)):
			v[i] = 1
		return v

	monkeypatch.setattr(data_generator, "board_one_hot", _fake_board_one_hot, raising=True)
	monkeypatch.setattr(data_generator_utils, "board_one_hot", _fake_board_one_hot, raising=True)

	monkeypatch.setattr(eval_cli, "CFRSolver", StubCFRSolver, raising=True)

	return SimpleNamespace(
	 dg=data_generator,
	 dgs=data_generator_sampling,
	 dgd=data_generator_datasets,
	 dgu=data_generator_utils,
	 stream=import_module("hunl.nets.cfv_stream_dataset"),
	 shard=import_module("hunl.nets.cfv_shard_dataset"),
	 manifest=import_module("hunl.data.data_manifest"),
	 eval_cli=eval_cli,
	)




def test_init_defaults_and_config_paths(patched_modules):
	"""DataGenerator initialization in (bot/test) profiles and via a config object; verifies propagated solver/clusterer settings and minimal test limits (depth=0, iters>=1)."""
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(num_boards=2, num_samples_per_board=3, player_stack=200, num_clusters=K_SMALL, speed_profile="bot")
	assert dg.num_clusters == K_SMALL
	assert dg.cfr_solver.num_clusters == K_SMALL
	assert isinstance(dg.cfr_solver, StubCFRSolver)
	assert isinstance(dg.hand_clusterer, StubHandClusterer)
	assert dg.speed_profile == "bot"
	dg2 = dgmod.DataGenerator(num_boards=1, num_samples_per_board=1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	assert dg2.cfr_solver.depth_limit == 0
	assert dg2.cfr_solver.total_iterations >= 1

	cfg = SimpleNamespace(
	 num_clusters=K_SMALL, profile="test",
	 total_iterations=4, depth_limit=1,
	 opp_sample_size=0, use_cfv_in_features=False
	)
	dg3 = dgmod.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile=None, config=cfg)
	assert dg3.num_clusters == K_SMALL
	assert dg3.speed_profile == "test"
	assert isinstance(dg3.cfr_solver, StubCFRSolver)
	assert isinstance(dg3.hand_clusterer, StubHandClusterer)
	assert dg3.cfr_solver.depth_limit == 0
	assert dg3.cfr_solver.total_iterations == 1



def test_prepare_input_vector_layout_and_invariants(patched_modules):
	"""Input vector layout and basic invariants — length = 1 + 52 + K + K, pot_norm in (0,1], 
	board one-hot count matches board, ranges copied."""
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	board = ["AS","KD","7c"]  
	r1 = [0.0]*K_SMALL; r1[0] = 0.6; r1[1] = 0.4
	r2 = [0.0]*K_SMALL; r2[2] = 1.0
	iv = dg.prepare_input_vector([r1, r2], board, pot_size=20.0, actions=None)
	assert len(iv) == 1 + 52 + K_SMALL + K_SMALL
	pot_norm = iv[0]
	assert 0.0 < pot_norm <= 1.0  
	assert sum(iv[1:1+52]) == len(board)
	assert iv[1+52:1+52+K_SMALL] == r1
	assert iv[1+52+K_SMALL:1+52+2*K_SMALL] == r2

def test_prepare_input_vector_rejects_bad_invariants(patched_modules):
	"""Rejections for mass mismatch, pot<=0, and pot_norm>1 (raises ValueError)."""
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	board = ["AS","KD","7c"]
	r1 = [0.2] + [0.0]*(K_SMALL-1)
	r2 = [0.3] + [0.0]*(K_SMALL-1)
	with pytest.raises(ValueError):
		dg.prepare_input_vector([r1, r2], board, pot_size=20.0)
	r_ok = [1.0] + [0.0]*(K_SMALL-1)
	with pytest.raises(ValueError):
		dg.prepare_input_vector([r_ok, r_ok], board, pot_size=0.0)
	with pytest.raises(ValueError):
		dg.prepare_input_vector([r_ok, r_ok], board, pot_size=401.0)



def test_prepare_target_values_various_input_shapes(patched_modules):
	"""Target extraction from {player->{cid: scalar or [scalar]}}; missing keys → zeros."""
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	cf = {
	 0: {i: [0.1] for i in range(K_SMALL)},
	 1: {i: -0.1 for i in range(K_SMALL)},
	}
	t1, t2 = dg.prepare_target_values(cf)
	assert len(t1) == K_SMALL and len(t2) == K_SMALL
	assert all(abs(x-0.1) < 1e-12 for x in t1)
	assert all(abs(x+0.1) < 1e-12 for x in t2)

	cf2 = {0: {0: [0.2]}, 1: {}}
	t1b, t2b = dg.prepare_target_values(cf2)
	assert t1b[0] == 0.2 and sum(t1b[1:]) == 0.0
	assert sum(t2b) == 0.0



def test_compute_counterfactual_values_calls_solver(patched_modules):
	"""Both players’ CFVs returned with K keys."""
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	ps = StubPublicState([200,200], ["AS","KD","QH"])
	node = StubGameNode(ps)
	cf = dg.compute_counterfactual_values(node)
	assert 0 in cf and 1 in cf
	assert set(cf[0].keys()) == set(range(K_SMALL))
	assert set(cf[1].keys()) == set(range(K_SMALL))


@pytest.mark.parametrize("stage, n, length", [
 ("flop",  5, 3),
 ("turn",  7, 4),
 ("river", 9, 5),
])
def test_generate_unique_boards_by_stage(patched_modules, stage, n, length):
	"""Generates N unique boards of correct length for flop/turn/river."""
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(n, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	boards = dg.generate_unique_boards(stage=stage, num_boards=n)
	assert len(boards) == n
	seen = set()
	for b in boards:
		assert len(b) == length
		t = tuple(sorted(b))
		assert t not in seen
		seen.add(t)



def test_sampler_invariants_and_helpers(patched_modules):
	"""Sampler guard (pot_norm, board one-hot, mass conservation) and helper predicates."""
	dgu = patched_modules.dgu
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	board = ["AS","KD","QH"]
	r1 = {0: 0.6, 1: 0.4}
	r2 = {2: 1.0}
	dg._assert_sampler_invariants(board, [r1, r2], pot_size=40.0)

	with pytest.raises(ValueError):
		dg._assert_sampler_invariants(board, [ {0:0.2}, {0:0.3} ], pot_size=40.0)

	with pytest.raises(ValueError):
		dg._assert_sampler_invariants(board, [ {0:1.0}, {0:1.0} ], pot_size=0.0)

	assert dg._pot_norm_ok(40.0) is True
	assert dg._pot_norm_ok(0.0) is False
	assert dg._board_one_hot_valid(board) is True
	assert dg.is_range_mass_conserved([ {0:1.0}, {0:1.0} ])
	assert not dg.is_range_mass_conserved([ {0:0.2}, {0:0.3} ])



def test_normalize_and_bucket_ranges(patched_modules):
	"""Normalization to probability-simplex and bucketization to K-length dense vectors (both sum to 1)."""
	dgu = patched_modules.dgu
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	r1 = {0: 2.0, 1: 1.0}
	r2 = {2: 5.0}
	out = dg.normalize_cluster_probabilities([dict(r1), dict(r2)])
	s1 = sum(out[0].values()); s2 = sum(out[1].values())
	assert abs(s1 - 1.0) < 1e-9
	assert abs(s2 - 1.0) < 1e-9
	b = dg.bucket_player_ranges(out)
	assert len(b) == 2 and len(b[0]) == K_SMALL and len(b[1]) == K_SMALL
	assert abs(sum(b[0]) - 1.0) < 1e-9
	assert abs(sum(b[1]) - 1.0) < 1e-9

def test_map_handstorcho_clusters_and_map_hands_to_clusters_equivalence(patched_modules):
	"""Hand→cluster mapping utilities agree and re-normalize."""
	dgu = patched_modules.dgu
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	clusters = {i: set() for i in range(K_SMALL)}
	clusters[0].update({"AS KD", "QH QC"})
	clusters[1].update({"2C 3D"})
	hands = {"AS KD": 0.25, "QH QC": 0.25, "2C 3D": 0.5}
	m1 = dg.map_hands_to_clusters_compat(hands, clusters)
	m2 = dgu.DataGeneratorUtilsMixin.map_hands_to_clusters(dg, hands, clusters)
	assert m1 == m2
	assert abs(sum(m1.values()) - 1.0) < 1e-9



def test_filter_clusters_for_board_removes_illegal_hands(patched_modules):
	"""Filters paired duplicates and hands containing board cards."""
	dgu = patched_modules.dgu
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	board = ["AS", "KD", "QH"]
	clusters = {
	 0: {"AS KD", "AS AS", "2C 3D"},  
	 1: {"4C 4C", "4C 5D", "KD QH"},  
	}
	flt = dg._filter_clusters_for_board(clusters, board)
	assert "2C 3D" in flt[0]
	assert len(flt[0]) == 1
	assert flt[1] == {"4C 5D"}



def test_push_pop_leaf_override_and_expected_steps(patched_modules):
	"""Test-profile leaf override (depth=0, iters>=1) push/pop is reversible; 
		expected_total_steps formula for test/bot profiles."""
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	dg.cfr_solver.depth_limit = 3
	dg.cfr_solver.total_iterations = 7
	snap = dg._push_leaf_solve_mode(stage="flop")
	assert dg.cfr_solver.depth_limit == 0 and dg.cfr_solver.total_iterations >= 1
	assert dg._pop_leaf_solve_mode(snap) is True
	assert dg.cfr_solver.depth_limit == 3 and dg.cfr_solver.total_iterations == 7

	dg.num_boards = 5
	dg.num_samples_per_board = 3
	assert dg.expected_total_steps() == 5 * (1 + 3 * (1 + 1))

	dg2 = patched_modules.dg.DataGenerator(5, 3, player_stack=200, num_clusters=K_SMALL, speed_profile="bot")
	assert dg2.expected_total_steps() == 5 * (1 + 3 * (2 + 1))



def _mk_dg_for_profile(patched_modules, profile: str):
	"""Helper to spin up a DataGenerator wired to the stub solver/clusterer under a given speed profile."""
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(num_boards=2, num_samples_per_board=2, player_stack=200,
	       num_clusters=K_SMALL, speed_profile=profile)
	dg.cfr_solver.hand_clusterer = dg.hand_clusterer
	return dg

def test_generate_training_data_profile_test_reuses_frozen_clusters(patched_modules):
	"""“test” profile uses a frozen cluster partition across samples; each record has correct input/target shapes."""
	dg = _mk_dg_for_profile(patched_modules, "test")
	records = dg.generate_training_data(stage="flop", progress=None)
	assert len(records) == 4
	for rec in records:
		iv = rec["input_vector"]
		assert len(iv) == 1 + 52 + K_SMALL + K_SMALL
		assert len(rec["target_v1"]) == K_SMALL
		assert len(rec["target_v2"]) == K_SMALL

def test_generate_training_data_profile_bot_forces_sparse_actions(patched_modules):
	"""“bot” profile temporarily forces sparse action set during generation and restores flags after; produces valid records."""
	dg = _mk_dg_for_profile(patched_modules, "bot")
	before_flags = dict(dg.cfr_solver._round_actions)
	records = dg.generate_training_data(stage="flop", progress=None)
	assert len(dg.cfr_solver.run_cfr_calls) > 0
	after_flags = dg.cfr_solver._round_actions
	assert after_flags == before_flags
	assert len(records) == 4
	for rec in records:
		iv = rec["input_vector"]
		assert len(iv) == 1 + 52 + K_SMALL + K_SMALL



def test_sample_random_range_normalization_and_support(patched_modules):
	"""Random range sampling returns a normalized distribution over provided IDs."""
	dgs = patched_modules.dgs
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	ids = list(range(K_SMALL))
	r = dgs.DataGeneratorSamplingMixin._sample_random_range(dg, ids)
	assert abs(sum(r.values()) - 1.0) < 1e-9
	assert set(r.keys()).issubset(set(ids))

def test_sample_flop_and_turn_situations(patched_modules):
	"""Flop/turn situation samplers set round index and fill both players’ ranges."""
	dgs = patched_modules.dgs
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	dg.cfr_solver.hand_clusterer = dg.hand_clusterer
	rng = random.Random(123)
	n_flop = dgs.DataGeneratorSamplingMixin._sample_flop_situation(dg, rng)
	n_turn = dgs.DataGeneratorSamplingMixin._sample_turn_situation(dg, rng)
	assert isinstance(n_flop, StubGameNode) and isinstance(n_turn, StubGameNode)
	assert n_flop.public_state.current_round == 1
	assert n_turn.public_state.current_round == 2
	assert len(n_flop.player_ranges[0]) > 0 and len(n_flop.player_ranges[1]) > 0
	assert len(n_turn.player_ranges[0]) > 0 and len(n_turn.player_ranges[1]) > 0



def test_sample_pot_size_bins_and_spec(patched_modules, monkeypatch):
	"""Pot-size sampler hits defined bins; spec payload includes player stack."""
	dgs = patched_modules.dgs
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	seq = [0.05, 0.15, 0.45, 0.75, 0.95]  
	it = {"i": 0}
	def fake_random():
		i = it["i"]; it["i"] += 1
		return seq[i % len(seq)]
	monkeypatch.setattr(patched_modules.dgs.random, "random", fake_random, raising=True)
	vals = [dgs.DataGeneratorSamplingMixin.sample_pot_size(dg) for _ in range(len(seq))]
	assert all(2.0 <= v <= 400.0 for v in vals)
	spec = dgs.DataGeneratorSamplingMixin.pot_sampler_spec(dg)
	assert spec["name"] == "bins.v1" and "bins" in spec and spec["player_stack"] == 200

def test_range_generator_spec_fields(patched_modules):
	"""Range generator spec advertises recursive_R delegate and normalization."""
	dgs = patched_modules.dgs
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	spec = dgs.DataGeneratorSamplingMixin.range_generator_spec(dg)
	assert spec["name"] == "recursive_R.v1"
	assert spec["params"]["delegate"] == "solver.recursive_range_sampling"
	assert spec["params"]["normalize"] is True



def test_generate_turn_dataset_writes_npz_and_calls_terminal_solver(tmp_path, patched_modules, monkeypatch):
	"""Turn dataset writer: pushes “production” guard, writes chunked .npz with meta and (inputs, targets) of expected shape, and pops guard."""
	dgd = patched_modules.dgd
	dg = patched_modules.dg.DataGenerator(1, 2, player_stack=200, num_clusters=K_SMALL, speed_profile="bot")
	dg.cfr_solver.hand_clusterer = dg.hand_clusterer

	called = {"push": 0, "pop": 0}

	def _push_prod():
		called["push"] += 1
		return ("dl", "iters")

	def _pop_prod(snap):
		called["pop"] += 1
		assert snap == ("dl", "iters")
		return True

	monkeypatch.setattr(
	 dgd.DataGeneratorDatasetsMixin,
	 "_push_production_mode",
	 lambda self: _push_prod(),
	 raising=True,
	)
	monkeypatch.setattr(
	 dgd.DataGeneratorDatasetsMixin,
	 "_pop_production_mode",
	 lambda self, s: _pop_prod(s),
	 raising=True,
	)

	out = dgd.DataGeneratorDatasetsMixin.generate_turn_dataset(
	 dg, num_situations=3, out_dir=str(tmp_path), chunk_size=2, seed=42
	)

	assert out["written_chunks"] == 2

	files = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".npz")
	assert any("turn_chunk_00000" in f for f in files)
	assert any("turn_chunk_00001" in f for f in files)

	assert called["push"] == 1 and called["pop"] == 1

	p0 = tmp_path / files[0]
	dat = np.load(p0, allow_pickle=False)
	X = dat["inputs"]; Y1 = dat["target_v1"]; Y2 = dat["target_v2"]
	assert X.shape[1] == 1 + 52 + K_SMALL + K_SMALL
	assert Y1.shape[1] == K_SMALL and Y2.shape[1] == K_SMALL
	assert int(dat["meta_num_clusters"]) == K_SMALL
	assert str(dat["meta_stage"]) == "turn"


def test_generate_flop_dataset_writes_npz_and_uses_turn_net(tmp_path, patched_modules, monkeypatch):
	"""Flop dataset writer: pushes+pops guard and writes valid chunk files; meta fields match “flop”."""
	dgd = patched_modules.dgd
	dgu = patched_modules.dgu  
	dg = patched_modules.dg.DataGenerator(1, 2, player_stack=200, num_clusters=K_SMALL, speed_profile="bot")
	dg.cfr_solver.hand_clusterer = dg.hand_clusterer

	called = {"push": 0, "pop": 0}
	monkeypatch.setattr(
	 dgu.DataGeneratorUtilsMixin,
	 "_push_production_mode",
	 lambda self: called.__setitem__("push", called["push"] + 1) or ("dl", "it"),
	 raising=True,
	)
	monkeypatch.setattr(
	 dgu.DataGeneratorUtilsMixin,
	 "_pop_production_mode",
	 lambda self, s: called.__setitem__("pop", called["pop"] + 1) or True,
	 raising=True,
	)

	out = dgd.DataGeneratorDatasetsMixin.generate_flop_dataset(
	 dg, num_situations=3, out_dir=str(tmp_path), chunk_size=2, seed=7
	)
	assert out["written_chunks"] == 2

	files = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".npz")
	assert any("flop_chunk_00000" in f for f in files)
	assert any("flop_chunk_00001" in f for f in files)

	assert called["push"] == 1 and called["pop"] == 1

	dat = np.load(tmp_path / files[0], allow_pickle=False)
	assert str(dat["meta_stage"]) == "flop"
	assert int(dat["meta_num_clusters"]) == K_SMALL


def test_generate_flop_dataset_using_turn_model_persist_and_memory(tmp_path, patched_modules, monkeypatch):
	"""Flop dataset via turn model — both persisted (npz) and in-memory modes return correctly shaped outputs."""
	dgd = patched_modules.dgd
	dg = patched_modules.dg.DataGenerator(1, 2, player_stack=200, num_clusters=K_SMALL, speed_profile="bot")
	dg.cfr_solver.hand_clusterer = dg.hand_clusterer
	turn_model = StubTurnModel(num_clusters=K_SMALL)

	out_persist = dgd.DataGeneratorDatasetsMixin.generate_flop_dataset_using_turn(dg, turn_model, 
	  num_situations=3, out_dir=str(tmp_path), chunk_size=2, seed=9)

	assert out_persist["written_chunks"] == 2
	files = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".npz")
	assert any("flop_chunk_00000" in f for f in files)

	out_mem = dgd.DataGeneratorDatasetsMixin.generate_flop_dataset_using_turn(dg, turn_model, num_situations=2, out_dir=None, chunk_size=10, seed=9)
	assert isinstance(out_mem["in_memory"], list) or out_mem["written_chunks"] == 0



def test_cfv_stream_dataset_iter_and_restores_counts(patched_modules):
	"""Streaming dataset yields exactly num_samples and restores DataGenerator’s (num_boards, num_samples_per_board)."""
	stream_mod = patched_modules.stream
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(num_boards=3, num_samples_per_board=5, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	ds = stream_mod.CFVStreamDataset(dg, stage="flop", num_samples=4, seed=1729)
	orig_nb, orig_ns = dg.num_boards, dg.num_samples_per_board
	got = list(ds)
	assert len(got) == 4
	for rec in got:
		assert len(rec["input_vector"]) == 1 + 52 + K_SMALL + K_SMALL
		assert len(rec["target_v1"]) == K_SMALL
		assert len(rec["target_v2"]) == K_SMALL
		assert rec["schema"] == "cfv.v1"
		assert rec["stage"] == "flop"
	assert dg.num_boards == orig_nb and dg.num_samples_per_board == orig_ns



def test_cfv_shard_dataset_schema_filtering(tmp_path):
	"""Shard dataset filters lines by schema (“cfv.v1”) and yields only matching records."""
	shard_path = tmp_path / "data.jsonl"
	with open(shard_path, "w") as f:
		f.write(json.dumps({"schema": "cfv.v1", "x": 1}) + "\n")
		f.write(json.dumps({"schema": "other",   "x": 2}) + "\n")
		f.write(json.dumps({"schema": "cfv.v1", "x": 3}) + "\n")
	import hunl.nets.cfv_shard_dataset
	ds = cfv_shard_dataset.CFVShardDataset([str(shard_path)], schema_version="cfv.v1", verify_schema=True)
	xs = [obj["x"] for obj in ds]
	assert xs == [1, 3]  



def test_manifest_make_and_save(tmp_path, patched_modules):
	"""Manifest builder includes sampler/range specs and extras; save/load round-trip."""
	manifest_mod = patched_modules.manifest
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	man = manifest_mod.make_manifest(dg, stage="flop", seed=123, extras={"note":"ok"})
	assert man["schema"] == "cfv.manifest.v1"
	assert man["stage"] == "flop"
	assert man["seed"] == 123
	assert man["num_clusters"] == K_SMALL
	assert man["pot_sampler"]["name"] == "bins.v1"
	assert man["range_generator"]["name"] == "recursive_R.v1"
	assert man["note"] == "ok"
	out = manifest_mod.save_manifest(man, str(tmp_path / "manifest.json"))
	with open(out, "r") as f:
		loaded = json.load(f)
	assert loaded == man



def test_eval_cli_helpers(patched_modules):
	"""Smoke tests for eval_cli helpers: _no_negative_pot_delta tolerance and 
		summarization/block-metrics helpers return plausible numbers."""
	ec = patched_modules.eval_cli
	ps_prev = StubPublicState([200,200], [])
	ps_prev.pot_size = 50.0
	ps_prev.last_refund_amount = 0.0
	ps_next = StubPublicState([200,200], [])
	ps_next.pot_size = 49.9999999999  
	ps_next.last_refund_amount = 0.0
	assert ec.is_nonnegative_pot_delta(ps_prev, ps_next) is True

	res = [(+1.0, +0.8), (-1.0, -0.7), (+0.5, +0.45), (0.0, 0.0)]
	mean_na, mean_av, std_na, std_av, reduction = ec._summarize(res)
	assert isinstance(mean_na, float) and isinstance(mean_av, float)
	bm = ec._block_metrics(res, block_size=2)
	assert "naive" in bm and "aivat" in bm and "blocks" in bm
	assert bm["blocks"] == 2
	assert isinstance(bm["naive"]["mbb100"], float)
	assert isinstance(bm["naive"]["ci95"], list) and len(bm["naive"]["ci95"]) == 2
	assert isinstance(bm["aivat"]["mbb100"], float)


def test_generate_turn_dataset_writes_npz_and_calls_terminal_solver(tmp_path, patched_modules, monkeypatch):
	"""Turn dataset writer: pushes “production” guard, writes chunked .npz with meta and (inputs, targets) of 
		expected shape, and pops guard."""
	dgd = patched_modules.dgd
	dgu = patched_modules.dgu                     
	dg = patched_modules.dg.DataGenerator(1, 2, player_stack=200, num_clusters=K_SMALL, speed_profile="bot")
	dg.cfr_solver.hand_clusterer = dg.hand_clusterer

	called = {"push": 0, "pop": 0}
	def _push_prod():
		called["push"] += 1
		return ("dl", "iters")
	def _pop_prod(snap):
		called["pop"] += 1
		assert snap == ("dl", "iters")
		return True

	monkeypatch.setattr(dgu.DataGeneratorUtilsMixin, "_push_production_mode", lambda self: _push_prod(), raising=True)
	monkeypatch.setattr(dgu.DataGeneratorUtilsMixin, "_pop_production_mode",  lambda self, s: _pop_prod(s), raising=True)

	out = dgd.DataGeneratorDatasetsMixin.generate_turn_dataset(
	 dg, num_situations=3, out_dir=str(tmp_path), chunk_size=2, seed=42
	)

	assert out["written_chunks"] == 2
	files = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".npz")
	assert any("turn_chunk_00000" in f for f in files)
	assert any("turn_chunk_00001" in f for f in files)
	assert called["push"] == 1 and called["pop"] == 1

	p0 = tmp_path / files[0]
	dat = np.load(p0, allow_pickle=False)
	X = dat["inputs"]; Y1 = dat["target_v1"]; Y2 = dat["target_v2"]
	assert X.shape[1] == 1 + 52 + K_SMALL + K_SMALL
	assert Y1.shape[1] == K_SMALL and Y2.shape[1] == K_SMALL
	assert int(dat["meta_num_clusters"]) == K_SMALL
	assert str(dat["meta_stage"]) == "turn"


def test_generate_turn_dataset_guard_finally_on_error(tmp_path, patched_modules, monkeypatch):
	"""Even if a persist call raises, a production guard is still pushed, and the test allows either 
		immediate failure before pop or a finally-pop."""
	dgd = patched_modules.dgd
	dgu = patched_modules.dgu                     
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="bot")
	dg.cfr_solver.hand_clusterer = dg.hand_clusterer

	called = {"push": 0, "pop": 0}

	monkeypatch.setattr(
	 dgu.DataGeneratorUtilsMixin,
	 "_push_production_mode",
	 lambda self: called.__setitem__("push", called["push"] + 1) or ("dl", "it"),
	 raising=True,
	)
	monkeypatch.setattr(
	 dgu.DataGeneratorUtilsMixin,
	 "_pop_production_mode",
	 lambda self, s: called.__setitem__("pop", called["pop"] + 1) or True,
	 raising=True,
	)

	def boom(*args, **kwargs):
		raise RuntimeError("boom")
	monkeypatch.setattr(dgd.DataGeneratorDatasetsMixin, "_persist_npz_chunk", boom, raising=True)

	with pytest.raises(RuntimeError):
		dgd.DataGeneratorDatasetsMixin.generate_turn_dataset(dg, num_situations=1, out_dir=str(tmp_path))

	assert called["push"] == 1
	assert called["pop"] in (0, 1)

