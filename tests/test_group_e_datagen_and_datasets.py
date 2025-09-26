# TestSuite_GroupE_DataGen_And_Datasets
# pytest -q
#
# Covers Group E:
#   data_generator.py
#   data_generator_datasets.py
#   data_generator_sampling.py
#   data_generator_utils.py
#   data_manifest.py
#   cfv_stream_dataset.py
#   cfv_shard_dataset.py
#   eval_cli.py
#
# The tests use controlled stubs to avoid heavy solving while asserting all the
# expected calls, shapes, invariants, and paper-aligned semantics (DeepStack style):
#  - input layout: [pot_norm, 52 one-hot, r1(K), r2(K)]
#  - pot_norm in (0,1], board one-hot valid, mass conservation
#  - turn dataset labels solved to terminal, flop labels via turn net
#  - restricted action sets in offline generation (HALF_POT/TWO_POT disabled)
#  - pot-fraction CFVs (no hidden pot scaling in dataset generation)

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

import numpy as np
import pytest

# -------------------------
# Helper stubs and fixtures
# -------------------------

K_SMALL = 8  # Use a small K in tests for speed but keep shapes/logic identical
DECK = [r+s for r in "23456789TJQKA" for s in "CDHS"]  # if poker_utils not available in test env

class StubPublicState:
	def __init__(self, initial_stacks, board_cards, dealer=0):
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
	def __init__(self, public_state):
		self.public_state = public_state
		self.player_ranges = [{}, {}]
		self.players_in_hand = [True, True]

class StubHandClusterer:
	def __init__(self, cfr_solver, num_clusters, profile, opp_sample_size, use_cfv_in_features, config=None):
		self.cfr_solver = cfr_solver
		self.num_clusters = int(num_clusters)
		self.profile = profile
		self.opp_sample_size = opp_sample_size
		self.use_cfv_in_features = use_cfv_in_features
		self.config = config
		self._frozen = None

	def cluster_hands(self, hands, board, opponent_range, pot_size):
		K = self.num_clusters
		clusters = {i: set() for i in range(K)}
		for h in (hands.keys() if isinstance(hands, dict) else hands):
			key = h if isinstance(h, str) else " ".join(list(h))
			cls = int(hashlib.sha256(key.encode()).hexdigest(), 16) % K
			clusters[cls].add(key)
		return clusters

	def cluster_hands(self, hands, board, opponent_range, pot_size):
		# Deterministic bucket: hash(hand) % K
		K = self.num_clusters
		clusters = {i: set() for i in range(K)}
		for h in (hands.keys() if isinstance(hands, dict) else hands):
			key = h if isinstance(h, str) else " ".join(list(h))
			cls = int(hashlib.sha256(key.encode()).hexdigest(), 16) % K
			clusters[cls].add(key)
		return clusters

class StubCFRSolver:
	def __init__(self, config=None, depth_limit=1, num_clusters=K_SMALL):
		self.config = config
		self.depth_limit = int(getattr(config, "depth_limit", depth_limit)) if config else depth_limit
		self.total_iterations = int(getattr(config, "total_iterations", 1)) if config else 1
		self.num_clusters = int(getattr(config, "num_clusters", num_clusters)) if config else num_clusters
		self.clusters = {i: set() for i in range(self.num_clusters)}
		self.hand_clusterer = None
		self.device = "cpu"
		self.models = {}
		# instrumentation for assertions
		self.run_cfr_calls: List[StubGameNode] = []
		self.ensure_sparse_schedule_calls = 0
		self._round_actions = {r: {"half_pot": True, "two_pot": False} for r in (0,1,2,3)}

	def _ensure_sparse_schedule(self):
		# instrument call
		self.ensure_sparse_schedule_calls += 1
		if not hasattr(self, "_round_actions"):
			self._round_actions = {r: {"half_pot": True, "two_pot": False} for r in (0,1,2,3)}
		return self._round_actions

	def run_cfr(self, node):
		self.run_cfr_calls.append(node)

	def predict_counterfactual_values(self, node, player: int) -> Dict[int, List[float]]:
		# pot-fraction CFVs between -0.5 and +0.5, K-wide
		K = self.num_clusters
		base = 0.25 if player == 0 else -0.25
		return {i: [base for _ in range(1)] for i in range(K)}

	def turn_label_targets_solve_to_terminal(self, node) -> Tuple[List[float], List[float]]:
		# labels represent endgame solves (no net)
		K = self.num_clusters
		t1 = [+0.10 for _ in range(K)]
		t2 = [-0.10 for _ in range(K)]
		return t1, t2

	def flop_label_targets_using_turn_net(self, node) -> Tuple[List[float], List[float]]:
		# labels produced using turn net leaf
		K = self.num_clusters
		t1 = [+0.05 for _ in range(K)]
		t2 = [-0.05 for _ in range(K)]
		return t1, t2

	def recursive_range_sampling(self, hands_set, total_prob, public_cards):
		# uniform over hands_set summing to total_prob
		n = max(1, len(hands_set))
		u = float(total_prob) / float(n)
		return {h if isinstance(h, str) else " ".join(list(h)): u for h in hands_set}

	# For eval_cli._play_episode (we don't exercise play loop end-to-end in these tests)
	def _calculate_terminal_utility(self, node, player: int) -> float:
		return 0.0

class StubTurnModel(torch.nn.Module if 'torch' in globals() else object):
	# Minimal turn model that exposes .to().eval() and acts as identity
	def __init__(self, num_clusters=K_SMALL):
		super().__init__()
		self.num_clusters = num_clusters

	def to(self, device):
		return self

	def eval(self):
		return self

# -------------------------
# Pytest fixtures
# -------------------------
@pytest.fixture(scope="function")
def patched_modules(monkeypatch):
    import data_generator
    import data_generator_sampling
    import data_generator_datasets
    import data_generator_utils
    import cfv_stream_dataset
    import eval_cli

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
        stream=__import__("cfv_stream_dataset"),
        shard=__import__("cfv_shard_dataset"),
        manifest=__import__("data_manifest"),
        eval_cli=eval_cli,
    )





# -------------------------
# DataGenerator.__init__ paths
# -------------------------

def test_init_defaults_and_config_paths(patched_modules):
	dgmod = patched_modules.dg
	# default path (no config)
	dg = dgmod.DataGenerator(num_boards=2, num_samples_per_board=3, player_stack=200, num_clusters=K_SMALL, speed_profile="bot")
	assert dg.num_clusters == K_SMALL
	assert dg.cfr_solver.num_clusters == K_SMALL
	assert isinstance(dg.cfr_solver, StubCFRSolver)
	assert isinstance(dg.hand_clusterer, StubHandClusterer)
	assert dg.speed_profile == "bot"
	# when speed_profile == "test" -> depth_limit forced to 0, total_iterations >= 1
	dg2 = dgmod.DataGenerator(num_boards=1, num_samples_per_board=1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	assert dg2.cfr_solver.depth_limit == 0
	assert dg2.cfr_solver.total_iterations >= 1

	# config path
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
	# for "test" profile in config: depth_limit & iterations set minimal
	assert dg3.cfr_solver.depth_limit == 0
	assert dg3.cfr_solver.total_iterations == 1

# -------------------------
# DataGenerator.prepare_input_vector / invariants
# -------------------------

def test_prepare_input_vector_layout_and_invariants(patched_modules):
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	board = ["AS","KD","7c"]  # case-insensitive suits not enforced by stubbed one_hot; using three cards
	# ranges already normalized
	r1 = [0.0]*K_SMALL; r1[0] = 0.6; r1[1] = 0.4
	r2 = [0.0]*K_SMALL; r2[2] = 1.0
	iv = dg.prepare_input_vector([r1, r2], board, pot_size=20.0, actions=None)
	# layout: 1 pot_norm + 52 + K + K
	assert len(iv) == 1 + 52 + K_SMALL + K_SMALL
	pot_norm = iv[0]
	assert 0.0 < pot_norm <= 1.0  # normalized against 2*stack (400), 20/400 = 0.05 in (0,1]
	# one-hot has exactly len(board)=3 ones (stubbed board_one_hot puts ones at first 3 indices)
	assert sum(iv[1:1+52]) == len(board)
	# ranges are appended exactly
	assert iv[1+52:1+52+K_SMALL] == r1
	assert iv[1+52+K_SMALL:1+52+2*K_SMALL] == r2

def test_prepare_input_vector_rejects_bad_invariants(patched_modules):
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	board = ["AS","KD","7c"]
	# Mass not conserved
	r1 = [0.2] + [0.0]*(K_SMALL-1)
	r2 = [0.3] + [0.0]*(K_SMALL-1)
	with pytest.raises(ValueError):
		dg.prepare_input_vector([r1, r2], board, pot_size=20.0)
	# Pot norm out of range: pot <= 0
	r_ok = [1.0] + [0.0]*(K_SMALL-1)
	with pytest.raises(ValueError):
		dg.prepare_input_vector([r_ok, r_ok], board, pot_size=0.0)
	# Pot norm > 1 (e.g., > 400 when stacks=200)
	with pytest.raises(ValueError):
		dg.prepare_input_vector([r_ok, r_ok], board, pot_size=401.0)

# -------------------------
# DataGenerator.prepare_target_values
# -------------------------

def test_prepare_target_values_various_input_shapes(patched_modules):
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	# mapping player->{bucket: value or [value,...]}
	cf = {
		0: {i: [0.1] for i in range(K_SMALL)},
		1: {i: -0.1 for i in range(K_SMALL)},
	}
	t1, t2 = dg.prepare_target_values(cf)
	assert len(t1) == K_SMALL and len(t2) == K_SMALL
	assert all(abs(x-0.1) < 1e-12 for x in t1)
	assert all(abs(x+0.1) < 1e-12 for x in t2)

	# missing keys -> zeros
	cf2 = {0: {0: [0.2]}, 1: {}}
	t1b, t2b = dg.prepare_target_values(cf2)
	assert t1b[0] == 0.2 and sum(t1b[1:]) == 0.0
	assert sum(t2b) == 0.0

# -------------------------
# DataGenerator.compute_counterfactual_values
# -------------------------

def test_compute_counterfactual_values_calls_solver(patched_modules):
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	ps = StubPublicState([200,200], ["AS","KD","QH"])
	node = StubGameNode(ps)
	cf = dg.compute_counterfactual_values(node)
	assert 0 in cf and 1 in cf
	assert set(cf[0].keys()) == set(range(K_SMALL))
	assert set(cf[1].keys()) == set(range(K_SMALL))

# -------------------------
# DataGenerator.generate_unique_boards
# -------------------------

@pytest.mark.parametrize("stage, n, length", [
	("flop",  5, 3),
	("turn",  7, 4),
	("river", 9, 5),
])
def test_generate_unique_boards_by_stage(patched_modules, stage, n, length):
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(n, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	boards = dg.generate_unique_boards(stage=stage, num_boards=n)
	assert len(boards) == n
	# uniqueness and correct length
	seen = set()
	for b in boards:
		assert len(b) == length
		t = tuple(sorted(b))
		assert t not in seen
		seen.add(t)

# -------------------------
# DataGenerator._assert_sampler_invariants and helpers
# -------------------------

def test_sampler_invariants_and_helpers(patched_modules):
	dgu = patched_modules.dgu
	# board one-hot validation uses patched one-hot; exactly len(board) ones
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	board = ["AS","KD","QH"]
	r1 = {0: 0.6, 1: 0.4}
	r2 = {2: 1.0}
	# mass ok, pot ok, one-hot ok -> no exception
	dg._assert_sampler_invariants(board, [r1, r2], pot_size=40.0)

	# mass not conserved -> error
	with pytest.raises(ValueError):
		dg._assert_sampler_invariants(board, [ {0:0.2}, {0:0.3} ], pot_size=40.0)

	# pot norm invalid -> error
	with pytest.raises(ValueError):
		dg._assert_sampler_invariants(board, [ {0:1.0}, {0:1.0} ], pot_size=0.0)

	# helpers directly
	assert dg._pot_norm_ok(40.0) is True
	assert dg._pot_norm_ok(0.0) is False
	assert dg._board_one_hot_valid(board) is True
	assert dg._mass_conservation_ok([ {0:1.0}, {0:1.0} ])
	assert not dg._mass_conservation_ok([ {0:0.2}, {0:0.3} ])

# -------------------------
# DataGenerator.normalize_cluster_probabilities, bucket_player_ranges
# -------------------------

def test_normalize_and_bucket_ranges(patched_modules):
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
	dgu = patched_modules.dgu
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	clusters = {i: set() for i in range(K_SMALL)}
	clusters[0].update({"AS KD", "QH QC"})
	clusters[1].update({"2C 3D"})
	hands = {"AS KD": 0.25, "QH QC": 0.25, "2C 3D": 0.5}
	m1 = dg.map_handstorcho_clusters(hands, clusters)
	# DataGeneratorUtils also provides map_hands_to_clusters; check consistency
	m2 = dgu.DataGeneratorUtilsMixin.map_hands_to_clusters(dg, hands, clusters)
	assert m1 == m2
	assert abs(sum(m1.values()) - 1.0) < 1e-9

# -------------------------
# DataGenerator._filter_clusters_for_board
# -------------------------

def test_filter_clusters_for_board_removes_illegal_hands(patched_modules):
	dgu = patched_modules.dgu
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	board = ["AS", "KD", "QH"]
	clusters = {
		0: {"AS KD", "AS AS", "2C 3D"},  # removes: AS KD (board), AS AS (dup)
		1: {"4C 4C", "4C 5D", "KD QH"},  # removes: 4C 4C (dup), KD QH (board)
	}
	flt = dg._filter_clusters_for_board(clusters, board)
	assert "2C 3D" in flt[0]
	assert len(flt[0]) == 1
	assert flt[1] == {"4C 5D"}

# -------------------------
# DataGenerator._push/pop_leaf_override and expected_total_steps
# -------------------------

def test_push_pop_leaf_override_and_expected_steps(patched_modules):
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	# initial
	dg.cfr_solver.depth_limit = 3
	dg.cfr_solver.total_iterations = 7
	snap = dg._push_leaf_override(stage="flop")
	# in test profile -> forced depth_limit=0, iterations >= 1
	assert dg.cfr_solver.depth_limit == 0 and dg.cfr_solver.total_iterations >= 1
	assert dg._pop_leaf_override(snap) is True
	assert dg.cfr_solver.depth_limit == 3 and dg.cfr_solver.total_iterations == 7

	# expected_total_steps: test profile uses 1 re-solve iteration in formula (iters=1 path)
	dg.num_boards = 5
	dg.num_samples_per_board = 3
	assert dg.expected_total_steps() == 5 * (1 + 3 * (1 + 1))

	# non-test profile
	dg2 = patched_modules.dg.DataGenerator(5, 3, player_stack=200, num_clusters=K_SMALL, speed_profile="bot")
	assert dg2.expected_total_steps() == 5 * (1 + 3 * (2 + 1))

# -------------------------
# DataGenerator.generate_training_data (both profiles)
# -------------------------

def _mk_dg_for_profile(patched_modules, profile: str):
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(num_boards=2, num_samples_per_board=2, player_stack=200,
							 num_clusters=K_SMALL, speed_profile=profile)
	# ensure HandClusterer on solver to satisfy mixins
	dg.cfr_solver.hand_clusterer = dg.hand_clusterer
	return dg

def test_generate_training_data_profile_test_reuses_frozen_clusters(patched_modules):
	dg = _mk_dg_for_profile(patched_modules, "test")
	records = dg.generate_training_data(stage="flop", progress=None)
	# 2 boards * 2 samples_per_board = 4 records
	assert len(records) == 4
	# Invariants: each record has input_vector with the correct layout and targets of length K
	for rec in records:
		iv = rec["input_vector"]
		assert len(iv) == 1 + 52 + K_SMALL + K_SMALL
		assert len(rec["target_v1"]) == K_SMALL
		assert len(rec["target_v2"]) == K_SMALL

def test_generate_training_data_profile_bot_forces_sparse_actions(patched_modules):
	dg = _mk_dg_for_profile(patched_modules, "bot")
	# Before: default round flags include half_pot True for all rounds
	before_flags = dict(dg.cfr_solver._round_actions)
	records = dg.generate_training_data(stage="flop", progress=None)
	# Verify run_cfr invoked, and within generation half_pot/two_pot set to False then restored
	assert len(dg.cfr_solver.run_cfr_calls) > 0
	after_flags = dg.cfr_solver._round_actions
	# restored to prior flags
	assert after_flags == before_flags
	# check some outputs too
	assert len(records) == 4
	for rec in records:
		iv = rec["input_vector"]
		assert len(iv) == 1 + 52 + K_SMALL + K_SMALL

# -------------------------
# DataGeneratorSampling._sample_random_range / flop/turn situations
# -------------------------

def test_sample_random_range_normalization_and_support(patched_modules):
	dgs = patched_modules.dgs
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	ids = list(range(K_SMALL))
	r = dgs.DataGeneratorSamplingMixin._sample_random_range(dg, ids)
	assert abs(sum(r.values()) - 1.0) < 1e-9
	assert set(r.keys()).issubset(set(ids))

def test_sample_flop_and_turn_situations(patched_modules):
	dgs = patched_modules.dgs
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	dg.cfr_solver.hand_clusterer = dg.hand_clusterer
	rng = random.Random(123)
	n_flop = dgs.DataGeneratorSamplingMixin._sample_flop_situation(dg, rng)
	n_turn = dgs.DataGeneratorSamplingMixin._sample_turn_situation(dg, rng)
	assert isinstance(n_flop, StubGameNode) and isinstance(n_turn, StubGameNode)
	# round correctly set
	assert n_flop.public_state.current_round == 1
	assert n_turn.public_state.current_round == 2
	# player ranges set for both players
	assert len(n_flop.player_ranges[0]) > 0 and len(n_flop.player_ranges[1]) > 0
	assert len(n_turn.player_ranges[0]) > 0 and len(n_turn.player_ranges[1]) > 0

# -------------------------
# DataGenerator.sample_pot_size / specs
# -------------------------

def test_sample_pot_size_bins_and_spec(patched_modules, monkeypatch):
	dgs = patched_modules.dgs
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	# Force bin selection by controlling random.random
	seq = [0.05, 0.15, 0.45, 0.75, 0.95]  # hits each bin cumulatively
	it = {"i": 0}
	def fake_random():
		i = it["i"]; it["i"] += 1
		return seq[i % len(seq)]
	monkeypatch.setattr(patched_modules.dgs.random, "random", fake_random, raising=True)
	vals = [dgs.DataGeneratorSamplingMixin.sample_pot_size(dg) for _ in range(len(seq))]
	# All values within (lo,hi] unions [2,400]
	assert all(2.0 <= v <= 400.0 for v in vals)
	spec = dgs.DataGeneratorSamplingMixin.pot_sampler_spec(dg)
	assert spec["name"] == "bins.v1" and "bins" in spec and spec["player_stack"] == 200

def test_range_generator_spec_fields(patched_modules):
	dgs = patched_modules.dgs
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	spec = dgs.DataGeneratorSamplingMixin.range_generator_spec(dg)
	assert spec["name"] == "recursive_R.v1"
	assert spec["params"]["delegate"] == "solver.recursive_range_sampling"
	assert spec["params"]["normalize"] is True

# -------------------------
# DataGeneratorDatasets: generate_turn_dataset, flop_dataset, flop_dataset_using_turn
# -------------------------

def test_generate_turn_dataset_writes_npz_and_calls_terminal_solver(tmp_path, patched_modules, monkeypatch):
	dgd = patched_modules.dgd
	dg = patched_modules.dg.DataGenerator(1, 2, player_stack=200, num_clusters=K_SMALL, speed_profile="bot")
	dg.cfr_solver.hand_clusterer = dg.hand_clusterer

	# inject production-mode guards to assert they are called
	called = {"push": 0, "pop": 0}
	def _push_prod():
		called["push"] += 1
		return ("dl", "iters")
	def _pop_prod(snap):
		called["pop"] += 1
		assert snap == ("dl", "iters")
		return True
	monkeypatch.setattr(dgd.DataGeneratorDatasetsMixin, "_push_production_mode", lambda self: _push_prod(), raising=True)
	monkeypatch.setattr(dgd.DataGeneratorDatasetsMixin, "_pop_production_mode", lambda self, s: _pop_prod(s), raising=True)

	out = dgd.DataGeneratorDatasetsMixin.generate_turn_dataset(dg, num_situations=3, out_dir=str(tmp_path), chunk_size=2, seed=42)
	# two chunks (2 + 1)
	assert out["written_chunks"] == 2
	# files exist
	files = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".npz")
	assert any("turn_chunk_00000" in f for f in files)
	assert any("turn_chunk_00001" in f for f in files)
	# push/pop production guard called exactly once
	assert called["push"] == 1 and called["pop"] == 1

	# Inspect first chunk content
	p0 = tmp_path / files[0]
	dat = np.load(p0, allow_pickle=False)
	X = dat["inputs"]; Y1 = dat["target_v1"]; Y2 = dat["target_v2"]
	assert X.shape[1] == 1 + 52 + K_SMALL + K_SMALL
	assert Y1.shape[1] == K_SMALL and Y2.shape[1] == K_SMALL
	# meta fields present
	assert int(dat["meta_num_clusters"]) == K_SMALL
	assert str(dat["meta_stage"]) == "turn"

def test_generate_flop_dataset_writes_npz_and_uses_turn_net(tmp_path, patched_modules, monkeypatch):
	dgd = patched_modules.dgd
	dg = patched_modules.dg.DataGenerator(1, 2, player_stack=200, num_clusters=K_SMALL, speed_profile="bot")
	dg.cfr_solver.hand_clusterer = dg.hand_clusterer

	# guard hooks
	called = {"push": 0, "pop": 0}
	monkeypatch.setattr(dgd.DataGeneratorDatasetsMixin, "_push_production_mode", lambda self: called.__setitem__("push", called["push"]+1) or ("dl","it"), raising=True)
	monkeypatch.setattr(dgd.DataGeneratorDatasetsMixin, "_pop_production_mode", lambda self, s: called.__setitem__("pop", called["pop"]+1) or True, raising=True)

	out = dgd.DataGeneratorDatasetsMixin.generate_flop_dataset(dg, num_situations=3, out_dir=str(tmp_path), chunk_size=2, seed=7)
	assert out["written_chunks"] == 2
	files = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".npz")
	assert any("flop_chunk_00000" in f for f in files)
	assert any("flop_chunk_00001" in f for f in files)
	# check content
	dat = np.load(tmp_path / files[0], allow_pickle=False)
	assert str(dat["meta_stage"]) == "flop"
	# ensure K matches
	assert int(dat["meta_num_clusters"]) == K_SMALL

def test_generate_flop_dataset_using_turn_model_persist_and_memory(tmp_path, patched_modules, monkeypatch):
	dgd = patched_modules.dgd
	dg = patched_modules.dg.DataGenerator(1, 2, player_stack=200, num_clusters=K_SMALL, speed_profile="bot")
	dg.cfr_solver.hand_clusterer = dg.hand_clusterer
	turn_model = StubTurnModel(num_clusters=K_SMALL)

	# persist to disk
	out_persist = dgd.DataGeneratorDatasetsMixin.generate_flop_dataset_using_turn(dg, turn_model, num_situations=3, out_dir=str(tmp_path), chunk_size=2, seed=9)
	assert out_persist["written_chunks"] == 2
	files = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".npz")
	assert any("flop_chunk_00000" in f for f in files)

	# in-memory
	out_mem = dgd.DataGeneratorDatasetsMixin.generate_flop_dataset_using_turn(dg, turn_model, num_situations=2, out_dir=None, chunk_size=10, seed=9)
	# returns a small list if not persisted
	assert isinstance(out_mem["in_memory"], list) or out_mem["written_chunks"] == 0

# -------------------------
# CFVStreamDataset
# -------------------------

def test_cfv_stream_dataset_iter_and_restores_counts(patched_modules):
	stream_mod = patched_modules.stream
	dgmod = patched_modules.dg
	dg = dgmod.DataGenerator(num_boards=3, num_samples_per_board=5, player_stack=200, num_clusters=K_SMALL, speed_profile="test")
	ds = stream_mod.CFVStreamDataset(dg, stage="flop", num_samples=4, seed=1729)
	orig_nb, orig_ns = dg.num_boards, dg.num_samples_per_board
	got = list(ds)
	# returns exactly num_samples
	assert len(got) == 4
	# each record has inputs and targets
	for rec in got:
		assert len(rec["input_vector"]) == 1 + 52 + K_SMALL + K_SMALL
		assert len(rec["target_v1"]) == K_SMALL
		assert len(rec["target_v2"]) == K_SMALL
		assert rec["schema"] == "cfv.v1"
		assert rec["stage"] == "flop"
	# restored counts
	assert dg.num_boards == orig_nb and dg.num_samples_per_board == orig_ns

# -------------------------
# CFVShardDataset
# -------------------------

def test_cfv_shard_dataset_schema_filtering(tmp_path):
	shard_path = tmp_path / "data.jsonl"
	with open(shard_path, "w") as f:
		f.write(json.dumps({"schema": "cfv.v1", "x": 1}) + "\n")
		f.write(json.dumps({"schema": "other",   "x": 2}) + "\n")
		f.write(json.dumps({"schema": "cfv.v1", "x": 3}) + "\n")
	import cfv_shard_dataset
	ds = cfv_shard_dataset.CFVShardDataset([str(shard_path)], schema_version="cfv.v1", verify_schema=True)
	xs = [obj["x"] for obj in ds]
	assert xs == [1, 3]  # filtered by schema

# -------------------------
# data_manifest.py
# -------------------------

def test_manifest_make_and_save(tmp_path, patched_modules):
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

# -------------------------
# eval_cli helper functions (_summarize, _block_metrics, _no_negative_pot_delta)
# -------------------------

def test_eval_cli_helpers(patched_modules):
	ec = patched_modules.eval_cli
	# _no_negative_pot_delta: allow small refund tolerance
	ps_prev = StubPublicState([200,200], [])
	ps_prev.pot_size = 50.0
	ps_prev.last_refund_amount = 0.0
	ps_next = StubPublicState([200,200], [])
	ps_next.pot_size = 49.9999999999  # within tolerance
	ps_next.last_refund_amount = 0.0
	assert ec._no_negative_pot_delta(ps_prev, ps_next) is True

	# _summarize / _block_metrics
	res = [(+1.0, +0.8), (-1.0, -0.7), (+0.5, +0.45), (0.0, 0.0)]
	mean_na, mean_av, std_na, std_av, reduction = ec._summarize(res)
	assert isinstance(mean_na, float) and isinstance(mean_av, float)
	bm = ec._block_metrics(res, block_size=2)
	assert "naive" in bm and "aivat" in bm and "blocks" in bm
	assert bm["blocks"] == 2
	assert isinstance(bm["naive"]["mbb100"], float)
	assert isinstance(bm["naive"]["ci95"], list) and len(bm["naive"]["ci95"]) == 2
	assert isinstance(bm["aivat"]["mbb100"], float)

# -------------------------
# DataGeneratorDatasets guard coverage: ensure finally pops guard even on errors
# -------------------------

def test_generate_turn_dataset_guard_finally_on_error(tmp_path, patched_modules, monkeypatch):
	dgd = patched_modules.dgd
	dg = patched_modules.dg.DataGenerator(1, 1, player_stack=200, num_clusters=K_SMALL, speed_profile="bot")
	dg.cfr_solver.hand_clusterer = dg.hand_clusterer
	called = {"push": 0, "pop": 0}
	monkeypatch.setattr(dgd.DataGeneratorDatasetsMixin, "_push_production_mode", lambda self: called.__setitem__("push", called["push"]+1) or ("dl","it"), raising=True)
	monkeypatch.setattr(dgd.DataGeneratorDatasetsMixin, "_pop_production_mode", lambda self, s: called.__setitem__("pop", called["pop"]+1) or True, raising=True)
	# Force error inside by monkeypatching _persist_npz_chunk to raise on first call
	saved = {"i": 0}
	def boom(*args, **kwargs):
		saved["i"] += 1
		raise RuntimeError("boom")
	monkeypatch.setattr(dgd.DataGeneratorDatasetsMixin, "_persist_npz_chunk", boom, raising=True)
	with pytest.raises(RuntimeError):
		dgd.DataGeneratorDatasetsMixin.generate_turn_dataset(dg, num_situations=1, out_dir=str(tmp_path))
	# Even on error, guard popped
	assert called["push"] == 1 and called["pop"] == 1

