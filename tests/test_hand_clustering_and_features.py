import os
import math
import types
import random
import pytest

import numpy as np

from hunl.ranges.hand_clusterer import HandClusterer
from hunl.ranges.hand_clusterer_features import HandClustererFeaturesMixin
from hunl.ranges.hand_clusterer_utils import HandClustererUtilsMixin




HANDS_SMALL = [
	"AH AC",
	"KD KC",
	"QH QC",
	"TS TD",
	"8H 8C",
	"7S 6S",
	"5H 5S",
	"3D 3H",
]

BOARD_FLOP = ["2C", "7D", "JH"]
BOARD_TURN = ["2C", "7D", "JH", "QS"]
BOARD_RIVER = ["2C", "7D", "JH", "QS", "9C"]

class DummyCFR:
	"""Lightweight stand-in for a CFR solver; supplies a simple _player_wins used by HandClusterer tests."""

	def __init__(self, clusters=None):
		"""Initialize with an optional {cluster_id: set(hands)} mapping consumed by the clusterer."""
		self.clusters = clusters or {}

	@staticmethod
	def _rank_val(card):
		"""Map a string card like 'AH' to an integer rank value used by the toy win comparator."""
		r = card[0].upper()
		order = "23456789TJQKA"
		return 2 + order.index(r)

	def _player_wins(self, player_hand, opponent_hand, board):
		"""Return 1/−1/0 by comparing rank-sum heuristics of (hand+board); deterministic and cheap."""
		ps = sum(self._rank_val(c) for c in player_hand + board)
		os = sum(self._rank_val(c) for c in opponent_hand + board)
		if ps > os:
			return 1
		if ps < os:
			return -1
		return 0


@pytest.fixture
def dummy_cfr():
	"""Provide a shared DummyCFR instance for clusterer/unit tests."""
	return DummyCFR()

@pytest.fixture
def clusterer_bot(dummy_cfr):
	"""HandClusterer in 'bot' profile (full MC defaults, CFV features on, no opponent sampling)."""
	return HandClusterer(
		cfr_solver=dummy_cfr,
		num_clusters=4,
		profile="bot",
		use_cfv_in_features=True,
		opp_sample_size=None,
	)

@pytest.fixture
def clusterer_test_env(dummy_cfr, monkeypatch):
	"""HandClusterer in 'test' profile with FAST_TESTS set: zero-MC path, no CFV features, opp_sample_size=0."""
	monkeypatch.setenv("FAST_TESTS", "1")
	return HandClusterer(
		cfr_solver=dummy_cfr,
		num_clusters=5,
		profile="test",
		use_cfv_in_features=False,
		opp_sample_size=0,
	)

@pytest.fixture
def clusterer_with_config(dummy_cfr):
	"""HandClusterer built from a config namespace to verify test-profile overrides and hyperparams."""
	Cfg = types.SimpleNamespace(
		num_clusters=6,
		mc_samples_win=17,
		mc_samples_potential=11,
		tau_re=0.42,
		drift_sample_size=13,
		profile="test",
		opp_sample_size=None,
		use_cfv_in_features=True,
		fast_test_seed=12345,
		debug_fast_tests=True,
	)
	return HandClusterer(cfr_solver=dummy_cfr, config=Cfg)


def test_init_bot_profile_defaults(clusterer_bot):
	"""test_init_bot_profile_defaults: “bot” profile picks full-MC defaults (win/potential=200), 
	allows CFV features, no opponent sampling, K respected."""
	c = clusterer_bot
	assert c.profile == "bot"
	assert c.num_clusters == 4
	assert c._mc_samples_win == 200
	assert c._mc_samples_potential == 200
	assert c.opp_sample_size is None
	assert c.use_cfv_in_features is True

def test_init_test_profile_enforced_flags(clusterer_test_env):
	"""test_init_test_profile_enforced_flags: “test” profile forces zero MC samples, 
	opp_sample_size=0, disables CFV features."""
	c = clusterer_test_env
	assert c.profile == "test"
	assert c._mc_samples_win == 0
	assert c._mc_samples_potential == 0
	assert c.opp_sample_size == 0
	assert c.use_cfv_in_features is False

def test_init_with_config_test_profile(clusterer_with_config):
	"""test_init_with_config_test_profile: config object in “test” profile overrides 
		K and hyperparameters; debug_fast_tests implies enforced test flags."""
	c = clusterer_with_config
	assert c.profile == "test"
	assert c.num_clusters == 6
	assert c._mc_samples_win == 0
	assert c._mc_samples_potential == 0
	assert c.use_cfv_in_features is False
	assert c.tau_re == pytest.approx(0.42)
	assert c.drift_sample_size == 13

def test_preflop_partition_and_fit(dummy_cfr):
	"""test_preflop_partition_and_fit: preflop fit returns categorical buckets (no centroids), 
	contains all input hands, 
	and assign() maps a suited hand into a valid bucket."""
	c = HandClusterer(dummy_cfr, num_clusters=1000, profile="bot")
	result = c.fit(hands={"AH AC", "KH KS", "7S 6S", "7H 6H", "TD 9C"}, board=[], opponent_range={}, pot_size=100.0)
	assert isinstance(result, dict)
	assert c.clusters == result
	assert c.centroids is None
	merged = set().union(*result.values()) if result else set()
	assert merged == {"AH AC", "KH KS", "7S 6S", "7H 6H", "TD 9C"}
	cid_76s = c.assign("7D 6D", board=[])
	assert cid_76s in result

def test_cluster_hands_test_profile_fast(monkeypatch, clusterer_test_env):
	"""test_cluster_hands_test_profile_fast: in “test” mode, cluster_hands is deterministic 
		and returns the same cached structure across calls."""
	c = clusterer_test_env
	clusters_first = c.cluster_hands(HANDS_SMALL, BOARD_FLOP, {}, 80.0)
	assert isinstance(clusters_first, dict)
	merged = set().union(*clusters_first.values()) if clusters_first else set()
	assert merged == set(HANDS_SMALL)
	clusters_second = c.cluster_hands(HANDS_SMALL, BOARD_FLOP, {}, 80.0)
	assert clusters_first is clusters_second

def test_cluster_hands_test_profile_without_env_uses_config(dummy_cfr, monkeypatch):
	"""test_cluster_hands_test_profile_without_env_uses_config: without FAST_TESTS env,
	a “test” config with debug_fast_tests still drives the fast test path."""
	monkeypatch.delenv("FAST_TESTS", raising=False)
	Cfg = types.SimpleNamespace(
		num_clusters=4,
		mc_samples_win=0,
		mc_samples_potential=0,
		tau_re=999.0,
		drift_sample_size=10,
		profile="test",
		opp_sample_size=0,
		use_cfv_in_features=False,
		fast_test_seed=77,
		debug_fast_tests=True,
	)
	c = HandClusterer(dummy_cfr, config=Cfg)
	clusters = c.cluster_hands(HANDS_SMALL, BOARD_FLOP, {}, 50.0)
	assert isinstance(clusters, dict)
	assert set().union(*clusters.values()) == set(HANDS_SMALL)

def test_cluster_hands_bot_profile_medoids_and_separation(monkeypatch, clusterer_bot):
	"""test_cluster_hands_bot_profile_medoids_and_separation: in “bot” mode, feature-based k-means-style 
		clustering separates easy two-mode data; result is stable across calls."""
	c = clusterer_bot
	BINS = 21
	sig_low = np.zeros(BINS, float); sig_low[0] = 1.0
	sig_high = np.zeros(BINS, float); sig_high[-1] = 1.0

	original_chf = HandClustererFeaturesMixin.calculate_hand_features

	def fake_features(self, hand, board, opponent_range, pot_size):
		idx = HANDS_SMALL.index(hand)
		return sig_low if idx < len(HANDS_SMALL)//2 else sig_high

	monkeypatch.setattr(HandClustererFeaturesMixin, "calculate_hand_features", fake_features)

	try:
		K = 2
		c.set_num_clusters(K)
		clusters = c.cluster_hands(HANDS_SMALL, BOARD_TURN, {}, 120.0)
		assert isinstance(clusters, dict)
		non_empty = [k for k, v in clusters.items() if len(v) > 0]
		assert len(non_empty) == K
		sizes = sorted(len(v) for v in clusters.values())
		assert sizes == [len(HANDS_SMALL)//2, len(HANDS_SMALL) - len(HANDS_SMALL)//2]
		clusters2 = c.cluster_hands(HANDS_SMALL, BOARD_TURN, {}, 120.0)
		assert clusters == clusters2
	finally:
		monkeypatch.setattr(HandClustererFeaturesMixin, "calculate_hand_features", original_chf)

def test_cluster_hands_bot_profile_k_adjusts_when_k_gt_n(clusterer_bot):
	"""test_cluster_hands_bot_profile_k_adjusts_when_k_gt_n: when K>N, effective K is
		reduced to N (one hand per non-empty cluster)."""
	c = clusterer_bot
	c.set_num_clusters(len(HANDS_SMALL) + 5)
	clusters = c.cluster_hands(HANDS_SMALL, BOARD_TURN, {}, 10.0)
	assert len([k for k, v in clusters.items()]) == len(HANDS_SMALL)

def test_cluster_hands_bot_profile_empty_input(clusterer_bot):
	"""test_cluster_hands_bot_profile_empty_input: empty input yields 
		empty clusters/centroids and resets last-features cache."""
	c = clusterer_bot
	out = c.cluster_hands([], BOARD_TURN, {}, 10.0)
	assert out == {}
	assert c.clusters == {}
	assert c._last_features == {}

def test_assign_postflop_membership(clusterer_test_env):
	"""test_assign_postflop_membership: assign() returns the exact cluster id 
		for a known member on the same board."""
	c = clusterer_test_env
	clusters = c.cluster_hands(HANDS_SMALL, BOARD_FLOP, {}, 80.0)
	for cid, hset in clusters.items():
		if hset:
			h0 = next(iter(hset))
			got = c.assign(h0, board=BOARD_FLOP)
			assert got == cid
			break

def test_assign_unknown_hand_falls_back_to_hash_bucket(clusterer_test_env):
	"""test_assign_unknown_hand_falls_back_to_hash_bucket: unknown hand maps to a 
		valid (hash-based) bucket id."""
	c = clusterer_test_env
	c.cluster_hands(HANDS_SMALL, BOARD_FLOP, {}, 80.0)
	cids = set(c.clusters.keys())
	out_cid = c.assign("2C 3D", board=BOARD_FLOP)
	assert out_cid in cids

def test_set_num_clusters_roundtrip(clusterer_bot):
	"""test_set_num_clusters_roundtrip: set_num_clusters mutates and returns the new K."""
	assert clusterer_bot.set_num_clusters(7) == 7
	assert clusterer_bot.num_clusters == 7

def test_get_cluster_ranges_uniform(clusterer_bot):
	"""test_get_cluster_ranges_uniform: get_cluster_ranges() returns a uniform simplex
		over K buckets and sums to 1."""
	c = clusterer_bot
	c.set_num_clusters(4)
	r = c.get_cluster_ranges()
	assert set(r.keys()) == {0,1,2,3}
	assert all(v == pytest.approx(0.25) for v in r.values())
	assert pytest.approx(sum(r.values()), rel=1e-9) == 1.0

def test_hands_to_bucket_range_and_normalization(clusterer_test_env):
	"""test_hands_to_bucket_range_and_normalization: hands→bucket distribution normalizes 
		to 1 and uses current clusters as support."""
	c = clusterer_test_env
	clusters = c.cluster_hands(HANDS_SMALL, BOARD_TURN, {}, 50.0)
	hprob = {h: (i+1) for i, h in enumerate(HANDS_SMALL)}
	br = c.hands_to_bucket_range(hprob)
	assert pytest.approx(sum(br.values()), rel=1e-9) == 1.0
	assert set(br.keys()).issubset(set(clusters.keys()))

def test_bucket_range_to_hand_weights_and_normalization(clusterer_test_env):
	"""test_bucket_range_to_hand_weights_and_normalization: bucket→hand weights distribute 
		mass uniformly across members and re-normalize to 1."""
	c = clusterer_test_env
	clusters = c.cluster_hands(HANDS_SMALL, BOARD_TURN, {}, 50.0)
	non_empty = next(k for k, v in clusters.items() if len(v) > 0)
	br = {non_empty: 1.0}
	hw = c.bucket_range_to_hand_weights(br)
	assert pytest.approx(sum(hw.values()), rel=1e-9) == 1.0
	members = sorted(list(clusters[non_empty]))
	for m in members:
		assert hw[m] == pytest.approx(1.0 / len(members))

def test_persist_and_load_mapping_roundtrip(clusterer_test_env):
	"""test_persist_and_load_mapping_roundtrip: persist_mapping/load_mapping
		round-trips cluster membership and preserves assign() semantics."""
	c = clusterer_test_env
	clusters = c.cluster_hands(HANDS_SMALL, BOARD_TURN, {}, 50.0)
	m = c.persist_mapping()
	c2 = HandClusterer(cfr_solver=DummyCFR(), num_clusters=1, profile="bot")
	ok = c2.load_mapping(m)
	assert ok is True
	assert c2.num_clusters == len(m)
	assert c2.centroids is None
	assert c2._last_features is None
	for cid, hands in clusters.items():
		for h in hands:
			assert c2.assign(h, board=BOARD_TURN) == cid

def test_calculate_hand_features_histogram_and_cache(clusterer_bot):
	"""test_calculate_hand_features_histogram_and_cache: features are a 21-bin 
		normalized histogram; repeated calls hit the cache (hits↑, misses unchanged)."""
	c = clusterer_bot
	c._cache.clear()
	c._cache_hits = 0
	c._cache_misses = 0
	h = HANDS_SMALL[0]
	hist1 = c.calculate_hand_features(h, BOARD_FLOP, opponent_range={"X": 0.7}, pot_size=20.0)
	assert isinstance(hist1, np.ndarray)
	assert hist1.shape == (21,)
	assert pytest.approx(hist1.sum(), rel=1e-9) == 1.0
	assert c._cache_misses == 1 and c._cache_hits == 0
	hist2 = c.calculate_hand_features(h, BOARD_FLOP, opponent_range={}, pot_size=999.0)
	assert c._cache_hits == 1
	assert np.allclose(hist1, hist2)

def test__calculate_equity_repeatability_and_bounds(clusterer_bot):
	"""test__calculate_equity_repeatability_and_bounds: equity is repeatable for
		same inputs and bounded in [0,1]."""
	c = clusterer_bot
	eq1 = c._calculate_equity(HANDS_SMALL[0], BOARD_RIVER, {})
	eq2 = c._calculate_equity(HANDS_SMALL[0], BOARD_RIVER, {})
	assert 0.0 <= eq1 <= 1.0
	assert eq1 == pytest.approx(eq2)

def test__calculate_potential_equity_improvement_zero_at_river(clusterer_bot):
	"""test__calculate_potential_equity_improvement_zero_at_river: potential equity 
		improvement is zero on river (no future cards)."""
	c = clusterer_bot
	pe = c._calculate_potential_equity_improvement(HANDS_SMALL[0], BOARD_RIVER, {})
	assert pe == 0.0

def test__calculate_counterfactual_value_is_signed_fraction(clusterer_bot):
	"""test__calculate_counterfactual_value_is_signed_fraction: CFV proxy on river lies in 
		[-1,1] (win/lose/tie payoffs normalized)."""
	c = clusterer_bot
	cv = c._calculate_counterfactual_value(HANDS_SMALL[0], BOARD_RIVER, {}, pot_size=200.0)
	assert -1.0 <= cv <= 1.0

def test__calculate_payoff_exact_values(clusterer_bot):
	"""test__calculate_payoff_exact_values: payoff sign matches win/lose/tie exactly at terminal."""
	c = clusterer_bot
	h1 = "AH AC"
	h2 = "2D 3C"
	p_win = c._calculate_payoff(h1.split(), h2.split(), BOARD_RIVER, 100.0)
	o_win = c._calculate_payoff(h2.split(), h1.split(), BOARD_RIVER, 100.0)
	tie = c._calculate_payoff(h1.split(), h1.split(), BOARD_RIVER, 100.0)
	assert p_win == 1.0
	assert o_win == -1.0
	assert tie == 0.0

def test__evaluate_win_percentage_terminal_and_sampling(clusterer_bot, monkeypatch):
	"""test__evaluate_win_percentage_terminal_and_sampling: terminal returns {0,0.5,1};
		flop sampling is stochastic but repeatable under fixed seed."""
	c = clusterer_bot
	e_term = c._evaluate_win_percentage(["AH","AC"], ["2D","3C"], BOARD_RIVER)
	assert e_term in (0.0, 0.5, 1.0)
	c._mc_samples_win = 13
	e1 = c._evaluate_win_percentage(["AH","AC"], ["2D","3C"], BOARD_FLOP)
	e2 = c._evaluate_win_percentage(["AH","AC"], ["2D","3C"], BOARD_FLOP)
	assert 0.0 <= e1 <= 1.0
	assert e1 == pytest.approx(e2)

def test_calculate_hand_distance_euclidean(clusterer_bot):
	"""test_calculate_hand_distance_euclidean: distance is standard L2 between 
		feature vectors."""
	c = clusterer_bot
	a = np.array([0.0, 1.0, 2.0])
	b = np.array([0.0, 1.0, 5.0])
	d = c.calculate_hand_distance(a, b)
	assert d == pytest.approx(np.linalg.norm(a - b))

def test_emd_distance_properties(clusterer_bot):
	"""test_emd_distance_properties: EMD helper is 0 on identical dists, symmetric, 
	and additive on simple three-atom chains."""
	c = clusterer_bot
	f = np.array([1.0, 0.0, 0.0])
	g = np.array([0.0, 1.0, 0.0])
	h = np.array([0.0, 0.0, 1.0])
	d_fg = c._emd_distance(f, g)
	d_gh = c._emd_distance(g, h)
	d_fh = c._emd_distance(f, h)
	assert c._emd_distance(f, f) == 0.0
	assert d_fg == pytest.approx(d_gh)
	assert d_fh == pytest.approx(d_fg + d_gh)
	assert c._emd_distance(f, g) == pytest.approx(c._emd_distance(g, f))

def test_compute_drift_behavior(clusterer_bot):
	"""test_compute_drift_behavior: drift is None without prior, then equals the mean L2 delta over sampled overlapping keys with drift_sample_size."""
	c = clusterer_bot
	nf = {"A": np.array([0.1, 0.2]), "B": np.array([0.0, 1.0])}
	assert c._compute_drift(nf) is None
	c._last_features = {"A": np.array([0.1, 0.1]), "B": np.array([0.0, 0.0]), "C": np.array([9,9])}
	c.drift_sample_size = 2
	d = c._compute_drift(nf)
	expect = (np.linalg.norm([0.1,0.2] - np.array([0.1,0.1])) + np.linalg.norm([0.0,1.0] - np.array([0.0,0.0]))) / 2.0
	assert d == pytest.approx(expect)

def test_deterministic_seed_for_clustering_insensitivity_to_range_and_pot(clusterer_bot):
	"""test_deterministic_seed_for_clustering_insensitivity_to_range_and_pot: clustering RNG seed depends only on board, not range/pot."""
	c = clusterer_bot
	s1 = c._deterministic_seed_for_clustering(BOARD_TURN, {"X": 0.3}, 10.0)
	s2 = c._deterministic_seed_for_clustering(BOARD_TURN, {"Y": 0.9, 1: 0.1}, 999.0)
	s3 = c._deterministic_seed_for_clustering(BOARD_FLOP, {}, 0.0)
	assert s1 == s2
	assert s1 != s3

def test_opponent_range_signature_canonicalization(clusterer_bot):
	"""test_opponent_range_signature_canonicalization: opponent-range signature 
		canonicalizes hand keys (“AC QH” == “QH AC”) and merges numeric keys deterministically."""
	c = clusterer_bot
	r1 = {"QH AC": 0.3, 5: 0.7, "AC QH": 0.0}
	r2 = {5: 0.7, "AC QH": 0.0, "QH AC": 0.3}
	sig1 = c._opponent_range_signature(r1)
	sig2 = c._opponent_range_signature(r2)
	assert sig1 == sig2

def test__maybe_sample_items_limit_and_determinism(clusterer_bot):
	"""test__maybe_sample_items_limit_and_determinism: downsampling uses a 
		deterministic RNG and respects opp_sample_size."""
	c = clusterer_bot
	items = {i: i for i in range(10)}
	out1 = c._maybe_sample_items(items, seed=1729)
	out2 = c._maybe_sample_items(items, seed=1729)
	assert out1 == out2
	c.opp_sample_size = 3
	out3 = c._maybe_sample_items(items, seed=1729)
	out4 = c._maybe_sample_items(items, seed=1729)
	assert len(out3) == 3 and out3 == out4

def test_fit_non_preflop_uses_medoids_and_respects_k(clusterer_bot, monkeypatch):
	"""test_fit_non_preflop_uses_medoids_and_respects_k: non-preflop fit clusters all
		hands into ≤K non-empty buckets covering the input set."""
	c = clusterer_bot
	result = c.fit(HANDS_SMALL, board=BOARD_TURN, opponent_range={}, pot_size=75.0)
	assert isinstance(result, dict)
	assert len(result) <= len(HANDS_SMALL)
	merged = set().union(*result.values()) if result else set()
	assert merged == set(HANDS_SMALL)

def test_preflop_not_bucketed_in_aux_network_context():
	"""test_preflop_not_bucketed_in_aux_network_context: preflop fit returns one cluster
		per hand-type class (no k-means/centroids in this path)."""
	c = HandClusterer(DummyCFR(), num_clusters=1000, profile="bot")
	res = c.fit({"AH AC", "KH KC", "7S 6S"}, board=[], opponent_range={}, pot_size=50.0)
	assert isinstance(res, dict)
	assert len(res) == 3

