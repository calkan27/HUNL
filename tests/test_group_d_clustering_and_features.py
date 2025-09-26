# Test Suite: GROUP D — Clustering & Features (DeepStack-faithful)
# File: tests/test_group_d_clustering_and_features.py

import os
import math
import types
import random
import pytest

import numpy as np

# Import units under test
from hand_clusterer import HandClusterer
from hand_clusterer_features import HandClustererFeaturesMixin
from hand_clusterer_utils import HandClustererUtilsMixin

# --- Minimal dummy CFR solver used by the feature mixin -----------------------

class DummyCFR:
    """
    Provides only the bits HandClustererFeaturesMixin needs:
      - self.clusters (when opponent_range uses integer bucket ids)
      - _player_wins(hand, opp_hand, board)
    """
    def __init__(self, clusters=None):
        self.clusters = clusters or {}

    @staticmethod
    def _rank_val(card):
        # card like 'AH', 'TD', '7s' etc (case-insensitive ranks/suits)
        r = card[0].upper()
        order = "23456789TJQKA"
        return 2 + order.index(r)

    def _player_wins(self, player_hand, opponent_hand, board):
        # Very simple deterministic scoring: sum rank values across hand + board.
        # Enough for consistent ordering in tests.
        ps = sum(self._rank_val(c) for c in player_hand + board)
        os = sum(self._rank_val(c) for c in opponent_hand + board)
        if ps > os:
            return 1
        if ps < os:
            return -1
        return 0


# ------------------------ Common small helper data ----------------------------

# Small pool of hands with no mutual card overlaps against default boards below.
HANDS_SMALL = [
    "AH AC",  # strong pair
    "KH KC",
    "2D 3C",
    "7S 6S",
    "QD QC",
    "TS 9S",
]

# Boards: flop, turn, river (avoid collision with the hands above)
BOARD_FLOP = ["4H", "5D", "8C"]
BOARD_TURN = ["4H", "5D", "8C", "TD"]
BOARD_RIVER = ["4H", "5D", "8C", "TD", "QS"]


# ------------------------------- Fixtures -------------------------------------

@pytest.fixture
def dummy_cfr():
    return DummyCFR()

@pytest.fixture
def clusterer_bot(dummy_cfr):
    # Production-like profile ("bot"): mc samples default to 200 in __init__
    return HandClusterer(
        cfr_solver=dummy_cfr,
        num_clusters=4,
        profile="bot",
        use_cfv_in_features=True,
        opp_sample_size=None,
    )

@pytest.fixture
def clusterer_test_env(dummy_cfr, monkeypatch):
    # Test profile path requires FAST_TESTS=1 or config.debug_fast_tests=True
    monkeypatch.setenv("FAST_TESTS", "1")
    return HandClusterer(
        cfr_solver=dummy_cfr,
        num_clusters=5,  # intentionally > len(HANDS_SMALL) to hit K>N branch
        profile="test",
        use_cfv_in_features=False,  # should be forced off by "test" anyway
        opp_sample_size=0,          # forced in "test" profile anyway
    )

@pytest.fixture
def clusterer_with_config(dummy_cfr):
    # Exercise the config-driven constructor branch (including debug_fast_tests)
    Cfg = types.SimpleNamespace(
        num_clusters=6,
        mc_samples_win=17,
        mc_samples_potential=11,
        tau_re=0.42,
        drift_sample_size=13,
        profile="test",
        opp_sample_size=None,
        use_cfv_in_features=True,   # will be forced False in "test"
        fast_test_seed=12345,
        debug_fast_tests=True,
    )
    return HandClusterer(cfr_solver=dummy_cfr, config=Cfg)


# ------------------------ Initialization & Config tests -----------------------

def test_init_bot_profile_defaults(clusterer_bot):
    c = clusterer_bot
    assert c.profile == "bot"
    assert c.num_clusters == 4
    # defaults in non-test mode
    assert c._mc_samples_win == 200
    assert c._mc_samples_potential == 200
    assert c.opp_sample_size is None
    assert c.use_cfv_in_features is True

def test_init_test_profile_enforced_flags(clusterer_test_env):
    c = clusterer_test_env
    assert c.profile == "test"
    # In test profile, sampling values are zeroed and CFV features disabled
    assert c._mc_samples_win == 0
    assert c._mc_samples_potential == 0
    assert c.opp_sample_size == 0
    assert c.use_cfv_in_features is False

def test_init_with_config_test_profile(clusterer_with_config):
    c = clusterer_with_config
    # Config overrides: profile=test; debug_fast_tests enabled
    assert c.profile == "test"
    # num_clusters is taken from config:
    assert c.num_clusters == 6
    # In test profile, enforce zero samples & disable CFV-in-features:
    assert c._mc_samples_win == 0
    assert c._mc_samples_potential == 0
    assert c.use_cfv_in_features is False
    # Drift and tau set from config
    assert c.tau_re == pytest.approx(0.42)
    assert c.drift_sample_size == 13


# ----------------------- Preflop partition & assignment -----------------------

def test_preflop_partition_and_fit(dummy_cfr):
    c = HandClusterer(dummy_cfr, num_clusters=1000, profile="bot")
    # Preflop stage is chosen when board length == 0
    result = c.fit(hands={"AH AC", "KH KS", "7S 6S", "7H 6H", "TD 9C"}, board=[], opponent_range={}, pot_size=100.0)
    # Should bucket by hand type strings (e.g., 'AA', 'KK', '76s', '76s', 'T9o')
    # And set c.num_clusters == number of distinct types present.
    assert isinstance(result, dict)
    assert c.clusters == result
    assert c.centroids is None
    # All original hands should appear exactly once across buckets
    merged = set().union(*result.values()) if result else set()
    assert merged == {"AH AC", "KH KS", "7S 6S", "7H 6H", "TD 9C"}
    # Assignment on preflop should find cluster by type, not exact card string
    cid_76s = c.assign("7D 6D", board=[])  # still "76s"
    assert cid_76s in result


# ----------------------- Test profile clustering (hash-based) -----------------

def test_cluster_hands_test_profile_fast(monkeypatch, clusterer_test_env):
    # cluster_hands in "test" profile requires FAST_TESTS=1 (already set by fixture)
    c = clusterer_test_env
    clusters_first = c.cluster_hands(HANDS_SMALL, BOARD_FLOP, {}, 80.0)
    assert isinstance(clusters_first, dict)
    # K>N path should have at least (K-N) empties, but all hands must be assigned exactly once.
    merged = set().union(*clusters_first.values()) if clusters_first else set()
    assert merged == set(HANDS_SMALL)
    # Frozen clusters reused on subsequent calls (object identity stable)
    clusters_second = c.cluster_hands(HANDS_SMALL, BOARD_FLOP, {}, 80.0)
    assert clusters_first is clusters_second  # same object reused

def test_cluster_hands_test_profile_without_env_uses_config(dummy_cfr, monkeypatch):
    # FAST_TESTS not set, but config.debug_fast_tests=True should permit test path
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


# ----------- Bot profile clustering (medoids + EMD + deterministic seed) ------

def test_cluster_hands_bot_profile_medoids_and_separation(monkeypatch, clusterer_bot):
    """
    Force two clearly separated feature "signatures" to verify:
    - EMD-based medoids create distinct clusters for distinct feature shapes.
    - Deterministic seed leads to stable clustering.
    """
    c = clusterer_bot
    # Monkeypatch calculate_hand_features to return 2 separated signatures
    # First half: concentrated near 0 (losing); Second half: near 1 (winning)
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
        # Expect two clusters, each non-empty, reflecting the two groups
        non_empty = [k for k, v in clusters.items() if len(v) > 0]
        assert len(non_empty) == K
        sizes = sorted(len(v) for v in clusters.values())
        assert sizes == [len(HANDS_SMALL)//2, len(HANDS_SMALL) - len(HANDS_SMALL)//2]
        # Stable under repeated calls (same board -> same seed -> same assignment)
        clusters2 = c.cluster_hands(HANDS_SMALL, BOARD_TURN, {}, 120.0)
        assert clusters == clusters2
    finally:
        # Restore original behavior
        monkeypatch.setattr(HandClustererFeaturesMixin, "calculate_hand_features", original_chf)


def test_cluster_hands_bot_profile_k_adjusts_when_k_gt_n(clusterer_bot):
    c = clusterer_bot
    # Request more clusters than hands; bot path reduces K to N
    c.set_num_clusters(len(HANDS_SMALL) + 5)
    clusters = c.cluster_hands(HANDS_SMALL, BOARD_TURN, {}, 10.0)
    assert len([k for k, v in clusters.items()]) == len(HANDS_SMALL)


def test_cluster_hands_bot_profile_empty_input(clusterer_bot):
    c = clusterer_bot
    out = c.cluster_hands([], BOARD_TURN, {}, 10.0)
    assert out == {}
    assert c.clusters == {}
    assert c._last_features == {}


# ----------------------------- Assign & mapping --------------------------------

def test_assign_postflop_membership(clusterer_test_env):
    c = clusterer_test_env
    clusters = c.cluster_hands(HANDS_SMALL, BOARD_FLOP, {}, 80.0)
    # Pick any hand; ensure assign(hand, board!=preflop) returns its cluster id
    for cid, hset in clusters.items():
        if hset:
            h0 = next(iter(hset))
            got = c.assign(h0, board=BOARD_FLOP)
            assert got == cid
            break

def test_assign_unknown_hand_falls_back_to_hash_bucket(clusterer_test_env):
    c = clusterer_test_env
    c.cluster_hands(HANDS_SMALL, BOARD_FLOP, {}, 80.0)
    # Hand not present: fallback to deterministic hash-based bucket index in [0, K)
    cids = set(c.clusters.keys())
    out_cid = c.assign("2C 3D", board=BOARD_FLOP)
    assert out_cid in cids

def test_set_num_clusters_roundtrip(clusterer_bot):
    assert clusterer_bot.set_num_clusters(7) == 7
    assert clusterer_bot.num_clusters == 7


# ----------------------- Ranges <-> Buckets transformations --------------------

def test_get_cluster_ranges_uniform(clusterer_bot):
    c = clusterer_bot
    c.set_num_clusters(4)
    r = c.get_cluster_ranges()
    assert set(r.keys()) == {0,1,2,3}
    assert all(v == pytest.approx(0.25) for v in r.values())
    assert pytest.approx(sum(r.values()), rel=1e-9) == 1.0

def test_hands_to_bucket_range_and_normalization(clusterer_test_env):
    c = clusterer_test_env
    clusters = c.cluster_hands(HANDS_SMALL, BOARD_TURN, {}, 50.0)
    # Build a non-uniform hand distribution
    hprob = {h: (i+1) for i, h in enumerate(HANDS_SMALL)}
    br = c.hands_to_bucket_range(hprob)
    assert pytest.approx(sum(br.values()), rel=1e-9) == 1.0
    assert set(br.keys()).issubset(set(clusters.keys()))

def test_bucket_range_to_hand_weights_and_normalization(clusterer_test_env):
    c = clusterer_test_env
    clusters = c.cluster_hands(HANDS_SMALL, BOARD_TURN, {}, 50.0)
    # Put all mass on one (non-empty) cluster -> should be uniform across its members
    non_empty = next(k for k, v in clusters.items() if len(v) > 0)
    br = {non_empty: 1.0}
    hw = c.bucket_range_to_hand_weights(br)
    assert pytest.approx(sum(hw.values()), rel=1e-9) == 1.0
    members = sorted(list(clusters[non_empty]))
    # All members in that cluster receive equal share
    for m in members:
        assert hw[m] == pytest.approx(1.0 / len(members))


def test_persist_and_load_mapping_roundtrip(clusterer_test_env):
    c = clusterer_test_env
    clusters = c.cluster_hands(HANDS_SMALL, BOARD_TURN, {}, 50.0)
    m = c.persist_mapping()
    # Start a fresh clusterer and load the mapping
    c2 = HandClusterer(cfr_solver=DummyCFR(), num_clusters=1, profile="bot")
    ok = c2.load_mapping(m)
    assert ok is True
    assert c2.num_clusters == len(m)
    assert c2.centroids is None
    assert c2._last_features is None
    # Assign finds the same cluster membership
    for cid, hands in clusters.items():
        for h in hands:
            assert c2.assign(h, board=BOARD_TURN) == cid


# ---------------------------- Feature & caching tests --------------------------

def test_calculate_hand_features_histogram_and_cache(clusterer_bot):
    c = clusterer_bot
    # Reset cache
    c._cache.clear()
    c._cache_hits = 0
    c._cache_misses = 0
    # same hand+board => hit on second call, even if opponent_range/pot differ
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
    c = clusterer_bot
    eq1 = c._calculate_equity(HANDS_SMALL[0], BOARD_RIVER, {})
    eq2 = c._calculate_equity(HANDS_SMALL[0], BOARD_RIVER, {})
    assert 0.0 <= eq1 <= 1.0
    assert eq1 == pytest.approx(eq2)

def test__calculate_potential_equity_improvement_zero_at_river(clusterer_bot):
    c = clusterer_bot
    pe = c._calculate_potential_equity_improvement(HANDS_SMALL[0], BOARD_RIVER, {})
    assert pe == 0.0

def test__calculate_counterfactual_value_is_signed_fraction(clusterer_bot):
    c = clusterer_bot
    # Returns average signed payoff in [-1, 1] under DummyCFR scoring
    cv = c._calculate_counterfactual_value(HANDS_SMALL[0], BOARD_RIVER, {}, pot_size=200.0)
    assert -1.0 <= cv <= 1.0

def test__calculate_payoff_exact_values(clusterer_bot):
    c = clusterer_bot
    # hand1 vs hand2 on a fixed river:
    h1 = "AH AC"
    h2 = "2D 3C"
    p_win = c._calculate_payoff(h1.split(), h2.split(), BOARD_RIVER, 100.0)
    o_win = c._calculate_payoff(h2.split(), h1.split(), BOARD_RIVER, 100.0)
    tie = c._calculate_payoff(h1.split(), h1.split(), BOARD_RIVER, 100.0)
    assert p_win == 1.0
    assert o_win == -1.0
    assert tie == 0.0

def test__evaluate_win_percentage_terminal_and_sampling(clusterer_bot, monkeypatch):
    c = clusterer_bot
    # Terminal
    e_term = c._evaluate_win_percentage(["AH","AC"], ["2D","3C"], BOARD_RIVER)
    assert e_term in (0.0, 0.5, 1.0)
    # Non-terminal: ensure value is within [0,1] and deterministic with seeded sampling
    # Reduce sampling to keep test fast and deterministic
    c._mc_samples_win = 13
    e1 = c._evaluate_win_percentage(["AH","AC"], ["2D","3C"], BOARD_FLOP)
    e2 = c._evaluate_win_percentage(["AH","AC"], ["2D","3C"], BOARD_FLOP)
    assert 0.0 <= e1 <= 1.0
    assert e1 == pytest.approx(e2)


# ----------------------------- Distance functions -----------------------------

def test_calculate_hand_distance_euclidean(clusterer_bot):
    c = clusterer_bot
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([0.0, 1.0, 5.0])
    d = c.calculate_hand_distance(a, b)
    assert d == pytest.approx(np.linalg.norm(a - b))

def test_emd_distance_properties(clusterer_bot):
    c = clusterer_bot
    f = np.array([1.0, 0.0, 0.0])
    g = np.array([0.0, 1.0, 0.0])
    h = np.array([0.0, 0.0, 1.0])
    d_fg = c._emd_distance(f, g)
    d_gh = c._emd_distance(g, h)
    d_fh = c._emd_distance(f, h)
    # Basic properties
    assert c._emd_distance(f, f) == 0.0
    assert d_fg == pytest.approx(d_gh)
    assert d_fh == pytest.approx(d_fg + d_gh)  # for these 3-point spikes, triangle is tight
    # Symmetry (with this 1D cumulative scheme it's symmetric)
    assert c._emd_distance(f, g) == pytest.approx(c._emd_distance(g, f))


# ---------------------------- Drift & seeding tests ---------------------------

def test_compute_drift_behavior(clusterer_bot):
    c = clusterer_bot
    # No previous features -> None
    nf = {"A": np.array([0.1, 0.2]), "B": np.array([0.0, 1.0])}
    assert c._compute_drift(nf) is None
    c._last_features = {"A": np.array([0.1, 0.1]), "B": np.array([0.0, 0.0]), "C": np.array([9,9])}
    c.drift_sample_size = 2  # only first two keys after sort considered
    d = c._compute_drift(nf)
    # average L2 over overlap {'A','B'}
    expect = (np.linalg.norm([0.1,0.2] - np.array([0.1,0.1])) + np.linalg.norm([0.0,1.0] - np.array([0.0,0.0]))) / 2.0
    assert d == pytest.approx(expect)

def test_deterministic_seed_for_clustering_insensitivity_to_range_and_pot(clusterer_bot):
    c = clusterer_bot
    s1 = c._deterministic_seed_for_clustering(BOARD_TURN, {"X": 0.3}, 10.0)
    s2 = c._deterministic_seed_for_clustering(BOARD_TURN, {"Y": 0.9, 1: 0.1}, 999.0)
    s3 = c._deterministic_seed_for_clustering(BOARD_FLOP, {}, 0.0)
    # With this implementation, seed depends only on board -> same for same board
    assert s1 == s2
    assert s1 != s3


# --------------------- Opponent-range signature & sampling --------------------

def test_opponent_range_signature_canonicalization(clusterer_bot):
    c = clusterer_bot
    r1 = {"QH AC": 0.3, 5: 0.7, "AC QH": 0.0}
    r2 = {5: 0.7, "AC QH": 0.0, "QH AC": 0.3}
    # canonicalization makes "AC QH" and "QH AC" identical ordering, ints labeled c#k
    sig1 = c._opponent_range_signature(r1)
    sig2 = c._opponent_range_signature(r2)
    assert sig1 == sig2

def test__maybe_sample_items_limit_and_determinism(clusterer_bot):
    c = clusterer_bot
    items = {i: i for i in range(10)}  # 10 items
    out1 = c._maybe_sample_items(items, seed=1729)
    out2 = c._maybe_sample_items(items, seed=1729)
    # No limit set -> sorted output (by key) and deterministic
    assert out1 == out2
    # With limit -> deterministic subset under given seed
    c.opp_sample_size = 3
    out3 = c._maybe_sample_items(items, seed=1729)
    out4 = c._maybe_sample_items(items, seed=1729)
    assert len(out3) == 3 and out3 == out4


# ------------------------------ Fit: non-preflop ------------------------------

def test_fit_non_preflop_uses_medoids_and_respects_k(clusterer_bot, monkeypatch):
    c = clusterer_bot
    # K default=1000 inside .fit; ensure it shrinks to N
    result = c.fit(HANDS_SMALL, board=BOARD_TURN, opponent_range={}, pot_size=75.0)
    assert isinstance(result, dict)
    # Number of clusters should not exceed len(HANDS_SMALL)
    assert len(result) <= len(HANDS_SMALL)
    # All hands assigned exactly once
    merged = set().union(*result.values()) if result else set()
    assert merged == set(HANDS_SMALL)


# ------------------------- Paper-aligned behavioral checks --------------------

def test_preflop_not_bucketed_in_aux_network_context():
    """
    Sanity check aligned with the paper statement that the pre-flop (auxiliary) network
    uses 169 strategically distinct hands and *no* bucket abstraction for inputs.
    Here we verify our HandClusterer returns preflop partitions by type, not medoids.
    (DeepStack supplement: k-means/EMD bucketing used for flop/turn; preflop distinct.) 
    See: arXiv:1701.01724 [supplement], “bucket abstraction ... earth mover’s distance ...
    used for flop/turn; pre-flop has 169 strategically distinct hands”, and your doc’s
    cross-reference.  :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
    """
    c = HandClusterer(DummyCFR(), num_clusters=1000, profile="bot")
    res = c.fit({"AH AC", "KH KC", "7S 6S"}, board=[], opponent_range={}, pot_size=50.0)
    # Partitions keyed by type (AA, KK, 76s...); number equals types present in input
    assert isinstance(res, dict)
    assert len(res) == 3

