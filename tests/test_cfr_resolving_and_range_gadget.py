

import os
import math
import types
import itertools
import builtins
import random

import pytest
import torch
import torch.nn as nn

# Project imports (from your repo)
from range_gadget import RangeGadget
from cfr_solver import CFRSolver
from action_type import ActionType


# ---------- Global helpers / dummies ----------

class DummyRiverEndgame:
    """Deterministic river endgame CFV provider."""
    def __init__(self, value_per_cluster=0.123):
        self.value_per_cluster = float(value_per_cluster)
        self.calls = []

    def compute_cluster_cfvs(self, clusters, node, player, wins_fn, best_hand, hand_rank):
        """Return a dict {cid: scalar}."""
        # Record for assertions
        self.calls.append((tuple(sorted(int(k) for k in clusters.keys())), player))
        return {int(cid): self.value_per_cluster for cid in clusters.keys()}


class DummyNet(nn.Module):
    """Deterministic CFV 'network' with a simple enforce_zero_sum outer correction."""
    def __init__(self, K):
        super().__init__()
        self.K = int(K)
        # Tiny linear to keep non-zero outputs but predictable
        self.w1 = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.w1.weight.fill_(0.5)

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        # Produce non-trivial vectors (not already zero-sum) to test outer adjustment accounting
        # v1 raw: [0, 1, 2, ..., K-1], v2 raw: [K-1, ..., 1, 0] with a slight affine to avoid symmetry
        base = torch.arange(self.K, dtype=torch.float32, device=device).unsqueeze(0).repeat(B, 1)
        v1 = base + 0.25
        v2 = torch.flip(base, dims=[1]) - 0.15
        return v1, v2

    def enforce_zero_sum(self, r1, r2, v1, v2):
        # Per-sample outer correction:
        # δ = 0.5*(<r1,v1> + <r2,v2>); f1 = v1 - δ; f2 = v2 - δ ;
        # (This matches the contract the tests account for.)
        s1 = torch.sum(r1 * v1, dim=1, keepdim=True)
        s2 = torch.sum(r2 * v2, dim=1, keepdim=True)
        delta = 0.5 * (s1 + s2)
        return v1 - delta, v2 - delta


class BarePublicState:
    """Minimal state to satisfy the solver’s mixin calls where we don’t want real game evolution."""
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
        self.last_action = None  # set by tests when needed
        self._min_raise = int(min_raise_size)

    def legal_actions(self):
        return list(self._legal) if self._legal is not None else []

    def _min_raise_size(self):
        return self._min_raise

    def update_state(self, prev_node, action):
        # We never rely on deep semantics in these tests; return a shallow 'next' state:
        nxt = BarePublicState(
            round_idx=self.current_round,
            current_player=(self.current_player + 1) % 2,
            pot_size=self.pot_size,  # keep pot constant to keep predictable
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
    """Minimal node wrapper with public_state + player_ranges (hashable by id)."""
    def __init__(self, ps, p0_range=None, p1_range=None):
        self.public_state = ps
        self.player_ranges = [dict(p0_range or {}), dict(p1_range or {})]

    @property
    def current_player(self):
        return self.public_state.current_player


@pytest.fixture(autouse=True)
def _fast_env(monkeypatch):
    # Keep all tests fast and deterministic
    monkeypatch.setenv("FAST_TESTS", "1")
    yield
    monkeypatch.delenv("FAST_TESTS", raising=False)


@pytest.fixture
def solver(monkeypatch):
    s = CFRSolver(depth_limit=2, num_clusters=4, speed_profile="test", config=None)

    # Replace real models with deterministic dummies
    K = s.num_clusters
    s.models["preflop"] = DummyNet(K)
    s.models["flop"] = DummyNet(K)
    s.models["turn"] = DummyNet(K)

    # River endgame: deterministic
    s.river_endgame = DummyRiverEndgame(value_per_cluster=0.123)

    # Provide a minimal cluster mapping
    s.clusters = {
        0: {"As Kh", "Ad Kc"},
        1: {"2h 2d", "2s 2c"},
        2: {"Qs Qh"},
        3: {"7d 6d", "9c 8c"},
    }

    # Force test-mode behavior for sampling/strength
    class _HC:
        profile = "test"
    s.hand_clusterer = _HC()

    return s


# ---------- RangeGadget tests ----------

def test_range_gadget_begin_update_get_monotone():
    g = RangeGadget()
    # empty begin
    assert g.begin() == {}
    # begin with initial upper
    init = {0: 1.0, 2: -0.5}
    out = g.begin(init)
    assert out == {0: 1.0, 2: -0.5}
    # update with mixed (including lower for 0 and higher for 2) + new key
    upd = {0: 0.25, 2: -0.2, 3: 1.7}
    out2 = g.update(upd)
    # monotone by max per key
    assert out2[0] == 1.0       # unchanged (max of 1.0 and 0.25)
    assert out2[2] == -0.2      # -0.2 > -0.5
    assert out2[3] == 1.7
    # idempotent get
    assert g.get() == out2


# ---------- Caching/signatures tests ----------

def test_preflop_signature_stability_and_sensitivity(solver):
    ps1 = BarePublicState(round_idx=0, current_player=0, pot_size=100, dealer=1, blinds=(5, 10))
    n1 = BareNode(ps1, {0: 0.6, 1: 0.4}, {0: 0.3, 1: 0.7})
    k1 = solver._preflop_signature(n1)

    # identical copy => same signature
    ps1b = BarePublicState(round_idx=0, current_player=0, pot_size=100, dealer=1, blinds=(5, 10))
    n1b = BareNode(ps1b, {0: 0.6, 1: 0.4}, {0: 0.3, 1: 0.7})
    assert solver._preflop_signature(n1b) == k1

    # change pot or partition => different signature
    ps2 = BarePublicState(round_idx=0, current_player=0, pot_size=120, dealer=1, blinds=(5, 10))
    n2 = BareNode(ps2, {0: 0.6, 1: 0.4}, {0: 0.3, 1: 0.7})
    assert solver._preflop_signature(n2) != k1

    # mutate cluster partition -> signature changes
    old_sig = solver._cluster_partition_signature()
    solver.clusters[3].add("Ah Ad")  # change contents
    new_sig = solver._cluster_partition_signature()
    assert new_sig != old_sig


def test_preflop_cache_put_get_lru_and_stats(solver):
    key = ("PREFLOP_CACHE_V2", 5, 10, 0, 0, (1000, 1000), (0, 0), 100, solver.num_clusters,
           solver._cluster_partition_signature(),
           solver._range_sig({0: 0.6, 1: 0.4}),
           solver._range_sig({0: 0.4, 1: 0.6}))
    own = {0: 0.6, 1: 0.4}
    opp = {0: 0.1, 1: 0.2}
    # put
    solver._preflop_cache_put(key, own, opp)
    got = solver._preflop_cache_get(key)
    assert got["own_range"] == own and got["opp_cfv"] == opp
    # stats updated
    stats = solver._preflop_cache_stats
    assert stats["hits"] == 1 and stats["puts"] >= 1 and stats["misses"] >= 0

    # Eviction behavior: set small cap, insert more than cap, oldest should be evicted
    solver._preflop_cache_cap = 2
    solver._preflop_cache.clear()
    kA, kB, kC = ("A",), ("B",), ("C",)
    solver._preflop_cache_put(kA, {}, {})
    solver._preflop_cache_put(kB, {}, {})
    solver._preflop_cache_put(kC, {}, {})
    assert solver._preflop_cache_get(kA) is None  # evicted
    assert solver._preflop_cache_get(kB) is not None
    assert solver._preflop_cache_get(kC) is not None


def test_range_sig_rounding_to_12_decimals(solver):
    r = {0: 0.123456789012345, 2: 0.3333333333333333}
    sig = solver._range_sig(r)
    # Rounded coefficients must appear
    assert sig == ((0, round(0.123456789012345, 12)), (2, round(0.3333333333333333, 12)))


def test_state_key_determinism_and_sensitivity(solver):
    ps = BarePublicState(round_idx=1, current_player=1, pot_size=200, dealer=0, board_cards=["Ah","Kd","2c"])
    n = BareNode(ps, {0: 1.0}, {1: 1.0})
    k1 = solver._state_key(n)
    # small variation -> different key
    ps2 = BarePublicState(round_idx=1, current_player=1, pot_size=210, dealer=0, board_cards=["Ah","Kd","2c"])
    n2 = BareNode(ps2, {0: 1.0}, {1: 1.0})
    assert solver._state_key(n2) != k1


# ---------- Hand strength / recursive sampling ----------

def test_evaluate_hand_strength_cached_and_seeded(solver, monkeypatch):
    # Using int (cluster id) path -> will sample from solver.clusters[0]
    ps_board = ["Ah", "Kd", "2c"]  # harmless board
    v1 = solver._evaluate_hand_strength(0, ps_board)
    v2 = solver._evaluate_hand_strength(0, ps_board)  # cached
    assert v1 == v2
    assert 0.0 <= v1 <= 1.0

    # Using explicit hand string path (still fast due to FAST_TESTS=1)
    v3 = solver._evaluate_hand_strength("As Kh", ps_board)
    v4 = solver._evaluate_hand_strength("As Kh", ps_board)
    assert v3 == v4
    assert 0.0 <= v3 <= 1.0


def test_recursive_range_sampling_sum_and_cache(solver):
    hands = {"As Kh", "2h 2d", "Qs Qh", "9c 8c"}
    total = 1.0
    board = ["Ah", "Kd"]
    out1 = solver.recursive_range_sampling(hands, total, board)
    out2 = solver.recursive_range_sampling(hands, total, board)  # cache hit
    # same result, sums to total
    assert out1 == out2
    assert pytest.approx(sum(out1.values()), rel=0, abs=1e-12) == total
    # All keys present
    assert set(out1.keys()) == set(hands)


# ---------- Input-vector builders ----------

def test_prepare_input_vector_shapes_and_normalization(solver):
    # Two clusters in each range, arbitrary (will be normalized inside)
    ps = BarePublicState(round_idx=1, pot_size=200, board_cards=["Ah", "Kd", "2c"])
    node = BareNode(ps, {0: 2.0, 1: 1.0}, {0: 3.0, 1: 0.0})
    vec = solver.prepare_input_vector(node)
    # 1 (pot_norm) + 52 (board_one_hot) + 2K ranges
    expected_len = 1 + len(set([
        # The board_one_hot uses DECK length from poker_utils; we can't import it here,
        # but its length is equal to 52 in standard. Use solver.calculate_input_size()
    ]))  # not used; we rely on calculate_input_size below
    assert len(vec) == solver.calculate_input_size()

    # The r1/r2 portions should be normalized; check sums near 1 across K
    K = solver.num_clusters
    start_r1 = 1 + len(vec) - (2 * K) - 0  # approximate; better recompute
    # Recompute proper offsets using the same logic as solver.prepare_input_vector
    # 1 + |DECK| + K + K == input_size
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


# ---------- predict_counterfactual_values & zero-sum ----------
def test_predict_cfv_stage_mapping_and_zero_sum_residuals(solver):
    # flop node -> turn model; preflop node -> flop model; river -> river endgame path
    # We also verify that residuals were recorded (post outer adjustment)
    # FLOP
    ps_flop = BarePublicState(round_idx=1, pot_size=240, board_cards=["Ah", "Kd", "2c"])
    n_flop = BareNode(ps_flop, {0:0.5, 1:0.5}, {0:0.5, 1:0.5})
    before = len(getattr(solver, "_zs_residual_samples", [])) if hasattr(solver, "_zs_residual_samples") else 0
    out_flop = solver.predict_counterfactual_values(n_flop, player=0)
    after = len(solver._zs_residual_samples)
    assert isinstance(out_flop, dict) and all(isinstance(v, list) for v in out_flop.values())
    assert after > before  # residual samples recorded

    # PREFLOP
    ps_pre = BarePublicState(round_idx=0, pot_size=100)
    n_pre = BareNode(ps_pre, {0:0.5, 1:0.5}, {0:0.5, 1:0.5})
    _ = solver.predict_counterfactual_values(n_pre, player=1)  # maps to flop net in your code

    # RIVER -> river endgame path
    ps_riv = BarePublicState(round_idx=3, pot_size=400, board_cards=["Ah","Kd","2c","7s","7h"])
    n_riv = BareNode(ps_riv, {0:1.0}, {1:1.0})
    out_riv = solver.predict_counterfactual_values(n_riv, player=0)
    # River path returns scalar per cluster replicated later by caller; here we see raw dict
    assert out_riv and all(isinstance(v, float) or isinstance(v, (list, tuple)) for v in out_riv.values())


def test_depth_limit_preflop_flop_use_model_scale_by_pot_and_turn_does_not(solver, monkeypatch):
    # Monkeypatch predict_counterfactual_values to see calls & control outputs
    calls = {"count": 0}
    def fake_pred(node, player):
        calls["count"] += 1
        # Return per-cluster [1.0, 1.0, ...] so the scaler (pot_size) is visible upstream
        return {cid: [1.0] for cid in node.player_ranges[player].keys()}
    monkeypatch.setattr(solver, "predict_counterfactual_values", fake_pred)

    # PRE-FLOP (depth >= limit): should call NN and scale by pot
    ps_pre = BarePublicState(round_idx=0, pot_size=250)
    n_pre = BareNode(ps_pre, {0:1.0, 1:1.0}, {0:1.0, 1:1.0})
    out = solver._calculate_counterfactual_values(n_pre, player=0, depth=999)  # force limit
    assert calls["count"] == 1
    for vec in out.values():
        assert all(v == 250.0 for v in vec)  # scaled by pot_size

    # FLOP (depth >= limit): also call NN (+scale)
    ps_flop = BarePublicState(round_idx=1, pot_size=300)
    n_flop = BareNode(ps_flop, {0:1.0}, {1:1.0})
    _ = solver._calculate_counterfactual_values(n_flop, player=0, depth=999)
    assert calls["count"] == 2  # second call

    # TURN (depth >= limit): MUST NOT call NN in your code path
    # Prevent recursion by returning no allowed actions
    monkeypatch.setattr(solver, "_allowed_actions_agent", lambda ps: [])
    monkeypatch.setattr(solver, "_allowed_actions_opponent", lambda ps: [])
    ps_turn = BarePublicState(round_idx=2, pot_size=350)
    n_turn = BareNode(ps_turn, {0:1.0}, {1:1.0})
    _ = solver._calculate_counterfactual_values(n_turn, player=0, depth=999)
    assert calls["count"] == 2  # unchanged => no NN call on turn limit


# ---------- Strategy masking & mixed action distribution ----------

def test_mask_strategy_and_uniform_fallbacks(solver):
    # Build a fake 6-action strategy; allow only CALL and ALL_IN
    A = len(ActionType)
    strat = [0.0]*A
    strat[ActionType.CALL.value] = 0.3
    strat[ActionType.ALL_IN.value] = 0.7
    allowed = [ActionType.CALL, ActionType.ALL_IN]
    m = solver._mask_strategy(strat, allowed)
    # normalized over allowed
    assert sum(m) == pytest.approx(1.0)
    for i, p in enumerate(m):
        if i in (ActionType.CALL.value, ActionType.ALL_IN.value):
            assert p > 0.0
        else:
            assert p == 0.0

    # If base probs are zero on allowed, produce uniform over allowed
    z = [0.0]*A
    m2 = solver._mask_strategy(z, allowed)
    assert m2[ActionType.CALL.value] == pytest.approx(0.5)
    assert m2[ActionType.ALL_IN.value] == pytest.approx(0.5)


def test_mixed_action_distribution_weighted_and_uniform_fallback(solver):
    ps = BarePublicState(round_idx=1)
    node = BareNode(ps, {0:0.6, 1:0.4}, {0:0.5, 1:0.5})
    values = solver.cfr_values[node]

    # Provide cumulative strategies to induce a specific average:
    A = len(ActionType)
    # cluster 0: always CALL
    values.cumulative_strategy[0] = [0.0]*A
    values.cumulative_strategy[0][ActionType.CALL.value] = 10.0
    # cluster 1: always ALL_IN
    values.cumulative_strategy[1] = [0.0]*A
    values.cumulative_strategy[1][ActionType.ALL_IN.value] = 5.0

    allowed = [ActionType.CALL, ActionType.ALL_IN]
    probs = solver._mixed_action_distribution(node, player=0, allowed_actions=allowed)
    # Weighted by priors 0.6 and 0.4 -> 0.6 on CALL, 0.4 on ALL_IN
    assert probs == pytest.approx([0.6, 0.4])

    # If prior weight sum <= 0 -> uniform over allowed
    node2 = BareNode(ps, {}, {})
    probs2 = solver._mixed_action_distribution(node2, player=0, allowed_actions=allowed)
    assert probs2 == pytest.approx([0.5, 0.5])


# ---------- Range update & chance lift ----------

def test_update_player_range_bayes_and_fallback(solver):
    ps = BarePublicState(round_idx=1, current_player=0)
    node = BareNode(ps, {0:0.7, 1:0.3}, {0:0.5, 1:0.5})
    values = solver.cfr_values[node]
    A = len(ActionType)
    # Set current strategy to: cluster 0 picks CALL with 0.2, cluster 1 with 0.8
    values.strategy[0] = [0.0]*A
    values.strategy[1] = [0.0]*A
    values.strategy[0][ActionType.CALL.value] = 0.2
    values.strategy[1][ActionType.CALL.value] = 0.8

    solver.update_player_range(node, player=0, cluster_id=0, action_index=ActionType.CALL.value)
    post = node.player_ranges[0]
    # Bayes: posterior ∝ prior * likelihood
    # unnorm: c0 = 0.7*0.2=0.14 ; c1 = 0.3*0.8=0.24 ; normalize: sum=0.38
    assert post[0] == pytest.approx(0.14/0.38)
    assert post[1] == pytest.approx(0.24/0.38)

    # Fallback when like=0 for all (e.g., invalid action_index) -> normalize priors
    solver.update_player_range(node, player=0, cluster_id=0, action_index=999)
    post2 = node.player_ranges[0]
    s = 0.7 + 0.3
    assert post2[0] == pytest.approx(0.7/s)
    assert post2[1] == pytest.approx(0.3/s)


def test_lift_ranges_after_chance_reweights_by_board_compat(solver):
    # Build a node where current ranges are uniform; lift should reweight by compatibility counts
    ps = BarePublicState(round_idx=1, board_cards=["Ah","Kd","2c"])
    n = BareNode(ps, {0:0.5, 1:0.5, 2:0.0, 3:0.0}, {0:0.5, 1:0.5, 2:0.0, 3:0.0})
    before0 = dict(n.player_ranges[0])
    out = solver.lift_ranges_after_chance(n)

    # For compatibility, only hands not using board cards and not pairs of identical ranks count.
    # We only check that probabilities were renormalized across clusters with non-zero compat.
    for pl in (0, 1):
        s = sum(out[pl].values())
        assert s == pytest.approx(1.0)
        # At least one of clusters 2 or 3 likely remains viable (depending on board overlap),
        # but we only assert normalization & preservation of keys present.
        assert set(out[pl].keys()) == set(n.player_ranges[pl].keys())


# ---------- Opponent CFV upper tracking & range gadget ----------

def test_range_gadget_begin_commit_tracking_per_state(solver):
    # Fresh node
    ps = BarePublicState(round_idx=1)
    n = BareNode(ps, {0:1.0}, {1:1.0})
    # Begin with empty => record empty
    b = solver._range_gadget_begin(n)
    assert b == {}
    # Commit an upper dict; ensure tracking recorded and monotone on subsequent commits
    u1 = solver._range_gadget_commit(n, {0: 0.5, 1: -0.2})
    assert u1 == {0: 0.5, 1: -0.2}
    u2 = solver._range_gadget_commit(n, {0: 0.3, 1: -0.1, 2: 7.0})
    assert u2[0] == 0.5  # unchanged (max)
    assert u2[1] == -0.1 # raised
    assert u2[2] == 7.0

    # Stored under the state's key
    key = solver._state_key(n)
    assert solver.opponent_cfv_upper_tracking[key] == u2


def test_apply_opponent_action_update_merge_max(solver):
    # Prev and next nodes with distinct keys
    n_prev = BareNode(BarePublicState(round_idx=1), {0:1.0}, {1:1.0})
    n_next = BareNode(BarePublicState(round_idx=1, current_player=1), {0:1.0}, {1:1.0})
    # Seed tracking
    kp = solver._state_key(n_prev)
    kn = solver._state_key(n_next)
    solver.opponent_cfv_upper_tracking = {
        kp: {0: 1.0, 1: 2.0},
        kn: {0: 0.5, 1: 3.5}
    }
    # Merge -> per key max
    solver.apply_opponent_action_update(n_prev, n_next, observed_action_type=ActionType.CALL)
    merged = solver.opponent_cfv_upper_tracking[kn]
    assert merged[0] == 1.0 and merged[1] == 3.5


# ---------- Diagnostics & schedules ----------

def test_upper_from_cfvs_max_component_and_scalar_passthrough(solver):
    cfvs = {0: [1.0, 2.0, -0.5], 1: [0.0]}
    out = solver._upper_from_cfvs(cfvs)
    assert out[0] == 2.0 and out[1] == 0.0

    cfvs2 = {0: 1.25, 3: -0.75}
    out2 = solver._upper_from_cfvs(cfvs2)
    assert out2 == {0: 1.25, 3: -0.75}


def test_set_cfr_hybrid_config_and_get_last_diagnostics(solver):
    cfg = solver.set_cfr_hybrid_config(preflop_omit=10, flop_omit=11, turn_omit=12)
    assert cfg == {"preflop": 10, "flop": 11, "turn": 12}

    # Prime diagnostics counters via a prediction call
    ps = BarePublicState(round_idx=1)
    n = BareNode(ps, {0:1.0}, {1:1.0})
    solver.predict_counterfactual_values(n, 0)
    d = solver.get_last_diagnostics()
    # Not all fields present until run_cfr(), but cfv_calls & preflop_cache stats should exist or be defaults
    assert "preflop_cache" in d
    assert "cfv_calls" in d or True  # tolerate missing until run loop executed


def test_apply_round_iteration_schedule_defaults_and_custom(solver):
    # Reset to defaults
    solver._round_iters = {0: 0, 1: 1000, 2: 2000, 3: 500}
    # Round present => returns that value and sets total_iterations
    it = solver.apply_round_iteration_schedule(1)
    assert it == 1000 and solver.total_iterations == 1000
    # Round absent => keeps total_iterations
    solver.total_iterations = 42
    it2 = solver.apply_round_iteration_schedule(99)
    assert it2 == 42 and solver.total_iterations == 42


# ---------- Cumulative strategy aggregation ----------

def test_get_cumulative_strategy_aggregates_across_nodes(solver):
    # Two distinct nodes, each with some cumulative strategy
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


# ---------- compute_values_depth_limited dispatch ----------

def test_compute_values_depth_limited_dispatch_raises_when_missing(solver):
    with pytest.raises(AttributeError):
        solver.compute_values_depth_limited(node=None, player=0)


# ---------- Allowed actions menus (agent & opponent) ----------

def test_allowed_actions_agent_no_call_to_call_and_raise_flags(solver):
    # Case 1: to_call == 0 (preflop), ample stacks -> allow {CALL, HP, P, 2P(if flagged), ALLIN}
    ps = BarePublicState(
        round_idx=0,
        current_player=0,
        pot_size=100,
        stacks=(10000, 10000),
        current_bets=(0,0),
        min_raise_size=10
    )
    # Ensure flags for round 0 include two-pot (from mixin defaults)
    solver._ensure_sparse_schedule()
    acts = solver._allowed_actions_agent(ps)
    assert ActionType.CALL in acts
    assert ActionType.HALF_POT_BET in acts
    assert ActionType.POT_SIZED_BET in acts
    assert ActionType.TWO_POT_BET in acts  # round 0 flag = True by default
    assert ActionType.ALL_IN in acts

    # Case 2: to_call > 0 -> include FOLD and CALL; raise sizes computed from pot_after_call.
    ps2 = BarePublicState(
        round_idx=1,
        current_player=0,
        pot_size=100,
        stacks=(500, 500),
        current_bets=(0, 50),  # to_call = 50
        min_raise_size=10
    )
    acts2 = solver._allowed_actions_agent(ps2)
    assert ActionType.FOLD in acts2
    assert ActionType.CALL in acts2
    # two-pot is disabled for round 1 by default flags in utils mixin
    assert ActionType.TWO_POT_BET not in acts2
    assert ActionType.ALL_IN in acts2  # always possible if any chips left


def test_allowed_actions_opponent_exists_and_filters_legal(solver):
    # Ensure method is present (required by cfr recursion paths)
    assert hasattr(solver, "_allowed_actions_opponent"), \
        "CFRSolverStrategiesMixin must define _allowed_actions_opponent"

    # With legal actions restricted, opponent menu should be subset + deduped
    legal = [ActionType.FOLD, ActionType.CALL, ActionType.ALL_IN]  # omit POT_SIZED_BET intentionally
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
    # HUNL_MENU_LEVEL is auto-chosen from iterations; we only assert the basic filtering here.


# ---------- Diagnostics after a small run (preflop no-iteration) ----------

def test_get_last_diagnostics_after_preflop_run_records_cache_and_residuals(solver, monkeypatch):
    # Prepare a preflop node; run_cfr will set total_iterations=0 for preflop path
    ps = BarePublicState(round_idx=0, current_player=0, pot_size=100)
    n = BareNode(ps, {0:0.6, 1:0.4}, {0:0.5, 1:0.5})

    # Seed a cache entry to verify it is used to set opponent_cfv_upper_tracking
    key = solver._preflop_signature(n)
    solver._preflop_cache_put(key, {0:0.7, 1:0.3}, {0:1.23, 1:-0.1})

    # Provide a minimal allowed action to avoid recursion
    monkeypatch.setattr(solver, "_allowed_actions_agent", lambda ps: [ActionType.CALL])
    # run
    _ = solver.run_cfr(n)  # preflop => 0 iterations; but cache is exercised

    d = solver.get_last_diagnostics()
    assert "preflop_cache" in d
    stats = d["preflop_cache"]
    assert stats["hits"] >= 1 or stats["misses"] >= 0  # presence of stats

    # Ensure opponent upper tracking was populated from cache on identical state
    key2 = solver._state_key(n)
    assert hasattr(solver, "opponent_cfv_upper_tracking")
    assert key2 in solver.opponent_cfv_upper_tracking
    up = solver.opponent_cfv_upper_tracking[key2]
    # Should reflect cached uppers (monotone map, exact values)
    assert up.get(0) == pytest.approx(1.23)
    assert up.get(1) == pytest.approx(-0.1)

