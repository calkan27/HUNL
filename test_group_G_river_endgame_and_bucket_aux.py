# -*- coding: utf-8 -*-
"""
Group G — River endgame & bucket auxiliary head
Comprehensive tests for:
  - river_endgame.RiverEndgame
  - river_bucket_aux.RiverBucketAux

Ground-truth requirements (from the attached papers):
- River is solved to terminal with a (possibly) bucketed abstraction; NO value
  nets are used on the river (nor after the final river card).
  (DeepStack: “we did not use a neural network after the final river card,
   but instead solved to the end of the game … bucketed abstraction for all
   actions on the river … no counterfactual value network was used.”)
  CITATION: :contentReference[oaicite:2]{index=2}

- Supervision/targets are pot-fraction CFVs: values are normalized by the pot;
  to convert to chips multiply by the pot size (P).
  (ResolveNet/DeepStack CFV networks spec.)
  CITATION: :contentReference[oaicite:3]{index=3}
"""
import math
import random
import itertools
from typing import Dict, List, Tuple

import pytest
import torch
import numpy as np

from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis import assume

# Import SUT modules
from river_endgame import RiverEndgame
from river_bucket_aux import RiverBucketAux

# Try to import the canonical 52-card deck supplied by the project; fall back to a local minimal deck.
try:
    from poker_utils import DECK as PROJECT_DECK
    CANONICAL_DECK = list(PROJECT_DECK)
except Exception:
    # ranks * suits (keep it small yet rich enough for unique board/hands)
    ranks = "23456789TJQKA"
    suits = "cdhs"
    CANONICAL_DECK = [f"{r}{s}" for r in ranks for s in suits]


# ----------------------------
# Helpers: ranking + stubs
# ----------------------------

RANK_ORDER = {r: i for i, r in enumerate("23456789TJQKA", start=2)}


class _FakePS:
	def __init__(self, board, pot, bets, stacks):
		self.board_cards = list(board)
		self.pot_size = float(pot)
		self.current_bets = [float(bets[0]), float(bets[1])]
		self.initial_stacks = [float(stacks[0]), float(stacks[1])]

class _FakeNode:
	def __init__(self, board, pot, bets, stacks, r0, r1):
		self.public_state = _FakePS(board, pot, bets, stacks)
		self.player_ranges = [dict(r0), dict(r1)]



def _score_7cards(cards: List[str]) -> int:
        """
        Simple deterministic 7-card strength surrogate (NOT poker-correct, but consistent).
        Higher is better. Ties possible.
        """
        counts = {}
        for c in cards:
                r = c[0]
                counts[r] = counts.get(r, 0) + 1
        rank_sum = sum(RANK_ORDER[c[0]] for c in cards)
        bonus = sum((cnt * cnt) for cnt in counts.values())
        _suit_idx = {"C": 0, "D": 1, "H": 2, "S": 3}
        suit_hash = sum(_suit_idx.get(c[1].upper(), 0) for c in cards)
        return 1000 * bonus + 10 * rank_sum + suit_hash

def best_hand_fn(full7: List[str]) -> int:
    return _score_7cards(full7)

def hand_rank_fn(s: int) -> int:
    return s  # identity (higher is stronger)

def wins_fn(my_hand: List[str], opp_hand: List[str], board: List[str]) -> int:
    my_score = hand_rank_fn(best_hand_fn(my_hand + board))
    opp_score = hand_rank_fn(best_hand_fn(opp_hand + board))
    if my_score > opp_score:
        return 1
    if my_score < opp_score:
        return -1
    return 0

def _mk_board(n: int = 5, rng: random.Random = None) -> List[str]:
    rng = rng or random.Random(2027)
    return rng.sample(CANONICAL_DECK, n)

def _hands_strings_from(deck: List[str]) -> List[str]:
    return [f"{a} {b}" for a, b in itertools.combinations(deck, 2)]

def _mk_clusters(board: List[str], K: int, hands_per_cluster: int, seed: int = 7) -> Dict[int, set]:
    """Build K clusters, each with a set of hole-card strings that don't touch the board."""
    rng = random.Random(seed)
    rest = [c for c in CANONICAL_DECK if c not in set(board)]
    all_hands = [f"{a} {b}" for a, b in itertools.combinations(rest, 2)]
    rng.shuffle(all_hands)
    clusters: Dict[int, set] = {i: set() for i in range(K)}
    i = 0
    for h in all_hands:
        if i >= K * hands_per_cluster:
            break
        clusters[i % K].add(h)
        i += 1
    return clusters


# ----------------------------
# Tests for RiverBucketAux
# ----------------------------

def test_bucket_aux_shapes_and_gradients():
    K = 8
    B = 5
    net = RiverBucketAux(num_buckets=B, num_clusters=K)
    net.train()

    # (batch, clusters, buckets)
    x = torch.randn(17, K, B, requires_grad=True)

    y = net(x)
    assert y.shape == (17, K), "Output must be (batch, num_clusters)."

    # Ensure gradients flow through
    s = y.sum()
    s.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "Backward should populate finite grads."

def test_bucket_aux_predict_matches_forward_under_no_grad():
    K = 6
    B = 3
    net = RiverBucketAux(num_buckets=B, num_clusters=K).eval()
    x = torch.randn(4, K, B)
    with torch.no_grad():
        y_forward = net(x)
    y_pred = net.predict(x)
    assert torch.allclose(y_forward, y_pred, atol=1e-6), "predict() must mirror forward() (no grad)."


# ----------------------------
# Tests for RiverEndgame internals
# ----------------------------

def test_filter_hands_removes_board_and_duplicates():
    re = RiverEndgame(num_buckets=None, max_sample_per_cluster=None)
    board = ["Ah", "Kd", "7s", "2c", "3c"]
    board_set = set(board)
    raw = [
        "Ah As",  # illegal: Ah on board
        "7s 7s",  # duplicate card (string variant)
        "Qc Qd",  # valid pair of distinct ranks/suits
        "2c 3c",  # both on board (illegal)
        "Jh Ts",  # valid
    ]
    out = re._filter_hands(raw, board_set)
    assert "Qc Qd" in out and "Jh Ts" in out
    assert all(h not in out for h in ["Ah As", "2c 3c"])
    # no malformed duplicates survived
    for h in out:
        a, b = h.split()
        assert a != b

def test_sample_is_deterministic_per_key_and_seed_and_bounds():
    re = RiverEndgame(num_buckets=None, max_sample_per_cluster=3, seed=123)
    items = list(range(20))
    # k >= len(items) => returns all items (order not guaranteed, set compare)
    assert set(re._sample(items, k=100, key=1)) == set(items)

    # k < len(items) => deterministic for a given key
    s1 = re._sample(items, k=5, key=42)
    s2 = re._sample(items, k=5, key=42)
    assert s1 == s2

    # different key => likely different sample
    s3 = re._sample(items, k=5, key=43)
    assert s3 != s1

def test_bucketize_edge_cases_and_monotonicity():
    re = RiverEndgame(num_buckets=4)
    strengths = [1, 2, 3, 4, 5, 6]
    bmap, B = re._bucketize(strengths)
    assert B == 4
    assert set(bmap.values()) <= set(range(B))
    # monotone: larger strength should map to same or larger bucket index (non-decreasing)
    prev_b = -1
    for s in sorted(set(strengths)):
        b = bmap[s]
        assert b >= prev_b
        prev_b = b

    # When #uniq <= #buckets, every unique strength gets its own (contiguous) bucket index
    re2 = RiverEndgame(num_buckets=10)
    strengths2 = [10, 11, 12]
    bmap2, B2 = re2._bucketize(strengths2)
    assert B2 == len(set(strengths2))
    assert [bmap2[s] for s in sorted(set(strengths2))] == list(range(B2))

def test_cluster_distribution_length_and_consistency():
    re = RiverEndgame(num_buckets=5)
    board = _mk_board()
    hands = ["Ah Kh", "Qs Jd", "2c 7c"]
    dist = re._cluster_distribution(hands, board, best_hand_fn, hand_rank_fn)
    assert len(dist) == len(hands)
    # If we force constant best_hand_fn, all strengths equal
    def const_best(hb): return 123
    dist2 = re._cluster_distribution(hands, board, const_best, hand_rank_fn)
    assert len(set(dist2)) == 1 and dist2[0] == 123

def test_pairwise_utility_zero_sum_and_formulas():
    re = RiverEndgame()
    for res in (-1, 0, 1):
        up = re._pairwise_util_p(res, pot_size=100.0, my_bet=10.0, opp_bet=25.0)
        if res > 0:
            assert up == 25.0
        elif res < 0:
            assert up == -10.0
        else:
            assert up == 0.5 * (25.0 - 10.0)
        # _expected_utility_pairwise must be zero-sum
        ep, eo = re._expected_utility_pairwise(["Ah", "Kd"], ["Qs", "Jh"], ["2c","3c","4d","5h","6s"], wins_fn, pot_size=100.0, my_bet=10.0, opp_bet=25.0)
        assert math.isclose(ep + eo, 0.0, rel_tol=0, abs_tol=1e-9)

def test_bucket_level_expectations_zero_sum_and_swap_invariance():
        print("\n================= DEBUG: bucket_level_zero_sum_and_swap =================")
        re = RiverEndgame()
        B = 6
        rng = random.Random(99)
        pb = np.asarray([rng.random() for _ in range(B)], dtype=float)
        qb = np.asarray([rng.random() for _ in range(B)], dtype=float)
        pb = pb / (pb.sum() if pb.sum() > 0 else 1.0)
        qb = qb / (qb.sum() if qb.sum() > 0 else 1.0)
        print(f"B={B}")
        print("p (ours) raw probs:", pb.tolist(), "sum=", float(pb.sum()))
        print("q (opp)  raw probs:", qb.tolist(), "sum=", float(qb.sum()))
        ev_p, ev_o = re._expected_utility_buckets_both(pb.tolist(), qb.tolist(), B, my_bet=20.0, opp_bet=20.0)
        print(f"ev_p(p,q)={ev_p:.12f}  ev_o(p,q)={ev_o:.12f}  ev_sum={ev_p+ev_o:.12f}")
        assert math.isclose(ev_p + ev_o, 0.0, abs_tol=1e-10)
        ev_p2, ev_o2 = re._expected_utility_buckets_both(qb.tolist(), pb.tolist(), B, my_bet=20.0, opp_bet=20.0)
        print(f"ev_p(q,p)={ev_p2:.12f}  ev_o(q,p)={ev_o2:.12f}  ev_sum2={ev_p2+ev_o2:.12f}")
        diff_p = ev_p + ev_p2
        diff_o = ev_o + ev_o2
        print(f"antisym check: ev_p(p,q) ?= -ev_p(q,p)  diff={diff_p:.12e}")
        print(f"antisym check: ev_o(p,q) ?= -ev_o(q,p)  diff={diff_o:.12e}")
        assert math.isclose(ev_p, -ev_p2, abs_tol=1e-10)
        assert math.isclose(ev_o, -ev_o2, abs_tol=1e-10)

def _normalized_dict(keys: List[int], rng_seed: int = 1) -> Dict[int, float]:
    rng = random.Random(rng_seed)
    vals = [rng.random() for _ in keys]
    s = sum(vals) if sum(vals) > 0 else 1.0
    return {k: v / s for k, v in zip(keys, vals)}

def _aggregate_ev(ev_by_cluster: Dict[int, List[float]], r: Dict[int, float]) -> float:
    # The implementation returns {cid: [ev]*4}; use the first component.
    return sum(r.get(cid, 0.0) * float(ev_by_cluster[cid][0]) for cid in ev_by_cluster.keys())

def _max_abs_ev(ev_by_cluster: Dict[int, List[float]]) -> float:
    return max(abs(float(vs[0])) for vs in ev_by_cluster.values()) if ev_by_cluster else 0.0


def test_compute_cluster_cfvs_unbucketed_structure_and_bounds():
    rng = random.Random(2027)
    board = _mk_board(5, rng)
    K = 4
    clusters = _mk_clusters(board, K=K, hands_per_cluster=8, seed=11)
    r_self = _normalized_dict(list(range(K)), rng_seed=7)
    r_opp  = _normalized_dict(list(range(K)), rng_seed=13)
    node = _FakeNode(board, pot=90.0, bets=(30.0, 30.0), stacks=(200.0, 200.0), r0=r_self, r1=r_opp)

    re = RiverEndgame(num_buckets=None, max_sample_per_cluster=6, seed=42)
    out = re.compute_cluster_cfvs(clusters, node, player=0, wins_fn=wins_fn,
                                  best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)
    # Structure
    assert set(out.keys()) == set(r_self.keys())
    for cid, vec in out.items():
        assert isinstance(vec, list) and len(vec) == 4, "Each cluster CFV should be a length-4 vector."
    # Magnitudes: since pairwise utilities are at most max(my_bet, opp_bet) in chips, pot-fraction should be <= 1 in magnitude.
    # NOTE: As per the paper, targets are pot-fraction CFVs (chip EV divided by P). CITATION: :contentReference[oaicite:4]{index=4}
    # Thus, |EV|/P <= max_bet/P <= 1 if bets <= pot.
    m = _max_abs_ev(out)
    assert m <= 1.0 + 1e-6, "Pot-fraction CFV magnitudes should be ≤ 1 when bets ≤ pot."

@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    K=st.integers(min_value=2, max_value=6),
    B=st.integers(min_value=2, max_value=6),
    pot=st.floats(min_value=20.0, max_value=400.0),
    bet=st.floats(min_value=2.0, max_value=200.0),
)
def test_compute_cluster_cfvs_bucketed_zero_sum_when_ranges_equal(K, B, pot, bet):
        rng = random.Random(99)
        board = _mk_board(5, rng)
        clusters = _mk_clusters(board, K=K, hands_per_cluster=8, seed=31)
        r = _normalized_dict(list(range(K)), rng_seed=5)
        node = _FakeNode(board, pot=float(pot), bets=(float(bet), float(bet)), stacks=(200.0, 200.0), r0=r, r1=r)
        re = RiverEndgame(num_buckets=B, max_sample_per_cluster=6, seed=1729)
        out = re.compute_cluster_cfvs(clusters, node, player=0, wins_fn=wins_fn, best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)
        agg_ev = _aggregate_ev(out, r)
        assert math.isfinite(agg_ev)
        assert abs(agg_ev) < 0.05

def test_compute_cluster_cfvs_sampling_consistency_when_k_geq_available():
    rng = random.Random(7)
    board = _mk_board(5, rng)
    K = 3
    clusters = _mk_clusters(board, K=K, hands_per_cluster=5, seed=101)
    r0 = _normalized_dict(list(range(K)), rng_seed=3)
    r1 = _normalized_dict(list(range(K)), rng_seed=4)
    node = _FakeNode(board, pot=60.0, bets=(20.0, 20.0), stacks=(200.0, 200.0), r0=r0, r1=r1)

    # When max_sample_per_cluster is None or ≥ available, both paths should enumerate all hands
    reA = RiverEndgame(num_buckets=None, max_sample_per_cluster=None, seed=101)
    reB = RiverEndgame(num_buckets=None, max_sample_per_cluster=9999, seed=101)
    outA = reA.compute_cluster_cfvs(clusters, node, player=0, wins_fn=wins_fn,
                                    best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)
    outB = reB.compute_cluster_cfvs(clusters, node, player=0, wins_fn=wins_fn,
                                    best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)
    for cid in outA.keys():
        assert math.isclose(outA[cid][0], outB[cid][0], rel_tol=0, abs_tol=1e-9)

# -------------------------------------------------------------------
# Pot-fraction normalization properties (paper-aligned expectations)
# -------------------------------------------------------------------

def _copy_node_with(node: _FakeNode, *, pot=None, bets=None, stacks=None) -> _FakeNode:
    return _FakeNode(
        board=list(node.public_state.board_cards),
        pot=node.public_state.pot_size if pot is None else pot,
        bets=tuple(node.public_state.current_bets if bets is None else bets),
        stacks=tuple(node.public_state.initial_stacks if stacks is None else stacks),
        r0=node.player_ranges[0],
        r1=node.player_ranges[1],
    )

@pytest.mark.parametrize("use_buckets", [False, True])
def test_pot_fraction_invariance_to_stack_scale(use_buckets: bool):
    """
    Expectation from the papers: targets are pot-fraction CFVs (chip EV divided by P),
    therefore changing the initial stacks (but not the pot or bets) must NOT change the output.
    CITATION: :contentReference[oaicite:5]{index=5}
    """
    rng = random.Random(123)
    board = _mk_board(5, rng)
    K = 4
    clusters = _mk_clusters(board, K=K, hands_per_cluster=8, seed=55)
    r0 = _normalized_dict(list(range(K)), rng_seed=8)
    r1 = _normalized_dict(list(range(K)), rng_seed=9)
    node = _FakeNode(board, pot=80.0, bets=(20.0, 25.0), stacks=(200.0, 200.0), r0=r0, r1=r1)

    re = RiverEndgame(num_buckets=(3 if use_buckets else None), max_sample_per_cluster=6, seed=2027)

    out1 = re.compute_cluster_cfvs(clusters, node, player=0, wins_fn=wins_fn,
                                   best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)
    # Double the stacks only (pot, bets unchanged)
    node2 = _copy_node_with(node, stacks=(400.0, 400.0))
    out2 = re.compute_cluster_cfvs(clusters, node2, player=0, wins_fn=wins_fn,
                                   best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)

    # Aggregate pot-fraction CFV over our range
    agg1 = _aggregate_ev(out1, r0)
    agg2 = _aggregate_ev(out2, r0)
    # Pot-fraction CFV must be unchanged by stack scale.
    assert math.isclose(agg1, agg2, rel_tol=0, abs_tol=1e-6), (
        "Pot-fraction CFV should not change when only initial stacks change."
    )

@pytest.mark.parametrize("use_buckets", [False, True])
def test_pot_fraction_invariance_to_pot_and_bet_scale(use_buckets: bool):
    """
    Expectation from the papers: if EVs are represented as fractions of the *pot*,
    then uniformly scaling (pot and both bets) by c leaves the pot-fraction CFVs unchanged.
    CITATION: :contentReference[oaicite:6]{index=6}
    """
    rng = random.Random(456)
    board = _mk_board(5, rng)
    K = 4
    clusters = _mk_clusters(board, K=K, hands_per_cluster=7, seed=66)
    r0 = _normalized_dict(list(range(K)), rng_seed=20)
    r1 = _normalized_dict(list(range(K)), rng_seed=21)
    node = _FakeNode(board, pot=100.0, bets=(30.0, 40.0), stacks=(200.0, 200.0), r0=r0, r1=r1)

    re = RiverEndgame(num_buckets=(4 if use_buckets else None), max_sample_per_cluster=6, seed=2028)
    out1 = re.compute_cluster_cfvs(clusters, node, player=0, wins_fn=wins_fn,
                                   best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)

    # Scale pot and both bets by c, stacks unchanged
    c = 1.7
    node2 = _copy_node_with(node, pot=node.public_state.pot_size * c, bets=(node.public_state.current_bets[0] * c,
                                                                            node.public_state.current_bets[1] * c))
    out2 = re.compute_cluster_cfvs(clusters, node2, player=0, wins_fn=wins_fn,
                                   best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)

    agg1 = _aggregate_ev(out1, r0)
    agg2 = _aggregate_ev(out2, r0)
    assert math.isclose(agg1, agg2, rel_tol=0, abs_tol=5e-3), (
        "Pot-fraction CFV should be invariant to uniform scaling of pot and bets."
    )

# ----------------------------
# Property tests (Hypothesis)
# ----------------------------

@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    K=st.integers(min_value=2, max_value=6),
    bet0=st.floats(min_value=1.0, max_value=200.0),
    bet1=st.floats(min_value=1.0, max_value=200.0),
    seed=st.integers(min_value=1, max_value=10_000),
)
def test_filter_hands_and_sampling_properties(K, bet0, bet1, seed):
    # Build a random board and clusters
    rng = random.Random(seed)
    board = _mk_board(5, rng)
    clusters = _mk_clusters(board, K=K, hands_per_cluster=6, seed=seed)
    # Ensure filter never yields hands touching the board or duplicate cards
    re = RiverEndgame(num_buckets=None, max_sample_per_cluster=4, seed=seed)
    for cid, hs in clusters.items():
        fh = re._filter_hands(hs, set(board))
        for h in fh:
            a, b = h.split()
            assert a != b
            assert a not in board and b not in board

    # Sampling never exceeds requested k and is deterministic for fixed key
    items = list(range(30))
    s1 = re._sample(items, k=7, key=cid + 1000)
    s2 = re._sample(items, k=7, key=cid + 1000)
    assert s1 == s2 and len(s1) == 7 and len(set(s1)) == 7


from hypothesis import assume

@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    K=st.integers(min_value=2, max_value=5),
    B=st.integers(min_value=2, max_value=6),
    pot=st.floats(min_value=20.0, max_value=400.0),
    b0=st.floats(min_value=1.0, max_value=100.0),
    b1=st.floats(min_value=1.0, max_value=100.0),
    seed=st.integers(min_value=1, max_value=9999),
)
def test_bucketed_vs_unbucketed_signals_agree_on_strict_dominance(K, B, pot, b0, b1, seed):
    # coarse bucketings can be noisy; require at least 3 buckets
    assume(B >= 3)

    # ---- robust formatting helpers (handle strings & floats) ----
    def _fmt_val(x, n=6):
        try:
            return f"{float(x):.{n}f}"
        except Exception:
            return str(x)

    def _fmt_dict(d, n=6, keys=None, sort_keys=True):
        items = d.items() if keys is None else [(k, d.get(k, 0.0)) for k in keys]
        if sort_keys:
            try:
                items = sorted(items, key=lambda kv: int(kv[0]))
            except Exception:
                items = sorted(items, key=lambda kv: str(kv[0]))
        return "{" + ", ".join(f"{str(k)}:{_fmt_val(v, n)}" for k, v in items) + "}"

    def _fmt_list(lst, n=6):
        return "[" + ", ".join(_fmt_val(x, n) for x in lst) + "]"

    rng = random.Random(seed)
    board = _mk_board(5, rng)
    clusters = _mk_clusters(board, K=K, hands_per_cluster=7, seed=seed)
    sizes = {i: len(clusters.get(i, set())) for i in range(K)}

    # Rank clusters by measured river strength on this fixed board
    def _avg_strength_for_cluster(cid: int) -> float:
        hs = sorted(list(clusters.get(int(cid), set())))
        if not hs:
            return float("-inf")
        acc, n = 0.0, 0
        for h in hs:
            a, b = h.split()
            acc += float(hand_rank_fn(best_hand_fn([a, b] + board)))
            n += 1
        return acc / float(max(1, n))

    ranked = sorted(range(K), key=lambda c: _avg_strength_for_cluster(c), reverse=True)
    half = max(1, K // 2)
    top = ranked[:half]
    bot = ranked[half:]

    # Uniform ranges over top vs bottom by *measured strength on this board*
    r0 = {i: 0.0 for i in range(K)}
    r1 = {i: 0.0 for i in range(K)}
    if top:
        u_top = 1.0 / float(len(top))
        for i in top:
            r0[i] = u_top
    if bot:
        u_bot = 1.0 / float(len(bot))
        for i in bot:
            r1[i] = u_bot

    node = _FakeNode(board, pot=float(pot), bets=(float(b0), float(b1)), stacks=(200.0, 200.0), r0=r0, r1=r1)

    reb = RiverEndgame(num_buckets=B, max_sample_per_cluster=6, seed=seed)
    reu = RiverEndgame(num_buckets=None, max_sample_per_cluster=6, seed=seed)

    out_b = reb.compute_cluster_cfvs(clusters, node, player=0, wins_fn=wins_fn, best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)
    out_u = reu.compute_cluster_cfvs(clusters, node, player=0, wins_fn=wins_fn, best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)

    def _first_comp_map(ev_by_cluster):
        return {int(k): (float(v[0]) if isinstance(v, list) else float(v)) for k, v in ev_by_cluster.items()}

    ev_b = _first_comp_map(out_b)
    ev_u = _first_comp_map(out_u)

    agg_b = _aggregate_ev(out_b, r0)
    agg_u = _aggregate_ev(out_u, r0)

    # ---- Compact yet abundant single-block debug print ----
    summary = []
    summary.append("\n================= DEBUG: bucketed_vs_unbucketed_dominance =================")
    summary.append(f"seed={int(seed)}  K={int(K)}  B={int(B)}")
    summary.append(f"pot={_fmt_val(pot, 6)}  bets=(p:{_fmt_val(b0,6)}, o:{_fmt_val(b1,6)})")
    summary.append("board=" + _fmt_list(board, 0))
    summary.append("cluster_sizes=" + _fmt_dict(sizes, n=0))
    summary.append("ranked(strong→weak)=" + _fmt_list(ranked, 0))
    summary.append("top=" + _fmt_list(top, 0) + "  bot=" + _fmt_list(bot, 0))
    summary.append("r0(top uniform)=" + _fmt_dict(r0, n=3))
    summary.append("r1(bot uniform)=" + _fmt_dict(r1, n=3))
    summary.append("bucketed_cfv[first] =" + _fmt_dict(ev_b, n=6))
    summary.append("unbucketed_cfv[first]=" + _fmt_dict(ev_u, n=6))
    summary.append(f"agg_b(top vs bot)={_fmt_val(agg_b, 12)}")
    summary.append(f"agg_u(top vs bot)={_fmt_val(agg_u, 12)}")
    summary.append("expectation: bucketed & unbucketed signals should have the SAME DIRECTION; "
                   "when both are sizable, magnitudes should be roughly consistent.")
    print("\n".join(summary))

    # ---- Paper-aligned checks ----
    # tiny signals: skip (not enough evidence to demand agreement)
    tiny = 0.03
    assume(abs(agg_b) > tiny or abs(agg_u) > tiny)

    # 1) DIRECTIONAL AGREEMENT (signs should match; allow tiny slack)
    sign_slack = 0.02  # tolerate micro disagreement near zero
    assert agg_b * agg_u >= -sign_slack, "Bucketed vs unbucketed disagree in direction."

    # 2) When both have decent magnitude, they shouldn’t be wildly different.
    if max(abs(agg_b), abs(agg_u)) > 0.10:
        assert abs(agg_b - agg_u) <= 0.20, "Magnitudes too far apart for a strong signal."


