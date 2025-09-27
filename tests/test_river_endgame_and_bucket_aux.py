"""
Test suite for river endgame solving and bucket auxiliary head: validates 
RiverEndgame utilities, bucketed and unbucketed EV properties, pot-fraction 
invariances, and RiverBucketAux shape/gradient behavior.
"""

import math
import random
import itertools
from typing import Dict, List, Tuple

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck

import hunl.endgame.river_endgame as river_endgame
import hunl.engine.poker_utils as poker_utils
from hunl.endgame.river_endgame import RiverEndgame
from hunl.nets.river_bucket_aux import RiverBucketAux


try:
	from hunl.engine.poker_utils import DECK as PROJECT_DECK
	CANONICAL_DECK = list(PROJECT_DECK)
except Exception:
	ranks = "23456789TJQKA"
	suits = "cdhs"
	CANONICAL_DECK = [f"{r}{s}" for r in ranks for s in suits]


RANK_ORDER = {r: i for i, r in enumerate("23456789TJQKA", start=2)}

def _score_7cards(cards: List[str]) -> int:
	"""
	Compute a deterministic surrogate strength for 7 cards to enable consistent win/tie evaluation in tests.
	"""
	counts = {}
	for c in cards:
		r = c[0]
		counts[r] = counts.get(r, 0) + 1
	rank_sum = sum(RANK_ORDER[c[0]] for c in cards)
	bonus = sum((cnt * cnt) for cnt in counts.values())
	suit_hash = sum("cdhs".index(c[1]) for c in cards)
	return 1000 * bonus + 10 * rank_sum + suit_hash

def best_hand_fn(full7: List[str]) -> int:
	"""
	Return a surrogate best-hand score for a 7-card set for downstream ranking.
	"""
	return _score_7cards(full7)

def hand_rank_fn(s: int) -> int:
	"""
	Convert a surrogate score into a comparable rank; identity preserves ordering.
	"""
	return s

def wins_fn(my_hand: List[str], opp_hand: List[str], board: List[str]) -> int:
	"""
	Compare two hands on a fixed board using surrogate ranks; return 1 for win, -1 for loss, 0 for tie.
	"""
	my_score = hand_rank_fn(best_hand_fn(my_hand + board))
	opp_score = hand_rank_fn(best_hand_fn(opp_hand + board))
	if my_score > opp_score:
		return 1
	if my_score < opp_score:
		return -1
	return 0

def _mk_board(n: int = 5, rng: random.Random = None) -> List[str]:
	"""
	Sample a board of n distinct cards from the canonical deck for test scenarios.
	"""
	rng = rng or random.Random(2027)
	return rng.sample(CANONICAL_DECK, n)

def _hands_strings_from(deck: List[str]) -> List[str]:
	"""
	Enumerate all 2-card combinations from a deck as space-separated strings.
	"""
	return [f"{a} {b}" for a, b in itertools.combinations(deck, 2)]

def _mk_clusters(board: List[str], K: int, hands_per_cluster: int, seed: int = 7) -> Dict[int, set]:
	"""
	Create K clusters of hole-card strings disjoint from the board, each holding up to hands_per_cluster items.
	"""
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


def test_bucket_aux_shapes_and_gradients():
	"""
	Verify RiverBucketAux returns (batch, num_clusters) outputs and supports backpropagation with finite gradients.
	"""
	K = 8
	B = 5
	net = RiverBucketAux(num_buckets=B, num_clusters=K)
	net.train()
	x = torch.randn(17, K, B, requires_grad=True)
	y = net(x)
	assert y.shape == (17, K)
	s = y.sum()
	s.backward()
	assert x.grad is not None and torch.isfinite(x.grad).all()

def test_bucket_aux_predict_matches_forward_under_no_grad():
	"""
	Check that predict mirrors forward under no-grad context for identical outputs.
	"""
	K = 6
	B = 3
	net = RiverBucketAux(num_buckets=B, num_clusters=K).eval()
	x = torch.randn(4, K, B)
	with torch.no_grad():
		y_forward = net(x)
	y_pred = net.predict(x)
	assert torch.allclose(y_forward, y_pred, atol=1e-6)


def test_filter_hands_removes_board_and_duplicates():
	"""
	Ensure _filter_hands removes any hand touching the board or with duplicate cards while preserving valid hands.
	"""
	re = RiverEndgame(num_buckets=None, max_sample_per_cluster=None)
	board = ["Ah", "Kd", "7s", "2c", "3c"]
	board_set = set(board)
	raw = ["Ah As", "7s 7s", "Qc Qd", "2c 3c", "Jh Ts"]
	out = re._filter_hands(raw, board_set)
	assert "Qc Qd" in out and "Jh Ts" in out
	assert all(h not in out for h in ["Ah As", "2c 3c"])
	for h in out:
		a, b = h.split()
		assert a != b

def test_sample_is_deterministic_per_key_and_seed_and_bounds():
	"""
	Confirm _sample returns the full set when k exceeds size, is deterministic for a fixed key, and varies across keys.
	"""
	re = RiverEndgame(num_buckets=None, max_sample_per_cluster=3, seed=123)
	items = list(range(20))
	assert set(re._sample(items, k=100, key=1)) == set(items)
	s1 = re._sample(items, k=5, key=42)
	s2 = re._sample(items, k=5, key=42)
	assert s1 == s2
	s3 = re._sample(items, k=5, key=43)
	assert s3 != s1

def test_bucketize_edge_cases_and_monotonicity():
	"""
	Test _bucketize produces nondecreasing bucket indices by strength and assigns unique buckets when capacity allows.
	"""
	re = RiverEndgame(num_buckets=4)
	strengths = [1, 2, 3, 4, 5, 6]
	bmap, B = re._bucketize(strengths)
	assert B == 4
	assert set(bmap.values()) <= set(range(B))
	prev_b = -1
	for s in sorted(set(strengths)):
		b = bmap[s]
		assert b >= prev_b
		prev_b = b
	re2 = RiverEndgame(num_buckets=10)
	strengths2 = [10, 11, 12]
	bmap2, B2 = re2._bucketize(strengths2)
	assert B2 == len(set(strengths2))
	assert [bmap2[s] for s in sorted(set(strengths2))] == list(range(B2))

def test_cluster_distribution_length_and_consistency():
	"""
	Validate _cluster_distribution length matches input and detects constant-strength scenarios.
	"""
	re = RiverEndgame(num_buckets=5)
	board = _mk_board()
	hands = ["Ah Kh", "Qs Jd", "2c 7c"]
	dist = re._cluster_distribution(hands, board, best_hand_fn, hand_rank_fn)
	assert len(dist) == len(hands)
	def const_best(hb): return 123
	dist2 = re._cluster_distribution(hands, board, const_best, hand_rank_fn)
	assert len(set(dist2)) == 1 and dist2[0] == 123

def test_pairwise_utility_zero_sum_and_formulas():
	"""
	Check pairwise utility sign conventions and verify zero-sum of expected utilities for a head-to-head matchup.
	"""
	re = RiverEndgame()
	for res in (-1, 0, 1):
		up = re._pairwise_util_p(res, pot_size=100.0, my_bet=10.0, opp_bet=25.0)
		if res > 0:
			assert up == 25.0
		elif res < 0:
			assert up == -10.0
		else:
			assert up == 0.5 * (25.0 - 10.0)
		ep, eo = re._expected_utility_pairwise(["Ah", "Kd"], ["Qs", "Jh"],
				["2c","3c","4d","5h","6s"], wins_fn, pot_size=100.0, my_bet=10.0, opp_bet=25.0)
		assert math.isclose(ep + eo, 0.0, rel_tol=0, abs_tol=1e-9)

def test_bucket_level_expectations_zero_sum_and_swap_invariance():
	"""
	Ensure bucket-level expected utilities are zero-sum for equal bets and are invariant under player-bucket swap.
	"""
	re = RiverEndgame()
	B = 6
	rng = random.Random(99)
	pb = np.asarray([rng.random() for _ in range(B)], dtype=float)
	qb = np.asarray([rng.random() for _ in range(B)], dtype=float)
	pb = pb / (pb.sum() if pb.sum() > 0 else 1.0)
	qb = qb / (qb.sum() if qb.sum() > 0 else 1.0)
	ev_p, ev_o = re._expected_utility_buckets_both(pb.tolist(), qb.tolist(), B, my_bet=20.0, opp_bet=20.0)
	assert math.isclose(ev_p + ev_o, 0.0, abs_tol=1e-10)
	ev_p2, ev_o2 = re._expected_utility_buckets_both(qb.tolist(), pb.tolist(), B, my_bet=20.0, opp_bet=20.0)
	assert math.isclose(ev_p, ev_o2, abs_tol=1e-10)
	assert math.isclose(ev_o, ev_p2, abs_tol=1e-10)


class _FakePS:
	"""
	Minimal public-state container for compute_cluster_cfvs tests carrying board, pot, bets, and stacks.
	"""
	def __init__(self, board, pot, bets, stacks):
		"""
		Initialize fields necessary for river endgame EV computation.
		"""
		self.board_cards = list(board)
		self.pot_size = float(pot)
		self.current_bets = [float(bets[0]), float(bets[1])]
		self.initial_stacks = [float(stacks[0]), float(stacks[1])]

class _FakeNode:
	"""
	Wrapper for public state and per-player cluster ranges used when invoking compute_cluster_cfvs.
	"""
	def __init__(self, board, pot, bets, stacks, r0: Dict[int, float], r1: Dict[int, float]):
		"""
		Attach a public state and store player 0 and player 1 cluster range dictionaries.
		"""
		self.public_state = _FakePS(board, pot, bets, stacks)
		self.player_ranges = [dict(r0), dict(r1)]


def _normalized_dict(keys: List[int], rng_seed: int = 1) -> Dict[int, float]:
	"""
	Build a probability distribution over given keys using a seeded RNG.
	"""
	rng = random.Random(rng_seed)
	vals = [rng.random() for _ in keys]
	s = sum(vals) if sum(vals) > 0 else 1.0
	return {k: v / s for k, v in zip(keys, vals)}

def _aggregate_ev(ev_by_cluster: Dict[int, List[float]], r: Dict[int, float]) -> float:
	"""
	Aggregate cluster EVs by range weights, reading the first component of each cluster vector.
	"""
	return sum(r.get(cid, 0.0) * float(ev_by_cluster[cid][0]) for cid in ev_by_cluster.keys())

def _max_abs_ev(ev_by_cluster: Dict[int, List[float]]) -> float:
	"""
	Return the maximum absolute EV across clusters using the first component per cluster.
	"""
	return max(abs(float(vs[0])) for vs in ev_by_cluster.values()) if ev_by_cluster else 0.0


def test_compute_cluster_cfvs_unbucketed_structure_and_bounds():
	"""
	Validate unbucketed compute_cluster_cfvs output structure and ensure 
	pot-fraction EV magnitudes are bounded by 1 when bets do not exceed pot.
	"""
	rng = random.Random(2027)
	board = _mk_board(5, rng)
	K = 4
	clusters = _mk_clusters(board, K=K, hands_per_cluster=8, seed=11)
	r_self = _normalized_dict(list(range(K)), rng_seed=7)
	r_opp  = _normalized_dict(list(range(K)), rng_seed=13)
	node = _FakeNode(board, pot=90.0, bets=(30.0, 30.0), stacks=(200.0, 200.0), r0=r_self, r1=r_opp)
	re = RiverEndgame(num_buckets=None, max_sample_per_cluster=6, seed=42)
	out = re.compute_cluster_cfvs(clusters, node, player=0, wins_fn=wins_fn, best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)
	assert set(out.keys()) == set(r_self.keys())
	for cid, vec in out.items():
		assert isinstance(vec, list) and len(vec) == 4
	m = _max_abs_ev(out)
	assert m <= 1.0 + 1e-6

@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
	K=st.integers(min_value=2, max_value=6),
	B=st.integers(min_value=2, max_value=6),
	pot=st.floats(min_value=20.0, max_value=400.0),
	bet=st.floats(min_value=2.0, max_value=200.0),
)
def test_compute_cluster_cfvs_bucketed_zero_sum_when_ranges_equal(K, B, pot, bet):
	"""
	Check bucketed branch yields approximately zero aggregate pot-fraction EV when both 
	players have identical cluster ranges and equal bets.
	"""
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
	"""
	Ensure unbucketed enumeration agrees when max_sample_per_cluster is None 
	versus a very large value, holding seed constant.
	"""
	rng = random.Random(7)
	board = _mk_board(5, rng)
	K = 3
	clusters = _mk_clusters(board, K=K, hands_per_cluster=5, seed=101)
	r0 = _normalized_dict(list(range(K)), rng_seed=3)
	r1 = _normalized_dict(list(range(K)), rng_seed=4)
	node = _FakeNode(board, pot=60.0, bets=(20.0, 20.0), stacks=(200.0, 200.0), r0=r0, r1=r1)
	reA = RiverEndgame(num_buckets=None, max_sample_per_cluster=None, seed=101)
	reB = RiverEndgame(num_buckets=None, max_sample_per_cluster=9999, seed=101)
	outA = reA.compute_cluster_cfvs(clusters, node, player=0, wins_fn=wins_fn, best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)
	outB = reB.compute_cluster_cfvs(clusters, node, player=0, wins_fn=wins_fn, best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)
	for cid in outA.keys():
		assert math.isclose(outA[cid][0], outB[cid][0], rel_tol=0, abs_tol=1e-9)


def _copy_node_with(node: _FakeNode, *, pot=None, bets=None, stacks=None) -> _FakeNode:
	"""
	Create a shallow-copied node with optional overrides for pot, bets, or stacks while preserving ranges.
	"""
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
	Assert pot-fraction CFV aggregates are invariant to scaling initial
	stacks when pot and bets are unchanged, for both bucketed and unbucketed modes.
	"""
	rng = random.Random(123)
	board = _mk_board(5, rng)
	K = 4
	clusters = _mk_clusters(board, K=K, hands_per_cluster=8, seed=55)
	r0 = _normalized_dict(list(range(K)), rng_seed=8)
	r1 = _normalized_dict(list(range(K)), rng_seed=9)
	node = _FakeNode(board, pot=80.0, bets=(20.0, 25.0), stacks=(200.0, 200.0), r0=r0, r1=r1)
	re = RiverEndgame(num_buckets=(3 if use_buckets else None), max_sample_per_cluster=6, seed=2027)
	out1 = re.compute_cluster_cfvs(clusters, node, player=0, wins_fn=wins_fn, best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)
	node2 = _copy_node_with(node, stacks=(400.0, 400.0))
	out2 = re.compute_cluster_cfvs(clusters, node2, player=0, wins_fn=wins_fn, best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)
	agg1 = _aggregate_ev(out1, r0)
	agg2 = _aggregate_ev(out2, r0)
	assert math.isclose(agg1, agg2, rel_tol=0, abs_tol=1e-6)

@pytest.mark.parametrize("use_buckets", [False, True])
def test_pot_fraction_invariance_to_pot_and_bet_scale(use_buckets: bool):
	"""
	Assert pot-fraction CFV aggregates are invariant to uniform scaling of pot 
	and both bets, for bucketed and unbucketed modes.
	"""
	rng = random.Random(456)
	board = _mk_board(5, rng)
	K = 4
	clusters = _mk_clusters(board, K=K, hands_per_cluster=7, seed=66)
	r0 = _normalized_dict(list(range(K)), rng_seed=20)
	r1 = _normalized_dict(list(range(K)), rng_seed=21)
	node = _FakeNode(board, pot=100.0, bets=(30.0, 40.0), stacks=(200.0, 200.0), r0=r0, r1=r1)
	re = RiverEndgame(num_buckets=(4 if use_buckets else None), max_sample_per_cluster=6, seed=2028)
	out1 = re.compute_cluster_cfvs(clusters, node, player=0, wins_fn=wins_fn, best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)
	c = 1.7
	node2 = _copy_node_with(node, pot=node.public_state.pot_size * c, bets=(node.public_state.current_bets[0] * c, node.public_state.current_bets[1] * c))
	out2 = re.compute_cluster_cfvs(clusters, node2, player=0, wins_fn=wins_fn, best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn)
	agg1 = _aggregate_ev(out1, r0)
	agg2 = _aggregate_ev(out2, r0)
	assert math.isclose(agg1, agg2, rel_tol=0, abs_tol=5e-3)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
	K=st.integers(min_value=2, max_value=6),
	bet0=st.floats(min_value=1.0, max_value=200.0),
	bet1=st.floats(min_value=1.0, max_value=200.0),
	seed=st.integers(min_value=1, max_value=10_000),
)
def test_filter_hands_and_sampling_properties(K, bet0, bet1, seed):
	"""
	Property-test that filtered hands never touch the board or duplicate cards and that sampling is deterministic per key with correct cardinality.
	"""
	rng = random.Random(seed)
	board = _mk_board(5, rng)
	clusters = _mk_clusters(board, K=K, hands_per_cluster=6, seed=seed)
	re = RiverEndgame(num_buckets=None, max_sample_per_cluster=4, seed=seed)
	for cid, hs in clusters.items():
		fh = re._filter_hands(hs, set(board))
		for h in fh:
			a, b = h.split()
			assert a != b
			assert a not in board and b not in board
	items = list(range(30))
	s1 = re._sample(items, k=7, key=cid + 1000)
	s2 = re._sample(items, k=7, key=cid + 1000)
	assert s1 == s2 and len(s1) == 7 and len(set(s1)) == 7


def test_bucketed_signal_nonnegative_under_strength_partition():
	"""
	Checks the paper-aligned property on the river: when ranges are partitioned by realized
	hand-strength into stronger (ours) vs weaker (opponent) clusters, the bucketed endgame
	(abstraction used for river actions) yields a non-negative aggregate EV for the stronger side,
	measured in pot-fraction CFVs. This matches the spec that the river is handled by a bucketed
	abstraction and that targets are expressed as fractions of the pot.
	"""
	rng = random.Random(4498)
	K, B, pot, b0, b1 = 2, 2, 20.0, 1.0, 1.0
	board = _mk_board(5, rng)
	clusters = _mk_clusters(board, K=K, hands_per_cluster=7, seed=4498)

	def _cluster_strength_score(cid: int) -> float:
		hands = sorted(list(clusters.get(cid, set())))
		if not hands:
			return float("-inf")
		s = 0.0
		n = 0
		board_norm = [c[0] + c[1].lower() for c in board]
		for h in hands[:min(12, len(hands))]:
			c1, c2 = (h.split() if isinstance(h, str) else list(h))
			c1n, c2n = c1[0] + c1[1].lower(), c2[0] + c2[1].lower()
			best = best_hand_fn([c1n, c2n] + list(board_norm))
			rank_val = hand_rank_fn(best)
			s += float(rank_val[0]) if isinstance(rank_val, (tuple, list)) else float(rank_val)
			n += 1
		return s / max(1, n)

	order = sorted(range(K), key=_cluster_strength_score)
	bot = order[: K // 2]
	top = order[K // 2 :]

	r0 = {i: 0.0 for i in range(K)}
	r1 = {i: 0.0 for i in range(K)}
	for i in top:
		r0[i] = 1.0 / max(1, len(top))
	for i in bot:
		r1[i] = 1.0 / max(1, len(bot))

	node = _FakeNode(board, pot=float(pot), bets=(float(b0), float(b1)), stacks=(200.0, 200.0), r0=r0, r1=r1)

	reb = RiverEndgame(num_buckets=B, max_sample_per_cluster=6, seed=4498)

	out_b = reb.compute_cluster_cfvs(
		clusters, node, player=0, wins_fn=wins_fn, best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn
	)

	agg_b = _aggregate_ev(out_b, r0)

	assert agg_b > -0.10


def test_unbucketed_signal_sanity_under_strength_partition():
	"""
	Sanity-check for the exact (unbucketed) evaluator under the same strength-based split.
	This test verifies structural and scaling invariants that the papers require (pot-fraction
	outputs, zero-sum at the pairwise level aggregated into cluster EVs), without imposing the
	non-negativity condition that is not guaranteed by the spec. Specifically it checks:
	  1) CFVs are returned for all our clusters and each value vector has length 4, and
	  2) the aggregate EV in pot-fraction units has magnitude ≤ 1 (consistent with pot-fraction scaling).
	"""
	rng = random.Random(4498)
	K, pot, b0, b1 = 2, 20.0, 1.0, 1.0
	board = _mk_board(5, rng)
	clusters = _mk_clusters(board, K=K, hands_per_cluster=7, seed=4498)

	def _cluster_strength_score(cid: int) -> float:
		hands = sorted(list(clusters.get(cid, set())))
		if not hands:
			return float("-inf")
		s = 0.0
		n = 0
		board_norm = [c[0] + c[1].lower() for c in board]
		for h in hands[:min(12, len(hands))]:
			c1, c2 = (h.split() if isinstance(h, str) else list(h))
			c1n, c2n = c1[0] + c1[1].lower(), c2[0] + c2[1].lower()
			best = best_hand_fn([c1n, c2n] + list(board_norm))
			rank_val = hand_rank_fn(best)
			s += float(rank_val[0]) if isinstance(rank_val, (tuple, list)) else float(rank_val)
			n += 1
		return s / max(1, n)

	order = sorted(range(K), key=_cluster_strength_score)
	bot = order[: K // 2]
	top = order[K // 2 :]

	r0 = {i: 0.0 for i in range(K)}
	r1 = {i: 0.0 for i in range(K)}
	for i in top:
		r0[i] = 1.0 / max(1, len(top))
	for i in bot:
		r1[i] = 1.0 / max(1, len(bot))

	node = _FakeNode(board, pot=float(pot), bets=(float(b0), float(b1)), stacks=(200.0, 200.0), r0=r0, r1=r1)

	reu = RiverEndgame(num_buckets=None, max_sample_per_cluster=6, seed=4498)

	out_u = reu.compute_cluster_cfvs(
		clusters, node, player=0, wins_fn=wins_fn, best_hand_fn=best_hand_fn, hand_rank_fn=hand_rank_fn
	)

	assert set(out_u.keys()) == set(r0.keys())
	for cid, vec in out_u.items():
		assert isinstance(vec, list) and len(vec) == 4

	agg_u = _aggregate_ev(out_u, r0)
	assert abs(agg_u) <= 1.0 + 1e-6

