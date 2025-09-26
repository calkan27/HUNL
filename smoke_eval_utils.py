import time
import random
from typing import Dict, Tuple

from public_state import PublicState
from game_node import GameNode
from cfr_solver import CFRSolver


def instrument_value_nets(solver: CFRSolver) -> Tuple[Dict[str, int], object]:

	counters: Dict[str, int] = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}
	orig = solver.predict_counterfactual_values

	def wrapped(node, player):
		stage = solver.get_stage(node)
		if stage in counters:
			counters[stage] += 1
		return orig(node, player)

	solver.predict_counterfactual_values = wrapped  
	return counters, orig


def measure_resolve_time(solver: CFRSolver, node: GameNode, trials: int = 5) -> float:

	total = 0.0
	for _ in range(max(0, int(trials))):
		start = time.time()
		solver.run_cfr(GameNode(node.public_state))
		total += (time.time() - start)
	return total / float(trials) if trials > 0 else 0.0


def preflop_cache_hit_rate(
	solver: CFRSolver,
	node: GameNode,
	trials: int = 6,
) -> Tuple[float, Dict[str, int]]:
	stats0_src = getattr(
		solver,
		"_preflop_cache_stats",
		{"hits": 0, "misses": 0},
	)
	stats0 = dict(stats0_src)

	h0 = int(stats0.get("hits", 0))
	m0 = int(stats0.get("misses", 0))

	tN = max(0, int(trials))
	i = 0
	while i < tN:
		ps_reset = PublicState(
			initial_stacks=list(node.public_state.initial_stacks),
			board_cards=[],
			dealer=node.public_state.dealer,
		)
		ps_reset.current_round = 0
		ps_reset.current_player = ps_reset.dealer

		n = GameNode(ps_reset)

		K = int(getattr(solver, "num_clusters", 0))

		if K > 0:
			u = 1.0 / float(K)
		else:
			u = 0.0

		r0: Dict[int, float] = {}
		r1: Dict[int, float] = {}

		j = 0
		while j < K:
			r0[j] = u
			r1[j] = u
			j += 1

		n.player_ranges = [r0, r1]

		_ = solver.run_cfr(n)

		i += 1

	stats1_src = getattr(
		solver,
		"_preflop_cache_stats",
		{"hits": 0, "misses": 0},
	)
	stats1 = dict(stats1_src)

	h1 = int(stats1.get("hits", 0))
	m1 = int(stats1.get("misses", 0))

	dh = max(0, h1 - h0)
	dm = max(0, m1 - m0)
	total = dh + dm

	if total > 0:
		hit_rate = float(dh) / float(total)
	else:
		hit_rate = 0.0

	out_stats: Dict[str, int] = {}
	out_stats["hits"] = h1
	out_stats["misses"] = m1
	out_stats["puts"] = int(stats1.get("puts", 0))
	out_stats["evictions"] = int(stats1.get("evictions", 0))

	return hit_rate, out_stats

def _make_initial_preflop(stack: int, seed: int) -> PublicState:

	random.seed(seed)
	ps = PublicState(initial_stacks=[int(stack), int(stack)], board_cards=[], dealer=0)
	ps.current_round = 0
	ps.current_player = ps.dealer
	return ps
