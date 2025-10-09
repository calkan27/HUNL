"""
I run an end-to-end smoke evaluation on a clean flop state to validate model wiring,
zero-sum residuals, cache behavior, time-per-resolve, and pot monotonicity. I can also
render a short summary that includes hit/miss stats for preflop caching and mean wall
times across streets.

Key functions: main — comprehensive smoke summary; main_basic — minimal checks for
zero-sum residual, mass conservation, and pot deltas; make_clean_state_on_flop —
deterministic flop PublicState; make_uniform_clusters — small fixed mapping;
uniform_ranges — helper; _zero_sum_residual_mag — compute residual magnitude;
nonnegative_pot_deltas_ok — single-step pot check.

Inputs: optional seeds and K; solver instances and GameNode contexts. Outputs: printed
lines and implicit return of earlier model functions to original state if patched for
instrumentation.

Dependencies: CFRSolver, engine (PublicState/GameNode/ActionType/Action/DECK), torch and
numpy through solver models. Invariants: ranges sum to one; chance cards come from
unused deck; residuals are measured after enforce_zero_sum. Performance: I set tiny
iteration counts and reuse nets to keep turnaround short.
"""

from hunl.constants import EPS_MASS, EPS_ZS
import random
import time
from typing import Dict, Set, Tuple

import numpy as np

from hunl.engine.public_state import PublicState
from hunl.engine.game_node import GameNode
from hunl.solving.cfr_solver import CFRSolver
from hunl.engine.action_type import ActionType
from hunl.engine.action import Action
from hunl.engine.poker_utils import DECK

from smoke_eval_checks import (
 zero_sum_residual_ok,
 mass_conservation_ok,
 nonnegative_pot_deltas_ok,
 pot_monotonicity_ok_sequence,
)
from smoke_eval_utils import (
 instrument_value_nets,
 measure_resolve_time,
 preflop_cache_hit_rate,
 _make_initial_preflop,
)



def make_clean_state_on_flop(seed: int = 1234) -> PublicState:
	random.seed(seed)
	board = ["AH", "KD", "2C"]  
	ps = PublicState(initial_stacks=[200, 200], board_cards=board, dealer=0)
	ps.current_round = 1
	ps.current_bets = [0, 0]
	ps.last_raiser = None
	ps.last_raise_increment = ps.big_blind
	ps.stacks = [200, 200]
	ps.pot_size = 0
	p0 = ["QS", "JC"]
	p1 = ["9H", "9D"]
	used = set(board + p0 + p1)
	ps.hole_cards = [p0[:], p1[:]]
	ps.deck = [c for c in DECK if c not in used]
	ps.current_player = (ps.dealer + 1) % 2  
	return ps


def make_uniform_clusters(
 solver: CFRSolver,
 K: int = 5,
) -> Dict[int, Set[str]]:
	hands: list[str] = []
	used: Set[str] = {"AH", "KD", "2C", "QS", "JC", "9H", "9D"}

	for c1 in DECK:
		if c1 in used:
			continue
		else:
			for c2 in DECK:
				if c2 == c1:
					continue
				else:
					if c2 in used:
						continue
					else:
						h = f"{c1} {c2}"
						h_rev = f"{c2} {c1}"

						if h in hands:
							pass
						else:
							if h_rev in hands:
								pass
							else:
								hands.append(h)

	clusters: Dict[int, Set[str]] = {}
	i = 0
	while i < K:
		clusters[i] = {hands[i]}
		i += 1

	solver.clusters = clusters
	return clusters

def uniform_ranges(K: int) -> Dict[int, float]:
	if K > 0:
		u = 1.0 / float(K)
	else:
		u = 0.0

	out: Dict[int, float] = {}
	i = 0
	while i < K:
		out[i] = u
		i += 1

	return out



def _zero_sum_residual_mag(
 solver: CFRSolver,
 node: GameNode,
) -> float:
	K = int(getattr(solver, "num_clusters", 0))

	r1 = [0.0] * K
	r2 = [0.0] * K

	for i, p in node.player_ranges[0].items():
		ii = int(i)
		if (0 <= ii) and (ii < K):
			r1[ii] = float(p)
		else:
			pass

	for i, p in node.player_ranges[1].items():
		ii = int(i)
		if (0 <= ii) and (ii < K):
			r2[ii] = float(p)
		else:
			pass

	cf1 = solver.predict_counterfactual_values(node, player=0)
	cf2 = solver.predict_counterfactual_values(node, player=1)

	v1 = []
	v2 = []

	i = 0
	while i < K:
		if i in cf1:
			v1.append(float(cf1[i][0]))
		else:
			v1.append(0.0)

		if i in cf2:
			v2.append(float(cf2[i][0]))
		else:
			v2.append(0.0)

		i += 1

	s = 0.0
	i = 0
	while i < K:
		s += v1[i] * r1[i] + v2[i] * r2[i]
		i += 1

	return abs(float(s))

def main_basic() -> None:
	ps = make_clean_state_on_flop(seed=1234)

	K = 5
	solver = CFRSolver(depth_limit=1, num_clusters=K)
	solver.load_models()
	make_uniform_clusters(solver, K=K)

	pr0 = uniform_ranges(K)
	pr1 = uniform_ranges(K)

	node = GameNode(ps)
	node.player_ranges[0] = pr0.copy()
	node.player_ranges[1] = pr1.copy()

	solver.total_iterations = 2
	solver.run_cfr(node)

	ok_zero_sum = zero_sum_residual_ok(solver, node, tol=EPS_ZS)
	ok_mass = mass_conservation_ok(node, tol=EPS_MASS)
	ok_pot = nonnegative_pot_deltas_ok(ps)

	if ok_zero_sum and ok_mass and ok_pot:
		print("PASS: zero-sum residual, bucket mass conservation, and non-negative pot deltas.")
	else:
		if not ok_zero_sum:
			print("FAIL: zero-sum residual check.")
		if not ok_mass:
			print("FAIL: bucket mass conservation check.")
		if not ok_pot:
			print("FAIL: non-negative pot deltas check.")


def nonnegative_pot_deltas_ok(ps) -> bool:
	before = ps.pot_size
	a_bet = Action(ActionType.POT_SIZED_BET)
	ps2 = ps.update_state(GameNode(ps), a_bet)
	d1 = ps2.pot_size - before
	allow1 = max(float(getattr(ps, "last_refund_amount", 0.0)), float(getattr(ps2, "last_refund_amount", 0.0)))
	if d1 + EPS_MASS < -allow1:
		return False
	a_call = Action(ActionType.CALL)
	ps3 = ps2.update_state(GameNode(ps2), a_call)
	d2 = ps3.pot_size - ps2.pot_size
	allow2 = max(float(getattr(ps2, "last_refund_amount", 0.0)), float(getattr(ps3, "last_refund_amount", 0.0)))
	if d2 + EPS_MASS < -allow2:
		return False
	return True

def main() -> None:
	ps = make_clean_state_on_flop(seed=1234)
	K = 5
	solver = CFRSolver(depth_limit=1, num_clusters=K)
	solver.load_models()
	make_uniform_clusters(solver, K=K)
	pr0 = uniform_ranges(K)
	pr1 = uniform_ranges(K)

	node = GameNode(ps)
	node.player_ranges[0] = pr0.copy()
	node.player_ranges[1] = pr1.copy()

	vn_counters, vn_orig = instrument_value_nets(solver)

	t0 = time.time()
	solver.total_iterations = 2
	solver.run_cfr(node)
	t1 = time.time()

	zmag = _zero_sum_residual_mag(solver, node)
	ok_mass = mass_conservation_ok(node, tol=EPS_MASS)
	ok_pot_once = nonnegative_pot_deltas_ok(ps)
	allowed_sparse = {ActionType.FOLD, ActionType.CALL, ActionType.POT_SIZED_BET, ActionType.ALL_IN}
	ok_pot_seq = pot_monotonicity_ok_sequence(ps, steps=6, allowed_actions=allowed_sparse)

	hr, cache_stats = preflop_cache_hit_rate(solver, GameNode(_make_initial_preflop(200, 777)), trials=6)

	ps_turn = make_clean_state_on_flop(seed=2222)
	ps_turn.current_round = 2
	ps_turn.current_bets = [0, 0]
	ps_turn.pot_size = 0
	n_turn = GameNode(ps_turn)
	n_turn.player_ranges[0] = pr0.copy()
	n_turn.player_ranges[1] = pr1.copy()
	mean_time_flop = measure_resolve_time(solver, node, trials=3)
	mean_time_turn = measure_resolve_time(solver, n_turn, trials=3)

	print("==================================== Smoke Evaluation ====================================")
	print(f"Zero-sum residual magnitude       : {zmag:.6e}")
	print(f"Bucket mass conservation (bool)   : {ok_mass}")
	print(f"Pot monotonicity (single step)    : {ok_pot_once}")
	print(f"Pot monotonicity (multi-step)     : {ok_pot_seq}")
	print("------------------------------------")
	print(f"Preflop cache hits                : {cache_stats.get('hits',0)}")
	print(f"Preflop cache misses              : {cache_stats.get('misses',0)}")
	print(f"Preflop cache puts                : {cache_stats.get('puts',0)}")
	print(f"Preflop cache evictions           : {cache_stats.get('evictions',0)}")
	print(f"Preflop cache hit rate (Δ window) : {hr*100.0:.2f}%")
	print("------------------------------------")
	print("Value-network call counts:")
	print(f"  preflop: {vn_counters.get('preflop',0)}  flop: {vn_counters.get('flop',0)}  turn: {vn_counters.get('turn',0)}  river: {vn_counters.get('river',0)}")
	print("------------------------------------")
	print(f"Mean re-solve wall time (flop)    : {mean_time_flop*1000.0:.2f} ms")
	print(f"Mean re-solve wall time (turn)    : {mean_time_turn*1000.0:.2f} ms")
	print(f"Single re-solve wall time (first) : {(t1 - t0)*1000.0:.2f} ms")
	print("==========================================================================================")

	solver.predict_counterfactual_values = vn_orig
