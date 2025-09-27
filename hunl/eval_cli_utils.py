from hunl.constants import EPS_MASS, EPS_ZS, SEED_RIVER
import random
from functools import partial
from typing import Callable, Dict, List, Tuple, Optional

from hunl.engine.public_state import PublicState
from hunl.engine.game_node import GameNode
from hunl.engine.action_type import ActionType
from hunl.engine.action import Action
from hunl.engine.poker_utils import DECK
from hunl.solving.cfr_solver import CFRSolver
from hunl.data.data_generator import DataGenerator


def _value_fn_apply(
 solver: CFRSolver,
 node: GameNode,
 player: int,
) -> Dict[int, List[float]]:
	return solver.predict_counterfactual_values(node, player)


def _value_fn_from_solver(
 solver: CFRSolver,
) -> Callable[[GameNode, int], Dict[int, List[float]]]:
	return partial(_value_fn_apply, solver)


def _make_tmp_solver_from_template(
 solver_template: CFRSolver,
 iters: int,
) -> CFRSolver:
	if hasattr(solver_template, "_config"):
		cfg = solver_template._config
	else:
		cfg = None

	if cfg is not None:
		tmp = CFRSolver(config=cfg)
	else:
		tmp = CFRSolver(
		 depth_limit=solver_template.depth_limit,
		 num_clusters=solver_template.num_clusters,
		)

	tmp.clusters = solver_template.clusters
	tmp.models = solver_template.models
	tmp.num_clusters = solver_template.num_clusters
	tmp.total_iterations = int(iters)

	return tmp


def _policy_from_resolve_apply(
 solver_template: CFRSolver,
 iters: int,
 node: GameNode,
 player: int,
) -> Dict[ActionType, float]:
	tmp = _make_tmp_solver_from_template(solver_template, iters)

	shadow = GameNode(node.public_state)
	shadow.player_ranges = [
	 dict(node.player_ranges[0]),
	 dict(node.player_ranges[1]),
	]

	if hasattr(tmp, "cfr_values"):
		tmp.cfr_values.clear()
	else:
		pass

	tmp._calculate_counterfactual_values(shadow, player)

	if player == node.public_state.current_player:
		allowed = tmp._allowed_actions_agent(node.public_state)
	else:
		allowed = tmp._allowed_actions_opponent(node.public_state)

	mixed = tmp._mixed_action_distribution(shadow, player, allowed)

	dist: Dict[ActionType, float] = {}

	for a, p in zip(allowed, mixed):
		dist[a] = float(p)

	s = 0.0
	for v in dist.values():
		s += float(v)

	if s <= 0.0:
		if len(allowed) > 0:
			u = 1.0 / float(len(allowed))
			out = {}
			for a in allowed:
				out[a] = u
			return out
		else:
			return {}
	else:
		for k in list(dist.keys()):
			dist[k] = dist[k] / s
		return dist


def _policy_from_resolve(
 solver_template: CFRSolver,
 iters: int = 2,
) -> Callable[[GameNode, int], Dict[ActionType, float]]:
	return partial(_policy_from_resolve_apply, solver_template, int(iters))


def _sample_from_policy(
 dist: Dict[ActionType, float],
) -> ActionType:
	r = random.random()
	c = 0.0
	choices = list(dist.items())

	if len(choices) == 0:
		return ActionType.CALL
	else:
		for a, p in choices:
			c += float(p)
			if r <= c:
				return a
		return choices[-1][0]


def _make_initial_preflop(
 stack: int,
 seed: int,
) -> PublicState:
	random.seed(seed)

	ps = PublicState(
	 initial_stacks=[stack, stack],
	 board_cards=[],
	 dealer=0,
	)
	ps.current_round = 0
	ps.current_player = ps.dealer

	return ps


def _wrapped_predict_cfv_apply(
 dg: DataGenerator,
 counters: Dict[str, int],
 orig_fn: Callable[[GameNode, int], Dict[int, List[float]]],
 nd: GameNode,
 pl: int,
) -> Dict[int, List[float]]:
	st = dg.cfr_solver.get_stage(nd)

	if st == "turn":
		counters["turn_leaf_calls"] += 1
	else:
		if st == "flop":
			counters["flop_calls"] += 1
		else:
			pass

	return orig_fn(nd, pl)


def flop_turn_leaf_sanity(
 samples: int = 5,
 seed: int = SEED_RIVER,
) -> Dict[str, int]:
	random.seed(seed)

	dg = DataGenerator(
	 num_boards=samples,
	 num_samples_per_board=1,
	 player_stack=200,
	 num_clusters=6,
	)

	dg.cfr_solver.depth_limit = 1
	dg.cfr_solver.total_iterations = 2

	counters = {
	 "turn_leaf_calls": 0,
	 "flop_calls": 0,
	}

	orig = dg.cfr_solver.predict_counterfactual_values
	wrapped = partial(_wrapped_predict_cfv_apply, dg, counters, orig)
	dg.cfr_solver.predict_counterfactual_values = wrapped

	i = 0
	while i < int(samples):
		node = dg._sample_flop_situation(random.Random(seed + i))
		dg.cfr_solver.run_cfr(node)
		i += 1

	dg.cfr_solver.predict_counterfactual_values = orig

	if int(counters["turn_leaf_calls"]) == 0:
		pass
	else:
		print("flop_turn_leaf_sanity: turn_leaf_calls != 0")

	return {
	 "samples": int(samples),
	 "turn_leaf_calls": int(counters["turn_leaf_calls"]),
	 "flop_calls": int(counters["flop_calls"]),
	}


def _sparse_menu(
 ps: PublicState,
) -> List[ActionType]:
	p = ps.current_player
	o = (p + 1) % 2
	my_bet = ps.current_bets[p]
	opp_bet = ps.current_bets[o]
	to_call = opp_bet - my_bet

	if to_call < 0:
		to_call = 0
	else:
		pass

	if hasattr(ps, "legal_actions"):
		legal = ps.legal_actions()
	else:
		legal = []

	cands: List[ActionType] = []

	if to_call > 0:
		if ActionType.FOLD in legal:
			cands.append(ActionType.FOLD)
		else:
			pass

	if ActionType.CALL in legal:
		cands.append(ActionType.CALL)
	else:
		pass

	if ActionType.POT_SIZED_BET in legal:
		cands.append(ActionType.POT_SIZED_BET)
	else:
		pass

	if ActionType.ALL_IN in legal:
		cands.append(ActionType.ALL_IN)
	else:
		pass

	return cands


def is_range_mass_conserved(
 r1: Dict[int, float],
 r2: Dict[int, float],
 tol: float = EPS_MASS,
) -> bool:
	s1 = 0.0
	for _, p in r1.items():
		s1 += float(p)

	s2 = 0.0
	for _, p in r2.items():
		s2 += float(p)

	if abs(s1 - 1.0) <= tol:
		if abs(s2 - 1.0) <= tol:
			return True
		else:
			return False
	else:
		return False


def is_zero_sum_residual_ok(
 solver: CFRSolver,
 tol: float = EPS_ZS,
) -> bool:
	d = solver.get_last_diagnostics()

	if isinstance(d, dict):
		val = float(d.get("zero_sum_residual", 0.0))
	else:
		val = 0.0

	if val <= float(tol):
		return True
	else:
		return False


def is_nonnegative_pot_delta(
 prev_ps: PublicState,
 next_ps: PublicState,
) -> bool:
	if float(next_ps.pot_size) + EPS_MASS >= float(prev_ps.pot_size):
		return True
	else:
		return False


def _init_diag_solver(
 ps: PublicState,
 K: int,
 depth: int,
 iters: int,
 k1: float,
 k2: float,
) -> Tuple[Optional[CFRSolver], str]:
	if K < 0:
		print("diag: invalid K")
		return None, "bad_K"
	else:
		pass

	solver = CFRSolver(
	 depth_limit=int(depth),
	 num_clusters=int(K),
	)
	solver.total_iterations = int(iters)

	if hasattr(solver, "set_soundness_constants"):
		solver.set_soundness_constants(float(k1), float(k2))
	else:
		pass

	return solver, ""


def _build_linear_clusters(
 ps: PublicState,
 K: int,
) -> Dict[int, set]:
	used = set(ps.board_cards)
	cards = []

	for c in DECK:
		if c in used:
			pass
		else:
			cards.append(c)

	clusters: Dict[int, set] = {}
	i = 0
	j = 0

	while (i < K) and (j + 1 < len(cards)):
		h = f"{cards[j]} {cards[j + 1]}"
		clusters[i] = {h}
		i += 1
		j += 2

	while i < K:
		clusters[i] = set()
		i += 1

	return clusters


def _make_node_with_ranges(
 ps: PublicState,
 K: int,
 r_us: Dict[int, float],
 r_opp: Dict[int, float],
) -> GameNode:
	node = GameNode(ps)

	cur = int(ps.current_player)
	opp = (cur + 1) % 2

	node.player_ranges[cur] = dict(r_us)
	node.player_ranges[opp] = dict(r_opp)

	return node


def _default_diag_spec(
 depth: int,
 iters: int,
 k1: float,
 k2: float,
) -> Dict[str, object]:
	return {
	 "depth_limit": int(depth),
	 "iterations": int(iters),
	 "zero_sum_residual": 0.0,
	 "zero_sum_residual_mean": 0.0,
	 "regret_l2": 0.0,
	 "avg_strategy_entropy": 0.0,
	 "cfv_calls": {},
	 "constraint_mode": "",
	 "preflop_cache": {},
	 "k1": float(k1),
	 "k2": float(k2),
	}


def _pack_solver_diag(
 diag: Dict[str, object],
 depth: int,
 iters: int,
 k1: float,
 k2: float,
) -> Dict[str, object]:
	out: Dict[str, object] = {}

	out["depth_limit"] = int(diag.get("depth_limit", depth))
	out["iterations"] = int(diag.get("iterations", iters))
	out["zero_sum_residual"] = float(diag.get("zero_sum_residual", 0.0))
	out["zero_sum_residual_mean"] = float(
	 diag.get("zero_sum_residual_mean", 0.0)
	)
	out["regret_l2"] = float(diag.get("regret_l2", 0.0))
	out["avg_strategy_entropy"] = float(
	 diag.get("avg_strategy_entropy", 0.0)
	)
	out["cfv_calls"] = dict(diag.get("cfv_calls", {}))
	out["constraint_mode"] = str(diag.get("constraint_mode", ""))
	out["preflop_cache"] = dict(diag.get("preflop_cache", {}))
	out["k1"] = float(diag.get("k1", k1))
	out["k2"] = float(diag.get("k2", k2))

	return out


def _diag_from_solver(
 ps: PublicState,
 K: int,
 r_us: Dict[int, float],
 r_opp: Dict[int, float],
 depth: int = 1,
 iters: int = 8,
 k1: float = 0.0,
 k2: float = 0.0,
) -> Dict[str, object]:
	solver, err = _init_diag_solver(ps, int(K), int(depth), int(iters), k1, k2)

	if solver is None:
		print(f"diag: init failed: {err}")
		return _default_diag_spec(depth, iters, k1, k2)
	else:
		pass

	clusters = _build_linear_clusters(ps, int(K))
	solver.clusters = clusters

	node = _make_node_with_ranges(ps, int(K), r_us, r_opp)

	ok_mass = is_range_mass_conserved(
	 node.player_ranges[0],
	 node.player_ranges[1],
	 tol=EPS_MASS,
	)

	if ok_mass:
		pass
	else:
		print("diag: mass conservation failed")
		return _default_diag_spec(depth, iters, k1, k2)

	_ = solver.run_cfr(node)

	diag = solver.get_last_diagnostics()

	if isinstance(diag, dict):
		return _pack_solver_diag(diag, depth, iters, k1, k2)
	else:
		print("diag: diagnostics missing or invalid")
		return _default_diag_spec(depth, iters, k1, k2)

