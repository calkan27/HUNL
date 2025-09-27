from hunl.constants import EPS_MASS, EPS_ZS, SEED_DEFAULT
import math
import random
from typing import Dict, List, Tuple, Optional

from hunl.engine.public_state import PublicState
from hunl.engine.game_node import GameNode
from hunl.engine.action_type import ActionType
from hunl.engine.action import Action
from hunl.engine.poker_utils import DECK  
from hunl.solving.cfr_solver import CFRSolver
from hunl.resolve_config import ResolveConfig


def _sparse_menu(ps: PublicState) -> List[ActionType]:
	p = ps.current_player
	o = (p + 1) % 2
	my_bet = ps.current_bets[p]
	opp_bet = ps.current_bets[o]
	to_call = opp_bet - my_bet

	if to_call < 0:
		to_call = 0

	if hasattr(ps, "legal_actions"):
		legal = ps.legal_actions()
	else:
		legal = []

	cands: List[ActionType] = []

	if to_call > 0:
		if ActionType.FOLD in legal:
			cands.append(ActionType.FOLD)

	if ActionType.CALL in legal:
		cands.append(ActionType.CALL)

	if ActionType.POT_SIZED_BET in legal:
		cands.append(ActionType.POT_SIZED_BET)

	if ActionType.ALL_IN in legal:
		cands.append(ActionType.ALL_IN)

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

	ok1 = abs(s1 - 1.0) <= tol
	ok2 = abs(s2 - 1.0) <= tol

	if ok1:
		if ok2:
			return True
		else:
			return False
	else:
		return False


def is_nonnegative_pot_delta(
 prev_ps: PublicState,
 next_ps: PublicState,
) -> bool:
	eps = EPS_MASS
	a = float(getattr(prev_ps, "last_refund_amount", 0.0))
	b = float(getattr(next_ps, "last_refund_amount", 0.0))
	allow = max(a, b)

	if float(next_ps.pot_size) + eps >= float(prev_ps.pot_size) - allow:
		return True
	else:
		return False


def _clone_solver_template(
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


def _engine_policy_action(
 solver_template: CFRSolver,
 node: GameNode,
 iters: int = 2,
) -> ActionType:
	tmp = _clone_solver_template(solver_template, iters)

	shadow = GameNode(node.public_state)
	shadow.player_ranges = [
	 dict(node.player_ranges[0]),
	 dict(node.player_ranges[1]),
	]

	if hasattr(tmp, "_calculate_counterfactual_values"):
		if hasattr(tmp, "cfr_values"):
			tmp.cfr_values.clear()
		else:
			pass
		curp = int(getattr(shadow.public_state, "current_player", 0))
		tmp._calculate_counterfactual_values(shadow, curp)
	else:
		pass

	curp = int(getattr(shadow.public_state, "current_player", 0))
	nodep = int(getattr(node.public_state, "current_player", 0))

	if curp == nodep:
		allowed = tmp._allowed_actions_agent(node.public_state)
		mixed = tmp._mixed_action_distribution(shadow, curp, allowed)
	else:
		allowed = tmp._allowed_actions_opponent(node.public_state)
		mixed = tmp._mixed_action_distribution(shadow, curp, allowed)

	if not mixed:
		return ActionType.CALL
	else:
		if not allowed:
			return ActionType.CALL
		else:
			r = random.random()
			c = 0.0
			for a, p in zip(allowed, mixed):
				c += float(p)
				if r <= c:
					return a
			return allowed[-1]


def _prio(a: ActionType) -> int:
	if a == ActionType.POT_SIZED_BET:
		return 3
	else:
		if a == ActionType.ALL_IN:
			return 2
		else:
			if a == ActionType.CALL:
				return 1
			else:
				return 0


def lbr_greedy_action(
 ps: PublicState,
 solver: CFRSolver,
 lbr_player: int,
 iters_after: int,
 freq_log: Dict[str, Dict[str, int]],
) -> ActionType:
	if int(ps.current_round) != 1:
		return _engine_policy_action(
		 solver,
		 GameNode(ps),
		 iters=iters_after,
		)
	else:
		if int(ps.current_player) != int(lbr_player):
			return _engine_policy_action(
			 solver,
			 GameNode(ps),
			 iters=iters_after,
			)
		else:
			menu = _sparse_menu(ps)

			if not menu:
				return ActionType.CALL
			else:
				ev_by_action: Dict[ActionType, float] = {}

				for a in menu:
					ps2 = ps.update_state(GameNode(ps), Action(a))

					if getattr(ps2, "is_terminal", False):
						if hasattr(ps2, "terminal_utility"):
							u = ps2.terminal_utility()
						else:
							u = [0.0, 0.0]

						if isinstance(u, (list, tuple)):
							if len(u) >= 2:
								ev_by_action[a] = float(u[int(lbr_player)])
							else:
								ev_by_action[a] = 0.0
						else:
							ev_by_action[a] = 0.0

						continue
					else:
						node = GameNode(ps2)
						K = solver.num_clusters
						u0 = 1.0 / float(K) if K > 0 else 0.0
						node.player_ranges[0] = {i: u0 for i in range(K)}
						node.player_ranges[1] = {i: u0 for i in range(K)}

						cur = node
						guard = 0

						while (not cur.public_state.is_terminal) and (guard < 200):
							guard += 1
							act = _engine_policy_action(
							 solver,
							 cur,
							 iters=iters_after,
							)
							new_ps = cur.public_state.update_state(
							 cur,
							 Action(act),
							)

							if is_nonnegative_pot_delta(cur.public_state, new_ps):
								cur = GameNode(new_ps)
								cur.player_ranges[0] = dict(node.player_ranges[0])
								cur.player_ranges[1] = dict(node.player_ranges[1])
							else:
								break

						if cur.public_state.is_terminal:
							ev = solver._calculate_terminal_utility(
							 cur,
							 player=int(lbr_player),
							)
							ev_by_action[a] = float(ev)
						else:
							ev_by_action[a] = 0.0

    # choose best by EV then priority
				best = None
				best_pair = None
				for k, v in ev_by_action.items():
					score = (v, _prio(k))
					if best_pair is None:
						best_pair = score
						best = k
					else:
						if score > best_pair:
							best_pair = score
							best = k
						else:
							pass

				if best is None:
					best = ActionType.CALL
				else:
					pass

				key = "flop" if int(ps.current_round) == 1 else "other"

				if key not in freq_log:
					freq_log[key] = {
					 "FOLD": 0,
					 "CALL": 0,
					 "POT": 0,
					 "ALL_IN": 0,
					}
				else:
					pass

				if best == ActionType.FOLD:
					name = "FOLD"
				elif best == ActionType.CALL:
					name = "CALL"
				elif best == ActionType.POT_SIZED_BET:
					name = "POT"
				else:
					name = "ALL_IN"

				freq_log[key][name] = freq_log[key].get(name, 0) + 1

				return best


def _init_agent_solver(
 cfg: Optional[ResolveConfig],
) -> Tuple[CFRSolver, float]:
	if cfg is None:
		cfg = ResolveConfig.from_env({})
	else:
		pass

	agent_solver = CFRSolver(config=cfg)
	agent_solver.load_models()

	K = agent_solver.num_clusters
	if K > 0:
		u = 1.0 / float(K)
	else:
		u = 0.0

	return agent_solver, u


def _init_episode_node(i: int, K: int, u: float) -> GameNode:
	ps = PublicState(
	 initial_stacks=[200, 200],
	 board_cards=None,
	 dealer=(i % 2),
	)
	ps.current_round = 0
	ps.current_player = ps.dealer

	node = GameNode(ps)
	node.player_ranges[0] = {j: u for j in range(K)}
	node.player_ranges[1] = {j: u for j in range(K)}

	return node


def _step_episode_once(
 node: GameNode,
 agent_solver: CFRSolver,
 policy_iters_agent: int,
 policy_iters_after_lbr: int,
 freq: Dict[str, Dict[str, int]],
) -> Tuple[GameNode, float, bool, bool, str]:
	cur_ps = node.public_state

	ok_mass = is_range_mass_conserved(
	 node.player_ranges[0],
	 node.player_ranges[1],
	 tol=EPS_MASS,
	)

	if ok_mass:
		pass
	else:
		print("LBRInvariantRangeMass")
		return node, 0.0, True, False, "err_mass"

	if int(cur_ps.current_round) == 1:
		if int(cur_ps.current_player) == 1:
			act = lbr_greedy_action(
			 cur_ps,
			 agent_solver,
			 lbr_player=1,
			 iters_after=policy_iters_after_lbr,
			 freq_log=freq,
			)
		else:
			act = _engine_policy_action(
			 agent_solver,
			 node,
			 iters=policy_iters_agent,
			)
			dg = agent_solver.get_last_diagnostics()
			if isinstance(dg, dict):
				if "zero_sum_residual" in dg:
					z = float(dg.get("zero_sum_residual", 0.0))
				else:
					z = 0.0
			else:
				z = 0.0
	else:
		act = _engine_policy_action(
		 agent_solver,
		 node,
		 iters=policy_iters_agent,
		)
		dg = agent_solver.get_last_diagnostics()
		if isinstance(dg, dict):
			if "zero_sum_residual" in dg:
				z = float(dg.get("zero_sum_residual", 0.0))
			else:
				z = 0.0
		else:
			z = 0.0

	new_ps = cur_ps.update_state(node, Action(act))

	cr_diff = int(getattr(new_ps, "current_round", 0)) - int(
	 getattr(cur_ps, "current_round", 0)
	)
	cr_back = int(getattr(new_ps, "current_round", 0)) < int(
	 getattr(cur_ps, "current_round", 0)
	)

	if getattr(new_ps, "is_terminal", False):
		illegal = False
	else:
		if cr_diff > 1:
			illegal = True
		else:
			if cr_back:
				illegal = True
			else:
				illegal = False

	if illegal:
		print("LBRIllegalStreetJump")
		return node, z, True, False, "err_jump"
	else:
		nnode = GameNode(new_ps)

		K = agent_solver.num_clusters
		if K > 0:
			u = 1.0 / float(K)
		else:
			u = 0.0

		nnode.player_ranges[0] = {j: u for j in range(K)}
		nnode.player_ranges[1] = {j: u for j in range(K)}

		return nnode, z, False, getattr(new_ps, "is_terminal", False), ""


def _summarize_results(
 results: List[float],
 residual_max: float,
 episodes: int,
 freq: Dict[str, Dict[str, int]],
) -> Dict[str, object]:
	if len(results) > 0:
		mean = sum(results) / float(len(results))
	else:
		mean = 0.0

	bb = 2.0
	mbbg = (mean / bb) * 1000.0

	var = 0.0
	for r in results:
		d = r - mean
		var += d * d

	if len(results) > 1:
		var = var / float(len(results) - 1)
	else:
		var = var / float(1)

	se = math.sqrt(var / float(len(results) if len(results) > 0 else 1))
	l95 = mbbg - 1.96 * ((se / bb) * 1000.0)
	u95 = mbbg + 1.96 * ((se / bb) * 1000.0)

	out = {
	 "episodes": int(episodes),
	 "mbb_per_game": float(mbbg),
	 "ci95": [float(l95), float(u95)],
	 "accept": bool(u95 <= -300.0),
	 "freq": {
	  "flop": {k: int(v) for k, v in freq.get("flop", {}).items()}
	 },
	 "sanity": {
	  "zero_sum_residual_max": float(residual_max),
	  "zero_sum_residual_max_ok": bool(residual_max <= EPS_ZS),
	 },
	}

	return out


def run_lbr_eval(
 episodes: int = 10000,
 seed: int = SEED_DEFAULT,
 cfg: Optional[ResolveConfig] = None,
 policy_iters_agent: int = 2,
 policy_iters_after_lbr: int = 2,
) -> Dict[str, object]:
	random.seed(seed)

	agent_solver, u = _init_agent_solver(cfg)

	K = agent_solver.num_clusters
	freq = {"flop": {"FOLD": 0, "CALL": 0, "POT": 0, "ALL_IN": 0}}
	results: List[float] = []
	residual_max = 0.0

	i = 1
	while i <= int(episodes):
		node = _init_episode_node(i, K, u)

		guard = 0
		done = False
		error = False

		while (not node.public_state.is_terminal) and (guard < 400):
			guard += 1

			node, zres, error, done, err = _step_episode_once(
			 node,
			 agent_solver,
			 policy_iters_agent,
			 policy_iters_after_lbr,
			 freq,
			)

			if zres > residual_max:
				residual_max = zres
			else:
				pass

			if error:
				print(f"Episode {i} aborted: {err}")
				done = True
			else:
				pass

			if done:
				break
			else:
				pass

		if hasattr(node.public_state, "terminal_utility"):
			res = node.public_state.terminal_utility()
		else:
			res = [0.0, 0.0]

		if isinstance(res, (list, tuple)):
			if len(res) >= 2:
				results.append(float(res[0]))
			else:
				results.append(0.0)
		else:
			results.append(0.0)

		i += 1

	return _summarize_results(results, residual_max, episodes, freq)


def run_lbr_acceptance(
 seeds: List[int],
 episodes: int = 10000,
 cfg: Optional[ResolveConfig] = None,
 policy_iters_agent: int = 2,
 policy_iters_after_lbr: int = 2,
) -> Dict[str, object]:
	union_low: Optional[float] = None
	union_high: Optional[float] = None
	per_seed: List[Dict[str, object]] = []

	for sd in list(seeds):
		res = run_lbr_eval(
		 episodes=episodes,
		 seed=int(sd),
		 cfg=cfg,
		 policy_iters_agent=policy_iters_agent,
		 policy_iters_after_lbr=policy_iters_after_lbr,
		)

		ci = res.get("ci95", [-1e9, 1e9])
		l = float(ci[0])
		u = float(ci[1])

		if union_low is None:
			union_low = l
		else:
			if l < union_low:
				union_low = l
			else:
				pass

		if union_high is None:
			union_high = u
		else:
			if u > union_high:
				union_high = u
			else:
				pass

		per_seed.append(
		 {
		  "seed": int(sd),
		  "mbb_per_game": float(res.get("mbb_per_game", 0.0)),
		  "ci95": [l, u],
		  "accept": bool(res.get("accept", False)),
		 }
		)

	if union_high is not None:
		if union_high <= -300.0:
			accept_union = True
		else:
			accept_union = False
	else:
		accept_union = False

	return {
	 "seeds": list(per_seed),
	 "union_ci95": [
	  float(union_low if union_low is not None else 0.0),
	  float(union_high if union_high is not None else 0.0),
	 ],
	 "accept_union": accept_union,
	}

