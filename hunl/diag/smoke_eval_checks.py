"""
I collect quick invariants and probes used in smoke tests and sanity evaluations. I
confirm zero-sum residuals after model enforcement, range mass conservation at nodes,
and non-negative pot deltas under sequences of legal actions. I also provide a
stochastic pot-monotonicity walk and a zero-sum verification for network outputs.

Key functions: zero_sum_residual_ok — check CFVs under ranges; mass_conservation_ok —
sum of ranges ≈ 1; nonnegative_pot_deltas_ok — no illegal pot decreases after bet/call;
pot_monotonicity_ok_sequence — random action walk that never reduces pot beyond
tolerance; verify_outer_zero_sum_residual — Monte-Carlo probe of model zero-sum.

Inputs: CFRSolver and GameNode (for value and range checks), PublicState (for pot
checks), torch Tensors for model probes. Outputs: booleans and small dicts with summary
stats.

Dependencies: torch for network evaluation; engine for actions and updates. Invariants:
all tolerances use EPS_ZS/EPS_MASS so I remain consistent with the rest of the system.
Performance: functions are designed for CI and quick local runs; sample sizes and loops
are capped to avoid GPU stalls.
"""

from hunl.constants import EPS_MASS, EPS_ZS, SEED_DEFAULT
import random
from typing import Dict, List
import torch

from hunl.engine.action_type import ActionType
from hunl.engine.action import Action
from hunl.engine.game_node import GameNode


def zero_sum_residual_ok(
 solver,
 node: GameNode,
 tol: float = EPS_ZS
) -> bool:
	K = int(getattr(solver, "num_clusters", 0))

	r1 = [0.0] * K
	r2 = [0.0] * K

	s1 = 0.0
	s2 = 0.0

	for i, p in node.player_ranges[0].items():
		ii = int(i)
		if (0 <= ii) and (ii < K):
			r1[ii] = float(p)
			s1 += float(p)

	for i, p in node.player_ranges[1].items():
		ii = int(i)
		if (0 <= ii) and (ii < K):
			r2[ii] = float(p)
			s2 += float(p)

	cf1 = solver.predict_counterfactual_values(node, player=0)
	cf2 = solver.predict_counterfactual_values(node, player=1)

	v1 = [0.0] * K
	v2 = [0.0] * K

	i = 0
	while i < K:
		if i in cf1:
			v1[i] = float(cf1[i][0])
		if i in cf2:
			v2[i] = float(cf2[i][0])
		i += 1

	s = 0.0

	i = 0
	while i < K:
		s += (v1[i] * r1[i]) + (v2[i] * r2[i])
		i += 1

	return abs(s) <= float(tol)


def mass_conservation_ok(
 node: GameNode,
 tol: float = EPS_MASS
) -> bool:
	s1 = 0.0
	for p in node.player_ranges[0].values():
		s1 += float(p)

	s2 = 0.0
	for p in node.player_ranges[1].values():
		s2 += float(p)

	ok1 = abs(s1 - 1.0) <= float(tol)
	ok2 = abs(s2 - 1.0) <= float(tol)

	return bool(ok1 and ok2)


def nonnegative_pot_deltas_ok(
 ps
) -> bool:
	before = float(getattr(ps, "pot_size", 0.0))

	a_bet = Action(ActionType.POT_SIZED_BET)
	ps2 = ps.update_state(GameNode(ps), a_bet)

	d1 = float(getattr(ps2, "pot_size", 0.0)) - before

	if d1 < -EPS_MASS:
		return False

	if getattr(ps2, "is_terminal", False):
		return True
	else:
		if not hasattr(ps2, "update_state"):
			return True

	a_call = Action(ActionType.CALL)
	ps3 = ps2.update_state(GameNode(ps2), a_call)

	d2 = float(getattr(ps3, "pot_size", 0.0)) - float(getattr(ps2, "pot_size", 0.0))

	if d2 < -EPS_MASS:
		return False
	else:
		return True


def pot_monotonicity_ok_sequence(
 ps,
 steps: int = 6,
 allowed_actions=None
) -> bool:
	seq_ok = True
	cur = ps

	i = 0

	default_allowed = {
	 ActionType.FOLD,
	 ActionType.CALL,
	 ActionType.POT_SIZED_BET,
	 ActionType.ALL_IN,
	}

	if allowed_actions is not None:
		allowed = set(allowed_actions)
	else:
		allowed = set(default_allowed)

	while (i < int(steps)) and (not bool(getattr(cur, "is_terminal", False))):
		before = float(getattr(cur, "pot_size", 0.0))

		if hasattr(cur, "legal_actions"):
			legal = list(cur.legal_actions())
		else:
			legal = []

		if legal:
			menu = []
			j = 0
			while j < len(legal):
				a = legal[j]
				if a in allowed:
					menu.append(a)
				j += 1

			if not menu:
				if ActionType.CALL in legal:
					menu = [ActionType.CALL]
				else:
					menu = list(legal)
		else:
			menu = [ActionType.CALL]

		a = random.choice(menu)

		cur = cur.update_state(GameNode(cur), Action(a))

		after = float(getattr(cur, "pot_size", 0.0))

		if (after + EPS_MASS) < before:
			seq_ok = False
			break

		i += 1

	return bool(seq_ok)


def _vzrs_rand_simplex(
 n: int
) -> List[float]:
	v = []

	i = 0
	while i < int(n):
		v.append(random.random())
		i += 1

	s = float(sum(v))
	if s <= 0.0:
		s = 1.0

	out = []
	i = 0
	while i < len(v):
		out.append(v[i] / s)
		i += 1

	return out


def _vzrs_rand_board_indices(
 n: int
) -> List[int]:
	idx = list(range(52))
	random.shuffle(idx)

	out = []
	i = 0
	while i < int(n):
		out.append(idx[i])
		i += 1

	return out


def _verify_zero_sum_stage(
 models: dict,
 stage_name: str,
 board_len: int,
 K: int,
 samples: int,
 tol: float
) -> Dict[str, object]:
	if stage_name not in models:
		return {
		 "stage": stage_name,
		 "checked": 0,
		 "max_residual": 0.0,
		 "ok": True,
		}

	net = models[stage_name]

	max_res = 0.0

	i = 0
	while i < int(samples):
		i += 1

		pn = random.random()
		if pn < EPS_ZS:
			pn = EPS_ZS
		if pn > 1.0:
			pn = 1.0

		board = [0] * 52

		j = 0
		idxs = _vzrs_rand_board_indices(int(board_len))
		while j < len(idxs):
			board[idxs[j]] = 1
			j += 1

		r1 = _vzrs_rand_simplex(int(K))
		r2 = _vzrs_rand_simplex(int(K))

		x = [pn] + list(board) + list(r1) + list(r2)

		xt = torch.tensor([x], dtype=torch.float32)
		r1t = torch.tensor([r1], dtype=torch.float32)
		r2t = torch.tensor([r2], dtype=torch.float32)

		with torch.no_grad():
			p1, p2 = net(xt)
			f1, f2 = net.enforce_zero_sum(r1t, r2t, p1, p2)

			s1 = torch.sum(r1t * f1, dim=1, keepdim=True)
			s2 = torch.sum(r2t * f2, dim=1, keepdim=True)

			ss = (s1 + s2).view(-1)[0]
			res = float(abs(float(ss.item())))

			if res > max_res:
				max_res = res

	ok = bool(float(max_res) <= float(tol))

	return {
	 "stage": stage_name,
	 "checked": int(samples),
	 "max_residual": float(max_res),
	 "ok": ok,
	}


def verify_outer_zero_sum_residual(
 models: dict,
 K: int,
 samples: int = 1000,
 tol: float = EPS_ZS,
 seed: int = SEED_DEFAULT
) -> Dict[str, dict]:
	random.seed(int(seed))

	out_flop = _verify_zero_sum_stage(
	 models,
	 "flop",
	 3,
	 int(K),
	 int(samples),
	 float(tol),
	)

	out_turn = _verify_zero_sum_stage(
	 models,
	 "turn",
	 4,
	 int(K),
	 int(samples),
	 float(tol),
	)

	return {
	 "flop": out_flop,
	 "turn": out_turn,
	}

