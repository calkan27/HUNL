from hunl.constants import SEED_DEFAULT
import argparse
import json
import random
import time
from typing import Dict, Any, Optional, Tuple

from hunl.engine.public_state import PublicState
from hunl.engine.game_node import GameNode
from hunl.engine.action_type import ActionType
from hunl.engine.action import Action
from hunl.solving.resolver_integration import resolve_at, resolve_at_with_diag
from hunl.engine.poker_utils import DECK
from hunl.solving.cfr_solver import CFRSolver


def _choose_action(
 policy: Dict[Any, float]
) -> Any:
	r = random.random()
	c = 0.0

	for a, p in policy.items():
		c += float(p)
		if r <= c:
			return a

	keys = list(policy.keys())

	if keys:
		return keys[-1]
	else:
		return ActionType.CALL


def _to_policy_dict(
 dist
):
	if isinstance(dist, dict):
		out = {}
		for k, v in dist.items():
			out[k] = float(v)
		return out
	else:
		return {}


def _uniform_cluster_range(
 K: int
) -> Dict[int, float]:
	if K <= 0:
		return {}

	u = 1.0 / float(K)
	out = {}

	i = 0
	while i < K:
		out[i] = u
		i += 1

	return out


def _heuristic_action(
 ps: PublicState
) -> ActionType:
	p = int(getattr(ps, "current_player", 0))
	o = (p + 1) % 2

	cb = list(getattr(ps, "current_bets", [0, 0]))
	if len(cb) < 2:
		cb = [0, 0]

	to_call = int(cb[o]) - int(cb[p])
	if to_call < 0:
		to_call = 0

	bb = int(getattr(ps, "big_blind", 2))

	stks = list(getattr(ps, "stacks", [bb * 100, bb * 100]))
	if len(stks) < 2:
		stks = [bb * 100, bb * 100]

	if isinstance(stks[p], (int, float)):
		stack_p = int(stks[p])
	else:
		stack_p = bb * 100

	if to_call == 0:
		if (float(getattr(ps, "pot_size", 0.0)) <= 0.0) or (stack_p <= 3 * bb):
			return ActionType.CALL
		else:
			if random.random() < 0.25:
				return ActionType.POT_SIZED_BET
			else:
				return ActionType.CALL
	else:
		if to_call >= stack_p:
			if random.random() < 0.5:
				return ActionType.ALL_IN
			else:
				return ActionType.FOLD
		else:
			if random.random() < 0.8:
				return ActionType.CALL
			else:
				return ActionType.FOLD


def _resolve_once(
 public_state: PublicState,
 r: Dict[int, float],
 w: Dict[int, float],
 depth: int,
 iters: int,
 bet_frac
) -> Tuple[Dict[Any, float], Dict[int, float], Dict[int, float]]:
	conf = {
	 "depth_limit": int(depth),
	 "iterations": int(iters),
	 "bet_fractions": list(bet_frac),
	}

	policy, w_next, our_cfv = resolve_at(
	 public_state,
	 r,
	 w,
	 config=conf,
	 value_server=None,
	)

	out_policy = _to_policy_dict(policy)

	out_w = {}
	for k, v in w_next.items():
		out_w[int(k)] = float(v)

	out_cfv = {}
	for k, v in our_cfv.items():
		out_cfv[int(k)] = float(v)

	return out_policy, out_w, out_cfv


def _log(
 rec: Dict[str, Any],
 log_path: str
):
	line = json.dumps(
	 rec,
	 separators=(",", ":"),
	 sort_keys=True,
	)
	print(line, flush=True)

	if log_path:
		with open(log_path, "a") as f:
			f.write(line + "\n")


def _available_cards_for_diag(
 ps
):
	used = set(list(getattr(ps, "board_cards", [])))

	deck_all = list(DECK)
	cards = []

	i = 0
	while i < len(deck_all):
		c = deck_all[i]
		if c not in used:
			cards.append(c)
		i += 1

	return cards


def _build_diag_solver(
 ps,
 K,
 r_us,
 r_opp,
 depth,
 iters,
 k1,
 k2
):
	if not isinstance(K, int):
		return None

	if K <= 0:
		return None

	solver = CFRSolver(
	 depth_limit=int(depth),
	 num_clusters=int(K),
	)
	solver.total_iterations = int(iters)

	if hasattr(solver, "set_soundness_constants"):
		solver.set_soundness_constants(
		 float(k1),
		 float(k2),
		)

	cards = _available_cards_for_diag(ps)

	clusters = {}
	i = 0
	j = 0

	while i < int(K):
		if j + 1 < len(cards):
			h = f"{cards[j]} {cards[j + 1]}"
			clusters[i] = {h}
			i += 1
			j += 2
		else:
			break

	while i < int(K):
		clusters[i] = set()
		i += 1

	solver.clusters = clusters

	node = GameNode(ps)

	node.player_ranges[int(ps.current_player)] = dict(r_us)
	node.player_ranges[(int(ps.current_player) + 1) % 2] = dict(r_opp)

	_ = solver.run_cfr(node)

	return solver


def _resolve_once_with_diag(
 ps,
 r,
 w,
 depth,
 iters,
 bet_frac,
 diag_solver
):
	conf = {
	 "depth_limit": int(depth),
	 "iterations": int(iters),
	 "bet_fractions": list(bet_frac),
	 "solver_for_diag": diag_solver,
	}

	out = resolve_at_with_diag(
	 ps,
	 r,
	 w,
	 config=conf,
	 value_server=None,
	)

	return out


def _parse_query_ps(
 msg,
 default_stack
):
	psd = dict(msg.get("public_state", {}))

	s0 = int(psd.get("s0", int(default_stack)))
	s1 = int(psd.get("s1", int(default_stack)))

	ps = PublicState(
	 initial_stacks=[s0, s1],
	 small_blind=int(psd.get("sb", 1)),
	 big_blind=int(psd.get("bb", 2)),
	 board_cards=list(psd.get("board", [])),
	 dealer=int(psd.get("dealer", 0)),
	)

	ps.current_round = int(psd.get("round", 0))

	b0 = int(psd.get("b0", 0))
	b1 = int(psd.get("b1", 0))
	ps.current_bets = [b0, b1]

	ps.pot_size = float(psd.get("pot", 0.0))
	ps.current_player = int(psd.get("player", 0))

	return ps


def _parse_json_line_safe(line: str):
	txt = str(line).strip()
	if not txt:
		return None
	if not (txt.startswith("{") and txt.endswith("}")):
		return None

	n = len(txt)
	i = 0
	depth = 0
	in_str = False
	esc = False
	colon_seen = False
	quote_count = 0

	while i < n:
		c = txt[i]
		if in_str:
			if esc:
				esc = False
			else:
				if c == "\\":
					esc = True
				elif c == "\"":
					in_str = False
					quote_count += 1
		else:
			if c == "\"":
				in_str = True
			elif c == "{":
				depth += 1
			elif c == "}":
				depth -= 1
				if depth < 0:
					return None
			elif c == ":":
				colon_seen = True
		i += 1

	if in_str:
		return None
	if depth != 0:
		return None
	if not colon_seen:
		return None
	if quote_count == 0:
		return None

	return json.loads(txt)


def _acpc_loop(args):
	for line in iter(input, ""):
		msg = _parse_json_line_safe(line)
		if msg is None:
			continue

		typ = str(msg.get("type", ""))
		if typ == "query":
			ps = _parse_query_ps(msg, args.stack)

			K_raw = msg.get("K", 6)
			if isinstance(K_raw, (int, float, str)):
				K = int(K_raw)
			else:
				K = 6
			if K < 0:
				K = 0

			r = _uniform_cluster_range(int(K))
			w = {i: 0.0 for i in range(int(K))}

			diag_solver = _build_diag_solver(
			 ps,
			 int(K),
			 r,
			 w,
			 int(args.depth),
			 int(args.iters),
			 float(args.k1),
			 float(args.k2),
			)

			t0 = time.time()
			policy, w_next, _, diag = _resolve_once_with_diag(
			 ps,
			 r,
			 w,
			 int(args.depth),
			 int(args.iters),
			 list(args.bet_frac),
			 diag_solver,
			)
			t1 = time.time()

			if policy:
				action = _choose_action(policy)
			else:
				action = ActionType.CALL

			_log(
			 {
			  "mode": "acpc-client",
			  "t_ms": int((t1 - t0) * 1000),
			  "policy": {
			   str(k): float(v) for k, v in policy.items()
			  },
			  "action": int(action.value),
			  "diag": diag,
			 },
			 args.log,
			)

			resp = {"type": "action", "action": int(action.value)}
			print(json.dumps(resp), flush=True)

		if typ == "close":
			break


def _init_hand_state(
 args,
 hand_index
):
	if (hand_index % 2) == 0:
		dealer_side = int(args.dealer)
	else:
		dealer_side = int(1 - int(args.dealer))

	ps = PublicState(
	 initial_stacks=[int(args.stack), int(args.stack)],
	 small_blind=int(args.sb),
	 big_blind=int(args.bb),
	 board_cards=None,
	 dealer=dealer_side,
	)

	node = GameNode(ps)

	return ps, node


def _resolve_step(
 args,
 ps,
 r,
 w,
 mode
):
	K = int(len(r))

	if mode == "baseline":
		if int(ps.current_player) != int(ps.dealer):
			return {}, dict(w), {"stage": "none"}, _heuristic_action(ps)

	diag_solver = _build_diag_solver(
	 ps,
	 K,
	 r,
	 w,
	 int(args.depth),
	 int(args.iters),
	 float(args.k1),
	 float(args.k2),
	)

	policy, w_next, _, diag = _resolve_once_with_diag(
	 ps,
	 r,
	 w,
	 int(args.depth),
	 int(args.iters),
	 list(args.bet_frac),
	 diag_solver,
	)

	if policy:
		act_type = _choose_action(policy)
	else:
		act_type = ActionType.CALL

	return policy, w_next, diag, act_type


def _log_decision(
 ps,
 t_ms,
 policy,
 act_type,
 diag,
 log_path
):
	rec = {
	 "round": int(getattr(ps, "current_round", 0)),
	 "player": int(getattr(ps, "current_player", 0)),
	 "pot": float(getattr(ps, "pot_size", 0.0)),
	 "bets": [
	  int(getattr(ps, "current_bets", [0, 0])[0]),
	  int(getattr(ps, "current_bets", [0, 0])[1]),
	 ],
	 "action": int(act_type.value),
	 "t_ms": int(t_ms),
	 "policy": {str(k): float(v) for k, v in dict(policy or {}).items()},
	 "diag": diag,
	}
	_log(rec, log_path)


def run_continual_cli(
 argv: Optional[list] = None
):
	parser = argparse.ArgumentParser(
	 prog="hunl-play",
	 add_help=True,
	)

	parser.add_argument(
	 "--mode",
	 choices=["self", "baseline", "acpc-client"],
	 default="baseline",
	)
	parser.add_argument("--hands", type=int, default=1)
	parser.add_argument("--stack", type=int, default=200)
	parser.add_argument("--sb", type=int, default=1)
	parser.add_argument("--bb", type=int, default=2)
	parser.add_argument("--dealer", type=int, default=0)
	parser.add_argument("--depth", type=int, default=1)
	parser.add_argument("--iters", type=int, default=400)
	parser.add_argument(
	 "--bet-frac",
	 type=float,
	 nargs="+",
	 default=[1.0],
	)
	parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
	parser.add_argument("--log", type=str, default="")
	parser.add_argument("--k1", type=float, default=0.0)
	parser.add_argument("--k2", type=float, default=0.0)

	args = parser.parse_args(argv)

	random.seed(int(args.seed))

	if args.mode == "acpc-client":
		_acpc_loop(args)
		return

	h = 0
	while h < int(args.hands):
		ps, node = _init_hand_state(args, h)

		K = 6
		r = _uniform_cluster_range(K)
		w = {i: 0.0 for i in range(K)}

		step_guard = 0

		while (not bool(getattr(ps, "is_terminal", False))) and (step_guard < 200):
			step_guard += 1

			t0 = time.time()

			policy, w_next, diag, act_type = _resolve_step(
			 args,
			 ps,
			 r,
			 w,
			 args.mode,
			)

			new_ps = ps.update_state(node, Action(act_type))

			t1 = time.time()

			_log_decision(
			 ps,
			 (t1 - t0) * 1000.0,
			 policy,
			 act_type,
			 diag,
			 args.log,
			)

			ps = new_ps
			node = GameNode(ps)
			r = r
			w = dict(w_next)

		if hasattr(ps, "terminal_utility"):
			res = ps.terminal_utility()
		else:
			res = [0.0, 0.0]

		_log(
		 {
		  "mode": str(args.mode),
		  "result": list(res),
		  "hand": int(h + 1),
		  "steps": int(step_guard),
		  "dealer": int(getattr(ps, "dealer", 0)),
		  "final_pot": float(getattr(ps, "pot_size", 0.0)),
		  "log": {"hand": int(h + 1), "decisions": []},
		 },
		 args.log,
		)

		h += 1
