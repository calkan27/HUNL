"""
I expose a practical API to run one-shot resolves at arbitrary public states and to
collect diagnostics. I adapt bet-size modes, depth limits, and constraint handling, then
call the internal CFR engine with a ValueServer leaf function. I return the root policy,
tightened opponent CFV upper bounds, our CFV vector, and a diagnostic record.

Key functions: resolve_at — return policy, updated upper bounds, and our CFVs;
resolve_at_with_diag — same with a detailed diag dict; _build_lookahead_tree — construct
a sparse public tree; _make_leaf_value_fn/_leaf_value_from_value_server — pot-fraction
CFVs from the value server; _depth_and_bets_from_config — map stage to depth/bet sizes;
_validate_root_policy_and_invariants — sanity checks.

Inputs: PublicState, our range r, opp CFV upper bounds w, config (iterations,
bet_size_mode), optional ValueServer. Outputs: dict policy (ActionType→prob), dict new
bounds by cluster, dict our CFVs, and diagnostics (stage, iterations, zero-sum stats,
cache info).

Dependencies: ValueServer, LookaheadTreeBuilder, ActionType, model I/O helpers.
Invariants: I normalize ranges to simplices, keep pot-fraction scaling, and never
produce illegal actions. Performance: I batch model queries inside ValueServer and keep
small action sets to meet interactive latency.
"""

from hunl.constants import EPS_SUM
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
import torch

from hunl.engine.poker_utils import board_one_hot
from hunl.nets.value_server import ValueServer
from hunl.solving.lookahead_tree import LookaheadTreeBuilder
from hunl.solving.cfr_core import PublicChanceCFR
from hunl.nets.model_io import load_cfv_bundle
from hunl.engine.action_type import ActionType


def _stage_from_round(cr: int) -> str:
	m = {
	 0: "preflop",
	 1: "flop",
	 2: "turn",
	 3: "river",
	}
	s = m.get(int(cr), "flop")

	if s in ("flop", "turn"):
		return s
	else:
		return "flop"


def _ranges_to_simplex_vector(d: Dict[int, float], k: int) -> List[float]:
	v = [0.0] * int(k)
	s = 0.0

	for i, p in dict(d).items():
		ii = int(i)

		if (0 <= ii) and (ii < int(k)):
			pi = float(p)
			v[ii] = pi
			s += pi

	if s > 0.0:
		i = 0
		while i < int(k):
			v[i] = v[i] / s
			i += 1

	return v


def _ensure_value_server(config: Dict[str, Any], value_server: Optional[ValueServer]) -> ValueServer:
	if value_server is not None:
		return value_server

	if ("value_server" in config) and isinstance(
	 config["value_server"],
	 ValueServer,
	):
		return config["value_server"]

	if ("models" in config) and isinstance(config["models"], dict):
		return ValueServer(
		 models=config["models"],
		 device=config.get("device", None),
		)

	if ("bundle_path" in config) and isinstance(
	 config["bundle_path"],
	 str,
	):
		loaded = load_cfv_bundle(
		 config["bundle_path"],
		 device=config.get("device", None),
		)
		return ValueServer(
		 models=loaded.get("models", {}),
		 device=config.get("device", None),
		)

	return ValueServer(models={})


def _bet_fraction_schedule_for_mode(mode: str, stage: str) -> List[float]:
	m = str(mode).strip().lower()
	if m == "sparse_2":
		return [0.5, 1.0]
	else:
		if (m == "sparse_3") or (m == "full"):
			return [0.5, 1.0, 2.0]
		else:
			if str(stage).strip().lower() == "turn":
				return [0.5, 1.0, 2.0]
			else:
				return [0.5, 1.0]



def _depth_and_bets_from_config(stage: str, user_depth: int, config: Dict[str, Any]) -> Tuple[int, List[float], bool, str]:
	if stage == "turn":
		depth_limit = 99
	else:
		depth_limit = int(user_depth)
	if "bet_fractions" in config:
		bet_fractions = list(config["bet_fractions"])
	else:
		mode = str(config.get("bet_size_mode", "")).strip().lower()
		if mode:
			bet_fractions = _bet_fraction_schedule_for_mode(mode, stage)
		else:
			if stage == "turn":
				bet_fractions = [0.5, 1.0, 2.0]
			else:
				bet_fractions = [0.5, 1.0]
	include_all_in = True
	constraint_mode = str(config.get("constraint_mode", "sp")).strip().lower()
	return (
	 depth_limit,
	 bet_fractions,
	 include_all_in,
	 constraint_mode,
	)


def _build_lookahead_tree(public_state, depth_limit: int, bet_fractions: List[float], include_all_in: bool):
	builder = LookaheadTreeBuilder(
	 depth_limit=depth_limit,
	 bet_fractions=bet_fractions,
	 include_all_in=include_all_in,
	)
	return builder.build(public_state)


def _leaf_value_from_value_server(vs: ValueServer, ps, pov_player: int, r1: List[float], r2: List[float]) -> torch.Tensor:
	if hasattr(ps, "initial_stacks"):
		total_initial = float(sum(getattr(ps, "initial_stacks", [200, 200])))
	else:
		total_initial = 1.0
	if total_initial <= 0.0:
		total_initial = 1.0
	if bool(getattr(ps, "is_terminal", False)):
		if hasattr(ps, "terminal_utility"):
			u = ps.terminal_utility()
		else:
			u = [0.0, 0.0]
		if isinstance(u, (list, tuple)):
			if len(u) >= 2:
				val = float(u[int(pov_player)])
			else:
				val = 0.0
		else:
			val = 0.0
		p = float(getattr(ps, "pot_size", 0.0))
		if p > 0.0:
			out = val / p
		else:
			out = 0.0
		return torch.tensor([out], dtype=torch.float32)
	else:
		cr_loc = int(getattr(ps, "current_round", 0))
		if cr_loc >= 2:
			return torch.tensor([0.0], dtype=torch.float32)
		else:
			pot_norm = float(getattr(ps, "pot_size", 0.0)) / total_initial
			bvec = board_one_hot(list(getattr(ps, "board_cards", [])))
			x = [pot_norm] + list(bvec) + list(r1) + list(r2)
			xt = torch.tensor([x], dtype=torch.float32)
			if cr_loc == 0:
				st = "flop"
			else:
				if cr_loc == 1:
					st = "flop"
				else:
					st = "turn"
			v1, v2 = vs.query(
			 st,
			 xt,
			 scale_to_pot=False,
			 as_numpy=False,
			)
			if int(pov_player) == 0:
				out = v1
			else:
				out = v2
			return out[0]


def _make_leaf_value_fn(vs: ValueServer):
	return lambda ps, pov_player, r1, r2: _leaf_value_from_value_server(vs, ps, pov_player, r1, r2)


def _solve_subgame(root, r_us_vec, r_opp_vec, w_vec, depth_limit, bet_fractions, include_all_in, T, leaf_value_fn, config):
	cfr = PublicChanceCFR(
	 depth_limit=depth_limit,
	 bet_fractions=list(bet_fractions),
	 include_all_in=bool(include_all_in),
	 regret_matching_plus=bool(config.get("rm_plus", True)),
	 importance_weighting=True,
	)

	warm = config.get("warm_start", None)

	if isinstance(warm, dict):
		cfr.set_warm_start(warm)

	return cfr.solve_subgame(
	 root_node=root,
	 r_us=r_us_vec,
	 r_opp=r_opp_vec,
	 opp_cfv_constraints=w_vec,
	 T=int(T),
	 leaf_value_fn=leaf_value_fn,
	)


def _build_resolve_context(public_state, r_us: Dict[int, float], w_opp: Dict[int, float], config: Optional[Dict[str, Any]], value_server: Optional[ValueServer]):
	if config is None:
		config = {}

	stage = _stage_from_round(
	 int(getattr(public_state, "current_round", 0))
	)

	K = int(len(r_us))

	r_our_vec = _ranges_to_simplex_vector(r_us, K)

	r_opp_init = config.get("r_opp_init", None)
	if isinstance(r_opp_init, dict):
		r_opp_vec = _ranges_to_simplex_vector(r_opp_init, K)
	else:
		if K > 0:
			r_opp_vec = [1.0 / float(K)] * K
		else:
			r_opp_vec = []

	w_vec = _ranges_to_simplex_vector(w_opp, K)

	vs = _ensure_value_server(config, value_server)

	if hasattr(vs, "get_counters"):
		c_before = dict(vs.get_counters())
	else:
		c_before = {}

	user_depth = int(config.get("depth_limit", 1))

	depth_limit, bet_fractions, include_all_in, constraint_mode = (
	 _depth_and_bets_from_config(stage, user_depth, config)
	)

	root = _build_lookahead_tree(
	 public_state,
	 depth_limit,
	 bet_fractions,
	 include_all_in,
	)

	return (
	 stage,
	 K,
	 r_our_vec,
	 r_opp_vec,
	 w_vec,
	 vs,
	 c_before,
	 depth_limit,
	 bet_fractions,
	 include_all_in,
	 constraint_mode,
	 root,
	)


def _solve_lookahead_with_constraints(root, r_our_vec, r_opp_vec, w_vec, constraint_mode, depth_limit, bet_fractions, include_all_in, T, leaf_value_fn, config):
	K = len(r_our_vec)

	if int(T) <= 0:
		T = 1

	mode = str(constraint_mode).strip().lower()

	if mode == "br":
		w_constraints = [-1e30] * K
	else:
		w_constraints = list(w_vec)

	root_policy, node_values, opp_cfv = _solve_subgame(
	 root,
	 r_our_vec,
	 r_opp_vec,
	 w_constraints,
	 int(depth_limit),
	 list(bet_fractions),
	 bool(include_all_in),
	 int(T),
	 leaf_value_fn,
	 config,
	)

	return root_policy, node_values, opp_cfv


def _allowed_actions_from_bet_fractions(bet_fractions: List[float], include_all_in: bool) -> set:
	allowed = set()

	i = 0
	while i < len(bet_fractions):
		f = bet_fractions[i]
		if abs(float(f) - 0.5) < EPS_SUM:
			allowed.add(ActionType.HALF_POT_BET)
		i += 1

	i = 0
	while i < len(bet_fractions):
		f = bet_fractions[i]
		if abs(float(f) - 1.0) < EPS_SUM:
			allowed.add(ActionType.POT_SIZED_BET)
		i += 1

	i = 0
	while i < len(bet_fractions):
		f = bet_fractions[i]
		if abs(float(f) - 2.0) < EPS_SUM:
			allowed.add(ActionType.TWO_POT_BET)
		i += 1

	allowed.add(ActionType.FOLD)
	allowed.add(ActionType.CALL)

	if bool(include_all_in):
		allowed.add(ActionType.ALL_IN)

	return allowed


def _validate_root_policy_and_invariants(root_policy: Dict[Any, float], r_our_vec: List[float], r_opp_vec: List[float], 
  stage: str, bet_fractions: List[float], include_all_in: bool, flop_queries: int, turn_queries: int) -> Dict[str, Any]:
	allowed_actions = _allowed_actions_from_bet_fractions(
	 bet_fractions,
	 include_all_in,
	)

	menu_ok = True
	for a in list(root_policy.keys()):
		if a in allowed_actions:
			pass
		else:
			menu_ok = False
			break

	rm1 = float(sum(r_our_vec))
	rm2 = float(sum(r_opp_vec))

	if abs(rm1 - 1.0) <= EPS_SUM:
		if abs(rm2 - 1.0) <= EPS_SUM:
			range_mass_ok = True
		else:
			range_mass_ok = False
	else:
		range_mass_ok = False

	if stage != "turn":
		turn_leaf_net_ok = True
	else:
		if int(turn_queries) == 0:
			turn_leaf_net_ok = True
		else:
			turn_leaf_net_ok = False

	return {
	 "policy_actions_ok": bool(menu_ok),
	 "range_mass_ok": bool(range_mass_ok),
	 "turn_leaf_net_ok": bool(turn_leaf_net_ok),
	 "flop_net_queries": int(flop_queries),
	 "turn_net_queries": int(turn_queries),
	}


def _scalarize_cfv_by_cluster(node_values: Dict[int, float]) -> Dict[int, float]:
	out: Dict[int, float] = {}

	if isinstance(node_values, dict):
		for i, v in node_values.items():
			ii = int(i)
			out[ii] = float(v)

	return out


def _merge_value_server_zero_sum_stats(vs: ValueServer, diag: Dict[str, Any]) -> Dict[str, Any]:
	out = dict(diag)

	if hasattr(vs, "get_zero_sum_stats"):
		zs = vs.get_zero_sum_stats()

		if isinstance(zs, dict):
			if "overall" in zs:
				overall = zs["overall"]

				out.setdefault(
				 "zero_sum_residual",
				 float(overall.get("max", 0.0)),
				)
				out.setdefault(
				 "zero_sum_residual_mean",
				 float(overall.get("mean", 0.0)),
				)

			stage_stats: Dict[str, Any] = {}

			for k, v in zs.items():
				stage_stats[k] = {
				 "max": float(v.get("max", 0.0)),
				 "mean": float(v.get("mean", 0.0)),
				 "count": float(v.get("count", 0.0)),
				}

			out["zero_sum_by_stage"] = stage_stats

	return out


def resolve_at_with_diag(public_state, r_us: Dict[int, float], w_opp: Dict[int, float], 
  config: Optional[Dict[str, Any]] = None, value_server: Optional[ValueServer] = None) -> Tuple[Dict[Any, float],
 Dict[int, float], Dict[int, float], Dict[str, Any]]:
	stage, K, r_our_vec, r_opp_vec, w_vec, vs, c_before, depth_limit, bet_fractions, include_all_in, constraint_mode, root = _build_resolve_context(  
	 public_state,
	 r_us,
	 w_opp,
	 config,
	 value_server,
	)

	leaf_value_fn = _make_leaf_value_fn(vs)

	if config is None:
		T = 1000
	else:
		T = int(config.get("iterations", 1000))

	root_policy, node_values, opp_cfv = _solve_lookahead_with_constraints(
	 root,
	 r_our_vec,
	 r_opp_vec,
	 w_vec,
	 constraint_mode,
	 depth_limit,
	 bet_fractions,
	 include_all_in,
	 T,
	 leaf_value_fn,
	 config or {},
	)

	if hasattr(vs, "get_counters"):
		c_after = dict(vs.get_counters())
	else:
		c_after = {}

	flop_queries = int(c_after.get("flop", 0) - int(c_before.get("flop", 0)))
	turn_queries = int(c_after.get("turn", 0) - int(c_before.get("turn", 0)))

	base_diag = {
	 "iterations": int(T),
	 "depth_limit": int(depth_limit),
	 "stage": str(stage),
	 "bet_size_mode": str((config or {}).get("bet_size_mode", "")),
	 "constraint_mode": str(constraint_mode),
	 "bet_fractions": [float(x) for x in bet_fractions],
	 "include_all_in": bool(include_all_in),
	 "K": int(K),
	}

	menu_diag = _validate_root_policy_and_invariants(
	 root_policy,
	 r_our_vec,
	 r_opp_vec,
	 stage,
	 bet_fractions,
	 include_all_in,
	 flop_queries,
	 turn_queries,
	)

	diag = dict(base_diag)
	diag.update(menu_diag)

	sf = (config or {}).get("solver_for_diag", None)

	if sf is not None:
		if hasattr(sf, "get_last_diagnostics"):
			ds = sf.get_last_diagnostics()

			if isinstance(ds, dict):
				for k in (
				 "regret_l2",
				 "avg_strategy_entropy",
				 "zero_sum_residual",
				 "zero_sum_residual_mean",
				):
					if k in ds:
						diag[k] = ds[k]

				if "cfv_calls" in ds:
					if isinstance(ds["cfv_calls"], dict):
						diag["cfv_calls"] = dict(ds["cfv_calls"])

				if "preflop_cache" in ds:
					if isinstance(ds["preflop_cache"], dict):
						diag["preflop_cache"] = dict(ds["preflop_cache"])

	diag = _merge_value_server_zero_sum_stats(vs, diag)

	our_cfv_vec = _scalarize_cfv_by_cluster(node_values)

	return (
	 {a: float(p) for a, p in root_policy.items()},
	 {int(k): float(v) for k, v in (opp_cfv or {}).items()},
	 our_cfv_vec,
	 diag,
	)


def _tighten_cfv_upper_bounds(prev_upper: Dict[int, float], proposed_upper: Dict[int, float]) -> Dict[int, float]:
	out: Dict[int, float] = {}

	keys = set(list(prev_upper.keys()) + list(proposed_upper.keys()))

	for k in keys:
		a = float(prev_upper.get(int(k), float("inf")))
		b = float(proposed_upper.get(int(k), float("inf")))

		if a < b:
			out[int(k)] = a
		else:
			out[int(k)] = b

	return out


def resolve_at(public_state, r_us: Dict[int, float], w_opp: Dict[int, float], config: Optional[Dict[str, Any]] = None, 
  value_server: Optional[ValueServer] = None) -> Tuple[Dict[Any, float], Dict[int, float], Dict[int, float]]:
	pol, w_next_raw, our_cfv, diag = resolve_at_with_diag(public_state, r_us, w_opp, config=config, value_server=value_server)
	cm = str(diag.get("constraint_mode", "sp")).strip().lower()
	if cm == "sp":
		w_next = _tighten_cfv_upper_bounds(dict(w_opp or {}), dict(w_next_raw or {}))
	else:
		w_next = dict(w_opp or {})
	if not diag.get("range_mass_ok", True):
		print("AcceptanceCheckFailed: range_mass")
	if not diag.get("policy_actions_ok", True):
		print("AcceptanceCheckFailed: action_menu")
	if not diag.get("turn_leaf_net_ok", True):
		print("AcceptanceCheckFailed: turn_leaf_invoked_net")
	return pol, w_next, our_cfv
