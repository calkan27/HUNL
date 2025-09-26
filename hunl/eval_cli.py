import argparse
import random
import statistics
from typing import Dict, List, Tuple, Optional, Any

from poker_utils import DECK
from public_state import PublicState
from game_node import GameNode
from action import Action
from cfr_solver import CFRSolver
from resolve_config import ResolveConfig
from aivat import AIVATEvaluator
from eval_cli_utils import (
	_value_fn_from_solver,
	_policy_from_resolve,
	_sample_from_policy,
	_make_initial_preflop,
)


def _blocks_from_series(
	series: List[float],
	block_size: int
) -> List[float]:
	blocks: List[float] = []

	i = 0
	while i < len(series):
		chunk = series[i : i + int(block_size)]
		i += int(block_size)

		if len(chunk) == int(block_size):
			total = 0.0
			j = 0
			while j < len(chunk):
				total += float(chunk[j])
				j += 1
			blocks.append(total / float(block_size))

	return blocks


def _ci_from_blocks(
	blocks: List[float],
	bb: float
) -> Dict[str, Any]:
	if not blocks:
		return {
			"mbb100": 0.0,
			"ci95": [0.0, 0.0],
		}

	m = 0.0
	i = 0
	while i < len(blocks):
		m += float(blocks[i])
		i += 1
	m = m / float(len(blocks))

	var = 0.0
	k = 0
	while k < len(blocks):
		d = float(blocks[k]) - float(m)
		var += d * d
		k += 1

	if len(blocks) > 1:
		var = var / float(len(blocks) - 1)
	else:
		var = var / 1.0

	se = (var / float(len(blocks))) ** 0.5

	mbb100 = (m / float(bb)) * 100.0
	half = 1.96 * ((se / float(bb)) * 100.0)

	return {
		"mbb100": float(mbb100),
		"ci95": [float(mbb100 - half), float(mbb100 + half)],
	}


def _play_episode(
	solver0: CFRSolver,
	solver1: CFRSolver,
	rng_seed: int,
	value_solver_for_aivat: CFRSolver,
	policy_iters_agent: int = 2,
	policy_iters_opp: int = 1,
) -> Tuple[float, Dict]:
	ps = _make_initial_preflop(
		stack=200,
		seed=int(rng_seed),
	)
	node = GameNode(ps)

	K = int(solver0.num_clusters)
	if K > 0:
		u = 1.0 / float(K)
	else:
		u = 0.0

	node.player_ranges[0] = {i: u for i in range(K)}
	node.player_ranges[1] = {i: u for i in range(K)}

	pol_agent = _policy_from_resolve(
		solver0,
		iters=int(policy_iters_agent),
	)
	pol_opp = _policy_from_resolve(
		solver1,
		iters=int(policy_iters_opp),
	)

	events = []
	step_guard = 0

	while (not node.public_state.is_terminal) and (step_guard < 200):
		step_guard += 1

		prev_ranges = [
			dict(node.player_ranges[0]),
			dict(node.player_ranges[1]),
		]
		prev_board = list(getattr(node.public_state, "board_cards", []))
		prev_round = int(getattr(node.public_state, "current_round", 0))
		cur = int(getattr(node.public_state, "current_player", 0))

		if cur == 0:
			dist = pol_agent(node, player=0)
			chosen = _sample_from_policy(dist)

			events.append(
				{
					"type": "agent",
					"action": chosen,
					"policy": {k: float(v) for k, v in dist.items()},
				}
			)

			new_ps = node.public_state.update_state(
				node,
				Action(chosen),
			)
			node = GameNode(new_ps)
			node.player_ranges = [
				dict(prev_ranges[0]),
				dict(prev_ranges[1]),
			]
		else:
			dist = pol_opp(node, player=1)
			chosen = _sample_from_policy(dist)

			events.append(
				{
					"type": "opponent",
					"action": chosen,
					"policy": {k: float(v) for k, v in dist.items()},
				}
			)

			new_ps = node.public_state.update_state(
				node,
				Action(chosen),
			)
			node = GameNode(new_ps)
			node.player_ranges = [
				dict(prev_ranges[0]),
				dict(prev_ranges[1]),
			]

		new_board = list(getattr(node.public_state, "board_cards", []))

		if int(getattr(node.public_state, "current_round", 0)) > prev_round:
			if len(new_board) > len(prev_board):
				seen = set(prev_board)

				for c in new_board:
					if c not in seen:
						events.append({"type": "chance", "action": c})
						seen.add(c)

	if not node.public_state.is_terminal:
		return 0.0, {"initial_node": None, "events": []}

	naive = solver0._calculate_terminal_utility(node, player=0)

	episode = {
		"initial_node": GameNode(
			_make_initial_preflop(
				stack=200,
				seed=int(rng_seed),
			)
		),
		"events": events,
	}

	value_fn = _value_fn_from_solver(value_solver_for_aivat)

	def policy_fn(nd, player):
		if player == 0:
			return pol_agent(nd, player)
		else:
			return pol_opp(nd, player)

	aiv = AIVATEvaluator(
		value_fn=value_fn,
		policy_fn=policy_fn,
		chance_policy_fn=_chance_policy_uniform,
		agent_player=0,
	)
	res = aiv.evaluate(episode)

	return float(naive), {"aivat": float(res["aivat"])}


def _run_matches(
	mode: str,
	episodes: int,
	seed: int,
	cfg: ResolveConfig,
) -> List[Tuple[float, float]]:
	random.seed(int(seed))

	solverA = CFRSolver(config=cfg)
	solverB = CFRSolver(config=cfg)

	all_hands: List[Tuple[float, float]] = []

	i = 0
	while i < int(episodes):
		i += 1

		rng_seed = random.randint(1, 10**9)

		if mode == "agent-vs-agent":
			naive, ares = _play_episode(solverA, solverA, rng_seed, solverA)
		else:
			naive, ares = _play_episode(solverA, solverB, rng_seed, solverA)

		all_hands.append((float(naive), float(ares["aivat"])))

	return all_hands


def _summarize(
	results: List[Tuple[float, float]]
) -> Tuple[float, float, float, float, float]:
	na = [x[0] for x in results]
	av = [x[1] for x in results]

	if na:
		mean_na = float(statistics.fmean(na))
	else:
		mean_na = 0.0

	if av:
		mean_av = float(statistics.fmean(av))
	else:
		mean_av = 0.0

	if len(na) > 1:
		std_na = float(statistics.pstdev(na))
	else:
		std_na = 0.0

	if len(av) > 1:
		std_av = float(statistics.pstdev(av))
	else:
		std_av = 0.0

	if std_na > 0.0:
		reduction = 1.0 - (std_av / std_na)
	else:
		reduction = 0.0

	return mean_na, mean_av, std_na, std_av, reduction


def _block_metrics(
	results: List[Tuple[float, float]],
	block_size: int = 100,
) -> Dict[str, Any]:
	n = len(results)

	if n == 0:
		return {
			"blocks": 0,
			"naive": {"mbb100": 0.0, "ci95": [0.0, 0.0]},
			"aivat": {"mbb100": 0.0, "ci95": [0.0, 0.0]},
		}

	bb = 2.0

	na = [x[0] for x in results]
	av = [x[1] for x in results]

	na_b = _blocks_from_series(na, int(block_size))
	av_b = _blocks_from_series(av, int(block_size))

	na_stats = _ci_from_blocks(na_b, float(bb))
	av_stats = _ci_from_blocks(av_b, float(bb))

	return {
		"blocks": int(len(na_b)),
		"naive": na_stats,
		"aivat": av_stats,
	}


def _chance_policy_uniform(node: GameNode) -> Dict[str, float]:
	ps = node.public_state

	avail = list(getattr(ps, "deck", []))

	if not avail:
		used = set(
			list(getattr(ps, "board_cards", []))
			+ sum(list(getattr(ps, "hole_cards", [[], []])), [])
		)
		avail = [c for c in DECK if c not in used]

	if not avail:
		return {}

	u = 1.0 / float(len(avail))

	out = {}
	i = 0
	while i < len(avail):
		out[avail[i]] = u
		i += 1

	return out


def _no_negative_pot_delta(prev_ps: PublicState, next_ps: PublicState) -> bool:
	eps = 1e-9

	ref_prev = float(getattr(prev_ps, "last_refund_amount", 0.0))
	ref_next = float(getattr(next_ps, "last_refund_amount", 0.0))
	allow = ref_prev if ref_prev > ref_next else ref_next

	return float(next_ps.pot_size) + eps >= float(prev_ps.pot_size) - allow


def main(argv: Optional[List[str]] = None) -> None:
	parser = argparse.ArgumentParser(
		description="Evaluate agents with naive vs AIVAT-corrected estimates.",
	)
	parser.add_argument(
		"--mode",
		choices=["agent-vs-agent", "agent-vs-policy"],
		default="agent-vs-agent",
	)
	parser.add_argument("--episodes", type=int, default=50)
	parser.add_argument("--seed", type=int, default=1729)
	parser.add_argument("--num-clusters", type=int, default=1000)
	parser.add_argument("--depth-limit", type=int, default=1)
	parser.add_argument("--iterations", type=int, default=8)
	parser.add_argument("--river-buckets", type=int, default=0)

	args = parser.parse_args(argv)

	cfg = ResolveConfig.from_env(
		{
			"num_clusters": int(args.num_clusters),
			"depth_limit": int(args.depth_limit),
			"total_iterations": int(args.iterations),
		}
	)

	if int(args.river_buckets) > 0:
		if hasattr(cfg, "__dict__"):
			setattr(cfg, "river_num_buckets", int(args.river_buckets))
		else:
			print("[INFO] ResolveConfig has no dict; skipping river buckets.")

	results = _run_matches(
		args.mode,
		int(args.episodes),
		int(args.seed),
		cfg,
	)

	mean_na, mean_av, std_na, std_av, reduction = _summarize(results)
	bm = _block_metrics(results, block_size=100)

	print("====================================")
	print(f"Mode: {args.mode}")
	print(f"Episodes: {args.episodes}")
	print("------------------------------------")
	print(f"Naive average reward (agent 0): {mean_na:.6f}")
	print(f"AIVAT-corrected estimate      : {mean_av:.6f}")
	print(f"Std dev (naive)               : {std_na:.6f}")
	print(f"Std dev (AIVAT)               : {std_av:.6f}")
	print(f"Std reduction vs naive        : {reduction * 100.0:.2f}%")
	print("------------------------------------")
	print(f"Blocks (100 hands)            : {bm['blocks']}")
	print(
		f"Naive mbb/100                 : {bm['naive']['mbb100']:.2f}  "
		f"CI95=({bm['naive']['ci95'][0]:.2f},{bm['naive']['ci95'][1]:.2f})"
	)
	print(
		f"AIVAT mbb/100                 : {bm['aivat']['mbb100']:.2f}  "
		f"CI95=({bm['aivat']['ci95'][0]:.2f},{bm['aivat']['ci95'][1]:.2f})"
	)
	print("====================================")

