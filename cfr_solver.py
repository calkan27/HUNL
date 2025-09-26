import os
import copy
import random
import itertools
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn as nn

from action_type import ActionType
from action import Action
from game_node import GameNode
from poker_utils import DECK, best_hand, hand_rank, board_one_hot
from hand_clusterer import HandClusterer
from cfv_network import CounterfactualValueNetwork
from cfr_values import CFRValues
from river_endgame import RiverEndgame

from cfr_solver_models import CFRSolverModelsMixin
from cfr_solver_strategies import CFRSolverStrategiesMixin
from cfr_solver_caching import CFRSolverCachingMixin
from cfr_solver_diagnostics import CFRSolverDiagnosticsMixin
from cfr_solver_utils import CFRSolverUtilsMixin


class CFRSolver(
	CFRSolverModelsMixin,
	CFRSolverStrategiesMixin,
	CFRSolverCachingMixin,
	CFRSolverDiagnosticsMixin,
	CFRSolverUtilsMixin,
):

	def __init__(
		self,
		depth_limit=4,
		num_clusters=1000,
		all_possible_hands=None,
		speed_profile: str = "bot",
		hand_clusterer=None,
		config=None
	):
		self._config = config
		if self._config is not None:
			self.depth_limit = int(self._config.depth_limit)
			self.num_clusters = int(self._config.num_clusters)
			self.speed_profile = self._config.profile
			self.total_iterations = int(self._config.total_iterations)
			self.constraint_mode = str(getattr(self._config, "constraint_mode", "sp"))
		else:
			self.depth_limit = depth_limit
			self.num_clusters = num_clusters
			self.speed_profile = speed_profile
			self.total_iterations = 20
			self.constraint_mode = "sp"

		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

		if (self._config is not None) and (self._config.profile == "test"):
			_fast_seed = int(getattr(self._config, "fast_test_seed", 1729))
			if hasattr(torch, "manual_seed"):
				torch.manual_seed(_fast_seed)
			else:
				print("[INFO] torch.manual_seed not available.")
			if hasattr(torch, "cuda"):
				if hasattr(torch.cuda, "is_available"):
					if torch.cuda.is_available():
						if hasattr(torch.cuda, "manual_seed_all"):
							torch.cuda.manual_seed_all(_fast_seed)
						else:
							print("[INFO] torch.cuda.manual_seed_all not available.")
					else:
						print("[INFO] CUDA not available; skipping CUDA seed.")
				else:
					print("[INFO] torch.cuda.is_available not present.")
			else:
				print("[INFO] torch.cuda not present.")

		self.cfr_values = defaultdict(CFRValues)
		self.all_possible_hands = (all_possible_hands or self.generate_all_possible_hands())
		self.opponent_counterfactual_values = {}
		self.iteration = 0

		if hand_clusterer is not None:
			self.hand_clusterer = hand_clusterer
		else:
			if (not hasattr(self, "hand_clusterer")) or (self.hand_clusterer is None):
				if self._config is not None:
					self.hand_clusterer = HandClusterer(
						self,
						num_clusters=self.num_clusters,
						profile=self.speed_profile,
						opp_sample_size=self._config.opp_sample_size,
						use_cfv_in_features=self._config.use_cfv_in_features,
						config=self._config,
					)
				else:
					self.hand_clusterer = HandClusterer(self, num_clusters=self.num_clusters, profile=self.speed_profile)

		self.clusters = {}
		self._evaluate_hand_strength_cache = {}
		self._recursive_range_sampling_cache = {}

		self.models = {
			"preflop": CounterfactualValueNetwork(self.calculate_input_size_preflop(), num_clusters=self.num_clusters).to(self.device),
			"flop":    CounterfactualValueNetwork(self.calculate_input_size(),          num_clusters=self.num_clusters).to(self.device),
			"turn":    CounterfactualValueNetwork(self.calculate_input_size(),          num_clusters=self.num_clusters).to(self.device),
		}

		if (self._config is not None) and (self._config.profile == "test"):
			for k in ("preflop", "flop", "turn"):
				if hasattr(self.models[k], "eval"):
					self.models[k].eval()
				if hasattr(self, "_zero_initialize_model"):
					self._zero_initialize_model(self.models[k])

		if (self._config is not None) and (self._config.profile != "test"):
			if hasattr(self.models["preflop"], "eval"):
				self.models["preflop"].eval()
			if hasattr(self.models["flop"], "eval"):
				self.models["flop"].eval()
			if hasattr(self.models["turn"], "eval"):
				self.models["turn"].eval()

		self._preflop_cache = OrderedDict()
		self._preflop_cache_cap = (
			int(getattr(self._config, "preflop_cache_max_entries", 10000))
			if getattr(self, "_config", None) is not None
			else 10000
		)
		self._preflop_cache_stats = {"hits": 0, "misses": 0, "puts": 0, "evictions": 0}

		if self._config is not None:
			self.river_endgame = RiverEndgame(
				num_buckets=getattr(self._config, "river_num_buckets", None),
				max_sample_per_cluster=getattr(self._config, "river_max_sample_per_cluster", None),
				seed=getattr(self._config, "fast_test_seed", 2027),
			)
		else:
			self.river_endgame = RiverEndgame()

	def reset(self):
		self.cfr_values = defaultdict(CFRValues)
		self.iteration = 0
		self.opponent_counterfactual_values = {}
		self._evaluate_hand_strength_cache = {}
		self._recursive_range_sampling_cache = {}

	def run_cfr(self, node):
		self._ensure_sparse_schedule()

		if getattr(self, "hand_clusterer", None) is not None:
			prof = str(getattr(self.hand_clusterer, "profile", ""))
			if prof == "test":
				ok_env = (os.getenv("FAST_TESTS") == "1")
				ok_cfg = bool(getattr(getattr(self, "_config", None), "debug_fast_tests", False))
				ok = ok_env or ok_cfg
				if not ok:
					print("FAST_TESTS not enabled in test profile; skipping run_cfr.")
					return None

		ps = node.public_state
		agent_player = ps.current_player
		if agent_player not in (0, 1):
			return None

		if not hasattr(self, "_soundness"):
			self._soundness = {}

		self._diag_cfv_calls = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}
		self._zs_residual_samples = []

		cache_hit = False
		if ps.current_round == 0:
			key0 = self._preflop_signature(node)
			hit = self._preflop_cache_get(key0)
			if hit is not None:
				cache_hit = True
				self._apply_preflop_cache_hit(node, hit)

		key = self._state_key(node) if hasattr(self, "_state_key") else None

		if hasattr(self, "own_range_tracking"):
			if key in getattr(self, "own_range_tracking", {}):
				node.player_ranges[agent_player] = dict(self.own_range_tracking[key])

		if hasattr(self, "opponent_cfv_upper_tracking"):
			if key in getattr(self, "opponent_cfv_upper_tracking", {}):
				pass

		self._prepare_clusters_for_run_cfr(ps, node, agent_player)
		self._normalize_ranges_for_run_cfr(node, ps)

		self.cfr_values = defaultdict(CFRValues)
		self.iteration = 0

		self.apply_round_iteration_schedule(ps.current_round)
		if int(ps.current_round) == 0:
			self.total_iterations = 0

		stage_name = self.get_stage(node)
		if not (isinstance(self._omit_prefix_iters, dict) and (stage_name in self._omit_prefix_iters)):
			self._omit_prefix_iters = {"preflop": 980, "flop": 500, "turn": 500, "river": 1000}

		_ = self._range_gadget_begin(node)
		self._do_iterations_for_run_cfr(node, agent_player)

		act = self._finalize_and_choose_action_for_run_cfr(node, agent_player, ps)
		return act

	def _calculate_counterfactual_values(self, node, player, depth=0, cache=None):
		if cache is None:
			cache = {}

		node_key = (id(node), player, depth)

		if node_key in cache:
			return cache[node_key]

		term = self._cfv_terminal_case(node, player, depth, cache, node_key)
		if term is not None:
			return term

		net = self._cfv_value_net_case(node, player, depth, cache, node_key)
		if net is not None:
			return net

		current_player = node.current_player

		if current_player == player:
			counterfactual_values = self._cfv_current_player_branch(node, player, depth)
		else:
			counterfactual_values = self._cfv_opponent_branch(node, player, depth)

		cache[node_key] = counterfactual_values
		return counterfactual_values

	def _calculate_counterfactual_utility(self, node, player, depth):
		if self._is_terminal(node):
			return self._calculate_terminal_utility(node, player)

		stage = self.get_stage(node)
		if (depth >= self.depth_limit) and (stage in ("preflop", "flop")):
			preds = self.predict_counterfactual_values(node, player)
			sc = 1.0 if bool(getattr(self, "_label_pot_fraction", False)) else float(node.public_state.pot_size)
			ev = 0.0
			total = 0.0
			for cid, p in node.player_ranges[player].items():
				if cid in preds:
					v = preds[cid][0]
					val = float(v[0]) * sc if isinstance(v, (list, tuple)) else float(v) * sc
					ev += p * val
					total += p
			if total > 0:
				return ev / total
			else:
				return 0.0

		current_player = node.current_player
		expected_value = 0.0

		if current_player == player:
			allowed_actions = self._allowed_actions_agent(node.public_state)
			for cluster_id, cluster_prob in node.player_ranges[player].items():
				if cluster_prob == 0.0:
					continue
				values = self.cfr_values[node]
				base_strategy = values.compute_strategy(cluster_id)
				strategy = self._mask_strategy(base_strategy, allowed_actions)
				action_utilities = []
				for a_type in allowed_actions:
					a_idx = a_type.value
					if a_idx in values.pruned_actions[cluster_id]:
						action_utilities.append(0.0)
						continue
					action = Action(a_type)
					ps2 = node.public_state.update_state(node, action)
					if ps2 is None:
						action_utilities.append(0.0)
						continue
					child_node = GameNode(ps2)
					child_node.player_ranges = copy.deepcopy(node.player_ranges)
					if int(ps2.current_round) > int(node.public_state.current_round):
						self.lift_ranges_after_chance(child_node)
					self.update_player_range(child_node, player, cluster_id, a_idx)
					utility = self._calculate_counterfactual_utility(child_node, player, depth + 1)
					action_utilities.append(utility)
				i = 0
				while i < len(allowed_actions):
					a_idx = allowed_actions[i].value
					expected_value += cluster_prob * strategy[a_idx] * action_utilities[i]
					i += 1
		else:
			allowed_actions = self._allowed_actions_opponent(node.public_state)
			best = None
			for a_type in allowed_actions:
				act = Action(a_type)
				ps2 = node.public_state.update_state(node, act)
				if ps2 is None:
					continue
				ch = GameNode(ps2)
				ch.player_ranges = copy.deepcopy(node.player_ranges)
				if int(ps2.current_round) > int(node.public_state.current_round):
					self.lift_ranges_after_chance(ch)
				val = self._calculate_counterfactual_utility(ch, player, depth + 1)
				if (best is None) or (val < best):
					best = val
			if best is not None:
				expected_value += float(best)
			else:
				expected_value += 0.0

		return expected_value

	def _update_regret(self, node, player, cfv_by_action):
		values = self.cfr_values[node]
		stage = self.get_stage(node)
		omit = 0
		if hasattr(self, "_omit_prefix_iters"):
			if isinstance(self._omit_prefix_iters, dict):
				omit = int(self._omit_prefix_iters.get(stage, 0))
		if player == node.public_state.current_player:
			allowed = self._allowed_actions_agent(node.public_state)
		else:
			allowed = self._allowed_actions_opponent(node.public_state)
		A = len(ActionType)

		for cid, pri in node.player_ranges[player].items():
			strat = values.compute_strategy(cid)
			m = self._mask_strategy(strat, allowed)

			action_vals = [0.0] * A
			i = 0
			while i < len(allowed):
				a_idx = allowed[i].value
				v = cfv_by_action.get(cid, [0.0] * A)[a_idx]
				action_vals[a_idx] = float(v[0] if isinstance(v, (list, tuple)) and (len(v) > 0) else v)
				i += 1

			exp = 0.0
			j = 0
			while j < A:
				exp += float(m[j]) * float(action_vals[j])
				j += 1

			k = 0
			while k < A:
				reg = float(action_vals[k]) - float(exp)
				values.cumulative_regret[cid][k] += reg
				if values.cumulative_regret[cid][k] > 0.0:
					values.cumulative_positive_regret[cid][k] = values.cumulative_regret[cid][k]
				else:
					values.cumulative_positive_regret[cid][k] = 0.0
				values.regret_squared_sums[cid][k] += float(reg * reg)
				k += 1

			if int(self.iteration) > omit:
				values.update_strategy(cid, m)

			values.prune_actions(cid, int(self.iteration), int(self.total_iterations))
			values.reassess_pruned_actions(cid, int(self.iteration))

	def _is_terminal(self, node) -> bool:
		ps = node.public_state
		return bool(getattr(ps, "is_terminal", False))

	def _calculate_terminal_utility(self, node, player: int) -> float:
		ps = node.public_state
		u = ps.terminal_utility() if hasattr(ps, "terminal_utility") else [0.0, 0.0]
		if isinstance(u, (list, tuple)):
			if len(u) >= 2:
				return float(u[int(player)])
		return 0.0

	def set_cfr_hybrid_config(
		self,
		preflop_omit=None,
		flop_omit=None,
		turn_omit=None,
		river_omit=None,
		round_iters=None,
		round_flags=None
	):
		self._ensure_sparse_schedule()

		if (not hasattr(self, "_omit_prefix_iters")) or (not isinstance(self._omit_prefix_iters, dict)):
			self._omit_prefix_iters = {"preflop": 980, "flop": 500, "turn": 500, "river": 1000}

		if preflop_omit is not None:
			self._omit_prefix_iters["preflop"] = int(preflop_omit)
		if flop_omit is not None:
			self._omit_prefix_iters["flop"] = int(flop_omit)
		if turn_omit is not None:
			self._omit_prefix_iters["turn"] = int(turn_omit)
		if river_omit is not None:
			self._omit_prefix_iters["river"] = int(river_omit)

		if isinstance(round_iters, dict):
			for k, v in round_iters.items():
				self._round_iters[int(k)] = int(v)

		if isinstance(round_flags, dict):
			out = {}
			for r, fl in round_flags.items():
				out[int(r)] = {"half_pot": bool(fl.get("half_pot", True)), "two_pot": bool(fl.get("two_pot", False))}
			self._round_actions = out

		if (round_iters is None) and (round_flags is None) and (river_omit is None):
			return {"preflop": int(self._omit_prefix_iters["preflop"]), 
				"flop": int(self._omit_prefix_iters["flop"]), "turn": int(self._omit_prefix_iters["turn"])}

		return {
			"omit_prefix": dict(self._omit_prefix_iters),
			"round_iters": dict(self._round_iters),
			"round_actions": {
				int(k): {"half_pot": bool(v.get("half_pot", True)), "two_pot": bool(v.get("two_pot", False))}
				for k, v in self._round_actions.items()
			},
		}

	def _apply_preflop_cache_hit(self, node, hit):
		own_cached = dict(hit.get("own_range", {}))
		opp_cfv_cached = dict(hit.get("opp_cfv", {}))

		pl = int(node.public_state.current_player)

		if own_cached:
			s = 0.0
			for v in own_cached.values():
				s += float(v)
			if s > 0.0:
				for k in list(own_cached.keys()):
					own_cached[k] = float(own_cached[k]) / s
			node.player_ranges[pl] = {int(k): float(v) for k, v in own_cached.items()}

		if not hasattr(self, "opponent_cfv_upper_tracking"):
			self.opponent_cfv_upper_tracking = {}

		sk = self._state_key(node) if hasattr(self, "_state_key") else None

		if sk is not None:
			prev = dict(self.opponent_cfv_upper_tracking.get(sk, {}))
			out = {}
			keys = set(list(prev.keys()) + list(opp_cfv_cached.keys()))
			for k in keys:
				a = float(prev.get(int(k), float("-inf")))
				b = float(opp_cfv_cached.get(int(k), float("-inf")))
				out[int(k)] = a if (a > b) else b
			self.opponent_cfv_upper_tracking[sk] = out

			if isinstance(sk, tuple):
				if len(sk) >= 5:
					sk_flip = list(sk)
					cp = int(sk_flip[4])
					if cp in (0, 1):
						sk_flip[4] = 1 - cp
						self.opponent_cfv_upper_tracking[tuple(sk_flip)] = dict(out)

		return True

	def flop_label_targets_using_turn_net(self, node):
		old_flag = bool(getattr(self, "_label_pot_fraction", False))
		self._label_pot_fraction = True

		old_depth = int(getattr(self, "depth_limit", 1))
		self._ensure_sparse_schedule()

		round_flags_backup = {
			int(k): {"half_pot": bool(v.get("half_pot", True)), "two_pot": bool(v.get("two_pot", False))}
			for k, v in getattr(self, "_round_actions", {}).items()
		}

		for r in (0, 1, 2, 3):
			self._round_actions[int(r)] = {"half_pot": False, "two_pot": False}

		snap_cards = self._push_no_card_abstraction_for_node(node)

		try_depth = max(1, old_depth)
		self.depth_limit = try_depth

		_ = self.run_cfr(node)

		v0 = self._expected_cfv_vector(node, player=0)
		v1 = self._expected_cfv_vector(node, player=1)

		K = int(self.num_clusters)
		out0 = [0.0] * K
		out1 = [0.0] * K

		for cid in range(K):
			a = v0.get(int(cid), [0.0])
			b = v1.get(int(cid), [0.0])
			out0[cid] = float(a[0]) if isinstance(a, (list, tuple)) and (len(a) > 0) else float(a)
			out1[cid] = float(b[0]) if isinstance(b, (list, tuple)) and (len(b) > 0) else float(b)

		self._pop_no_card_abstraction(snap_cards, node)
		self._round_actions = round_flags_backup
		self._label_pot_fraction = old_flag
		self.depth_limit = old_depth

		return out0, out1

	def _expected_cfv_vector(self, node, player):
		A = len(ActionType)
		ps = node.public_state
		cfv_by_action = self._calculate_counterfactual_values(node, player, depth=0)
		out = {}

		if int(ps.current_player) == int(player):
			allowed = self._allowed_actions_agent(ps)
			values = self.cfr_values.get(node, None)
			for cid in node.player_ranges[player].keys():
				if values is None:
					u = (1.0 / float(len(allowed))) if (len(allowed) > 0) else 0.0
					ev = 0.0
					for a in allowed:
						v = cfv_by_action.get(cid, [0.0] * A)[int(a.value)]
						val = float(v[0]) if isinstance(v, (list, tuple)) and (len(v) > 0) else float(v)
						ev += u * val
					out[int(cid)] = [float(ev)]
				else:
					base = values.get_average_strategy(int(cid))
					msk = self._mask_strategy(base, allowed)
					ev = 0.0
					for a in allowed:
						ai = int(a.value)
						v = cfv_by_action.get(cid, [0.0] * A)[ai]
						val = float(v[0]) if isinstance(v, (list, tuple)) and (len(v) > 0) else float(v)
						ev += float(msk[ai]) * val
					out[int(cid)] = [float(ev)]
		else:
			allowed = self._allowed_actions_opponent(ps)
			for cid in node.player_ranges[player].keys():
				vals = []
				for a in allowed:
					ai = int(a.value)
					v = cfv_by_action.get(cid, [0.0] * A)[ai]
					val = float(v[0]) if isinstance(v, (list, tuple)) and (len(v) > 0) else float(v)
					vals.append(val)
				if len(vals) > 0:
					out[int(cid)] = [float(min(vals))]
				else:
					out[int(cid)] = [0.0]

		return out

	def turn_label_targets_solve_to_terminal(self, node):
		old_depth = int(getattr(self, "depth_limit", 1))
		self._ensure_sparse_schedule()

		round_flags_backup = {
			int(k): {"half_pot": bool(v.get("half_pot", True)), "two_pot": bool(v.get("two_pot", False))}
			for k, v in getattr(self, "_round_actions", {}).items()
		}

		for r in (0, 1, 2, 3):
			self._round_actions[int(r)] = {"half_pot": False, "two_pot": False}

		snap_cards = self._push_no_card_abstraction_for_node(node)

		self.depth_limit = 99
		_ = self.run_cfr(node)

		v0 = self._expected_cfv_vector(node, player=0)
		v1 = self._expected_cfv_vector(node, player=1)

		K = int(self.num_clusters)
		out0 = [0.0] * K
		out1 = [0.0] * K

		for cid in range(K):
			a = v0.get(int(cid), [0.0])
			b = v1.get(int(cid), [0.0])
			va = float(a[0]) if isinstance(a, (list, tuple)) and (len(a) > 0) else float(a)
			vb = float(b[0]) if isinstance(b, (list, tuple)) and (len(b) > 0) else float(b)
			out0[cid] = va
			out1[cid] = vb

		p = float(getattr(node.public_state, "pot_size", 0.0))
		if p > 0.0:
			i = 0
			while i < K:
				out0[i] = out0[i] / p
				out1[i] = out1[i] / p
				i += 1

		self._pop_no_card_abstraction(snap_cards, node)
		self._round_actions = round_flags_backup
		self.depth_limit = old_depth

		return out0, out1


	def _prepare_clusters_for_run_cfr(self, ps, node, agent_player):
		test_mode = (getattr(self.hand_clusterer, "profile", "bot") == "test")
		if test_mode:
			if (not self.clusters) or (len(self.clusters) == 0):
				board = list(ps.board_cards)
				used = set(board)
				hands = []
				for i in range(len(DECK)):
					a = DECK[i]
					if a in used:
						continue
					for j in range(i + 1, len(DECK)):
						b = DECK[j]
						if b in used:
							continue
						if a == b:
							continue
						hands.append(f"{a} {b}")
						if len(hands) >= self.num_clusters:
							break
					if len(hands) >= self.num_clusters:
						break
				self.clusters = {i: ({hands[i]} if i < len(hands) else set()) for i in range(self.num_clusters)}
		else:
			if not self.clusters:
				all_hands = self.generate_all_possible_hands()
				self.clusters = self.hand_clusterer.cluster_hands(
					all_hands,
					ps.board_cards,
					node.player_ranges[(agent_player + 1) % 2],
					ps.pot_size,
				)

	def _normalize_ranges_for_run_cfr(self, node, ps):
		fast_env = (os.getenv("FAST_TESTS") == "1")
		if fast_env:
			for player in [0, 1]:
				total = 0.0
				for v in node.player_ranges[player].values():
					total += float(v)
				if total > 0.0:
					for cid in list(node.player_ranges[player].keys()):
						node.player_ranges[player][cid] = node.player_ranges[player][cid] / total
				else:
					keys = list(node.player_ranges[player].keys())
					k = len(keys)
					if k > 0:
						u = 1.0 / float(k)
						for cid in keys:
							node.player_ranges[player][cid] = u
		else:
			for player in [0, 1]:
				total_prob = 0.0
				for v in node.player_ranges[player].values():
					total_prob += float(v)
				node.player_ranges[player] = self.recursive_range_sampling(
					set(node.player_ranges[player].keys()),
					total_prob,
					ps.board_cards,
				)

	def _do_iterations_for_run_cfr(self, node, agent_player):
		total_iters = int(self.total_iterations)
		for _ in range(total_iters):
			self.iteration += 1

			cfvs = {}
			for pl in [0, 1]:
				cfvs[pl] = self._calculate_counterfactual_values(node, pl)

			for pl in [0, 1]:
				self._update_regret(node, pl, cfvs[pl])

			if node not in self.opponent_counterfactual_values:
				self.opponent_counterfactual_values[node] = {}

			self.opponent_counterfactual_values[node][0] = cfvs[1]
			self.opponent_counterfactual_values[node][1] = cfvs[0]

			opp_player = (agent_player + 1) % 2
			upper = self._upper_from_cfvs(cfvs.get(opp_player, {}))
			self._range_gadget_commit(node, upper)

	def _finalize_and_choose_action_for_run_cfr(self, node, agent_player, ps):
		regret_l2 = self._compute_regret_l2(node)
		avg_ent = self._compute_avg_strategy_entropy(node)
		zero_sum_res = self._compute_zero_sum_residual(node)
		if getattr(self, "_zs_residual_samples", None):
			total_abs = 0.0
			for x in self._zs_residual_samples:
				total_abs += abs(x)
			zero_sum_res_mean = float(total_abs / float(len(self._zs_residual_samples)))
		else:
			zero_sum_res_mean = 0.0
		self._last_diagnostics = {
			"depth_limit": int(self.depth_limit),
			"iterations": int(self.total_iterations),
			"k1": float(self._soundness.get("k1", 0.0)),
			"k2": float(self._soundness.get("k2", 0.0)),
			"regret_l2": float(regret_l2),
			"avg_strategy_entropy": float(avg_ent),
			"cfv_calls": dict(self._diag_cfv_calls),
			"zero_sum_residual": float(zero_sum_res),
			"zero_sum_residual_mean": float(zero_sum_res_mean),
			"constraint_mode": str(getattr(self, "constraint_mode", "sp")),
		}
		if ps.current_round == 0:
			key1 = self._preflop_signature(node)
			own = dict(node.player_ranges[agent_player])
			s = 0.0
			for v in own.values():
				s += float(v)
			if s > 0.0:
				for k2 in list(own.keys()):
					own[k2] = own[k2] / s
			opp = (agent_player + 1) % 2
			opp_cfvs = self.opponent_counterfactual_values.get(node, {}).get(opp, {})
			upper = self._upper_from_cfvs(opp_cfvs)
			self._preflop_cache_put(key1, own, upper)
		allowed_actions = self._allowed_actions_agent(ps)
		action_probs = self._mixed_action_distribution(node, agent_player, allowed_actions)
		r = random.random()
		cum = 0.0
		chosen = allowed_actions[-1]
		for a_type, p in zip(allowed_actions, action_probs):
			cum += p
			if r <= cum:
				chosen = a_type
				break
		act = Action(chosen)
		tmp_parent = GameNode(ps)
		tmp_parent.player_ranges = [dict(node.player_ranges[0]), dict(node.player_ranges[1])]
		tmp_parent.public_state.last_action = act
		if hasattr(self, "update_tracking_on_own_action"):
			self.update_tracking_on_own_action(
				tmp_parent,
				agent_player=agent_player,
				counterfactual_values=self.opponent_counterfactual_values.get(node, {}),
			)
		new_ps = ps.update_state(node, act)
		if new_ps is None:
			fallback_order = [
				ActionType.CALL,
				ActionType.POT_SIZED_BET,
				ActionType.HALF_POT_BET,
				ActionType.TWO_POT_BET,
				ActionType.ALL_IN,
				ActionType.FOLD,
			]
			legal_set = set(allowed_actions)
			chosen_fb = None
			i = 0
			while i < len(fallback_order):
				a = fallback_order[i]
				if a in legal_set:
					candidate = Action(a)
					ps2 = ps.update_state(node, candidate)
					if ps2 is not None:
						act = candidate
						new_ps = ps2
						chosen_fb = a
						break
				i += 1
			if new_ps is None and allowed_actions:
				a = allowed_actions[0]
				ps2 = ps.update_state(node, Action(a))
				if ps2 is not None:
					act = Action(a)
					new_ps = ps2
		if new_ps is None:
			return act
		node.public_state = new_ps
		self.cfr_values = defaultdict(CFRValues)
		self.iteration = 0
		return act

	def _cfv_terminal_case(self, node, player, depth, cache, node_key):
		if self._is_terminal(node):
			cf_values = {}
			utility = self._calculate_terminal_utility(node, player)
			for cluster_id in node.player_ranges[player]:
				cf_values[cluster_id] = [utility] * len(ActionType)
			cache[node_key] = cf_values
			return cf_values
		return None

	def _cfv_value_net_case(self, node, player, depth, cache, node_key):
		stage = self.get_stage(node)
		if (depth >= self.depth_limit) and (stage in ("preflop", "flop")):
			preds = self.predict_counterfactual_values(node, player)
			sc = 1.0 if bool(getattr(self, "_label_pot_fraction", False)) else float(node.public_state.pot_size)
			scaled = {}
			for cid, vec in preds.items():
				if isinstance(vec, (list, tuple)):
					scaled[int(cid)] = [float(x) * sc for x in vec]
				else:
					scaled[int(cid)] = [float(vec) * sc] * len(ActionType)
			cache[node_key] = scaled
			return scaled
		return None

	def _cfv_current_player_branch(self, node, player, depth):
		counterfactual_values = defaultdict(lambda: [0.0] * len(ActionType))
		allowed_actions = self._allowed_actions_agent(node.public_state)

		for cluster_id, cluster_prob in node.player_ranges[player].items():
			if cluster_prob == 0.0:
				continue

			values = self.cfr_values[node]
			base_strategy = values.compute_strategy(cluster_id)
			strategy = self._mask_strategy(base_strategy, allowed_actions)

			for a_type in allowed_actions:
				a_idx = a_type.value
				if a_idx in values.pruned_actions[cluster_id]:
					continue

				action = Action(a_type)
				new_public_state = node.public_state.update_state(node, action)
				if new_public_state is None:
					continue

				child_node = GameNode(new_public_state)
				child_node.player_ranges = copy.deepcopy(node.player_ranges)

				if int(new_public_state.current_round) > int(node.public_state.current_round):
					self.lift_ranges_after_chance(child_node)

				self.update_player_range(child_node, player, cluster_id, a_idx)

				utility = self._calculate_counterfactual_utility(child_node, player, depth + 1)

				counterfactual_values[cluster_id][a_idx] += strategy[a_idx] * utility

		return counterfactual_values

	def _cfv_opponent_branch(self, node, player, depth):
		counterfactual_values = defaultdict(lambda: [0.0] * len(ActionType))
		allowed_actions = self._allowed_actions_opponent(node.public_state)

		per_action_util = {}

		for a_type in allowed_actions:
			act = Action(a_type)
			ps2 = node.public_state.update_state(node, act)
			if ps2 is None:
				continue

			ch = GameNode(ps2)
			ch.player_ranges = copy.deepcopy(node.player_ranges)

			if int(ps2.current_round) > int(node.public_state.current_round):
				self.lift_ranges_after_chance(ch)

			per_action_util[a_type.value] = self._calculate_counterfactual_utility(ch, player, depth + 1)

		for cid in node.player_ranges[player].keys():
			for a_type in allowed_actions:
				a_idx = a_type.value
				if a_idx in per_action_util:
					counterfactual_values[cid][a_idx] = float(per_action_util.get(a_idx, 0.0))

		return counterfactual_values
