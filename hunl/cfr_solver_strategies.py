import os
from collections import defaultdict
from typing import Dict, List

from action_type import ActionType
from action import Action
from game_node import GameNode


class CFRSolverStrategiesMixin:

	def get_cumulative_strategy(self, player):
		cumulative_strategy = {}
		for node, values in self.cfr_values.items():
			for cluster_id, strategy in values.cumulative_strategy.items():
				if cluster_id not in cumulative_strategy:
					cumulative_strategy[cluster_id] = [0.0] * len(strategy)
				for action in range(len(strategy)):
					cumulative_strategy[cluster_id][action] += strategy[action]
		return cumulative_strategy

	def update_player_range(self, node: GameNode, player: int, cluster_id: int, action_index: int):
		values = self.cfr_values[node]
		num_actions = len(ActionType)

		if player not in (0, 1):
			return

		if not hasattr(node, "_priors_baseline"):
			node._priors_baseline = [{}, {}]

		if not node._priors_baseline[player]:
			node._priors_baseline[player] = dict(node.player_ranges[player])

		priors = node.player_ranges[player]
		post = {}
		norm = 0.0

		for cid, prior in priors.items():
			if cid in values.cumulative_positive_regret:
				current_strategy = values.compute_strategy(cid)
			else:
				current_strategy = values.strategy.get(cid, [1.0 / num_actions] * num_actions)

			if player == (node.public_state.current_player + 1) % 2:
				like = 1.0
			else:
				if (action_index < 0) or (action_index >= len(current_strategy)):
					like = 0.0
				else:
					like = current_strategy[action_index]

			weight = prior * like
			post[cid] = weight
			norm += weight

		if norm <= 0.0:
			if isinstance(getattr(node, "_priors_baseline", None), list):
				base = node._priors_baseline[player]
			else:
				base = priors

			total_prior = 0.0
			for _, v in base.items():
				total_prior += float(v)

			if total_prior <= 0.0:
				k = len(base)
				if k > 0:
					u = 1.0 / k
					for cid in base:
						post[cid] = u
				else:
					post = {}
			else:
				for cid in base:
					post[cid] = float(base[cid]) / float(total_prior)
		else:
			for cid in post:
				post[cid] = post[cid] / norm

		node.player_ranges[player] = post

	def _allowed_actions_agent(self, ps):
		self._ensure_sparse_schedule()

		ridx = int(getattr(ps, "current_round", getattr(ps, "round_idx", 0)))
		flags = self._round_actions.get(ridx, {"half_pot": True, "two_pot": False})

		p = int(getattr(ps, "current_player", 0))
		cb = tuple(getattr(ps, "current_bets", (0, 0)))
		my_bet = cb[p] if p < len(cb) else 0
		opp_bet = cb[(p + 1) % 2] if len(cb) > 1 else 0
		to_call = opp_bet - my_bet
		if to_call < 0:
			to_call = 0

		if hasattr(ps, "_min_raise_size") and callable(ps._min_raise_size):
			min_raise_inc = int(ps._min_raise_size())
		else:
			min_raise_inc = int(getattr(ps, "min_raise_size", 0))

		st0 = getattr(ps, "stacks", (0, 0))
		if isinstance(st0, (tuple, list)) and (len(st0) >= 2):
			my_stack = int(st0[p])
		else:
			my_stack = int(getattr(ps, "stack", 0))

		legal_list = None
		if hasattr(ps, "legal_actions") and callable(ps.legal_actions):
			l = ps.legal_actions()
			if isinstance(l, list):
				legal_list = l

		if legal_list is not None:
			out = []
			seen = set()

			for a in legal_list:
				if a == ActionType.HALF_POT_BET and not flags.get("half_pot", False):
					continue
				if a == ActionType.TWO_POT_BET and not flags.get("two_pot", False):
					continue
				if a not in seen:
					seen.add(a)
					out.append(a)

			if ActionType.CALL not in out:
				out.append(ActionType.CALL)

			if (to_call > 0) and (ActionType.FOLD not in out):
				out.append(ActionType.FOLD)

			if my_stack > 0:
				if to_call == 0:
					if flags.get("half_pot", False) and (ActionType.HALF_POT_BET not in out):
						out.append(ActionType.HALF_POT_BET)
					if ActionType.POT_SIZED_BET not in out:
						out.append(ActionType.POT_SIZED_BET)
					if flags.get("two_pot", False) and (ActionType.TWO_POT_BET not in out):
						out.append(ActionType.TWO_POT_BET)
					if ActionType.ALL_IN not in out:
						out.append(ActionType.ALL_IN)
				else:
					if my_stack <= to_call:
						if ActionType.ALL_IN not in out:
							out.append(ActionType.ALL_IN)
					else:
						if flags.get("half_pot", False) and (ActionType.HALF_POT_BET not in out):
							out.append(ActionType.HALF_POT_BET)
						if ActionType.POT_SIZED_BET not in out:
							out.append(ActionType.POT_SIZED_BET)
						if flags.get("two_pot", False) and (ActionType.TWO_POT_BET not in out):
							out.append(ActionType.TWO_POT_BET)
						if ActionType.ALL_IN not in out:
							out.append(ActionType.ALL_IN)

			return out

		out = []

		if to_call > 0:
			out.append(ActionType.FOLD)

		out.append(ActionType.CALL)

		if my_stack > 0:
			if to_call == 0:
				if flags.get("half_pot", False):
					out.append(ActionType.HALF_POT_BET)
				out.append(ActionType.POT_SIZED_BET)
				if flags.get("two_pot", False):
					out.append(ActionType.TWO_POT_BET)
				out.append(ActionType.ALL_IN)
			else:
				if my_stack <= to_call:
					out.append(ActionType.ALL_IN)
				else:
					if flags.get("half_pot", False):
						out.append(ActionType.HALF_POT_BET)
					out.append(ActionType.POT_SIZED_BET)
					if flags.get("two_pot", False):
						out.append(ActionType.TWO_POT_BET)
					out.append(ActionType.ALL_IN)

		seen = set()
		filt = []
		for a in out:
			if a not in seen:
				seen.add(a)
				filt.append(a)

		if not filt:
			return [ActionType.CALL]

		return filt

	def _allowed_actions_opponent(self, ps):
		return self._allowed_actions_agent(ps)

	def _mask_strategy(self, base_strategy, allowed_actions):
		A = len(base_strategy)
		keep = [False] * A

		for a in allowed_actions:
			ai = int(getattr(a, "value", 0))
			if 0 <= ai < A:
				keep[ai] = True

		out = [0.0] * A
		s = 0.0

		i = 0
		while i < A:
			if keep[i]:
				v = float(base_strategy[i])
				if v < 0.0:
					v = 0.0
				out[i] = v
				s += v
			i += 1

		if s <= 0.0:
			if allowed_actions:
				u = 1.0 / float(len(allowed_actions))
				for a in allowed_actions:
					out[int(a.value)] = u
		else:
			i = 0
			while i < A:
				if keep[i]:
					out[i] = out[i] / s
				i += 1

		return out

	def _mixed_action_distribution(self, node: GameNode, player: int, allowed_actions: list):
		if not allowed_actions:
			return []

		values = self.cfr_values.get(node, None)
		if values is None:
			u = 1.0 / float(len(allowed_actions))
			return [u] * len(allowed_actions)

		if hasattr(node, "player_ranges"):
			priors = dict(node.player_ranges[player])
		else:
			priors = {}

		total_mass = 0.0
		for _, p in priors.items():
			total_mass += float(p)

		if total_mass <= 0.0:
			u = 1.0 / float(len(allowed_actions))
			return [u] * len(allowed_actions)

		A = len(ActionType)
		acc = [0.0] * A

		for cid, prior in priors.items():
			w = float(prior)
			if w <= 0.0:
				continue
			if hasattr(values, "get_average_strategy"):
				base = values.get_average_strategy(int(cid))
			else:
				base = [1.0 / A] * A
			msk = self._mask_strategy(base, allowed_actions)
			i = 0
			while i < A:
				acc[i] += w * msk[i]
				i += 1

		out = []
		s = 0.0
		for a in allowed_actions:
			p = float(acc[int(a.value)])
			out.append(p)
			s += p

		if s <= 0.0:
			u = 1.0 / float(len(allowed_actions))
			return [u] * len(allowed_actions)

		return [x / s for x in out]

