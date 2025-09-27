"""
Preflop and computation caches and lightweight signatures for reuse within continual
re-solving. This mixin encapsulates a content-addressed preflop cache, partition
signatures for clusters, and helpers for efficient range sampling/equity proxies used by
clustering and recursion.

Key responsibilities: build stable preflop keys combining blinds/dealer/bets/pot/cluster
partition and current bucketed ranges; LRU-style cache get/put with stats; cluster
partition signature (count and FNV-like hash per cluster); compact state keys for
tracking; fast equity sampling and recursive range splitting with board awareness.

Inputs: a GameNode with PublicState and current player ranges; solver attributes such as
num_clusters, clusters, and internal caches. Outputs: cache entries with normalized own
range and opponent CFV upper vectors; updated cache statistics
(hits/misses/puts/evictions).

Invariants: signatures ignore private cards; range normalization occurs when rehydrating
cache hits; negative or inconsistent counters are coerced to safe bounds. Performance:
ordered dict with capacity limits, constant-time recency updates, and parameterized
sampling (test vs production) to keep preflop latency low.
"""


import os
import random
import itertools
from collections import OrderedDict
from typing import Any, Dict, Tuple

import numpy as np

from hunl.engine.poker_utils import DECK


class CFRSolverCachingMixin:

	def _preflop_signature(self, node):
		ps = node.public_state

		sb = int(ps.small_blind)
		bb = int(ps.big_blind)
		dealer = int(ps.dealer)

		if ps.current_player is not None:
			current_player = int(ps.current_player)
		else:
			current_player = -1

		if getattr(ps, "initial_stacks", None) is not None:
			src_stacks = ps.initial_stacks
		else:
			src_stacks = ps.stacks

		init_stacks = tuple(int(x) for x in src_stacks)
		curr_bets = (int(ps.current_bets[0]), int(ps.current_bets[1]))
		pot = int(ps.pot_size)
		K = int(self.num_clusters)

		part_sig = self._cluster_partition_signature()

		if node.player_ranges and (len(node.player_ranges) > 0):
			r1_sig = self._range_sig(node.player_ranges[0])
		else:
			r1_sig = tuple()

		if node.player_ranges and (len(node.player_ranges) > 1):
			r2_sig = self._range_sig(node.player_ranges[1])
		else:
			r2_sig = tuple()

		return (
			"PREFLOP_CACHE_V2",
			sb, bb, dealer, current_player,
			init_stacks, curr_bets, pot, K, part_sig, r1_sig, r2_sig,
		)

	def _preflop_cache_get(self, key):
		if not isinstance(self._preflop_cache, OrderedDict):
			self._preflop_cache_stats["misses"] += 1
			return None

		if key in self._preflop_cache:
			val = self._preflop_cache.pop(key)
			self._preflop_cache[key] = val
			self._preflop_cache_stats["hits"] += 1
			return val

		self._preflop_cache_stats["misses"] += 1
		return None

	def _preflop_cache_put(self, key, own_range_norm, opp_cfv_vector):
		if not isinstance(self._preflop_cache, OrderedDict):
			return

		entry = {
			"own_range": {int(k): float(v) for k, v in dict(own_range_norm).items()},
			"opp_cfv": {int(k): float(v) for k, v in dict(opp_cfv_vector).items()},
		}

		if key in self._preflop_cache:
			self._preflop_cache.pop(key)

		self._preflop_cache[key] = entry
		self._preflop_cache_stats["puts"] += 1

		while len(self._preflop_cache) > int(self._preflop_cache_cap):
			self._preflop_cache.popitem(last=False)
			self._preflop_cache_stats["evictions"] += 1

	def _cluster_partition_signature(self):
		items = []

		if self.clusters:
			pairs = self.clusters.items()
		else:
			pairs = []

		for cid, hset in pairs:
			n = 0
			acc = 0

			for h in hset:
				if isinstance(h, str):
					s = h
				else:
					s = " ".join(list(h))

				val = 2166136261
				i = 0

				while i < len(s):
					val ^= ord(s[i])
					val = (val * 16777619) & 0xFFFFFFFF
					i += 1

				acc = (acc ^ val) & 0xFFFFFFFF
				n += 1

			items.append((int(cid), int(n), int(acc)))

		items.sort(key=lambda x: x[0])
		return tuple(items)

	def _range_sig(self, r):
		items = [(int(k), float(v)) for k, v in r.items()]
		items.sort(key=lambda x: x[0])
		return tuple((k, round(v, 12)) for k, v in items)

	def _state_key(self, node):
		if hasattr(node, "_public_signature"):
			if hasattr(getattr(node, "public_state", None), "actions"):
				return node._public_signature()

		ps = node.public_state
		cb = getattr(ps, "current_bets", (0, 0))

		if (not isinstance(cb, (tuple, list))) or (len(cb) < 2):
			cb = (int(getattr(ps, "bet0", 0)), int(getattr(ps, "bet1", 0)))

		if getattr(ps, "current_player", None) is not None:
			curp = int(getattr(ps, "current_player", -1))
		else:
			curp = -1

		is_term = bool(getattr(ps, "is_terminal", False))
		is_show = bool(getattr(ps, "is_showdown", False))

		return (
			tuple(getattr(ps, "board_cards", [])),
			int(getattr(ps, "current_round", getattr(ps, "round_idx", 0))),
			(int(cb[0]), int(cb[1])),
			int(getattr(ps, "pot_size", 0)),
			curp,
			int(getattr(ps, "dealer", 0)),
			is_term,
			is_show,
			(True, True),
		)

	def _evaluate_hand_strength(self, hand, public_cards):
		cache_key = (hand, tuple(public_cards))

		if cache_key in self._evaluate_hand_strength_cache:
			return self._evaluate_hand_strength_cache[cache_key]

		test_mode = (
			getattr(self, "hand_clusterer", None) is not None
			and getattr(self.hand_clusterer, "profile", "bot") == "test"
		)
		fast_env = (os.getenv("FAST_TESTS") == "1")

		if test_mode or fast_env:
			max_cluster_samples = 6
			opponent_hand_sample_size = 16
			future_board_samples = 2
		else:
			max_cluster_samples = None
			opponent_hand_sample_size = 100
			future_board_samples = 10

		rng = random.Random(self._stable_seed(hand, public_cards))

		if isinstance(hand, int) or isinstance(hand, np.integer):
			hands_in_cluster = list(self.clusters.get(int(hand), []))

			if len(hands_in_cluster) == 0:
				self._evaluate_hand_strength_cache[cache_key] = 0.0
				return 0.0

			if max_cluster_samples is not None:
				if len(hands_in_cluster) > max_cluster_samples:
					hands_in_cluster = rng.sample(hands_in_cluster, max_cluster_samples)
		else:
			hands_in_cluster = [hand]

		num_hands = len(hands_in_cluster)
		total_strength = 0.0

		for hand_str in hands_in_cluster:
			if isinstance(hand_str, str):
				hand_cards = hand_str.split()
			else:
				hand_cards = list(hand_str)

			used_cards = set(hand_cards + public_cards)
			available_cards = [card for card in DECK if card not in used_cards]

			all_opponent_hands = list(itertools.combinations(available_cards, 2))

			if len(all_opponent_hands) == 0:
				total_strength += 0.5
				continue

			if len(all_opponent_hands) > opponent_hand_sample_size:
				sampled_opponent_hands = rng.sample(all_opponent_hands, opponent_hand_sample_size)
			else:
				sampled_opponent_hands = all_opponent_hands

			win_count = 0
			tie_count = 0
			total_simulations = 0

			cards_to_come = 5 - len(public_cards)

			if cards_to_come <= 0:
				for opp_hand in sampled_opponent_hands:
					result = self._player_wins(hand_cards, list(opp_hand), public_cards)

					if result == 1:
						win_count += 1
					else:
						if result == 0:
							tie_count += 1

					total_simulations += 1
			else:
				for opp_hand in sampled_opponent_hands:
					opp_hand_set = set(opp_hand)
					safe_available_cards = [c for c in available_cards if c not in opp_hand_set]
					possible_future_boards = list(itertools.combinations(safe_available_cards, cards_to_come))

					if not possible_future_boards:
						possible_future_boards = [()]

					if len(possible_future_boards) > future_board_samples:
						sampled_future_boards = rng.sample(possible_future_boards, future_board_samples)
					else:
						sampled_future_boards = possible_future_boards

					for future_board in sampled_future_boards:
						full_board = public_cards + list(future_board)
						result = self._player_wins(hand_cards, list(opp_hand), full_board)

						if result == 1:
							win_count += 1
						else:
							if result == 0:
								tie_count += 1

						total_simulations += 1

			if total_simulations > 0:
				strength = (win_count + 0.5 * tie_count) / total_simulations
			else:
				strength = 0.5

			total_strength += strength

		if num_hands > 0:
			avg_strength = total_strength / num_hands
		else:
			avg_strength = 0.5

		self._evaluate_hand_strength_cache[cache_key] = avg_strength
		return avg_strength

	def recursive_range_sampling(self, hands_set, total_prob, public_cards=None):
		test_mode = (
			getattr(self, "hand_clusterer", None) is not None
			and getattr(self.hand_clusterer, "profile", "bot") == "test"
		)
		fast_env = (os.getenv("FAST_TESTS") == "1")

		if test_mode or fast_env:
			n = len(hands_set)

			if n == 0:
				return {}

			u = total_prob / float(n)
			out = {}

			for h in hands_set:
				out[h] = u

			return out

		if public_cards is None:
			pub_cards_tuple = tuple([])
		else:
			pub_cards_tuple = tuple(public_cards)

		cache_key = (frozenset(hands_set), total_prob, pub_cards_tuple)

		if cache_key in self._recursive_range_sampling_cache:
			return self._recursive_range_sampling_cache[cache_key]

		if not hands_set:
			result = {}
			self._recursive_range_sampling_cache[cache_key] = result
			return result

		if len(hands_set) == 1:
			hand = list(hands_set)[0]
			result = {hand: total_prob}
			self._recursive_range_sampling_cache[cache_key] = result
			return result

		prob_subset_1 = random.uniform(0, total_prob)
		prob_subset_2 = total_prob - prob_subset_1

		hands_with_strength = []

		for hand in hands_set:
			if public_cards is None:
				_pc = []
			else:
				_pc = public_cards
			strength = self._evaluate_hand_strength(hand, _pc)
			hands_with_strength.append((hand, strength))

		hands_with_strength.sort(key=lambda x: x[1])
		midpoint = len(hands_with_strength) // 2

		weaker_hands = set()
		stronger_hands = set()

		i = 0
		while i < midpoint:
			weaker_hands.add(hands_with_strength[i][0])
			i += 1

		j = midpoint
		while j < len(hands_with_strength):
			stronger_hands.add(hands_with_strength[j][0])
			j += 1

		sampled_range_1 = {}
		sampled_range_2 = {}

		if len(weaker_hands) > 0:
			sampled_range_1 = self.recursive_range_sampling(weaker_hands, prob_subset_1, public_cards)

		if len(stronger_hands) > 0:
			sampled_range_2 = self.recursive_range_sampling(stronger_hands, prob_subset_2, public_cards)

		result = {}
		for k, v in sampled_range_1.items():
			result[k] = v
		for k, v in sampled_range_2.items():
			result[k] = v

		self._recursive_range_sampling_cache[cache_key] = result
		return result

	def compute_values_depth_limited(self, node, player):
		for base in self.__class__.__mro__:
			if base.__name__ == "CFRSolverCachingMixin":
				continue

			m = base.__dict__.get("compute_values_depth_limited")

			if m is not None:
				return m(self, node, player)

		print("compute_values_depth_limited not found in MRO")
		return None

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








