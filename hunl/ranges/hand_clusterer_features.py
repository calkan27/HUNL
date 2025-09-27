"""
Feature extraction and small Monte Carlo utilities for clustering private hands given a
public board and an opponent range. This mixin constructs equity-like descriptors and
histograms that drive k-means-style bucketing, while honoring card availability and
board collisions.

Key methods: _deck_without (filter used cards), _all_pairs_from_deck and _sample_pairs
(candidate/MC sampling), _weight_for_opp_hand (range weight lookup), _hist_len (fixed
histogram size), _seed_for (deterministic per-hand seed), calculate_hand_features (main
descriptor pipeline), _evaluate_win_percentage (MC equity over future cards), and small
helpers.

Inputs: hand string, board card list, opponent range over hands, solver hooks for
win/hand ranking; configuration fields for MC sample counts. Outputs: numpy arrays
(fixed-length features or histograms) suitable for clustering and drift checks.

Invariants: excludes duplicate/suited-invalid pairs and board collisions; reproducible
RNG seeded by hand+board; normalization of histograms when total weight is positive.
Performance: test/production modes adjust MC sample counts; sampling avoids heavy
allocations by reusing local buffers.
"""


import random
from typing import Any, Dict, List

import numpy as np
from hunl.engine.poker_utils import DECK


class HandClustererFeaturesMixin:
	def _deck_without(self, exclude: List[str]) -> List[str]:
		out: List[str] = []
		for c in DECK:
			if c in exclude:
				pass
			else:
				out.append(c)
		return out

	def _all_pairs_from_deck(self, deck: List[str]) -> List[List[str]]:
		pairs: List[List[str]] = []
		i = 0
		while i + 1 < len(deck):
			a = deck[i]
			j = i + 1
			while j < len(deck):
				b = deck[j]
				if a != b:
					pairs.append([a, b])
				else:
					pass
				j += 1
			i += 1
		return pairs

	def _sample_pairs(self, rng: random.Random, pairs: List[List[str]], n: int) -> List[List[str]]:
		if len(pairs) > n:
			return rng.sample(pairs, n)
		else:
			return list(pairs)

	def _weight_for_opp_hand(self, opponent_range: Dict[Any, float], a: str, b: str) -> float:
		key_ab = f"{a} {b}"
		key_ba = f"{b} {a}"

		if key_ab in opponent_range:
			return float(opponent_range.get(key_ab, 1.0))
		else:
			return float(opponent_range.get(key_ba, 1.0))

	def _hist_len(self) -> int:
		return 21

	def _seed_for(self, token: str, board: List[str]) -> int:
		return int(self._stable_seed(token, board))

	def calculate_hand_features(
		self,
		hand: str,
		board: List[str],
		opponent_range: Dict[Any, float],
		pot_size: float,
	) -> np.ndarray:
		board_key = ",".join(board)
		cache_key = f"specP3.1|hand={hand}|board={board_key}"

		if cache_key in self._cache:
			self._cache_hits += 1
			return self._cache[cache_key]
		else:
			self._cache_misses += 1

		if isinstance(hand, str):
			my = hand.split()
		else:
			my = list(hand)

		used = set(my + board)
		deck = self._deck_without(list(used))

		opp_hands_all = self._all_pairs_from_deck(deck)

		if len(opp_hands_all) == 0:
			hist = np.zeros((self._hist_len(),), dtype=float)
			hist[10] = 1.0
			self._cache[cache_key] = hist
			return hist
		else:
			pass

		rng = random.Random(self._seed_for(hand, board))

		mc = int(getattr(self, "_mc_samples_win", 200))
		sample_n = min(len(opp_hands_all), max(1, mc))
		opp_hands = self._sample_pairs(rng, opp_hands_all, sample_n)

		hist = np.zeros((self._hist_len(),), dtype=float)
		total_w = 0.0

		for oh in opp_hands:
			w = self._weight_for_opp_hand(opponent_range, oh[0], oh[1])
			p = self._evaluate_win_percentage(my, oh, board)

			p_clip = max(0.0, min(1.0, float(p)))
			bi = int(round(p_clip * 20.0))

			hist[bi] += w
			total_w += w

		if total_w > 0.0:
			hist = hist / total_w
		else:
			pass

		self._cache[cache_key] = hist
		return hist

	def _evaluate_win_percentage(
		self,
		player_hand: List[str],
		opponent_hand: List[str],
		board: List[str],
	) -> float:
		if len(board) >= 5:
			r = self.cfr_solver._player_wins(player_hand, opponent_hand, board)

			if r > 0:
				return 1.0
			else:
				if r < 0:
					return 0.0
				else:
					return 0.5
		else:
			used = set(player_hand + opponent_hand + board)
			avail = self._deck_without(list(used))
			need = 5 - len(board)

			if need <= 0:
				return 0.5
			else:
				token = "".join(player_hand) + "|" + "".join(opponent_hand)
				rng = random.Random(self._seed_for(token, board))

				mc = int(getattr(self, "_mc_samples_win", 200))
				samples = min(len(avail), max(1, mc))

				if samples < need:
					samples = need
				else:
					pass

				wins = 0.0
				ties = 0.0
				trials = 0

				i = 0
				while i < samples:
					draw = rng.sample(avail, need)
					res = self.cfr_solver._player_wins(
						player_hand,
						opponent_hand,
						list(board) + list(draw),
					)

					if res > 0:
						wins += 1.0
					else:
						if res == 0:
							ties += 1.0
						else:
							pass

					trials += 1
					i += 1

				if trials <= 0:
					return 0.5
				else:
					return (wins + 0.5 * ties) / float(trials)

	def _calculate_payoff(
		self,
		player_hand: List[str],
		opponent_hand: List[str],
		board: List[str],
		pot_size: float,
	) -> float:
		r = self.cfr_solver._player_wins(player_hand, opponent_hand, board)

		if r > 0:
			return 1.0
		else:
			if r < 0:
				return -1.0
			else:
				return 0.0

	def _calculate_equity(
		self,
		hand: str,
		board: List[str],
		opponent_range: Dict[Any, float],
	) -> float:
		if isinstance(hand, str):
			my = hand.split()
		else:
			my = list(hand)

		used = set(my + board)
		deck = self._deck_without(list(used))

		opp_hands_all = self._all_pairs_from_deck(deck)

		if len(opp_hands_all) == 0:
			return 0.5
		else:
			rng = random.Random(self._seed_for(hand, board))

			mc = int(getattr(self, "_mc_samples_win", 200))
			sample_n = min(len(opp_hands_all), max(1, mc))
			opp_hands = self._sample_pairs(rng, opp_hands_all, sample_n)

			total = 0.0
			den = 0.0

			for oh in opp_hands:
				w = self._weight_for_opp_hand(opponent_range, oh[0], oh[1])
				p = self._evaluate_win_percentage(my, oh, board)
				total += w * p
				den += w

			if den <= 0.0:
				den = float(len(opp_hands))
			else:
				pass

			return total / den

	def _calculate_potential_equity_improvement(
		self,
		hand: str,
		board: List[str],
		opponent_range: Dict[Any, float],
	) -> float:
		if len(board) >= 5:
			return 0.0
		else:
			eq_now = self._calculate_equity(hand, board, opponent_range)

			if isinstance(hand, str):
				hand_list = hand.split()
			else:
				hand_list = list(hand)

			used = set(hand_list + board)
			avail = self._deck_without(list(used))

			seed = self._seed_for(hand, board) ^ 0x9E3779B1
			rng = random.Random(int(seed))

			mc = int(getattr(self, "_mc_samples_potential", 200))
			samples = min(len(avail), max(1, mc))

			if samples <= 0:
				return 0.0
			else:
				acc = 0.0
				trials = 0

				i = 0
				while i < samples:
					draw = rng.choice(avail)
					eq_next = self._calculate_equity(
						hand,
						board + [draw],
						opponent_range,
					)
					inc = eq_next - eq_now

					if inc > 0.0:
						acc += inc
					else:
						pass

					trials += 1
					i += 1

				if trials > 0:
					return acc / float(trials)
				else:
					return 0.0

	def _calculate_counterfactual_value(
		self,
		hand: str,
		board: List[str],
		opponent_range: Dict[Any, float],
		pot_size: float,
	) -> float:
		if isinstance(hand, str):
			my = hand.split()
		else:
			my = list(hand)

		used = set(my + board)
		avail = self._deck_without(list(used))

		opp_hands = self._all_pairs_from_deck(avail)

		if len(opp_hands) == 0:
			return 0.0
		else:
			rng = random.Random(self._seed_for(hand, board))

			mc = int(getattr(self, "_mc_samples_win", 200))
			sample_n = min(len(opp_hands), max(1, mc))
			opp_hands = self._sample_pairs(rng, opp_hands, sample_n)

			total = 0.0
			den = 0.0

			for oh in opp_hands:
				w = self._weight_for_opp_hand(opponent_range, oh[0], oh[1])
				payoff = self._calculate_payoff(my, oh, board, pot_size)
				total += w * payoff
				den += w

			if den <= 0.0:
				den = float(len(opp_hands))
			else:
				pass

			return total / den

	def _emd_distance(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
		a = np.asarray(sig1, dtype=np.float64).ravel()
		b = np.asarray(sig2, dtype=np.float64).ravel()

		sa = float(np.sum(a))
		sb = float(np.sum(b))

		if sa > 0.0:
			a = a / sa
		else:
			pass

		if sb > 0.0:
			b = b / sb
		else:
			pass

		ca = np.cumsum(a)
		cb = np.cumsum(b)

		return float(np.sum(np.abs(ca - cb)))

	def calculate_hand_distance(
		self,
		features1: np.ndarray,
		features2: np.ndarray,
	) -> float:
		a = np.asarray(features1, dtype=np.float64).ravel()
		b = np.asarray(features2, dtype=np.float64).ravel()

		d = a - b
		return float(np.sqrt(np.dot(d, d)))

