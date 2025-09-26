import random
import itertools
from typing import Dict, List

from poker_utils import DECK
from public_state import PublicState
from game_node import GameNode


class DataGeneratorSamplingMixin:

	def _sample_random_range(
		self,
		cluster_ids: List[int],
	) -> Dict[int, float]:
		def _rec(ids, p):
			if not ids:
				return {}

			if len(ids) == 1:
				return {int(ids[0]): float(p)}

			sids = sorted(ids)
			m = len(sids) // 2
			left = sids[:m]
			right = sids[m:]

			p1 = random.random() * float(p)
			p2 = float(p) - p1

			a = _rec(left, p1)
			b = _rec(right, p2)

			for k, v in b.items():
				a[int(k)] = a.get(int(k), 0.0) + float(v)

			return a

		K = int(self.num_clusters)
		if cluster_ids:
			ids = list(cluster_ids)
		else:
			ids = list(range(K))

		r = _rec(ids, 1.0)
		s = sum(r.values()) or 0.0

		if s > 0.0:
			for k in list(r.keys()):
				r[k] = float(r[k]) / float(s)

		return r

	def _sample_flop_situation(
		self,
		rng: random.Random,
	) -> GameNode:
		board = rng.sample([c for c in DECK], 3)

		public_state = PublicState(
			initial_stacks=[self.player_stack, self.player_stack],
			board_cards=list(board),
		)
		public_state.current_round = 1
		public_state.current_bets = [0, 0]
		public_state.last_raiser = None
		public_state.stacks = [self.player_stack, self.player_stack]
		public_state.pot_size = self.sample_pot_size()
		public_state.current_player = (public_state.dealer + 1) % 2

		used = set(public_state.board_cards)
		deck_remaining = [c for c in DECK if c not in used]
		rng.shuffle(deck_remaining)

		public_state.hole_cards[0] = deck_remaining[:2]
		public_state.hole_cards[1] = deck_remaining[2:4]

		used2 = set(
			public_state.board_cards
			+ public_state.hole_cards[0]
			+ public_state.hole_cards[1]
		)
		public_state.deck = [c for c in DECK if c not in used2]
		rng.shuffle(public_state.deck)

		game_node = GameNode(public_state)

		deck_wo_board = [c for c in DECK if c not in public_state.board_cards]
		all_hands = [
			" ".join(h) for h in itertools.combinations(deck_wo_board, 2)
		]
		hands_set = set(
			h for h in all_hands
			if all(x not in public_state.board_cards for x in h.split())
		)

		if hands_set:
			u = 1.0 / float(len(hands_set))
			opp_range_over_hands = {h: u for h in hands_set}
		else:
			opp_range_over_hands = {}

		clusters = self.hand_clusterer.cluster_hands(
			hands_set,
			public_state.board_cards,
			opp_range_over_hands,
			public_state.pot_size,
		)
		self.cfr_solver.clusters = clusters
		self.clusters = clusters

		hand_probs_self = self._recursive_R(
			sorted(list(hands_set)),
			1.0,
			public_state.board_cards,
		)
		hand_probs_opp = self._recursive_R(
			sorted(list(hands_set)),
			1.0,
			public_state.board_cards,
		)

		r_self = self.map_hands_to_clusters(hand_probs_self, self.clusters)
		r_opp = self.map_hands_to_clusters(hand_probs_opp, self.clusters)

		self.normalize_cluster_probabilities([r_self, r_opp])

		game_node.player_ranges[public_state.current_player] = r_self
		game_node.player_ranges[(public_state.current_player + 1) % 2] = r_opp

		return game_node

	def _sample_turn_situation(
		self,
		rng: random.Random,
	) -> GameNode:
		board = rng.sample([c for c in DECK], 4)

		public_state = PublicState(
			initial_stacks=[self.player_stack, self.player_stack],
			board_cards=list(board),
		)
		public_state.current_round = 2
		public_state.current_bets = [0, 0]
		public_state.last_raiser = None
		public_state.stacks = [self.player_stack, self.player_stack]
		public_state.pot_size = self.sample_pot_size()
		public_state.current_player = (public_state.dealer + 1) % 2

		used = set(public_state.board_cards)
		deck_remaining = [c for c in DECK if c not in used]
		rng.shuffle(deck_remaining)

		public_state.hole_cards[0] = deck_remaining[:2]
		public_state.hole_cards[1] = deck_remaining[2:4]

		used2 = set(
			public_state.board_cards
			+ public_state.hole_cards[0]
			+ public_state.hole_cards[1]
		)
		public_state.deck = [c for c in DECK if c not in used2]
		rng.shuffle(public_state.deck)

		game_node = GameNode(public_state)

		deck_wo_board = [c for c in DECK if c not in public_state.board_cards]
		all_hands = [
			" ".join(h) for h in itertools.combinations(deck_wo_board, 2)
		]
		hands_set = set(
			h for h in all_hands
			if all(x not in public_state.board_cards for x in h.split())
		)

		if hands_set:
			u = 1.0 / float(len(hands_set))
			opp_range_over_hands = {h: u for h in hands_set}
		else:
			opp_range_over_hands = {}

		clusters = self.hand_clusterer.cluster_hands(
			hands_set,
			public_state.board_cards,
			opp_range_over_hands,
			public_state.pot_size,
		)
		self.cfr_solver.clusters = clusters
		self.clusters = clusters

		hand_probs_self = self._recursive_R(
			sorted(list(hands_set)),
			1.0,
			public_state.board_cards,
		)
		hand_probs_opp = self._recursive_R(
			sorted(list(hands_set)),
			1.0,
			public_state.board_cards,
		)

		r_self = self.map_hands_to_clusters(hand_probs_self, self.clusters)
		r_opp = self.map_hands_to_clusters(hand_probs_opp, self.clusters)

		self.normalize_cluster_probabilities([r_self, r_opp])

		game_node.player_ranges[public_state.current_player] = r_self
		game_node.player_ranges[(public_state.current_player + 1) % 2] = r_opp

		return game_node

	def sample_pot_size(self) -> float:
		bins = [
			(0.10, 2.0, 6.0),
			(0.30, 6.0, 20.0),
			(0.30, 20.0, 60.0),
			(0.20, 60.0, 150.0),
			(0.10, 150.0, 400.0),
		]

		r = random.random()
		acc = 0.0
		lo = 2.0
		hi = 6.0

		for p, a, b in bins:
			acc += p
			if r <= acc:
				lo = a
				hi = b
				break

		return float(random.uniform(lo, hi))

	def pot_sampler_spec(self) -> Dict[str, object]:
		return {
			"name": "bins.v1",
			"bins": [
				{"p": 0.10, "lo": 2.0,   "hi": 6.0},
				{"p": 0.30, "lo": 6.0,   "hi": 20.0},
				{"p": 0.30, "lo": 20.0,  "hi": 60.0},
				{"p": 0.20, "lo": 60.0,  "hi": 150.0},
				{"p": 0.10, "lo": 150.0, "hi": 400.0},
			],
			"player_stack": int(self.player_stack),
		}

	def range_generator_spec(self) -> Dict[str, object]:
		return {
			"name": "recursive_R.v1",
			"params": {
				"delegate": "solver.recursive_range_sampling",
				"public_cards_masking": True,
				"normalize": True,
			},
		}

