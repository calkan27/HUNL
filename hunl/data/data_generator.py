"""
Generate supervised training examples for counterfactual value (CFV) networks by
sampling public states, clustering legal hands, running a depth-limited re-solver, and
packing input/target tensors. A DataGenerator is configurable for fast test profiles and
production runs and can emit in-memory samples or persist NPZ shards with rich metadata.

Key class: DataGenerator. Key methods: generate_training_data (return list of records),
generate_turn_dataset/generate_flop_dataset (persisted shards),
generate_flop_dataset_using_turn (teacher–student flow using a turn net),
compute_counterfactual_values (query solver for both players), prepare_input_vector
(pot_norm | 52-d board one-hot | two K-d ranges), prepare_target_values (scalar CFVs per
cluster, normalized by pot), generate_unique_boards and helpers for seed/profile and
invariants.

Inputs: number of boards, samples per board, player stack, num_clusters, and optional
ResolveConfig driving solver limits, clustering profile, and zero-sum enforcement.
Outputs: lists of dicts with input_vector and target_v1/target_v2 or NPZ shards with
meta (schema, stage, action set, outer zero-sum flag).

Invariants: bucket mass conservation, valid board one-hot encoding, pot normalization in
(0,1], and scalar targets per cluster. Performance: reuses clusters in test mode, caps
per-update iterations, optionally forces sparse action sets, caches flop features, and
uses device-aware tensors.
"""



from hunl.constants import EPS_SUM, SEED_DEFAULT
import copy
import random
import itertools
import time
import os
import hashlib
from typing import List, Dict, Any

import torch
import numpy as np

from hunl.engine.poker_utils import board_one_hot, DECK
from hunl.engine.public_state import PublicState
from hunl.engine.game_node import GameNode
from hunl.engine.action_type import ActionType
from hunl.engine.action import Action
from hunl.solving.cfr_solver import CFRSolver
from hunl.ranges.hand_clusterer import HandClusterer

from hunl.data.data_generator_sampling import DataGeneratorSamplingMixin
from hunl.data.data_generator_datasets import DataGeneratorDatasetsMixin
from hunl.data.data_generator_utils import DataGeneratorUtilsMixin


class DataGenerator(
 DataGeneratorSamplingMixin,
 DataGeneratorDatasetsMixin,
 DataGeneratorUtilsMixin,
):
	def __init__(
	 self,
	 num_boards,
	 num_samples_per_board,
	 player_stack=200,
	 num_clusters=1000,
	 speed_profile: str = None,
	 config=None,
	):
		self._config = config
		self.num_boards = num_boards
		self.num_samples_per_board = num_samples_per_board
		self.player_stack = player_stack
		if self._config is not None:
			self.num_clusters = int(self._config.num_clusters)
			self.speed_profile = self._config.profile
			self.cfr_solver = CFRSolver(config=self._config)
			if str(self._config.profile) == "test":
				self.cfr_solver.depth_limit = 0
				self.cfr_solver.total_iterations = 1
			else:
				self.cfr_solver.total_iterations = int(self._config.total_iterations)
			self.hand_clusterer = HandClusterer(
			 self.cfr_solver,
			 num_clusters=self.num_clusters,
			 profile=self.speed_profile,
			 opp_sample_size=self._config.opp_sample_size,
			 use_cfv_in_features=self._config.use_cfv_in_features,
			 config=self._config,
			)
			self.cfr_solver.hand_clusterer = self.hand_clusterer
			self.cfr_solver.num_clusters = self.num_clusters
			self.torch_frozen_clusters_base = None
			if torch.cuda.is_available():
				self.device = torch.device("cuda")
			else:
				self.device = torch.device("cpu")
			fs = getattr(self._config, "fast_test_seed", SEED_DEFAULT)
			if isinstance(fs, (int, float, str)):
				seed_val = int(fs) if not isinstance(fs, int) else fs
				self.set_seed(int(seed_val))
			else:
				print("[INFO] fast_test_seed missing; using default seed.")
			return
		self.num_clusters = int(num_clusters)
		env_fast = (os.getenv("FAST_TESTS") == "1")
		if speed_profile is not None:
			self.speed_profile = speed_profile
		else:
			if env_fast:
				self.speed_profile = "test"
			else:
				self.speed_profile = "bot"
		self.cfr_solver = CFRSolver(
		 depth_limit=4,
		 num_clusters=self.num_clusters,
		)
		self.cfr_solver.total_iterations = 20
		self.hand_clusterer = HandClusterer(
		 self.cfr_solver,
		 num_clusters=self.num_clusters,
		 profile=self.speed_profile,
		 opp_sample_size=None,
		 use_cfv_in_features=(self.speed_profile != "test"),
		)
		self.cfr_solver.hand_clusterer = self.hand_clusterer
		self.cfr_solver.num_clusters = self.num_clusters
		if self.speed_profile == "test":
			self.cfr_solver.depth_limit = 0
			self.cfr_solver.total_iterations = 1
		self.torch_frozen_clusters_base = None
		if torch.cuda.is_available():
			self.device = torch.device("cuda")
		else:
			self.device = torch.device("cpu")

	def generate_training_data(
	 self,
	 stage: str = "flop",
	 progress=None,
	) -> List[Dict[str, Any]]:
		data = []

		boards = self.generate_unique_boards(stage, self.num_boards)
		leaf_snap = self._push_leaf_solve_mode(stage)

		for board_index, public_cards in enumerate(boards):
			deck_without_board = [c for c in DECK if c not in public_cards]
			possible_hands = [" ".join(h) for h in itertools.combinations(deck_without_board, 2)]
			hands_set = set(possible_hands)

			if hands_set:
				u = 1.0 / float(len(hands_set))
				opponent_range_over_hands = {h: u for h in hands_set}
			else:
				opponent_range_over_hands = {}

			nominal_pot = 1.0

			if self.speed_profile == "test":
				if self.torch_frozen_clusters_base is None:
					self.torch_frozen_clusters_base = self.hand_clusterer.cluster_hands(
					 hands_set,
					 board=public_cards,
					 opponent_range=opponent_range_over_hands,
					 pot_size=nominal_pot,
					)
				clusters = self._filter_clusters_for_board(
				 self.torch_frozen_clusters_base,
				 public_cards,
				)
				self.cfr_solver.clusters = clusters
				self.clusters = clusters

				max_opponent_iterations = 1
				cfr_iterations_per_update = 1
				force_fcp_only = False
			else:
				clusters = self.hand_clusterer.cluster_hands(
				 hands_set,
				 board=public_cards,
				 opponent_range=opponent_range_over_hands,
				 pot_size=nominal_pot,
				)
				self.cfr_solver.clusters = clusters
				self.clusters = clusters

				max_opponent_iterations = 1
				cfr_iterations_per_update = 1000
				force_fcp_only = True

			if callable(progress):
				progress(1)

			for sample_index in range(self.num_samples_per_board):
				pot_size = self.sample_pot_size()

				hand_probs_self = self._recursive_range_split(sorted(list(hands_set)), 1.0, public_cards)
				hand_probs_opp = self._recursive_range_split(sorted(list(hands_set)), 1.0, public_cards)

				r_self = self.map_hands_to_clusters(hand_probs_self, self.clusters)
				r_opp = self.map_hands_to_clusters(hand_probs_opp, self.clusters)
				self.normalize_cluster_probabilities([r_self, r_opp])

				opponent_range = dict(r_opp)
				previous_opponent_range = dict(opponent_range)

				kref = 0
				while kref < max_opponent_iterations:
					kref += 1

					public_state = PublicState(
					 initial_stacks=[self.player_stack, self.player_stack],
					 board_cards=public_cards,
					)
					public_state.pot_size = pot_size
					target_round = self.get_round_from_stage(stage)
					public_state.current_round = target_round

					if target_round >= 1:
						public_state.current_bets = [0, 0]
						public_state.last_raiser = None
						public_state.stacks = [self.player_stack, self.player_stack]
						public_state.current_player = (public_state.dealer + 1) % 2

						while len(public_state.board_cards) > (2 + target_round):
							public_state.board_cards.pop()

						while len(public_state.board_cards) < (2 + target_round):
							for c in DECK:
								if c not in public_state.board_cards:
									if c not in public_state.hole_cards[0]:
										if c not in public_state.hole_cards[1]:
											public_state.board_cards.append(c)
											break

					used = set(public_state.board_cards)
					deck_remaining = [c for c in DECK if c not in used]
					random.shuffle(deck_remaining)

					public_state.hole_cards[0] = deck_remaining[:2]
					public_state.hole_cards[1] = deck_remaining[2:4]

					used2 = set(
					 public_state.board_cards
					 + public_state.hole_cards[0]
					 + public_state.hole_cards[1]
					)
					public_state.deck = [c for c in DECK if c not in used2]
					random.shuffle(public_state.deck)

					game_node = GameNode(public_state)
					game_node.player_ranges[0] = dict(r_self)
					game_node.player_ranges[1] = dict(opponent_range)
					game_node.players_in_hand = [True, True]

					if force_fcp_only:
						round_flags_backup = {
						 int(k): {
						  "half_pot": bool(v.get("half_pot", True)),
						  "two_pot": bool(v.get("two_pot", False)),
						 }
						 for k, v in getattr(self.cfr_solver, "_round_actions", {}).items()
						}
						self.cfr_solver._ensure_sparse_schedule()
						for r in (0, 1, 2, 3):
							self.cfr_solver._round_actions[int(r)] = {
							 "half_pot": False,
							 "two_pot": False,
							}

					self.cfr_solver.total_iterations = int(cfr_iterations_per_update)
					_ = self.cfr_solver.run_cfr(game_node)

					if force_fcp_only:
						self.cfr_solver._round_actions = round_flags_backup

					if callable(progress):
						progress(1)

				game_node.player_ranges[1] = dict(opponent_range)

				player_ranges_bucketed = self.bucket_player_ranges(
				 [dict(r_self), dict(opponent_range)]
				)

				ok_inv = self._assert_sampler_invariants(
				 public_cards,
				 [
				  {i: player_ranges_bucketed[0][i] for i in range(self.num_clusters)},
				  {i: player_ranges_bucketed[1][i] for i in range(self.num_clusters)},
				 ],
				 pot_size,
				)
				if not ok_inv:
					print("[SKIP] Invariants failed; skipping sample.")
					if callable(progress):
						progress(1)
					continue

				input_vector = self.prepare_input_vector(
				 player_ranges_bucketed,
				 public_cards,
				 pot_size,
				 game_node.public_state.actions,
				)
				if not input_vector:
					print("[SKIP] Input vector invalid; skipping sample.")
					if callable(progress):
						progress(1)
					continue

				counterfactual_values = self.compute_counterfactual_values(game_node)
				target_v1, target_v2 = self.prepare_target_values(
				 counterfactual_values,
				 pot_size,
				)
				if (not target_v1) or (not target_v2):
					print("[SKIP] Targets invalid; skipping sample.")
					if callable(progress):
						progress(1)
					continue

				data.append(
				 {
				  "input_vector": input_vector,
				  "target_v1": target_v1,
				  "target_v2": target_v2,
				 }
				)

				if callable(progress):
					progress(1)

		data.sort(key=lambda rec: tuple(round(x, 12) for x in rec["input_vector"]))
		_ = self._pop_leaf_solve_mode(leaf_snap)
		return data

	def compute_counterfactual_values(
	 self,
	 node: GameNode,
	) -> Dict[int, Dict[int, List[float]]]:
		cf0 = self.cfr_solver.predict_counterfactual_values(node, player=0)
		cf1 = self.cfr_solver.predict_counterfactual_values(node, player=1)
		return {0: cf0, 1: cf1}

	def prepare_input_vector(
	 self,
	 player_ranges_bucketed: List[List[float]],
	 public_cards: List[str],
	 pot_size: float,
	 actions=None,
	) -> List[float]:
		total_initial = float(sum([self.player_stack, self.player_stack])) if self.player_stack is not None else 1.0
		if total_initial <= 0.0:
			total_initial = 1.0
		pot_norm = float(pot_size) / total_initial
		if (not (pot_norm > 0.0)) or (not (pot_norm <= 1.0)):
			print(f"[ERROR] PotNormalizationOutOfRange pot_norm={pot_norm}")
			raise ValueError("PotNormalizationOutOfRange")
		bvec = board_one_hot(public_cards)
		ones = 0
		i = 0
		while i < len(bvec):
			if bvec[i] not in (0, 1):
				print("[ERROR] BoardOneHotInvalid")
				raise ValueError("BoardOneHotInvalid")
			ones += bvec[i]
			i += 1
		if ones != len(public_cards):
			print("[ERROR] BoardOneHotCardCountMismatch")
			raise ValueError("BoardOneHotCardCountMismatch")
		r1, r2 = player_ranges_bucketed
		s1 = 0.0
		for v in r1:
			s1 += float(v)
		s2 = 0.0
		for v in r2:
			s2 += float(v)
		if (abs(s1 - 1.0) > EPS_SUM) or (abs(s2 - 1.0) > EPS_SUM):
			print("[ERROR] RangeMassNotConserved")
			raise ValueError("RangeMassNotConserved")
		return [pot_norm] + list(bvec) + list(r1) + list(r2)


	def prepare_target_values(
	 self,
	 counterfactual_values: Dict[int, Dict[int, List[float]]],
	 pot_size: float = None,
	) -> List[List[float]]:
		K = int(self.num_clusters)

		if (pot_size is not None) and (float(pot_size) > 0.0):
			scale = 1.0 / float(pot_size)
		else:
			scale = None

		t1 = [0.0] * K
		t2 = [0.0] * K

		cf0 = dict(counterfactual_values.get(0, {}))
		cf1 = dict(counterfactual_values.get(1, {}))

		A = len(ActionType)

		i = 0
		while i < K:
			v0 = cf0.get(i, 0.0)

			if isinstance(v0, (list, tuple)):
				if len(v0) == 1:
					x0 = float(v0[0])
				else:
					if (len(v0) == A):
						ok = True
						j = 0
						while j < A:
							if abs(float(v0[j]) - float(v0[0])) > EPS_SUM:
								ok = False
							j += 1
						if ok:
							x0 = float(v0[0])
						else:
							print("[ERROR] NonScalarCFVTarget_v1")
							return [], []
					else:
						print("[ERROR] NonScalarCFVTarget_v1")
						return [], []
			else:
				x0 = float(v0)

			v1 = cf1.get(i, 0.0)

			if isinstance(v1, (list, tuple)):
				if len(v1) == 1:
					x1 = float(v1[0])
				else:
					if (len(v1) == A):
						ok2 = True
						j2 = 0
						while j2 < A:
							if abs(float(v1[j2]) - float(v1[0])) > EPS_SUM:
								ok2 = False
							j2 += 1
						if ok2:
							x1 = float(v1[0])
						else:
							print("[ERROR] NonScalarCFVTarget_v2")
							return [], []
					else:
						print("[ERROR] NonScalarCFVTarget_v2")
						return [], []
			else:
				x1 = float(v1)

			if scale is not None:
				x0 = x0 * scale
				x1 = x1 * scale

			t1[i] = x0
			t2[i] = x1

			i += 1

		return t1, t2
