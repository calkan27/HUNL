"""
I provide normalization, mapping, and guardrails for the data pipeline. I convert
between hand probabilities and cluster vectors, filter clusters by public boards, and
verify sampler invariants (pot normalization, board one-hot validity, range mass). I
also control temporary production/test modes for the solver.

Key class: DataGeneratorUtilsMixin. Key methods: normalize_cluster_probabilities —
within-list normalization; bucket_player_ranges — dict→dense ranges;
_filter_clusters_for_board — drop colliding hands;
_push_leaf_solve_mode/_pop_leaf_solve_mode — adjust solver budgets for leaf labeling;
_assert_sampler_invariants — enforce guards;
map_hands_to_clusters/map_hands_to_clusters_compat — string mapping with stable hashing
fallback; _push_production_mode/_pop_production_mode — bulk mode toggles.

Inputs: clusters, ranges, board cards, pot sizes, and owning generator fields. Outputs:
normalized vectors, filtered mappings, and booleans or exceptions for invariant checks.

Dependencies: numpy/torch for seeding and arrays; engine for board encodings. Edge
cases: empty ranges or clusters return safe defaults; I clip pot normalization to a sane
domain to avoid degenerate targets. Performance: I avoid repeated hashing and reuse
filtered sets when possible.
"""

from hunl.constants import EPS_SUM
import random
import hashlib
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

from hunl.engine.poker_utils import board_one_hot


class DataGeneratorUtilsMixin:

	def normalize_cluster_probabilities(
	 self,
	 ranges_list: List[Dict[int, float]],
	) -> List[Dict[int, float]]:
		K = int(self.num_clusters)

		for r in ranges_list:
			s = sum(float(v) for v in r.values()) or 0.0

			if s > 0.0:
				for k in list(r.keys()):
					r[k] = float(r[k]) / float(s)
			else:
				if K > 0:
					u = 1.0 / float(K)
					for i in range(K):
						r[i] = u

		return ranges_list

	def bucket_player_ranges(
	 self,
	 ranges_pair: List[Dict[int, float]],
	) -> List[List[float]]:
		K = int(self.num_clusters)
		r1 = [0.0] * K
		r2 = [0.0] * K

		for cid, p in dict(ranges_pair[0]).items():
			i = int(cid)
			if 0 <= i < K:
				r1[i] = float(p)

		for cid, p in dict(ranges_pair[1]).items():
			i = int(cid)
			if 0 <= i < K:
				r2[i] = float(p)

		s1 = sum(r1) or 0.0
		s2 = sum(r2) or 0.0

		if s1 > 0.0:
			for i in range(K):
				r1[i] = r1[i] / s1

		if s2 > 0.0:
			for i in range(K):
				r2[i] = r2[i] / s2

		return [r1, r2]

	def _filter_clusters_for_board(
	 self,
	 clusters: Dict[int, set],
	 board_cards: List[str],
	) -> Dict[int, set]:
		board_set = set(board_cards)
		filtered: Dict[int, set] = {}

		for cid, hand_set in (clusters or {}).items():
			keep: set[str] = set()

			for h in hand_set:
				if isinstance(h, str):
					parts = h.split()
				else:
					parts = list(h)

				if len(parts) != 2:
					continue

				c1, c2 = parts[0], parts[1]

				if c1 == c2:
					continue

				if (c1 in board_set) or (c2 in board_set):
					continue

				keep.add(f"{c1} {c2}")

			filtered[int(cid)] = keep

		for cid in clusters.keys():
			if int(cid) not in filtered:
				filtered[int(cid)] = set()

		return filtered

	def _push_leaf_solve_mode(
	 self,
	 stage: str,
	):
		if (not hasattr(self, "cfr_solver")) or (self.cfr_solver is None):
			return None

		snap = (
		 int(getattr(self.cfr_solver, "depth_limit", 0)),
		 int(getattr(self.cfr_solver, "total_iterations", 1)),
		)

		if str(getattr(self, "speed_profile", "")) == "test":
			self.cfr_solver.depth_limit = 0
			if int(getattr(self.cfr_solver, "total_iterations", 0)) < 1:
				self.cfr_solver.total_iterations = 1

		return snap

	def _pop_leaf_solve_mode(
	 self,
	 snap: Tuple[int, int],
	):
		if (not hasattr(self, "cfr_solver")) or (self.cfr_solver is None):
			return False

		if isinstance(snap, tuple):
			if len(snap) == 2:
				self.cfr_solver.depth_limit = int(snap[0])
				self.cfr_solver.total_iterations = int(snap[1])
				return True

		return False

	def get_round_from_stage(
	 self,
	 stage: Any,
	) -> int:
		s = str(stage).lower()

		if s == "preflop":
			return 0
		if s == "flop":
			return 1
		if s == "turn":
			return 2
		if s == "river":
			return 3

		return 0

	def set_seed(
	 self,
	 seed: int,
	) -> int:
		seed_int = int(seed)

		random.seed(seed_int)

		if hasattr(np, "random"):
			if hasattr(np.random, "seed"):
				np.random.seed(seed_int)
			else:
				print("[INFO] numpy.random.seed not available.")
		else:
			print("[INFO] numpy not available for seeding.")

		if hasattr(torch, "manual_seed"):
			torch.manual_seed(seed_int)
		else:
			print("[INFO] torch.manual_seed not available.")

		if hasattr(torch, "cuda"):
			if hasattr(torch.cuda, "is_available"):
				if torch.cuda.is_available():
					if hasattr(torch.cuda, "manual_seed_all"):
						torch.cuda.manual_seed_all(seed_int)
					else:
						print("[INFO] torch.cuda.manual_seed_all not available.")
				else:
					print("[INFO] CUDA not available; skipping CUDA seed.")
			else:
				print("[INFO] torch.cuda.is_available not present.")
		else:
			print("[INFO] torch.cuda not present.")

		return int(seed_int)

	def expected_total_steps(
	 self
	) -> int:
		if getattr(self, "speed_profile", None) == "test":
			iters = 1
		else:
			iters = 2

		return int(self.num_boards) * (
		 1 + int(self.num_samples_per_board) * (iters + 1)
		)

	def _recursive_range_split(
	 self,
	 hands_set: List[str],
	 total_prob: float,
	 public_cards: List[str],
	) -> Dict[str, float]:
		if hasattr(self, "cfr_solver"):
			if self.cfr_solver is not None:
				return self.cfr_solver.recursive_range_sampling(
				 set(hands_set),
				 float(total_prob),
				 list(public_cards or []),
				)

		if not hands_set:
			return {}

		n = len(hands_set)
		u = float(total_prob) / float(n)

		out = {}
		for h in hands_set:
			out[h] = u

		return out

	def _assert_sampler_invariants(
	 self,
	 public_cards: List[str],
	 ranges_pair: List[Dict[int, float]],
	 pot_size: float,
	):
		ok_p = self._pot_norm_ok(pot_size)
		ok_b = self._board_one_hot_valid(public_cards)
		ok_m = self.is_range_mass_conserved(ranges_pair)
		if (not ok_p) or (not ok_b) or (not ok_m):
			msg = (
			 f"SamplerInvariantError "
			 f"pot_norm_ok={ok_p} "
			 f"board_1hot_ok={ok_b} "
			 f"mass_ok={ok_m}"
			)
			print(f"[ERROR] {msg}")
			raise ValueError(str(msg))
		return True


	def _board_one_hot_valid(
	 self,
	 board: List[str],
	) -> bool:
		v = board_one_hot(board)

		s = 0
		i = 0
		while i < len(v):
			if v[i] not in (0, 1):
				return False
			s += v[i]
			i += 1

		return s == len(board)

	def is_range_mass_conserved(
	 self,
	 ranges_pair: List[Dict[int, float]],
	 tol: float = EPS_SUM,
	) -> bool:
		r1, r2 = ranges_pair

		s1 = 0.0
		for _, p in dict(r1).items():
			s1 += float(p)

		s2 = 0.0
		for _, p in dict(r2).items():
			s2 += float(p)

		return (abs(s1 - 1.0) <= tol) and (abs(s2 - 1.0) <= tol)

	def _pot_norm_ok(
	 self,
	 pot_size: float,
	) -> bool:
		if self.player_stack is not None:
			total_initial = float(self.player_stack + self.player_stack)
		else:
			total_initial = 1.0

		if total_initial <= 0.0:
			total_initial = 1.0

		pn = float(pot_size) / total_initial
		return (pn > 0.0) and (pn <= 1.0)

	def map_hands_to_clusters(
	 self,
	 hand_probs: Dict[str, float],
	 clusters: Dict[int, set],
	) -> Dict[int, float]:
		out: Dict[int, float] = {}
		if not hand_probs:
			return out

		items = []
		for h, p in hand_probs.items():
			if isinstance(h, str):
				key = h
			else:
				key = " ".join(list(h))
			items.append((key, float(p)))
		items.sort(key=lambda t: t[0])

		cluster_ids_sorted = sorted(int(c) for c in clusters.keys())

		for key, p in items:
			cid = None
			for c in cluster_ids_sorted:
				hs = clusters.get(int(c), set())
				if key in hs:
					cid = int(c)
					break
			if cid is None:
				hh = hashlib.sha256(key.encode("utf-8")).hexdigest()
				cid = int(int(hh, 16) % int(self.num_clusters))
			if cid not in out:
				out[cid] = 0.0
			out[cid] += p

		s = 0.0
		for v in out.values():
			s += float(v)
		if s > 0.0:
			for k in list(out.keys()):
				out[k] = out[k] / s

		return out

	def map_hands_to_clusters_compat(
	 self,
	 hand_probs: Dict[str, float],
	 clusters: Dict[int, set],
	) -> Dict[int, float]:
		return self.map_hands_to_clusters(hand_probs, clusters)

	def _push_production_mode(self):
		snap = {
		 "speed_profile": getattr(self, "speed_profile", None),
		 "solver_depth_limit": getattr(
		  getattr(self, "cfr_solver", None),
		  "depth_limit",
		  None,
		 ),
		 "solver_total_iterations": getattr(
		  getattr(self, "cfr_solver", None),
		  "total_iterations",
		  None,
		 ),
		 "hc_profile": getattr(
		  getattr(self, "hand_clusterer", None),
		  "profile",
		  None,
		 ),
		 "label_pot_fraction": getattr(
		  getattr(self, "cfr_solver", None),
		  "_label_pot_fraction",
		  None,
		 ),
		}

		if hasattr(self, "speed_profile"):
			self.speed_profile = "bot"

		if hasattr(self, "hand_clusterer"):
			if hasattr(self.hand_clusterer, "profile"):
				self.hand_clusterer.profile = "bot"
			else:
				print("[INFO] hand_clusterer.profile not settable.")

		if hasattr(self, "cfr_solver"):
			if hasattr(self, "hand_clusterer"):
				self.cfr_solver.hand_clusterer = getattr(
				 self,
				 "hand_clusterer",
				 getattr(self.cfr_solver, "hand_clusterer", None),
				)
			if hasattr(self.cfr_solver, "__dict__"):
				self.cfr_solver._label_pot_fraction = True
			else:
				print("[INFO] cfr_solver has no dict; skipping label flag.")

		return snap

	def _pop_production_mode(
	 self,
	 snap,
	):
		if not isinstance(snap, dict):
			return False

		if "speed_profile" in snap:
			if hasattr(self, "speed_profile"):
				self.speed_profile = snap["speed_profile"]

		if "hc_profile" in snap:
			if hasattr(self, "hand_clusterer"):
				if hasattr(self.hand_clusterer, "profile"):
					self.hand_clusterer.profile = snap["hc_profile"]

		if hasattr(self, "cfr_solver"):
			if "solver_depth_limit" in snap:
				self.cfr_solver.depth_limit = snap["solver_depth_limit"]
			if "solver_total_iterations" in snap:
				self.cfr_solver.total_iterations = snap["solver_total_iterations"]
			if "label_pot_fraction" in snap:
				self.cfr_solver._label_pot_fraction = snap["label_pot_fraction"]

		return True
