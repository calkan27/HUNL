from hunl.constants import SEED_DEFAULT
import os
import time
import random
import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Set, Optional

import numpy as np
from hunl.engine.poker_utils import DECK  

from hunl.ranges.hand_clusterer_features import HandClustererFeaturesMixin
from hunl.ranges.hand_clusterer_utils import HandClustererUtilsMixin

if TYPE_CHECKING:
	from hunl.solving.cfr_solver import CFRSolver


class HandClusterer(HandClustererFeaturesMixin, HandClustererUtilsMixin):
	def __init__(
	 self,
	 cfr_solver: "CFRSolver",
	 num_clusters: int = 1000,
	 max_iterations: int = 3,
	 tau_re: float = 0.12,
	 drift_sample_size: int = 200,
	 profile: Optional[str] = None,
	 opp_sample_size: Optional[int] = None,
	 use_cfv_in_features: bool = True,
	 config: Optional[Any] = None,
	):
		self._config = config
		if self._config is not None:
			self.cfr_solver = cfr_solver
			self.num_clusters = int(self._config.num_clusters)
			self.max_iterations = int(max_iterations)
			self.clusters = None
			self.centroids = None
			self._cache = {}
			self._cache_hits = 0
			self._cache_misses = 0
			self._mc_samples_win = int(self._config.mc_samples_win)
			self._mc_samples_potential = int(self._config.mc_samples_potential)
			self.tau_re = float(self._config.tau_re)
			self.drift_sample_size = int(self._config.drift_sample_size)
			self._last_features = None
			self.profile = str(self._config.profile)
			self._fast_test_frozen_clusters = None
			self._fast_test_seed = int(getattr(self._config, "fast_test_seed", SEED_DEFAULT))
			self._fast_test_initialized = False
			if str(self.profile) == "test":
				self.opp_sample_size = 0 if self._config.opp_sample_size is None else int(self._config.opp_sample_size)
				self.use_cfv_in_features = False
				self._mc_samples_win = 0
				self._mc_samples_potential = 0
			else:
				self.opp_sample_size = int(self._config.opp_sample_size) if self._config.opp_sample_size is not None else None
				self.use_cfv_in_features = bool(self._config.use_cfv_in_features)
		else:
			self.cfr_solver = cfr_solver
			self.num_clusters = int(num_clusters)
			self.max_iterations = int(max_iterations)
			self.clusters = None
			self.centroids = None
			self._cache = {}
			self._cache_hits = 0
			self._cache_misses = 0
			self._mc_samples_win = 200
			self._mc_samples_potential = 200
			self.tau_re = float(tau_re)
			self.drift_sample_size = int(drift_sample_size)
			self._last_features = None
			env_fast = os.getenv("FAST_TESTS") == "1"
			self.profile = profile if profile is not None else ("test" if env_fast else "bot")
			self._fast_test_frozen_clusters = None
			self._fast_test_seed = int(os.getenv("FAST_TEST_SEED", str(SEED_DEFAULT)))
			self._fast_test_initialized = False
			if self.profile == "test":
				self._mc_samples_win = 0
				self._mc_samples_potential = 0
				if opp_sample_size is None:
					self.opp_sample_size = 0
				else:
					self.opp_sample_size = int(opp_sample_size)
				self.use_cfv_in_features = False
			else:
				self.opp_sample_size = int(opp_sample_size) if opp_sample_size is not None else None
				self.use_cfv_in_features = bool(use_cfv_in_features)


	def cluster_hands(
	 self,
	 hands: Any,
	 board: List[str],
	 opponent_range: Dict[Any, float],
	 pot_size: float
	) -> Dict[int, Set[str]]:
		if self.profile == "test":
			ft = self._cluster_hands_fast_test(hands)
			if ft is not None:
				return ft
			else:
				print("[FAST-TEST] Proceeding with full clustering path.")

		start_time = time.time()
		hits_before = self._cache_hits
		misses_before = self._cache_misses

		feats = self._features_for_hands(
		 hands=hands,
		 board=board,
		 opponent_range=opponent_range,
		 pot_size=pot_size
		)

		N = len(feats)

		if N == 0:
			self.clusters = {}
			self.centroids = None
			self._last_features = {}
			return {}

		Kcfg = int(self.num_clusters)
		if N < Kcfg:
			Kcfg = N
			self.num_clusters = Kcfg
		else:
			pass

		reused = self._reuse_previous_clusters_if_drift_small(
		 feats=feats,
		 hands=hands
		)
		if reused is not None:
			return reused
		else:
			pass

		keys = list(feats.keys())
		X = np.stack([feats[k] for k in keys], axis=0)

		C = self._kmeanspp_init(
		 X=X,
		 Kcfg=Kcfg,
		 board=board,
		 opponent_range=opponent_range,
		 pot_size=pot_size
		)

		assign, C = self._lloyd_assign_and_update(
		 X=X,
		 C=C,
		 max_iterations=int(self.max_iterations)
		)

		clusters_raw: Dict[int, Set[str]] = {i: set() for i in range(Kcfg)}

		for i in range(N):
			if assign[i] >= 0:
				k = int(assign[i])
			else:
				d2 = np.sum((X[i:i + 1] - C) ** 2, axis=1)
				k = int(np.argmin(d2))
			clusters_raw[k].add(keys[i])

		for k in range(Kcfg):
			if k not in clusters_raw:
				clusters_raw[k] = set()

		self.clusters = {int(k): set(v) for k, v in clusters_raw.items()}
		self.centroids = C.copy()
		self._last_features = feats

		total_time2 = time.time() - start_time
		dh = self._cache_hits - hits_before
		dm = self._cache_misses - misses_before

		print(f"[INFO] Clustering completed in {total_time2:.4f} seconds")
		print(f"[CACHE] hits={self._cache_hits} misses={self._cache_misses} (+{dh}/+{dm})")

		return self.clusters

	def fit(
	 self,
	 hands: Any,
	 board: List[str],
	 opponent_range: Dict[Any, float],
	 pot_size: float,
	 K: Optional[int] = None,
	 stage: Optional[str] = None,
	) -> Dict[int, Set[str]]:

		b = list(board) if board is not None else []
		if stage is None:
			stage = "preflop" if len(b) == 0 else ("flop" if len(b) == 3 else ("turn" if len(b) == 4 else "river"))

		if stage == "preflop":
			part = self._preflop_partition(hands)
			self.clusters = part
			self.num_clusters = len(self.clusters)
			self.centroids = None
			self._last_features = None
			return dict(self.clusters)

		k_default = 1000
		K_eff = int(K if K is not None else k_default)

		if isinstance(hands, (set, dict)):
			N = len(hands)
		else:
			N = len(list(hands))

		if K_eff > N:
			K_eff = N
		self.num_clusters = int(K_eff)

		cl = self.cluster_hands(hands, b, opponent_range, pot_size)
		self.clusters = cl
		return dict(self.clusters)

	def assign(self, hand: str, board: Optional[List[str]]) -> int:

		if self.clusters is None or len(self.clusters) == 0:
			return 0
		key = hand if isinstance(hand, str) else " ".join(list(hand))
		b = list(board) if board is not None else []

		if len(b) == 0:
			h1, h2 = key.split()
			t = self._preflop_handtype(h1, h2)
			for cid, hs in self.clusters.items():
				if t in hs:
					return int(cid)

		for cid, hs in self.clusters.items():
			if key in hs or hand in hs:
				return int(cid)

		return self.hand_to_bucket(hand)

	def set_num_clusters(self, K: int) -> int:
		self.num_clusters = int(K)
		return int(self.num_clusters)

	def hand_to_bucket_on_board(self, hand, board):
		key = hand if isinstance(hand, str) else " ".join(list(hand))
		b = list(board) if board is not None else []
		if hasattr(self, "assign"):
			return int(self.assign(key, b))
		return int(self.hand_to_bucket(key))

	def _cluster_hands_fast_test(
	 self,
	 hands: Any
	) -> Optional[Dict[int, Set[str]]]:
		ok_env = (os.getenv("FAST_TESTS") == "1")
		if getattr(self, "_config", None) is not None:
			ok_cfg = bool(getattr(self._config, "debug_fast_tests", False))
		else:
			ok_cfg = False

		ok = (ok_env or ok_cfg)

		if not ok:
			print("[FAST-TEST] Disabled: FAST_TESTS not set and debug_fast_tests is False.")
			return None

		if self._fast_test_frozen_clusters is not None:
			return self._fast_test_frozen_clusters

		if isinstance(hands, (set, dict)):
			if isinstance(hands, dict):
				hands_list = sorted(list(hands.keys()))
			else:
				hands_list = sorted(list(hands))
		else:
			hands_list = list(hands)
			hands_list.sort()

		K = int(self.num_clusters)
		N = len(hands_list)
		clusters = {i: set() for i in range(K)}

		if N == 0:
			self._fast_test_frozen_clusters = clusters
			self.clusters = clusters
			self.centroids = None
			self._fast_test_initialized = True
			return clusters

		rng = random.Random(self._fast_test_seed)

		if N < K:
			perm = list(range(K))
			rng.shuffle(perm)
			for idx, hand in enumerate(hands_list):
				clusters[perm[idx]] = {hand}
		else:
			for hand in hands_list:
				if isinstance(hand, str):
					key = hand
				else:
					key = " ".join(list(hand))
				h = hashlib.sha256(key.encode("utf-8")).hexdigest()
				cls = int(h, 16) % K
				clusters[cls].add(hand)

		for k in range(K):
			if k not in clusters:
				clusters[k] = set()

		empties = [k for k, v in clusters.items() if len(v) == 0]

		if len(empties) > 0:
			donors = sorted(
			 [(k, len(v)) for k, v in clusters.items() if len(v) > 1],
			 key=lambda x: -x[1]
			)
			for e in empties:
				if len(donors) == 0:
					break
				dk, _ = donors[0]
				move = sorted(clusters[dk])[0]
				clusters[dk].remove(move)
				clusters[e].add(move)
				donors = sorted(
				 [(k, len(v)) for k, v in clusters.items() if len(v) > 1],
				 key=lambda x: -x[1]
				)

		self._fast_test_frozen_clusters = clusters
		self.clusters = clusters
		self.centroids = None
		self._last_features = None
		self._fast_test_initialized = True
		return clusters

	def _features_for_hands(
	 self,
	 hands: Any,
	 board: List[str],
	 opponent_range: Dict[Any, float],
	 pot_size: float
	) -> Dict[str, np.ndarray]:
		feats: Dict[str, np.ndarray] = {}

		if isinstance(hands, dict):
			iterable = hands.keys()
		else:
			iterable = hands

		for h in iterable:
			f = self.calculate_hand_features(
			 h,
			 board,
			 opponent_range,
			 pot_size
			)
			feats[h] = np.asarray(f, dtype=float)

		return feats

	def _reuse_previous_clusters_if_drift_small(
	 self,
	 feats: Dict[str, np.ndarray],
	 hands: Any
	) -> Optional[Dict[int, Set[str]]]:
		drift = self._compute_drift(feats)
		ok_clusters = isinstance(self.clusters, dict) and bool(self.clusters)

		if drift is None:
			return None
		else:
			if drift >= float(getattr(self, "tau_re", 0.12)):
				return None

		if not ok_clusters:
			return None

		if isinstance(hands, dict):
			current_set = set(hands.keys())
		else:
			current_set = set(hands)

		reused: Dict[int, Set[str]] = {int(k): set() for k in range(self.num_clusters)}
		assigned = set()

		for cid, hset in self.clusters.items():
			keep = set()
			for h in hset:
				if h in current_set:
					keep.add(h)
					assigned.add(h)
			reused[int(cid)] = keep

		unassigned = current_set - assigned

		if len(unassigned) > 0:
			keys = list(feats.keys())
			X = np.stack([feats[k] for k in keys], axis=0)

			if isinstance(self.centroids, np.ndarray):
				C = np.asarray(self.centroids, dtype=float)
			else:
				C = None

			for h in sorted(unassigned):
				if C is not None:
					if C.size > 0:
						x = feats[h].reshape(1, -1)
						diff = C - x
						d2 = np.sum(diff * diff, axis=1)
						cid = int(np.argmin(d2))
					else:
						if isinstance(h, str):
							s = h
						else:
							s = " ".join(list(h))
						hh = hashlib.sha256(s.encode("utf-8")).hexdigest()
						cid = int(int(hh, 16) % int(self.num_clusters))
				else:
					if isinstance(h, str):
						s = h
					else:
						s = " ".join(list(h))
					hh = hashlib.sha256(s.encode("utf-8")).hexdigest()
					cid = int(int(hh, 16) % int(self.num_clusters))

				reused[cid].add(h)

		for k in range(self.num_clusters):
			if k not in reused:
				reused[k] = set()

		self._last_features = feats
		self.clusters = reused
		return self.clusters

	def _kmeanspp_init(
	 self,
	 X: np.ndarray,
	 Kcfg: int,
	 board: List[str],
	 opponent_range: Dict[Any, float],
	 pot_size: float
	) -> np.ndarray:
		N = X.shape[0]
		rs = np.random.RandomState(
		 self._deterministic_seed_for_clustering(
		  board,
		  opponent_range,
		  pot_size
		 )
		)

		cent_idx: List[int] = []
		i0 = int(rs.randint(0, N))
		cent_idx.append(i0)

		D = np.full((N,), np.inf, dtype=float)

		while len(cent_idx) < Kcfg:
			c = X[cent_idx[-1]]
			diff = X - c
			d2 = np.sum(diff * diff, axis=1)
			D = np.minimum(D, d2)

			total = float(np.sum(D))
			if total > 0.0:
				denom = total
			else:
				denom = 1.0

			probs = D / denom
			r = float(rs.rand())
			acc = 0.0
			pick = 0

			for i in range(N):
				acc += probs[i]
				if r <= acc:
					pick = i
					break

			if pick in cent_idx:
				pick = int(rs.randint(0, N))

			cent_idx.append(int(pick))

		C = X[cent_idx, :].copy()
		return C

	def _lloyd_assign_and_update(
	 self,
	 X: np.ndarray,
	 C: np.ndarray,
	 max_iterations: int
	) -> (np.ndarray, np.ndarray):
		N = X.shape[0]
		Kcfg = C.shape[0]
		assign = np.full((N,), -1, dtype=int)
		it = 0
		max_it = int(max_iterations)

		while it < max_it:
			it += 1

			diff = X[:, None, :] - C[None, :, :]
			dist2 = np.sum(diff * diff, axis=2)
			new_assign = np.argmin(dist2, axis=1).astype(int)

			if np.array_equal(assign, new_assign):
				break
			else:
				assign = new_assign

			for k in range(Kcfg):
				idx = np.where(assign == k)[0]
				if idx.size > 0:
					C[k, :] = np.mean(X[idx, :], axis=0)
				else:
					pass

		return assign, C

