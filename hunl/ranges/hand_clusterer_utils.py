import hashlib
import random
from typing import Any, Dict, List, Set, Tuple

import numpy as np


class HandClustererUtilsMixin:

	def get_cluster_ranges(self) -> Dict[int, float]:
		K = int(self.num_clusters)

		if K <= 0:
			return {}

		u = 1.0 / float(K)
		out: Dict[int, float] = {}

		i = 0
		while i < K:
			out[int(i)] = float(u)
			i += 1

		return out

	def hand_to_bucket(
		self,
		hand: str
	) -> int:
		if isinstance(hand, str):
			key = hand
		else:
			key = " ".join(list(hand))

		if getattr(self, "clusters", None):
			for cid, hset in self.clusters.items():
				if key in hset:
					return int(cid)
				else:
					if hand in hset:
						return int(cid)

		if getattr(self, "num_clusters", 0):
			h = hashlib.sha256(key.encode("utf-8")).hexdigest()
			return int(int(h, 16) % int(self.num_clusters))

		return 0

	def hands_to_bucket_range(
		self,
		hand_probs: Dict[str, float]
	) -> Dict[int, float]:
		out: Dict[int, float] = {}

		if not hand_probs:
			return out

		for h, p in list(hand_probs.items()):
			cid = self.hand_to_bucket(h)
			if cid in out:
				out[cid] = out[cid] + float(p)
			else:
				out[cid] = float(p)

		s = 0.0
		for v in out.values():
			s += float(v)

		if s > 0.0:
			for k in list(out.keys()):
				out[k] = out[k] / s

		return out

	def bucket_range_to_hand_weights(
		self,
		bucket_probs: Dict[int, float]
	) -> Dict[str, float]:
		out: Dict[str, float] = {}
		if not bucket_probs:
			return out

		if getattr(self, "clusters", None):
			for cid in sorted(int(k) for k in bucket_probs.keys()):
				p = float(bucket_probs[cid])
				hands = sorted(list(self.clusters.get(int(cid), [])))
				if not hands:
					continue
				w = p / float(len(hands))
				i = 0
				while i < len(hands):
					h = hands[i]
					if h in out:
						out[h] = out[h] + w
					else:
						out[h] = w
					i += 1

		s = 0.0
		for v in out.values():
			s += float(v)
		if s > 0.0:
			for k in list(out.keys()):
				out[k] = out[k] / s

		return out

	def persist_mapping(self) -> Dict[int, List[str]]:
		m: Dict[int, List[str]] = {}

		if not getattr(self, "clusters", None):
			return m

		for cid, hset in self.clusters.items():
			m[int(cid)] = sorted(list(hset))

		return m

	def load_mapping(
		self,
		mapping: Dict[int, List[str]]
	) -> bool:
		self.clusters = {}

		for k, v in dict(mapping).items():
			self.clusters[int(k)] = set(list(v))

		self.num_clusters = int(len(self.clusters))
		self.centroids = None
		self._last_features = None

		return True

	def _preflop_partition(
		self,
		hands: Any
	) -> Dict[int, Set[str]]:
		types: Dict[str, Set[str]] = {}

		if isinstance(hands, dict):
			iterable = hands.keys()
		else:
			iterable = hands

		for item in iterable:
			if isinstance(item, str):
				c1, c2 = item.split()
			else:
				c1, c2 = list(item)

			t = self._preflop_handtype(c1, c2)

			if t in types:
				types[t].add(f"{c1} {c2}")
			else:
				types[t] = {f"{c1} {c2}"}

		keys = sorted(types.keys())

		clusters: Dict[int, Set[str]] = {}

		i = 0
		while i < len(keys):
			k = keys[i]
			clusters[int(i)] = types[k]
			i += 1

		return clusters

	def _preflop_handtype(
		self,
		c1: str,
		c2: str
	) -> str:
		ranks = "23456789TJQKA"

		def r(c):
			return ranks.index(c[0])

		def suited(a, b):
			return a[1] == b[1]

		i = r(c1)
		j = r(c2)

		if i == j:
			return f"{ranks[i]}{ranks[j]}"

		x = max(i, j)
		y = min(i, j)
		s = "s" if suited(c1, c2) else "o"

		return f"{ranks[x]}{ranks[y]}{s}"

	def _opponent_range_signature(
		self,
		opponent_range: Dict[Any, float]
	) -> str:
		if opponent_range is None:
			return "none"

		acc: Dict[str, float] = {}

		for k, v in dict(opponent_range).items():
			if isinstance(k, (int, np.integer)):
				key = f"c#{int(k)}"
			else:
				s = str(k)

				if " " in s:
					p = s.split()
					if len(p) == 2:
						a, b = p[0], p[1]
						if a > b:
							tmp = a
							a = b
							b = tmp
						s = f"{a} {b}"

				key = s

			if key in acc:
				acc[key] = acc[key] + float(v)
			else:
				acc[key] = float(v)

		items = []
		for key, w in acc.items():
			items.append((key, f"{float(w):.6f}"))

		items.sort(key=lambda x: x[0])

		payload_parts = []
		i = 0
		while i < len(items):
			k, pv = items[i]
			payload_parts.append(f"{k}:{pv}")
			i += 1

		payload = "|".join(payload_parts).encode("utf-8")
		return hashlib.md5(payload).hexdigest()

	def _compute_drift(
		self,
		new_features: Dict[str, np.ndarray]
	) -> float:
		if getattr(self, "_last_features", None) is None:
			return None

		prev = self._last_features
		keys = sorted(set(prev.keys()) & set(new_features.keys()))

		if not keys:
			return None

		target_n = int(getattr(self, "drift_sample_size", 200))
		if len(keys) < target_n:
			n = len(keys)
		else:
			n = target_n

		sel = keys[:n]

		acc = 0.0
		i = 0
		while i < len(sel):
			h = sel[i]
			f0 = np.asarray(prev[h], dtype=float)
			f1 = np.asarray(new_features[h], dtype=float)
			d = np.linalg.norm(f1 - f0)
			acc += float(d)
			i += 1

		if sel:
			return acc / float(len(sel))
		else:
			return None

	def _deterministic_seed_for_clustering(
		self,
		board: List[str],
		opponent_range: Dict[Any, float],
		pot_size: float
	) -> int:
		board_key = ",".join(list(board or []))
		h = hashlib.md5(board_key.encode("utf-8")).hexdigest()[:8]
		val = int(h, 16)
		return int(val % (2**32 - 1))

	def _stable_seed(
		self,
		hand: Any,
		board: List[str]
	) -> int:
		if isinstance(hand, str):
			hand_str = hand
		else:
			hand_str = " ".join(hand)

		key_bytes = (hand_str + "|" + ",".join(board)).encode("utf-8")
		h = hashlib.md5(key_bytes).hexdigest()[:8]
		return int(h, 16)

	def _maybe_sample_items(
		self,
		items: Any,
		seed: int
	):
		if isinstance(items, dict):
			pairs = list(items.items())
		else:
			pairs = list(items)

		def _key_of(x):
			k = x[0]
			if isinstance(k, (int, np.integer)):
				return (0, int(k))
			else:
				return (1, str(k))

		pairs.sort(key=_key_of)

		limit = getattr(self, "opp_sample_size", None)

		if limit is not None:
			if int(limit) > 0:
				if len(pairs) > int(limit):
					rng = random.Random(seed)
					pairs = rng.sample(pairs, int(limit))

		return pairs
