"""
Diagnostics and soundness-related summaries for the CFR solver. This mixin surfaces
recent run metadata (depth, iterations, constraint mode), regret/entropy measures,
zero-sum residual statistics, and preflop cache counters. It also aggregates opponent
CFV upper bounds and worst-case values used in range gadgets.

Key methods: get_last_diagnostics (emit dict with metrics and cache stats),
set_soundness_constants (record k1/k2-style constants), internal helpers for regret L2
norm, average strategy entropy, zero-sum residual magnitude over collected samples,
mapping CFV vectors to per-cluster maxima, and extracting worst-case values from tracked
constraints.

Inputs: recent CFRValues state, range gadget tracking, zero-sum residual samples, and
preflop cache stats. Outputs: serializable dicts suitable for logs and CLI displays.

Invariants: numeric fields are floats/ints ready for JSON; absent data yields
conservative zeros; upper-bound aggregations are monotone with respect to observed
maxima. Performance: computations are linear in clusters/actions for the current node
and designed to be negligible compared to re-solving iterations.
"""


from hunl.constants import EPS_MASS
from typing import Dict, Any
import math


class CFRSolverDiagnosticsMixin:

	def get_last_diagnostics(self) -> Dict[str, Any]:
		out: Dict[str, Any] = {}

		if hasattr(self, "_last_diagnostics"):
			if isinstance(self._last_diagnostics, dict):
				for k, v in self._last_diagnostics.items():
					out[k] = v

		if hasattr(self, "_soundness"):
			if isinstance(self._soundness, dict):
				if "k1" in self._soundness:
					out["k1"] = float(self._soundness.get("k1", 0.0))
				if "k2" in self._soundness:
					out["k2"] = float(self._soundness.get("k2", 0.0))

		stats = dict(getattr(self, "_preflop_cache_stats", {}))

		if stats:
			out["preflop_cache"] = {
			 "hits": int(stats.get("hits", 0)),
			 "misses": int(stats.get("misses", 0)),
			 "puts": int(stats.get("puts", 0)),
			 "evictions": int(stats.get("evictions", 0)),
			}

		return out

	def set_soundness_constants(self, k1: float, k2: float) -> None:
		self._soundness = {"k1": float(k1), "k2": float(k2)}

	def _compute_regret_l2(self, node) -> float:
		values = self.cfr_values.get(node, None)

		if values is None:
			return 0.0

		acc = 0.0

		for _, reg in values.cumulative_positive_regret.items():
			for r in reg:
				f = float(r)
				acc += f * f

		return float(acc ** 0.5)

	def _compute_avg_strategy_entropy(self, node) -> float:
		values = self.cfr_values.get(node, None)

		if values is None:
			return 0.0

		total_h = 0.0
		count = 0

		for cid in values.cumulative_strategy.keys():
			p = values.get_average_strategy(cid)
			s = 0.0

			for x in p:
				px = float(x)

				if px > 0.0:
					s -= px * math.log(px + EPS_MASS)

			total_h += s
			count += 1

		if count > 0:
			return float(total_h / float(count))
		else:
			return 0.0

	def _compute_zero_sum_residual(self, node) -> float:
		buf = getattr(self, "_zs_residual_samples", None)

		if not buf:
			return 0.0

		mx = 0.0

		for v in buf:
			av = abs(float(v))

			if av > mx:
				mx = av

		return float(mx)

	def _upper_from_cfvs(self, cfv_dict) -> dict:
		out = {}

		for cid, vec in dict(cfv_dict).items():
			if isinstance(vec, (list, tuple)):
				if len(vec) > 0:
					mx = None

					for v in vec:
						fv = float(v)

						if (mx is None) or (fv > mx):
							mx = fv

					if mx is not None:
						out[int(cid)] = float(mx)
					else:
						out[int(cid)] = 0.0
				else:
					out[int(cid)] = 0.0
			else:
				if isinstance(vec, (int, float)):
					out[int(cid)] = float(vec)
				else:
					out[int(cid)] = 0.0

		return out

	def _worst_from_constraints(self, node, us: int) -> float:
		key = self._state_key(node)

		if not hasattr(self, "opponent_cfv_upper_tracking"):
			return 0.0

		tracking = getattr(self, "opponent_cfv_upper_tracking", {})

		if key not in tracking:
			return 0.0

		upper = tracking[key]
		mx = None

		for v in upper.values():
			fv = float(v)

			if (mx is None) or (fv > mx):
				mx = fv

		if mx is None:
			return 0.0
		else:
			return -float(mx)

	def _opponent_node_value_from_upper_bounds(self, node, agent_player):
		if not hasattr(self, "opponent_cfv_upper_tracking"):
			return 0.0

		if hasattr(self, "_state_key"):
			key = self._state_key(node)
		else:
			key = None

		if key is None:
			return 0.0

		if key not in self.opponent_cfv_upper_tracking:
			return 0.0

		upper = self.opponent_cfv_upper_tracking[key]

		if not upper:
			return 0.0

		mx = None

		for v in upper.values():
			fv = float(v)

			if (mx is None) or (fv > mx):
				mx = fv

		if mx is None:
			return 0.0
		else:
			return -float(mx)

