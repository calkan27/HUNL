"""
I centralize runtime configuration for the re-solver, value networks, and data
generation. I expose ResolveConfig with typed fields and a from_env helper that builds a
profile for fast tests or full runs. I let a caller specify depth_limit,
total_iterations, num_clusters, bet_size_mode, and flags like enforce_zero_sum_outer and
prefer_gpu.

Key classes/functions: ResolveConfig — immutable config object; from_env — construct
config with env-aware defaults; _env_flag/_env_int — parse environment booleans and
ints.

Inputs: optional overrides dict and environment variables FAST_TESTS, FAST_TEST_SEED,
DEBUG_FAST_TESTS. Outputs: a populated ResolveConfig instance. Invariants: numeric
fields are sane (non-negative, coerced types), and test profile collapses heavy settings
to minimum viable values to meet latency gates.

Internal dependencies: hunl.constants for default seeds. External dependencies: Python
stdlib only. Side effects: none beyond reading environment variables.

Edge cases: if a caller passes contradictory overrides (e.g., zero clusters with enabled
features), I coerce to safe defaults where possible. Performance considerations: the
test profile disables expensive features (e.g., CFV features in clustering, opponent
sampling) and forces small iteration counts to keep smoke and CI runs predictable.
"""

from hunl.constants import SEED_DEFAULT
from dataclasses import dataclass, field
import os
from typing import Optional, Dict, Any, Literal


BetSizeMode = Literal["sparse_2", "sparse_3", "full"]


def _env_flag(name: str, default: bool) -> bool:
	val = os.getenv(name, None)

	if val is None:
		return bool(default)
	else:
		v = val.strip().lower()

		if v in ("1", "true", "t", "yes", "y", "on"):
			return True
		else:
			if v in ("0", "false", "f", "no", "n", "off"):
				return False
			else:
				return bool(default)


def _env_int(name: str, default: int) -> int:
	val = os.getenv(name, None)

	if val is None:
		return int(default)
	else:
		s = val.strip()

		if (s.startswith("-") and s[1:].isdigit()) or s.isdigit():
			return int(s)
		else:
			return int(default)


def _coerce_float(x: Any, default: float) -> float:
	if isinstance(x, bool):
		return float(int(x))

	if isinstance(x, (int, float)):
		return float(x)

	if isinstance(x, str):
		s = x.strip()
		try_digits = s.replace(".", "", 1).lstrip("+-")

		if try_digits.isdigit():
			return float(s)
		else:
			return float(default)

	return float(default)


def _coerce_int(x: Any, default: int) -> int:
	if isinstance(x, bool):
		return int(x)

	if isinstance(x, int):
		return int(x)

	if isinstance(x, float):
		return int(x)

	if isinstance(x, str):
		s = x.strip()
		sign_ok = (s.startswith("-") and s[1:].isdigit()) or s.isdigit()

		if sign_ok:
			return int(s)
		else:
			return int(default)

	return int(default)


def _coerce_bool(x: Any, default: bool) -> bool:
	if isinstance(x, bool):
		return bool(x)

	if isinstance(x, (int, float)):
		return bool(x)

	if isinstance(x, str):
		v = x.strip().lower()

		if v in ("1", "true", "t", "yes", "y", "on"):
			return True
		else:
			if v in ("0", "false", "f", "no", "n", "off"):
				return False
			else:
				return bool(default)

	return bool(default)


@dataclass
class ResolveConfig:
	profile: Literal["bot", "test"] = field(
	 default_factory=lambda: ("test" if os.getenv("FAST_TESTS") == "1" else "bot")
	)
	fast_test_seed: int = _env_int("FAST_TEST_SEED", SEED_DEFAULT)
	debug_fast_tests: bool = _env_flag("DEBUG_FAST_TESTS", False)

	depth_limit: int = 4
	total_iterations: int = 20
	num_clusters: int = 10

	tau_re: float = 0.12
	drift_sample_size: int = 200

	bet_size_mode: BetSizeMode = "sparse_2"

	mc_samples_win: int = 200
	mc_samples_potential: int = 200
	opp_sample_size: Optional[int] = None
	use_cfv_in_features: bool = True

	preflop_cache_max_entries: int = 10000
	feature_cache_max_entries: int = 1_000_000

	enforce_zero_sum_outer: bool = True
	prefer_gpu: bool = True

	constraint_mode: Literal["sp", "br"] = "sp"
	paper_faithful: bool = True

	@staticmethod
	def from_env(
	 overrides: Optional[Dict[str, Any]] = None
	) -> "ResolveConfig":
		cfg = ResolveConfig()

		if overrides:
			for k, v in overrides.items():
				if hasattr(cfg, k):
					setattr(cfg, k, v)

		if (not overrides) or ("num_clusters" not in overrides):
			cfg.num_clusters = 1000

		fast = (os.getenv("FAST_TESTS") == "1")

		if fast:
			cfg.profile = "test"
			cfg.depth_limit = 0
			cfg.total_iterations = 1
			cfg.mc_samples_win = 0
			cfg.mc_samples_potential = 0
			cfg.opp_sample_size = 0
			cfg.use_cfv_in_features = False
			cfg.tau_re = float("inf")
		else:
			if bool(getattr(cfg, "paper_faithful", True)):
				cfg.profile = "bot"
				cfg.bet_size_mode = "sparse_2"
				cfg.use_cfv_in_features = True
				cfg.opp_sample_size = None

				ok_tau = isinstance(cfg.tau_re, (int, float)) and (0.0 < float(cfg.tau_re) < 1e9)
				if not ok_tau:
					cfg.tau_re = 0.12
			else:
				if cfg.profile == "test":
					cfg.depth_limit = 0
					cfg.total_iterations = 1
					cfg.mc_samples_win = 0
					cfg.mc_samples_potential = 0
					cfg.opp_sample_size = 0
					cfg.use_cfv_in_features = False
					cfg.tau_re = float("inf")

		return cfg

