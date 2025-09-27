from hunl.constants import SEED_DEFAULT
import json
import os
import yaml
from hunl.utils.seeded_rng import set_global_seed
from hunl.resolve_config import ResolveConfig


def _safe_int(x, default_val=None):
	if isinstance(x, bool):
		return int(x)
	if isinstance(x, int):
		return x
	if isinstance(x, float):
		return int(x)
	if isinstance(x, str):
		s = x.strip()
		sign_ok = (s.startswith("-") and s[1:].isdigit()) or s.isdigit()
		if sign_ok:
			return int(s)
		else:
			return default_val
	if hasattr(x, "__int__"):
		v = x.__int__()
		if isinstance(v, int):
			return v
		else:
			return default_val
	return default_val


def _safe_float(x, default_val=None):
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
			return default_val
	return default_val


def _safe_bool(x, default_val=False):
	if isinstance(x, bool):
		return x
	if isinstance(x, (int, float)):
		return bool(x)
	if isinstance(x, str):
		v = x.strip().lower()
		if v in ("1", "true", "t", "yes", "y", "on"):
			return True
		if v in ("0", "false", "f", "no", "n", "off"):
			return False
	return default_val


def _safe_str(x, default_val=""):
	if isinstance(x, str):
		return x
	if x is None:
		return default_val
	return str(x)


def _can_use_yaml():
	return hasattr(yaml, "safe_dump") and hasattr(yaml, "safe_load")


def save_config(config, path):
	if hasattr(config, "__dict__"):
		data = {k: getattr(config, k) for k in vars(config).keys()}
	else:
		if isinstance(config, dict):
			data = dict(config)
		else:
			data = {}

	dirn = os.path.dirname(path)
	if dirn:
		if not os.path.isdir(dirn):
			os.makedirs(dirn, exist_ok=True)

	ext = os.path.splitext(path)[1].lower()

	if ext in (".yml", ".yaml"):
		if _can_use_yaml():
			with open(path, "w") as f:
				yaml.safe_dump(data, f, sort_keys=True)
		else:
			with open(path, "w") as f:
				json.dump(data, f, indent=2, sort_keys=True)
	else:
		with open(path, "w") as f:
			json.dump(data, f, indent=2, sort_keys=True)

	return path


def load_config(path):
	ext = os.path.splitext(path)[1].lower()

	if ext in (".yml", ".yaml"):
		if _can_use_yaml():
			with open(path, "r") as f:
				out = yaml.safe_load(f)
			if out:
				return out
			else:
				return {}
		else:
			with open(path, "r") as f:
				return json.load(f)

	with open(path, "r") as f:
		return json.load(f)


def compose_resolve_config_from_yaml(
 abstraction_yaml_path,
 value_nets_yaml_path,
 solver_yaml_path,
 overrides=None,
):
	abst = load_config(abstraction_yaml_path) if abstraction_yaml_path else {}
	vnets = load_config(value_nets_yaml_path) if value_nets_yaml_path else {}
	solv = load_config(solver_yaml_path) if solver_yaml_path else {}

	if not isinstance(abst, dict):
		abst = {}
	if not isinstance(vnets, dict):
		vnets = {}
	if not isinstance(solv, dict):
		solv = {}

	seed = _resolve_seed_from_dicts(abst, vnets, solv)
	set_global_seed(seed)

	cfg = ResolveConfig.from_env(overrides or {})

	cfg = _apply_abstraction_to_cfg(cfg, abst, overrides=overrides or {})
	cfg = _apply_value_nets_to_cfg(cfg, vnets)
	cfg = _apply_solver_to_cfg(cfg, solv)

	runtime_overrides = _extract_runtime_overrides(solv)

	return {
	 "seed": int(seed),
	 "config": cfg,
	 "runtime_overrides": runtime_overrides,
	}


def _resolve_seed_from_dicts(abst, vnets, solv):
	seed = None

	for d in (abst, vnets, solv):
		if isinstance(d, dict):
			if "seed" in d:
				if d["seed"] is not None:
					cand = _safe_int(d["seed"], None)
					if cand is not None:
						seed = cand
						break

	if seed is None:
		env_seed = os.environ.get("FAST_TEST_SEED", str(SEED_DEFAULT))
		seed = _safe_int(env_seed, SEED_DEFAULT)

	return int(seed)


def _choose_num_clusters_from_abst(abst: dict) -> int | None:
	K_flop = None
	K_turn = None

	if isinstance(abst, dict):
		if "bucket_counts" in abst:
			if isinstance(abst["bucket_counts"], dict):
				if "flop" in abst["bucket_counts"]:
					K_flop = _safe_int(abst["bucket_counts"]["flop"], None)
				if "turn" in abst["bucket_counts"]:
					K_turn = _safe_int(abst["bucket_counts"]["turn"], None)

	if K_turn is not None:
		return int(K_turn)
	if K_flop is not None:
		return int(K_flop)
	return None


def _apply_abstraction_to_cfg(cfg, abst: dict, overrides: dict | None = None):
	K = _choose_num_clusters_from_abst(abst if isinstance(abst, dict) else {})

	if K is not None:
		if not (isinstance(overrides, dict) and ("num_clusters" in overrides)):
			cfg.num_clusters = int(K)

	if isinstance(abst, dict):
		if "tau_re" in abst:
			val = _safe_float(abst["tau_re"], None)
			if val is not None:
				cfg.tau_re = float(val)

		if "drift_sample_size" in abst:
			val = _safe_int(abst["drift_sample_size"], None)
			if val is not None:
				cfg.drift_sample_size = int(val)

		if "use_cfv_in_features" in abst:
			val = _safe_bool(abst["use_cfv_in_features"], None)
			if val is not None:
				cfg.use_cfv_in_features = bool(val)

	return cfg


def _apply_value_nets_to_cfg(cfg, vnets: dict):
	if isinstance(vnets, dict):
		if "outer_zero_sum" in vnets:
			val = _safe_bool(vnets["outer_zero_sum"], True)
			cfg.enforce_zero_sum_outer = bool(val)

		if "mc_samples_win" in vnets:
			val = _safe_int(vnets["mc_samples_win"], None)
			if val is not None:
				cfg.mc_samples_win = int(val)

		if "mc_samples_potential" in vnets:
			val = _safe_int(vnets["mc_samples_potential"], None)
			if val is not None:
				cfg.mc_samples_potential = int(val)

	lr_sched = vnets.get("lr_schedule", {}) if isinstance(vnets, dict) else {}

	if isinstance(lr_sched, dict):
		if "initial" in lr_sched:
			val = _safe_float(lr_sched["initial"], None)
			if val is not None:
				cfg.lr_initial = float(val)

		if "after" in lr_sched:
			val = _safe_float(lr_sched["after"], None)
			if val is not None:
				cfg.lr_after = float(val)

		if "drop_epoch" in lr_sched:
			val = _safe_int(lr_sched["drop_epoch"], None)
			if val is not None:
				cfg.lr_drop_epoch = int(val)

	if isinstance(vnets, dict):
		if "batch_size" in vnets:
			val = _safe_int(vnets["batch_size"], None)
			if val is not None:
				cfg.batch_size = int(val)

	return cfg


def _extract_runtime_overrides(solv: dict):
	runtime_overrides = {}

	if isinstance(solv, dict):
		if "iterations_per_round" in solv:
			if isinstance(solv["iterations_per_round"], dict):
				out = {}
				for k, v in solv["iterations_per_round"].items():
					ki = _safe_int(k, None)
					vi = _safe_int(v, None)
					if (ki is not None) and (vi is not None):
						out[int(ki)] = int(vi)
				if out:
					runtime_overrides["_round_iters"] = out

		if "round_actions" in solv:
			if isinstance(solv["round_actions"], dict):
				out = {}
				for r, vv in solv["round_actions"].items():
					ri = _safe_int(r, None)
					if ri is None:
						continue
					if isinstance(vv, dict):
						half_pot = _safe_bool(vv.get("half_pot", True), True)
						two_pot = _safe_bool(vv.get("two_pot", False), False)
						out[int(ri)] = {
						 "half_pot": bool(half_pot),
						 "two_pot": bool(two_pot),
						}
				if out:
					runtime_overrides["_round_actions"] = out

	return runtime_overrides


def _apply_solver_to_cfg(cfg, solv: dict):
	if isinstance(solv, dict):
		if "depth_limit" in solv:
			val = _safe_int(solv["depth_limit"], None)
			if val is not None:
				cfg.depth_limit = int(val)

		if "total_iterations" in solv:
			val = _safe_int(solv["total_iterations"], None)
			if val is not None:
				cfg.total_iterations = int(val)

		if "bet_size_mode" in solv:
			cfg.bet_size_mode = _safe_str(solv["bet_size_mode"], cfg.bet_size_mode)

		if "profile" in solv:
			cfg.profile = _safe_str(solv["profile"], cfg.profile)

	base_bf = _small_sparse_bet_fractions()

	if isinstance(solv, dict):
		if "bet_fractions" in solv:
			if isinstance(solv["bet_fractions"], dict):
				for r in (0, 1, 2, 3):
					key = str(r)
					if key in solv["bet_fractions"]:
						src = solv["bet_fractions"][key]
						if isinstance(src, (list, tuple)):
							vals = []
							for x in src:
								if x is not None:
									valf = _safe_float(x, None)
									if valf is not None:
										vals.append(float(valf))
							if vals:
								base_bf[int(r)] = list(vals)

	cfg.bet_fractions = base_bf
	return cfg


def _small_sparse_bet_fractions():
	return {
	 0: [0.5, 1.0],
	 1: [0.5, 1.0],
	 2: [0.5, 1.0],
	 3: [1.0],
	}

