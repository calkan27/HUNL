"""
I build and persist metadata records that describe dataset generation runs. I encode
action sets (bet fractions and per-round flags), outer zero-sum configuration, cluster
counts, stage, and seeds. I normalize inputs and defend against missing fields to keep
manifests stable across profiles.

Key functions: make_manifest — compose schema fields, action_set, and extras;
save_manifest — write JSON; internal helpers (_get_outer_zero_sum_flag,
_get_bet_fractions, _get_round_flags) that pull information from the owning
generator/solver; _normalize_bet_fractions — sanitize structures.

Inputs: a data generator instance, stage label, seed, and optional extras dict; output
path for saving. Outputs: dict manifest and JSON file.

Dependencies: stdlib json/os/time. Invariants: numeric keys are serialized as strings
when appropriate; default action set includes CALL/FOLD and sized bets; include_all_in
is always present for clarity. Performance: I avoid touching large tensors and only
traverse small config dicts.
"""

import os
import json
import time
from typing import Dict, Any


def _get_outer_zero_sum_flag(data_generator) -> bool:
	cfg = getattr(data_generator, "_config", None)

	if cfg is not None:
		if hasattr(cfg, "enforce_zero_sum_outer"):
			return bool(getattr(cfg, "enforce_zero_sum_outer"))
		else:
			return True
	else:
		return True


def _default_bet_fractions() -> Dict[int, list]:
	return {0: [1.0], 1: [1.0], 2: [1.0], 3: [1.0]}


def _normalize_bet_fractions(bf) -> Dict[int, list]:
	out: Dict[int, list] = {}
	if isinstance(bf, dict):
		for k, v in bf.items():
			key = str(int(k))
			if isinstance(v, (list, tuple)):
				out[key] = [float(x) for x in v]
			else:
				out[key] = [1.0]
	else:
		out = {str(i): [1.0] for i in (0, 1, 2, 3)}
	if len(out) == 0:
		out = {str(i): [1.0] for i in (0, 1, 2, 3)}
	return out



def _get_bet_fractions(data_generator) -> Dict[int, list]:
	cfg = getattr(data_generator, "_config", None)

	if cfg is not None:
		if hasattr(cfg, "bet_fractions"):
			return _normalize_bet_fractions(getattr(cfg, "bet_fractions"))
		else:
			return _default_bet_fractions()
	else:
		return _default_bet_fractions()


def _default_round_flags() -> Dict[int, Dict[str, bool]]:
	return {
		0: {"half_pot": True, "two_pot": False},
		1: {"half_pot": False, "two_pot": False},
		2: {"half_pot": False, "two_pot": False},
		3: {"half_pot": False, "two_pot": False},
	}


def _get_round_flags(data_generator) -> Dict[int, Dict[str, bool]]:
	solver = getattr(data_generator, "cfr_solver", None)
	if solver is None:
		return {str(i): {"half_pot": True, "two_pot": False} for i in (0, 1, 2, 3)}
	else:
		ra = getattr(solver, "_round_actions", None)
		if isinstance(ra, dict):
			flags: Dict[int, Dict[str, bool]] = {}
			for r, v in ra.items():
				if isinstance(v, dict):
					half = bool(v.get("half_pot", True))
					two = bool(v.get("two_pot", False))
					flags[str(int(r))] = {"half_pot": half, "two_pot": two}
				else:
					flags[str(int(r))] = {"half_pot": True, "two_pot": False}
			if len(flags) == 0:
				return {str(i): {"half_pot": True, "two_pot": False} for i in (0, 1, 2, 3)}
			else:
				return flags
		else:
			return {str(i): {"half_pot": True, "two_pot": False} for i in (0, 1, 2, 3)}



def _spec_base(data_generator, stage, seed) -> Dict[str, Any]:
	spec: Dict[str, Any] = {}
	spec["schema"] = "cfv.manifest.v1"
	spec["created_at"] = int(time.time())
	spec["stage"] = str(stage)
	spec["seed"] = int(seed)
	spec["num_clusters"] = int(getattr(data_generator, "num_clusters", 0))

	if hasattr(data_generator, "pot_sampler_spec"):
		spec["pot_sampler"] = data_generator.pot_sampler_spec()
	else:
		spec["pot_sampler"] = []

	if hasattr(data_generator, "range_generator_spec"):
		spec["range_generator"] = data_generator.range_generator_spec()
	else:
		spec["range_generator"] = {"name": "", "params": {}}

	return spec


def _attach_action_set(spec: Dict[str, Any], data_generator) -> None:
	outer = _get_outer_zero_sum_flag(data_generator)
	bf = _get_bet_fractions(data_generator)
	flags = _get_round_flags(data_generator)
	spec["outer_zero_sum"] = bool(outer)
	spec["action_set"] = {
			"bet_fractions": {str(k): [float(x) for x in v] for k, v in bf.items()},
			"round_flags": {str(k): {"half_pot": bool(v.get("half_pot", True)),
				"two_pot": bool(v.get("two_pot", False))} for k, v in flags.items()},
			"include_all_in": True,
	}

def _merge_extras(spec: Dict[str, Any], extras) -> None:
	if isinstance(extras, dict):
		for k, v in extras.items():
			spec[k] = v
	else:
		pass


def make_manifest(data_generator, stage, seed, extras=None) -> Dict[str, Any]:
	spec = _spec_base(data_generator, stage, seed)
	_attach_action_set(spec, data_generator)
	_merge_extras(spec, extras)
	return spec


def save_manifest(manifest, path):
	dirn = os.path.dirname(path)

	if dirn:
		if not os.path.isdir(dirn):
			os.makedirs(dirn, exist_ok=True)
	else:
		pass

	with open(path, "w") as f:
		json.dump(dict(manifest), f, indent=2, sort_keys=True)

	return path
