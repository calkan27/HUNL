"""
I save and load counterfactual-value bundles that package stage models, input layout
metadata, and optional cluster mappings. I help the solver replace its nets and clusters
at runtime without changing shapes by computing input slice boundaries and performing
shape-aware state-dict filtering.

Key functions: save_cfv_bundle — persist stage nets, meta (num_clusters,
board_one_hot_dim, range layout), and optional mapping; load_cfv_bundle — load tensors
and build CounterfactualValueNetwork or a CompatLinearCFV fallback;
_compute_input_slices/_upgrade_input_meta — derive consistent input indices and meta
fields.

Inputs: dict of stage→nn.Module, mapping dict, input meta, output file path; or a path
to load. Outputs: a file on disk (torch.save) or a dict {models, meta} suitable for
CFRSolver.apply_cfv_bundle.

Dependencies: torch; CounterfactualValueNetwork and CompatLinearCFV. Invariants: I never
trust shapes blindly; I check strict compatibility, then try relaxed loads, then
shape-filtered loads. Performance: I keep everything on CPU on load by default and let
the caller move to a device; this avoids accidental GPU memory spikes.
"""

from hunl.constants import SEED_DEFAULT
import os
import time
import torch
from hunl.nets.cfv_network import CounterfactualValueNetwork
from hunl.nets.compat_linear_cfv import CompatLinearCFV


def save_cfv_bundle(
 models,
 cluster_mapping,
 input_meta,
 path,
 seed=None,
):
	bundle = {}

	bundle["version"] = "1.0"
	bundle["created_at"] = int(time.time())

	if seed is not None:
		bundle["seed"] = int(seed)
	else:
		bundle["seed"] = int(os.environ.get("FAST_TEST_SEED", str(SEED_DEFAULT)))

	bundle["stages"] = {}

	max_K = 0

	for stage, net in dict(models).items():
		if net is None:
			continue

		stage_rec = {}
		stage_rec["input_size"] = int(getattr(net, "input_size", 0))
		stage_rec["num_clusters"] = int(getattr(net, "num_clusters", 0))
		stage_rec["state_dict"] = {
		 k: v.detach().cpu()
		 for k, v in net.state_dict().items()
		}

		bundle["stages"][str(stage)] = stage_rec

		if stage_rec["num_clusters"] > max_K:
			max_K = stage_rec["num_clusters"]

	if input_meta is None:
		im = {}
	else:
		im = dict(input_meta)

	if ("num_clusters" not in im) or (int(im.get("num_clusters", 0)) <= 0):
		im["num_clusters"] = int(max_K)

	if "board_one_hot_dim" not in im:
		im["board_one_hot_dim"] = 52

	if "uses_pot_norm" not in im:
		im["uses_pot_norm"] = True

	if "target_units" not in im:
		im["target_units"] = "pot_fraction"

	if "outer_zero_sum" not in im:
		im["outer_zero_sum"] = True

	K = int(im.get("num_clusters", 0))
	B = int(im.get("board_one_hot_dim", 52))

	start_pn = 0
	start_b = start_pn + 1
	start_r1 = start_b + B
	start_r2 = start_r1 + K
	end_all = start_r2 + K

	layout = {
	 "pot_norm": 1,
	 "board_one_hot": B,
	 "range_dims": K,
	 "ranges": {"r1": K, "r2": K},
	}
	slices = {
	 "pot_norm": [start_pn, start_pn + 1],
	 "board_one_hot": [start_b, start_b + B],
	 "r1": [start_r1, start_r1 + K],
	 "r2": [start_r2, start_r2 + K],
	 "total_input_size": end_all,
	}

	im["input_layout"] = layout
	im["input_slices"] = slices
	im["range_dims"] = {"r1": K, "r2": K}

	if cluster_mapping is not None:
		bundle["cluster_mapping"] = dict(cluster_mapping)
	else:
		bundle["cluster_mapping"] = {}

	bundle["input_meta"] = im

	out_path = str(path)
	dirn = os.path.dirname(out_path)

	if dirn:
		if not os.path.isdir(dirn):
			os.makedirs(dirn, exist_ok=True)

	torch.save(bundle, out_path)
	return out_path


def _state_dict_compatible(
 net,
 state_dict,
) -> bool:
	ok = True

	net_sd = net.state_dict()

	for k, v in net_sd.items():
		if k in state_dict:
			t = state_dict[k]
			if hasattr(t, "shape"):
				if tuple(t.shape) != tuple(v.shape):
					ok = False
					break
			else:
				ok = False
				break
		else:
			ok = False
			break

	return ok


def _filter_state_dict_matching(
 net,
 state_dict,
):
	out = {}

	net_sd = net.state_dict()

	for k, v in net_sd.items():
		if k in state_dict:
			t = state_dict[k]
			if hasattr(t, "shape"):
				if tuple(t.shape) == tuple(v.shape):
					out[k] = t

	return out


def _load_stage_model(stage_rec, device):
	insz = int(stage_rec.get("input_size", 0))
	K = int(stage_rec.get("num_clusters", 0))
	state_dict = dict(stage_rec.get("state_dict", {}))

	net = CounterfactualValueNetwork(insz, num_clusters=K)

	if _strict_compatible(net, state_dict):
		net.load_state_dict(state_dict, strict=True)
	else:
		has_bias = False

		for k in state_dict.keys():
			if str(k).endswith("bias"):
				has_bias = True
				break
			else:
				pass

		net = CompatLinearCFV(insz, K, has_bias)

		if _strict_compatible(net, state_dict):
				net.load_state_dict(state_dict, strict=True)
		else:
			if _overlap_shapes_ok(net, state_dict):
				net.load_state_dict(state_dict, strict=False)
			else:
				filt = _filtered_by_shape(net, state_dict)
				net.load_state_dict(filt, strict=False)

	if device is not None:
		net = net.to(device)
	else:
		pass

	net.eval()
	return net

def _compute_input_slices(
 num_clusters,
 board_one_hot_dim
):
	K = int(num_clusters)
	B = int(board_one_hot_dim)

	start_pn = 0
	start_b = start_pn + 1
	start_r1 = start_b + B
	start_r2 = start_r1 + K
	end_all = start_r2 + K

	return {
	 "pot_norm": [start_pn, start_pn + 1],
	 "board_one_hot": [start_b, start_b + B],
	 "r1": [start_r1, start_r1 + K],
	 "r2": [start_r2, start_r2 + K],
	 "total_input_size": end_all,
	}


def _upgrade_input_meta(
 im
):
	out = dict(im or {})

	if "board_one_hot_dim" not in out:
		out["board_one_hot_dim"] = int(out.get("board_one_hot_dim", 52))

	if "num_clusters" not in out:
		out["num_clusters"] = int(out.get("num_clusters", 0))

	if "input_layout" not in out:
		out["input_layout"] = {
		 "pot_norm": 1,
		 "board_one_hot": int(out["board_one_hot_dim"]),
		 "range_dims": int(out["num_clusters"]),
		 "ranges": {
		  "r1": int(out["num_clusters"]),
		  "r2": int(out["num_clusters"]),
		 },
		}
	else:
		il = dict(out["input_layout"])
		if "range_dims" not in il:
			il["range_dims"] = int(out.get("num_clusters", 0))
		out["input_layout"] = il

	if isinstance(out.get("range_dims", None), int):
		K = int(out["range_dims"])
		out["range_dims"] = {"r1": K, "r2": K}

	if "input_slices" not in out:
		Ks = int(out.get("num_clusters", 0))
		B = int(out.get("board_one_hot_dim", 52))

		start_pn = 0
		start_b = start_pn + 1
		start_r1 = start_b + B
		start_r2 = start_r1 + Ks
		end_all = start_r2 + Ks

		out["input_slices"] = {
		 "pot_norm": [start_pn, start_pn + 1],
		 "board_one_hot": [start_b, start_b + B],
		 "r1": [start_r1, start_r1 + Ks],
		 "r2": [start_r2, start_r2 + Ks],
		 "total_input_size": end_all,
		}

	if "target_units" not in out:
		out["target_units"] = "pot_fraction"

	if "outer_zero_sum" not in out:
		out["outer_zero_sum"] = True

	return out


def load_cfv_bundle(
 path,
 device=None
):
	if device is None:
		map_loc = "cpu"
	else:
		map_loc = device

	bundle = torch.load(path, map_location=map_loc)

	out_models = {}
	stages = dict(bundle.get("stages", {}))

	for stage, rec in stages.items():
		out_models[str(stage)] = _load_stage_model(rec, device)

	im = _upgrade_input_meta(bundle.get("input_meta", {}))

	meta = {
	 "cluster_mapping": dict(bundle.get("cluster_mapping", {})),
	 "input_meta": im,
	 "version": str(bundle.get("version", "")),
	 "created_at": int(bundle.get("created_at", 0)),
	}

	return {
	 "models": out_models,
	 "meta": meta,
	}


def _strict_compatible(net, sd):
	net_sd = net.state_dict()

	for k, v in net_sd.items():
		if k in sd:
			pass
		else:
			return False

		t = sd[k]

		if hasattr(t, "shape"):
			pass
		else:
			return False

		if tuple(t.shape) == tuple(v.shape):
			pass
		else:
			return False

	return True


def _overlap_shapes_ok(net, sd):
	net_sd = net.state_dict()

	for k, v in net_sd.items():
		if k in sd:
			t = sd[k]

			if hasattr(t, "shape"):
				pass
			else:
				return False

			if tuple(t.shape) == tuple(v.shape):
				pass
			else:
				return False
		else:
			pass

	return True


def _filtered_by_shape(net, sd):
	out = {}
	net_sd = net.state_dict()

	for k, v in net_sd.items():
		if k in sd:
			t = sd[k]

			if hasattr(t, "shape"):
				if tuple(t.shape) == tuple(v.shape):
					out[k] = t
				else:
					pass
			else:
				pass
		else:
			pass

	return out




