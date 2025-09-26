import os
import time
import json
import random
from typing import Dict, Any, Callable, Tuple

import numpy as np
from poker_utils import DECK


class DataGeneratorDatasetsMixin:
	def _default_meta(self, stage: str, schema: str) -> Dict[str, Any]:
		meta: Dict[str, Any] = {}
		meta["schema"] = str(schema)
		meta["created_at"] = int(time.time())
		meta["stage"] = str(stage)
		meta["num_clusters"] = int(self.num_clusters)
		meta["pot_sampler"] = self.pot_sampler_spec()
		meta["range_generator"] = self.range_generator_spec()

		cfg = getattr(self, "_config", None)
		if cfg is not None:
			flag = bool(getattr(cfg, "enforce_zero_sum_outer", True))
		else:
			flag = True
		meta["outer_zero_sum"] = bool(flag)

		action_set: Dict[str, Any] = {}
		bf: Dict[int, list] = {}
		i = 0
		while i < 4:
			bf[i] = [1.0]
			i += 1
		action_set["bet_fractions"] = bf

		rflags: Dict[int, Dict[str, bool]] = {}
		i = 0
		while i < 4:
			rflags[i] = {"half_pot": False, "two_pot": False}
			i += 1
		action_set["round_flags"] = rflags

		action_set["include_all_in"] = True
		meta["action_set"] = action_set

		return meta

	def _production_guard_begin(self) -> Tuple[object, Dict[int, Dict[str, bool]]]:
		guard = self._push_production_mode()
		self.cfr_solver._ensure_sparse_schedule()

		round_flags_backup: Dict[int, Dict[str, bool]] = {}
		src = getattr(self.cfr_solver, "_round_actions", {})
		for k, v in src.items():
			if isinstance(v, dict):
				round_flags_backup[int(k)] = {
					"half_pot": bool(v.get("half_pot", True)),
					"two_pot": bool(v.get("two_pot", False)),
				}
			else:
				round_flags_backup[int(k)] = {
					"half_pot": True,
					"two_pot": False,
				}

		r = 0
		while r < 4:
			self.cfr_solver._round_actions[int(r)] = {
				"half_pot": False,
				"two_pot": False,
			}
			r += 1

		return guard, round_flags_backup

	def _production_guard_end(self, guard, round_flags_backup):
		self.cfr_solver._round_actions = round_flags_backup
		self._pop_production_mode(guard)

	def _prepare_sampler_invariants(
		self,
		board_cards: list,
		bucketed_ranges: list,
		pot_size: float,
	) -> Tuple[Dict[int, float], Dict[int, float]]:
		pr0: Dict[int, float] = {}
		i = 0
		while i < self.num_clusters:
			pr0[i] = bucketed_ranges[0][i]
			i += 1

		pr1: Dict[int, float] = {}
		j = 0
		while j < self.num_clusters:
			pr1[j] = bucketed_ranges[1][j]
			j += 1

		self._assert_sampler_invariants(board_cards, [pr0, pr1], pot_size)
		return pr0, pr1

	def _prepare_iv_and_targets(
		self,
		node,
		target_provider: Callable[[object], Tuple[list, list]],
	) -> Tuple[list, list, list]:
		self.cfr_solver.total_iterations = 1000
		self.cfr_solver.depth_limit = max(1, int(getattr(self, "depth_limit", 1)))

		t1, t2 = target_provider(node)

		bucketed = self.bucket_player_ranges(
			[node.player_ranges[0], node.player_ranges[1]]
		)

		_ = self._prepare_sampler_invariants(
			node.public_state.board_cards,
			bucketed,
			node.public_state.pot_size,
		)

		iv = self.prepare_input_vector(
			bucketed,
			node.public_state.board_cards,
			node.public_state.pot_size,
			node.public_state.actions,
		)

		return iv, t1, t2

	def _append_and_maybe_flush(
		self,
		chunk: list,
		rec: Dict[str, Any],
		chunk_size: int,
		out_dir: str,
		stage: str,
		chunk_idx: int,
		meta: Dict[str, Any],
	) -> Tuple[list, int]:
		chunk.append(rec)

		if len(chunk) >= int(chunk_size):
			self._persist_npz_chunk(
				chunk,
				out_dir,
				stage,
				chunk_idx,
				meta,
			)
			return [], chunk_idx + 1
		else:
			return chunk, chunk_idx

	def _mk_rec(self, iv: list, t1: list, t2: list) -> Dict[str, Any]:
		rec: Dict[str, Any] = {}
		rec["input_vector"] = iv
		rec["target_v1"] = [float(x) for x in t1]
		rec["target_v2"] = [float(x) for x in t2]
		return rec

	def generate_turn_dataset(
		self,
		num_situations,
		out_dir,
		chunk_size: int = 50000,
		seed: int = 2027,
	):
		rng = random.Random(int(seed))
		meta = self._default_meta("turn", "cfv.dataset.turn.v1")
		guard, flags = self._production_guard_begin()

		count = 0
		chunk: list = []
		chunk_idx = 0

		while count < int(num_situations):
			node = self._sample_turn_situation(rng)

			iv, t1, t2 = self._prepare_iv_and_targets(
				node,
				lambda nd: self.cfr_solver.turn_label_targets_solve_to_terminal(nd),
			)

			rec = self._mk_rec(iv, t1, t2)

			chunk, chunk_idx = self._append_and_maybe_flush(
				chunk,
				rec,
				int(chunk_size),
				out_dir,
				"turn",
				chunk_idx,
				meta,
			)

			count += 1

		if len(chunk) > 0:
			self._persist_npz_chunk(chunk, out_dir, "turn", chunk_idx, meta)
		else:
			pass

		self._production_guard_end(guard, flags)

		written = int(chunk_idx + (1 if len(chunk) > 0 else 0))
		return {"written_chunks": written}

	def generate_flop_dataset(
		self,
		num_situations,
		out_dir,
		chunk_size: int = 50000,
		seed: int = 2027,
		persist_format: str = "npz",
	):
		_ = persist_format  

		rng = random.Random(int(seed))
		meta = self._default_meta("flop", "cfv.dataset.flop.v2")
		guard, flags = self._production_guard_begin()

		count = 0
		chunk: list = []
		chunk_idx = 0

		while count < int(num_situations):
			node_flop = self._sample_flop_situation(rng)

			iv, t1, t2 = self._prepare_iv_and_targets(
				node_flop,
				lambda nd: self.cfr_solver.flop_label_targets_using_turn_net(nd),
			)

			rec = self._mk_rec(iv, t1, t2)

			chunk, chunk_idx = self._append_and_maybe_flush(
				chunk,
				rec,
				int(chunk_size),
				out_dir,
				"flop",
				chunk_idx,
				meta,
			)

			count += 1

		if len(chunk) > 0:
			self._persist_npz_chunk(chunk, out_dir, "flop", chunk_idx, meta)
		else:
			pass

		self._production_guard_end(guard, flags)

		written = int(chunk_idx + (1 if len(chunk) > 0 else 0))
		return {"written_chunks": written}

	def _targets_flop_using_turn(
		self,
		node,
		turn_model,
	) -> Tuple[list, list]:
		self.cfr_solver.models["turn"] = (
			turn_model.to(self.cfr_solver.device).eval()
		)

		snap_cards = None
		if hasattr(self.cfr_solver, "_push_no_card_abstraction_for_node"):
			snap_cards = self.cfr_solver._push_no_card_abstraction_for_node(node)
		else:
			snap_cards = None

		self.cfr_solver.run_cfr(node)
		cf = self.compute_counterfactual_values(node)
		t1, t2 = self.prepare_target_values(cf, node.public_state.pot_size)

		if hasattr(self.cfr_solver, "_pop_no_card_abstraction"):
			self.cfr_solver._pop_no_card_abstraction(snap_cards, node)
		else:
			pass

		return t1, t2

	def generate_flop_dataset_using_turn(
		self,
		turn_model,
		num_situations,
		out_dir=None,
		chunk_size: int = 50000,
		seed: int = 2027,
	):
		rng = random.Random(int(seed))
		meta = self._default_meta("flop", "cfv.dataset.flop.v1")
		guard, flags = self._production_guard_begin()

		count = 0
		chunk: list = []
		chunk_idx = 0

		while count < int(num_situations):
			node = self._sample_flop_situation(rng)

			self.cfr_solver.total_iterations = 1000
			self.cfr_solver.depth_limit = max(
				1,
				int(getattr(self, "depth_limit", 1)),
			)

			t1, t2 = self._targets_flop_using_turn(node, turn_model)

			bucketed = self.bucket_player_ranges(
				[node.player_ranges[0], node.player_ranges[1]]
			)

			_ = self._prepare_sampler_invariants(
				node.public_state.board_cards,
				bucketed,
				node.public_state.pot_size,
			)

			iv = self.prepare_input_vector(
				bucketed,
				node.public_state.board_cards,
				node.public_state.pot_size,
				node.public_state.actions,
			)

			rec = self._mk_rec(iv, t1, t2)

			if out_dir:
				chunk, chunk_idx = self._append_and_maybe_flush(
					chunk,
					rec,
					int(chunk_size),
					out_dir,
					"flop",
					chunk_idx,
					meta,
				)
			else:
				chunk.append(rec)

			count += 1

		result: Dict[str, Any] = {}

		if out_dir:
			if len(chunk) > 0:
				self._persist_npz_chunk(chunk, out_dir, "flop", chunk_idx, meta)
			else:
				pass
			result["written_chunks"] = int(chunk_idx + (1 if len(chunk) > 0 else 0))
			result["in_memory"] = []
		else:
			result["written_chunks"] = 0
			result["in_memory"] = list(chunk)

		self._production_guard_end(guard, flags)
		return result

	def generate_unique_boards(
		self,
		stage: str = "flop",
		num_boards: int = 10,
	):
		stage_lc = str(stage).lower()

		if stage_lc == "flop":
			target_len = 3
		elif stage_lc == "turn":
			target_len = 4
		elif stage_lc == "river":
			target_len = 5
		else:
			target_len = 0

		out: list = []
		seen: set = set()

		deck_list = list(np.array(DECK))

		while len(out) < int(num_boards):
			sample = random.sample(deck_list, target_len)
			b = tuple(sorted(sample))

			if b in seen:
				pass
			else:
				seen.add(b)
				out.append(list(b))

		return out

	def _persist_npz_chunk(
		self,
		records,
		out_dir,
		stage,
		chunk_idx,
		meta: Dict[str, Any],
	):
		os.makedirs(str(out_dir), exist_ok=True)

		x_list: list = []
		y1_list: list = []
		y2_list: list = []

		for rec in list(records):
			x_list.append(list(rec["input_vector"]))
			y1_list.append(list(rec["target_v1"]))
			y2_list.append(list(rec["target_v2"]))

		if len(x_list) > 0:
			X = np.asarray(x_list, dtype=np.float32)
		else:
			X = np.zeros((0, 1), dtype=np.float32)

		if len(y1_list) > 0:
			Y1 = np.asarray(y1_list, dtype=np.float32)
		else:
			Y1 = np.zeros((0, self.num_clusters), dtype=np.float32)

		if len(y2_list) > 0:
			Y2 = np.asarray(y2_list, dtype=np.float32)
		else:
			Y2 = np.zeros((0, self.num_clusters), dtype=np.float32)

		out_path = os.path.join(
			str(out_dir),
			f"{str(stage)}_chunk_{int(chunk_idx):05d}.npz",
		)

		meta_schema = str(meta.get("schema", ""))
		meta_created_at = int(meta.get("created_at", int(time.time())))
		meta_stage = str(meta.get("stage", ""))
		meta_num_clusters = int(meta.get("num_clusters", self.num_clusters))
		meta_pot_spec = str(meta.get("pot_sampler", ""))
		meta_range_spec = str(meta.get("range_generator", ""))
		meta_outer_zero_sum = int(1 if bool(meta.get("outer_zero_sum", True)) else 0)
		meta_action_set = json.dumps(meta.get("action_set", {}))

		np.savez(
			out_path,
			inputs=X,
			target_v1=Y1,
			target_v2=Y2,
			meta_schema=meta_schema,
			meta_created_at=meta_created_at,
			meta_stage=meta_stage,
			meta_num_clusters=meta_num_clusters,
			meta_pot_spec=meta_pot_spec,
			meta_range_spec=meta_range_spec,
			meta_outer_zero_sum=meta_outer_zero_sum,
			meta_action_set=meta_action_set,
		)

		return out_path

