import time
from seeded_rng import SeededRNG


class CFVStreamDataset:
	def __init__(
		self,
		data_generator,
		stage,
		num_samples,
		seed: int = 1729,
		schema_version: str = "cfv.v1",
		shard_meta=None,
	):
		self.dg = data_generator
		self.stage = str(stage)
		self.num_samples = int(num_samples)
		self.rng = SeededRNG(seed)
		self.schema_version = str(schema_version)
		self.meta = dict(shard_meta) if shard_meta is not None else {}
		self._init_specs()

	def _init_specs(self):
		if hasattr(self.dg, "set_seed"):
			if callable(self.dg.set_seed):
				self.dg.set_seed(self.rng.seed)
			else:
				print("[INFO] data_generator.set_seed is not callable.")
		else:
			print("[INFO] data_generator has no set_seed; continuing without seeding.")

		if hasattr(self.dg, "pot_sampler_spec"):
			if callable(self.dg.pot_sampler_spec):
				pot_spec = self.dg.pot_sampler_spec()
			else:
				pot_spec = []
		else:
			pot_spec = []

		if hasattr(self.dg, "range_generator_spec"):
			if callable(self.dg.range_generator_spec):
				range_spec = self.dg.range_generator_spec()
			else:
				range_spec = {"name": "", "params": {}}
		else:
			range_spec = {"name": "", "params": {}}

		self.spec = {
			"schema": self.schema_version,
			"seed": int(self.rng.seed),
			"stage": self.stage,
			"num_clusters": int(getattr(self.dg, "num_clusters", 0)),
			"pot_sampler": pot_spec,
			"range_generator": range_spec,
		}

		for k, v in self.meta.items():
			self.spec[k] = v

	def __iter__(self):
		emitted = 0

		orig_nb = int(getattr(self.dg, "num_boards", 1))
		orig_ns = int(getattr(self.dg, "num_samples_per_board", 1))

		self._set_stream_mode(num_boards=1, num_samples_per_board=1)

		while emitted < self.num_samples:
			if hasattr(self.dg, "generate_training_data"):
				if callable(self.dg.generate_training_data):
					data = self.dg.generate_training_data(
						stage=self.stage,
						progress=None,
					)
				else:
					print("[WARN] data_generator.generate_training_data not callable.")
					break
			else:
				print("[WARN] data_generator has no generate_training_data.")
				break

			for rec in data:
				out_obj = self._record_from_rec(rec)
				yield out_obj
				emitted += 1

				if emitted >= self.num_samples:
					break

		self._restore_stream_mode(orig_nb, orig_ns)

	def _set_stream_mode(self, num_boards: int, num_samples_per_board: int):
		if hasattr(self.dg, "num_boards"):
			self.dg.num_boards = int(num_boards)
		else:
			print("[INFO] data_generator has no num_boards attribute.")

		if hasattr(self.dg, "num_samples_per_board"):
			self.dg.num_samples_per_board = int(num_samples_per_board)
		else:
			print("[INFO] data_generator has no num_samples_per_board attribute.")

	def _restore_stream_mode(self, orig_nb: int, orig_ns: int):
		if hasattr(self.dg, "num_boards"):
			self.dg.num_boards = int(orig_nb)
		if hasattr(self.dg, "num_samples_per_board"):
			self.dg.num_samples_per_board = int(orig_ns)

	def _safe_pot_spec(self):
		if hasattr(self.dg, "pot_sampler_spec"):
			if callable(self.dg.pot_sampler_spec):
				return self.dg.pot_sampler_spec()
			else:
				return []
		else:
			return []

	def _safe_range_spec(self):
		if hasattr(self.dg, "range_generator_spec"):
			if callable(self.dg.range_generator_spec):
				return self.dg.range_generator_spec()
			else:
				return {"name": "", "params": {}}
		else:
			return {"name": "", "params": {}}

	def _record_from_rec(self, rec):
		if isinstance(rec, dict):
			iv = list(rec.get("input_vector", []))
			t1 = list(rec.get("target_v1", []))
			t2 = list(rec.get("target_v2", []))
		else:
			iv = []
			t1 = []
			t2 = []

		return {
			"schema": self.schema_version,
			"stage": self.stage,
			"seed": int(self.rng.seed),
			"input_vector": iv,
			"target_v1": t1,
			"target_v2": t2,
			"pot_spec": self._safe_pot_spec(),
			"range_spec": self._safe_range_spec(),
			"generated_at": int(time.time()),
		}

