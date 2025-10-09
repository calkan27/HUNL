"""
I provide a thin RNG wrapper that seeds Python, NumPy, and Torch (including CUDA)
consistently, plus a couple of convenience draws. I also expose set_global_seed to
initialize global libraries for reproducible tests, data generation, and solver
randomness.

Key classes/functions: SeededRNG — per-instance seeded generator with .rand, .randint,
.choice; set_global_seed — one-shot seeding for global state.

Inputs: integer seeds. Outputs: deterministic streams of random numbers through wrapped
libraries. Invariants: I always coerce seeds to int and tolerate environments without
CUDA; I report missing APIs with informational prints to avoid breaking tests.

Dependencies: random, numpy, torch. Performance: initialization is cheap; callers should
reuse instances to avoid reseeding global state mid-experiment.
"""

import random
import numpy as _np
import torch as _t


class SeededRNG:
	def __init__(
		self,
		seed
	):
		self.seed = int(seed)
		self._py = random.Random(int(self.seed))

		if hasattr(_np, "random"):
			if hasattr(_np.random, "seed"):
				_np.random.seed(int(self.seed))
			else:
				print("[INFO] numpy.random.seed not available.")
		else:
			print("[INFO] numpy not available.")

		if hasattr(_t, "manual_seed"):
			_t.manual_seed(int(self.seed))
		else:
			print("[INFO] torch.manual_seed not available.")

		if hasattr(_t, "cuda"):
			if hasattr(_t.cuda, "is_available"):
				if _t.cuda.is_available():
					if hasattr(_t.cuda, "manual_seed_all"):
						_t.cuda.manual_seed_all(int(self.seed))
					else:
						print("[INFO] torch.cuda.manual_seed_all not available.")
				else:
					print("[INFO] CUDA not available; skipping CUDA seeding.")
			else:
				print("[INFO] torch.cuda.is_available not present.")
		else:
			print("[INFO] torch.cuda not present.")

	def rand(
		self
	):
		return self._py.random()

	def randint(
		self,
		a,
		b
	):
		return self._py.randint(a, b)

	def choice(
		self,
		seq
	):
		return self._py.choice(seq)


def set_global_seed(
	seed
):
	r = SeededRNG(seed)
	return int(r.seed)

