"""
Test suite for ValueServer and ResultHandle: validates zero-sum enforcement, batching and threading behavior, unknown-stage handling, and blocking semantics.
"""

import hunl.nets.value_server as value_server
import time
import threading
import numpy as np
import torch
import pytest

from hunl.nets.value_server import ValueServer
from hunl.utils.result_handle import ResultHandle


class _DummyCFVNet(torch.nn.Module):
	"""
	Minimal CFV network stub used to exercise ValueServer logic without learning; defines input sizing from pot normalization, board one-hot, and two K-length ranges.
	"""
	def __init__(self, K, board_dim=52):
		"""
		Initialize the stub model with K clusters and derived input_size matching [pot_norm, board_one_hot, r1, r2].
		"""
		super().__init__()
		self.num_clusters = int(K)
		self.input_size = 1 + board_dim + 2 * K

	def forward(self, x: torch.Tensor):
		"""
		Produce deterministic outputs (p1, p2) of shape [N, K] to allow downstream zero-sum adjustment and counter checks.
		"""
		N = x.shape[0]
		K = self.num_clusters
		p1 = torch.full((N, K), 0.2, dtype=torch.float32, device=x.device)
		p2 = torch.full((N, K), -0.1, dtype=torch.float32, device=x.device)
		return p1, p2

	@torch.no_grad()
	def enforce_zero_sum(self, r1, r2, p1, p2):
		"""
		Adjust predictions so that for each sample the range-weighted expectations satisfy (r1·f1 + r2·f2) == 0.
		"""
		sum_r1 = torch.clamp(torch.sum(r1, dim=1, keepdim=True), min=1e-9)
		sum_r2 = torch.clamp(torch.sum(r2, dim=1, keepdim=True), min=1e-9)
		s = torch.sum(r1 * p1, dim=1, keepdim=True) + torch.sum(r2 * p2, dim=1, keepdim=True)
		a = -0.5 * s / sum_r1
		b = -0.5 * s / sum_r2
		return p1 + a, p2 + b


def _make_inputs(K, board_dim=52, pot_norm=0.5):
	"""
	Construct a single-row input tensor in the layout [pot_norm, board_one_hot(board_dim), r1(K), r2(K)] with uniform ranges.
	"""
	r1 = np.zeros((K,), dtype=np.float32)
	r2 = np.zeros((K,), dtype=np.float32)
	r1[:] = 1.0 / K
	r2[:] = 1.0 / K
	board = np.zeros((board_dim,), dtype=np.float32)
	x = np.concatenate([[pot_norm], board, r1, r2]).astype(np.float32)
	return torch.tensor([x], dtype=torch.float32)


def test_value_server_zero_sum_and_counters():
	"""
	Ensure ValueServer returns correctly shaped outputs, enforces zero-sum under provided ranges, and increments per-stage query counters.
	"""
	K = 8
	net_flop = _DummyCFVNet(K)
	net_turn = _DummyCFVNet(K)
	vs = ValueServer(models={"flop": net_flop, "turn": net_turn}, max_wait_ms=1)
	try:
		xb = _make_inputs(K, 52, pot_norm=0.4)
		v1, v2 = vs.query("flop", xb, scale_to_pot=False, as_numpy=False)
		assert v1.shape == (1, K) and v2.shape == (1, K)
		r1 = xb[:, 1+52:1+52+K]
		r2 = xb[:, 1+52+K:1+52+2*K]
		res = torch.sum(r1 * v1, dim=1) + torch.sum(r2 * v2, dim=1)
		assert torch.allclose(res, torch.zeros_like(res), atol=1e-6)
		cnt = vs.get_counters()
		assert int(cnt.get("flop", 0)) >= 1
	finally:
		vs.stop()


def test_value_server_batching_and_threading():
	"""
	Verify that concurrent queries are accepted, results are returned as numpy arrays with expected shapes, and counters reflect the number of queries.
	"""
	K = 4
	net = _DummyCFVNet(K)
	vs = ValueServer(models={"flop": net}, max_batch_size=8, max_wait_ms=2)
	try:
		outs = []
		def _worker():
			xb = _make_inputs(K)
			v1n, v2n = vs.query("flop", xb, as_numpy=True)
			outs.append((v1n, v2n))
		threads = [threading.Thread(target=_worker) for _ in range(5)]
		for t in threads:
			t.start()
		for t in threads:
			t.join(timeout=2)
		assert len(outs) == 5
		for v1n, v2n in outs:
			assert isinstance(v1n, np.ndarray) and isinstance(v2n, np.ndarray)
			assert v1n.shape == (1, K) and v2n.shape == (1, K)
		cnt = vs.get_counters()
		assert int(cnt.get("flop", 0)) >= 5
	finally:
		vs.stop()


def test_value_server_unknown_stage_returns_empty_tensors():
	"""
	Check that querying a stage with no registered model returns tensors with zero columns for both players.
	"""
	K = 4
	net = _DummyCFVNet(K)
	vs = ValueServer(models={"flop": net}, max_wait_ms=1)
	try:
		xb = _make_inputs(K)
		v1, v2 = vs.query("river", xb, as_numpy=False)
		assert v1.shape == (1, 0) and v2.shape == (1, 0)
	finally:
		vs.stop()


def test_result_handle_blocks_until_set():
	"""
	Confirm ResultHandle blocks until a value is set and returns arrays with expected shapes and contents when requested as numpy.
	"""
	h = ResultHandle()
	out = []
	def _setter():
		time.sleep(0.01)
		a = torch.tensor([[1.0]]); b = torch.tensor([[2.0]])
		h.set((a, b))
	threading.Thread(target=_setter, daemon=True).start()
	v1n, v2n = h.result(as_numpy=True)
	assert v1n.shape == (1, 1) and v2n.shape == (1, 1)
	assert float(v1n[0, 0]) == 1.0 and float(v2n[0, 0]) == 2.0

