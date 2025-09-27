"""
Test suite for CFV bundle save/load: verifies that a minimal CFV network bundle can be serialized and deserialized 
with model shapes and input metadata preserved.
"""

import hunl.nets.model_io as model_io
import os
import torch
import numpy as np
import pytest

from hunl.nets.model_io import save_cfv_bundle, load_cfv_bundle


class _MiniCFVNet(torch.nn.Module):
	"""
	Minimal CFV network producing two K-length outputs and supporting zero-sum enforcement; used to validate bundle I/O.
	"""
	def __init__(self, input_size, K):
		"""
		Initialize a linear head mapping input_size to 2*K outputs and record cluster/input sizes for roundtrip checks.
		"""
		super().__init__()
		self.fc = torch.nn.Linear(input_size, 2*K, bias=False)
		self.num_clusters = int(K)
		self.input_size = int(input_size)

	def forward(self, x):
		"""
		Compute raw predictions and split into (player1, player2) tensors of shape [B, K].
		"""
		out = self.fc(x)
		K = self.num_clusters
		return out[:, :K], out[:, K:]

	@torch.no_grad()
	def enforce_zero_sum(self, r1, r2, p1, p2):
		"""
		Shift predictions so that range-weighted expectations sum to zero per sample for both players.
		"""
		sum_r1 = torch.clamp(torch.sum(r1, dim=1, keepdim=True), min=1e-9)
		sum_r2 = torch.clamp(torch.sum(r2, dim=1, keepdim=True), min=1e-9)
		s = torch.sum(r1 * p1, dim=1, keepdim=True) + torch.sum(r2 * p2, dim=1, keepdim=True)
		a = -0.5 * s / sum_r1
		b = -0.5 * s / sum_r2
		return p1 + a, p2 + b


def test_save_and_load_cfv_bundle_roundtrip(tmp_path):
	"""
	Ensure that save_cfv_bundle writes a valid bundle and load_cfv_bundle restores models and meta with correct keys and dimensions.
	"""
	K = 8
	insz = 1 + 52 + 2*K
	flop = _MiniCFVNet(insz, K)
	turn = _MiniCFVNet(insz, K)
	models = {"flop": flop, "turn": turn}
	cluster_mapping = {i: [f"H{i}"] for i in range(K)}
	input_meta = {"num_clusters": K, "board_one_hot_dim": 52, "uses_pot_norm": True}
	out_path = tmp_path / "bundle.pt"
	path = save_cfv_bundle(models, cluster_mapping, input_meta, str(out_path), seed=123)
	assert os.path.isfile(path)
	loaded = load_cfv_bundle(str(out_path), device=torch.device("cpu"))
	lm = loaded["models"]
	meta = loaded["meta"]
	assert "flop" in lm and "turn" in lm
	assert int(getattr(lm["flop"], "num_clusters", -1)) == K
	assert int(getattr(lm["flop"], "input_size", -1)) == insz
	assert meta["input_meta"]["num_clusters"] == K
	assert meta["input_meta"]["board_one_hot_dim"] == 52
	assert meta["input_meta"]["uses_pot_norm"] is True

