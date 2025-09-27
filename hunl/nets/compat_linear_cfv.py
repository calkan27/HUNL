"""
Minimal, shape-compatible CFV head used as a fallback when loading bundles with unknown
or simplified architectures. It exposes the same forward and outer zero-sum interface as
the deep CFV network while collapsing representation to a single linear layer over the
concatenated input.

Key class: CompatLinearCFV. Key methods: forward (returns two K-length predictions),
enforce_zero_sum (per-sample outer adjustment using bucket-weighted sums), predict
parity with full models (via shared interface).

Inputs: input feature vector shaped [N, 1 + |board| + 2K] where the first element is
normalized pot, followed by a 52-d one-hot board and two K-d bucketed ranges;
constructor parameters fix input_size, K, and bias usage. Outputs: two K-d tensors of
bucket CFVs per player in fractions of pot, optionally outer-adjusted.

Internal dependencies: torch; constants for epsilon tolerance. External dependencies:
none beyond PyTorch.

Invariants: outer zero-sum enforces ⟨r1,f1⟩ + ⟨r2,f2⟩ ≈ 0 per sample; inputs and outputs
stay on the same device; broadcasting and shapes are guarded. Performance: single matmul
is fast and memory-light; useful for quick compatibility loads or constrained evaluation
environments.
"""


from hunl.constants import EPS_SUM
import torch
from hunl.nets.cfv_network import CounterfactualValueNetwork

class CompatLinearCFV(torch.nn.Module):
	def __init__(self, input_size, num_clusters, use_bias):
		super().__init__()
		self.fc = torch.nn.Linear(int(input_size), int(2 * num_clusters), bias=bool(use_bias))
		self.num_clusters = int(num_clusters)
		self.input_size = int(input_size)
	def forward(self, x):
		out = self.fc(x)
		K = self.num_clusters
		return out[:, :K], out[:, K:]
	@torch.no_grad()
	def enforce_zero_sum(self, r1, r2, p1, p2):
		eps = EPS_SUM
		s = torch.sum(r1 * p1, dim=1, keepdim=True) + torch.sum(r2 * p2, dim=1, keepdim=True)
		a = -0.5 * s / torch.clamp(torch.sum(r1, dim=1, keepdim=True), min=eps)
		b = -0.5 * s / torch.clamp(torch.sum(r2, dim=1, keepdim=True), min=eps)
		return p1 + a, p2 + b


