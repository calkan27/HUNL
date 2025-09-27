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


