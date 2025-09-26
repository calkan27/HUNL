import torch
import torch.nn as nn

class RiverBucketAux(nn.Module):
	def __init__(self, num_buckets, num_clusters):
		super().__init__()
		self.num_buckets = int(num_buckets)
		self.num_clusters = int(num_clusters)
		self.fc1 = nn.Linear(self.num_buckets * self.num_clusters, 128)
		self.act1 = nn.PReLU()
		self.fc2 = nn.Linear(128, 128)
		self.act2 = nn.PReLU()
		self.out = nn.Linear(128, self.num_clusters)

	def forward(self, bucket_mix_by_cluster):
		x = bucket_mix_by_cluster.view(-1, self.num_clusters * self.num_buckets)
		h = self.act1(self.fc1(x))
		h = self.act2(self.fc2(h))
		y = self.out(h)
		return y

	def predict(self, bucket_mix_by_cluster):
		with torch.no_grad():
			return self.forward(bucket_mix_by_cluster)

	def enforce_zero_sum(self, r1, r2, v1, v2):
		eps = 1e-7
		s1 = torch.sum(r1, dim=1, keepdim=True)
		s2 = torch.sum(r2, dim=1, keepdim=True)
		u = v1.new_full((v1.shape[0], v1.shape[1]), 1.0 / max(1, v1.shape[1]))
		w1 = torch.where(s1 > eps, r1 / torch.clamp(s1, min=eps), u)
		w2 = torch.where(s2 > eps, r2 / torch.clamp(s2, min=eps), u)
		sv1 = torch.sum(w1 * v1, dim=1, keepdim=True)
		sv2 = torch.sum(w2 * v2, dim=1, keepdim=True)
		delta = 0.5 * (sv1 + sv2)
		f1 = v1 - delta
		f2 = v2 - delta
		return f1, f2

	def predict_with_zero_sum(self, x1, x2, r1, r2):
		p1 = self.forward(x1)
		p2 = self.forward(x2)
		f1, f2 = self.enforce_zero_sum(r1, r2, p1, p2)
		return f1, f2

	def zero_sum_residual(self, r1, r2, v1, v2):
		s1 = torch.sum(r1 * v1, dim=1)
		s2 = torch.sum(r2 * v2, dim=1)
		return torch.abs(s1 + s2)

