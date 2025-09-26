import torch
import torch.nn as nn
from torch.optim import Adam
from seeded_rng import set_global_seed
import math
import random

import torch
import torch.nn as nn

class CounterfactualValueNetwork(nn.Module):
	def __init__(self, input_size, num_clusters=1000, input_layout=None):
		super(CounterfactualValueNetwork, self).__init__()
		self.num_clusters = int(num_clusters)
		self.input_size = int(input_size)
		self.input_layout = dict(input_layout) if input_layout is not None else None
		self.hidden_layer_1 = nn.Linear(self.input_size, 500)
		self.activation_1 = nn.PReLU()
		self.hidden_layer_2 = nn.Linear(500, 500)
		self.activation_2 = nn.PReLU()
		self.hidden_layer_3 = nn.Linear(500, 500)
		self.activation_3 = nn.PReLU()
		self.hidden_layer_4 = nn.Linear(500, 500)
		self.activation_4 = nn.PReLU()
		self.hidden_layer_5 = nn.Linear(500, 500)
		self.activation_5 = nn.PReLU()
		self.hidden_layer_6 = nn.Linear(500, 500)
		self.activation_6 = nn.PReLU()
		self.hidden_layer_7 = nn.Linear(500, 500)
		self.activation_7 = nn.PReLU()
		self.output_player1_values = nn.Linear(500, self.num_clusters)
		self.output_player2_values = nn.Linear(500, self.num_clusters)
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu", a=0.25)
				if m.bias is not None:
					nn.init.zeros_(m.bias)

	def forward(self, input_tensor):
		h = self.activation_1(self.hidden_layer_1(input_tensor))
		h = self.activation_2(self.hidden_layer_2(h))
		h = self.activation_3(self.hidden_layer_3(h))
		h = self.activation_4(self.hidden_layer_4(h))
		h = self.activation_5(self.hidden_layer_5(h))
		h = self.activation_6(self.hidden_layer_6(h))
		h = self.activation_7(self.hidden_layer_7(h))
		p1 = self.output_player1_values(h)
		p2 = self.output_player2_values(h)
		return p1, p2

	def enforce_zero_sum(self, player1_range, player2_range, player1_values, player2_values):
		eps = 1e-9
		s1 = torch.sum(player1_range, dim=1, keepdim=True)
		s2 = torch.sum(player2_range, dim=1, keepdim=True)
		u1 = player1_values.new_full((player1_values.shape[0], player1_values.shape[1]),
									 1.0 / max(1, player1_values.shape[1]))
		u2 = player2_values.new_full((player2_values.shape[0], player2_values.shape[1]),
									 1.0 / max(1, player2_values.shape[1]))
		w1 = torch.where(s1 > eps, player1_range / torch.clamp(s1, min=eps), u1)
		w2 = torch.where(s2 > eps, player2_range / torch.clamp(s2, min=eps), u2)
		sv1 = torch.sum(w1 * player1_values, dim=1, keepdim=True)
		sv2 = torch.sum(w2 * player2_values, dim=1, keepdim=True)
		delta = 0.5 * (sv1 + sv2)
		f1 = player1_values - delta
		f2 = player2_values - delta
		return f1, f2

	def predict_with_zero_sum(self, input_tensor, player1_range, player2_range):
		p1, p2 = self(input_tensor)
		f1, f2 = self.enforce_zero_sum(player1_range, player2_range, p1, p2)
		return f1, f2




def make_cfv_network(input_size, num_clusters):
	return CounterfactualValueNetwork(int(input_size), int(num_clusters))
def build_three_stage_cfv(input_size_preflop, input_size_flop, input_size_turn, num_clusters, device=None):
	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	models = {
		'preflop': CounterfactualValueNetwork(int(input_size_preflop), int(num_clusters)).to(device),
		'flop': CounterfactualValueNetwork(int(input_size_flop), int(num_clusters)).to(device),
		'turn': CounterfactualValueNetwork(int(input_size_turn), int(num_clusters)).to(device),
	}
	return models
