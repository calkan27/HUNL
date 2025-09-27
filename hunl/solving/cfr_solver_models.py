from typing import Dict, List, Tuple, Any
import copy
import os
import torch
import torch.nn as nn

from hunl.engine.action_type import ActionType
from hunl.engine.poker_utils import DECK, board_one_hot, best_hand, hand_rank
from hunl.nets.cfv_network import CounterfactualValueNetwork
from hunl.engine.game_node import GameNode


class CFRSolverModelsMixin:

	def load_models(self):
		files = {
			"preflop": ["counterfactual_value_network_preflop.pt", "preflop.pt"],
			"flop":    ["counterfactual_value_network_flop.pt", "flop.pt", "counterfactual_value_network.pt"],
			"turn":    ["counterfactual_value_network_turn.pt", "turn.pt"],
		}

		loaded = {"preflop": False, "flop": False, "turn": False}

		for stage, candidates in files.items():
			ok = self._load_stage_model(stage, candidates)
			loaded[stage] = bool(ok)

		if (not loaded["flop"]) or (not loaded["turn"]):
			self._share_flop_turn_if_missing()

		if not loaded["preflop"]:
			self.models["preflop"].eval()
			self._zero_initialize_model(self.models["preflop"])
			print("[WARN] No preflop weights found; using zero-initialized stub in eval mode.")

		return loaded

	def _load_stage_model(self, stage, candidates):
		for path in candidates:
			if isinstance(path, str):
				if hasattr(os, "path"):
					if os.path.isfile(path):
						state_dict = torch.load(path, map_location=self.device)
						self.models[stage].load_state_dict(state_dict)
						self.models[stage].eval()
						print(f"[OK] Loaded {stage} CFV model from {path}.")
						return True

		self.models[stage].eval()
		self._zero_initialize_model(self.models[stage])
		print(f"[WARN] Missing CFV model for {stage}: tried {candidates}. Using deterministic zero-initialized weights.")
		return False

	def _zero_initialize_model(self, model):
		for m in model.modules():
			if isinstance(m, nn.Linear):
				with torch.no_grad():
					m.weight.zero_()
					if m.bias is not None:
						m.bias.zero_()

	def _same_shapes(self, mA, mB):
		dA = {k: v.shape for k, v in mA.state_dict().items()}
		dB = {k: v.shape for k, v in mB.state_dict().items()}
		return dA == dB

	def _share_flop_turn_if_missing(self):
		def _model_nonzero(m):
			with torch.no_grad():
				for p in m.parameters():
					if p is None:
						continue
					if p.abs().sum().item() != 0.0:
						return True
			return False

		if not self._same_shapes(self.models["flop"], self.models["turn"]):
			return

		flop_nonzero = _model_nonzero(self.models["flop"])
		turn_nonzero = _model_nonzero(self.models["turn"])

		if (not flop_nonzero) and turn_nonzero:
			self.models["flop"].load_state_dict(copy.deepcopy(self.models["turn"].state_dict()))
			self.models["flop"].eval()
			print("[INFO] Using turn weights for flop (compatibility fallback).")
		else:
			if (not turn_nonzero) and flop_nonzero:
				self.models["turn"].load_state_dict(copy.deepcopy(self.models["flop"].state_dict()))
				self.models["turn"].eval()
				print("[INFO] Using flop weights for turn (compatibility fallback).")

	def calculate_input_size(self):
		pot_size_input = 1
		public_card_input = len(DECK)
		range_input = 2 * self.num_clusters
		return pot_size_input + public_card_input + range_input

	def calculate_input_size_preflop(self):
		pot_size_input = 1
		range_input = 2 * self.num_clusters
		return pot_size_input + range_input

	def prepare_input_vector(self, node):
		if getattr(node.public_state, "initial_stacks", None):
			total_initial = sum(node.public_state.initial_stacks)
		else:
			total_initial = 1.0

		if total_initial <= 0:
			total_initial = 1.0

		pot_vec = [node.public_state.pot_size / float(total_initial)]
		board_vec = board_one_hot(node.public_state.board_cards)

		K = self.num_clusters
		r1 = [0.0] * K
		r2 = [0.0] * K

		range_p1 = dict(node.player_ranges[0])
		range_p2 = dict(node.player_ranges[1])

		total1 = 0.0
		for v in range_p1.values():
			total1 += float(v)

		total2 = 0.0
		for v in range_p2.values():
			total2 += float(v)

		if total1 > 0:
			for k in list(range_p1.keys()):
				range_p1[k] = range_p1[k] / total1
		else:
			for k in list(range_p1.keys()):
				range_p1[k] = 0.0

		if total2 > 0:
			for k in list(range_p2.keys()):
				range_p2[k] = range_p2[k] / total2
		else:
			for k in list(range_p2.keys()):
				range_p2[k] = 0.0

		for cluster_id, prob in range_p1.items():
			if (0 <= int(cluster_id)) and (int(cluster_id) < K):
				r1[int(cluster_id)] = float(prob)

		for cluster_id, prob in range_p2.items():
			if (0 <= int(cluster_id)) and (int(cluster_id) < K):
				r2[int(cluster_id)] = float(prob)

		return pot_vec + board_vec + r1 + r2

	def prepare_input_vector_preflop(self, node):
		if getattr(node.public_state, "initial_stacks", None):
			total_initial = sum(node.public_state.initial_stacks)
		else:
			total_initial = 1.0

		if total_initial <= 0:
			total_initial = 1.0

		pot_vec = [node.public_state.pot_size / float(total_initial)]

		K = self.num_clusters
		r1 = [0.0] * K
		r2 = [0.0] * K

		range_p1 = dict(node.player_ranges[0])
		range_p2 = dict(node.player_ranges[1])

		total1 = 0.0
		for v in range_p1.values():
			total1 += float(v)

		total2 = 0.0
		for v in range_p2.values():
			total2 += float(v)

		if total1 > 0:
			for k in list(range_p1.keys()):
				range_p1[k] = range_p1[k] / total1
		else:
			for k in list(range_p1.keys()):
				range_p1[k] = 0.0

		if total2 > 0:
			for k in list(range_p2.keys()):
				range_p2[k] = range_p2[k] / total2
		else:
			for k in list(range_p2.keys()):
				range_p2[k] = 0.0

		for cluster_id, prob in range_p1.items():
			if (0 <= int(cluster_id)) and (int(cluster_id) < K):
				r1[int(cluster_id)] = float(prob)

		for cluster_id, prob in range_p2.items():
			if (0 <= int(cluster_id)) and (int(cluster_id) < K):
				r2[int(cluster_id)] = float(prob)

		return pot_vec + r1 + r2

	def predict_counterfactual_values(self, node, player):
		if not hasattr(self, "_diag_cfv_calls"):
			self._diag_cfv_calls = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}

		if not hasattr(self, "_zs_residual_samples"):
			self._zs_residual_samples = []

		stage = self.get_stage(node)

		if stage in self._diag_cfv_calls:
			self._diag_cfv_calls[stage] += 1

		if stage == "river":
			def wins_fn(ph, oh, board):
				return self._player_wins(ph, oh, board)

			cf = self.river_endgame.compute_cluster_cfvs(self.clusters, node, player, wins_fn, best_hand, hand_rank)

			out = {}
			for cid, val in cf.items():
				out[int(cid)] = val

			if hasattr(self, "_zs_residual_samples"):
				self._zs_residual_samples.append(0.0)

			return out

		if stage not in ("preflop", "flop", "turn"):
			counterfactual_values = {}
			for cluster_id in node.player_ranges[player]:
				counterfactual_values[cluster_id] = [0.0] * len(ActionType)
			return counterfactual_values
		else:
			if (("preflop" not in self.models) and ("flop" not in self.models) and ("turn" not in self.models)):
				counterfactual_values = {}
				for cluster_id in node.player_ranges[player]:
					counterfactual_values[cluster_id] = [0.0] * len(ActionType)
				return counterfactual_values

		input_vector = self.prepare_input_vector(node)

		if stage == "preflop":
			stage_model = self.models["flop"]
		elif stage == "flop":
			stage_model = self.models["turn"]
		else:
			stage_model = self.models["turn"]

		K = self.num_clusters
		start_r1 = 1 + len(DECK)
		end_r1 = start_r1 + K
		start_r2 = end_r1
		end_r2 = start_r2 + K

		input_tensor = torch.tensor([input_vector], dtype=torch.float32).to(self.device)

		with torch.no_grad():
			v1, v2 = stage_model(input_tensor)
			r1 = input_tensor[:, start_r1:end_r1]
			r2 = input_tensor[:, start_r2:end_r2]
			v1_adj, v2_adj = stage_model.enforce_zero_sum(r1, r2, v1, v2)

			s1 = torch.sum(r1 * v1_adj, dim=1, keepdim=True)
			s2 = torch.sum(r2 * v2_adj, dim=1, keepdim=True)
			res = torch.abs(s1 + s2).view(-1).detach().cpu().tolist()

			if hasattr(self, "_zs_residual_samples"):
				for x in res:
					self._zs_residual_samples.append(float(x))

		if player == 0:
			pred = v1_adj
		else:
			pred = v2_adj

		counterfactual_values = {}
		for cluster_id in node.player_ranges[player]:
			idx = int(cluster_id)

			if (0 <= idx) and (idx < self.num_clusters):
				scalar = float(pred[0][idx].item())
			else:
				scalar = 0.0

			counterfactual_values[cluster_id] = [scalar] * len(ActionType)

		return counterfactual_values

	def apply_cfv_bundle(self, bundle, device=None):
		models = {}
		meta = {}

		if isinstance(bundle, dict):
			models = dict(bundle.get("models", {}))
			meta = dict(bundle.get("meta", {}))

		if device is None:
			device = getattr(self, "device", None)

		if device is None:
			if torch.cuda.is_available():
				device = torch.device("cuda")
			else:
				device = torch.device("cpu")

		applied = False

		for stage, net in models.items():
			if stage in ("preflop", "flop", "turn"):
				if hasattr(net, "to"):
					net = net.to(device)
				if hasattr(net, "eval"):
					net.eval()
				self.models[stage] = net
				applied = True
			else:
				print(f"[INFO] Skipping unknown stage '{stage}' in apply_cfv_bundle.")

		if "input_meta" in meta:
			im = dict(meta.get("input_meta", {}))

			if "num_clusters" in im:
				K_val = im.get("num_clusters", None)

				if K_val is not None:
					K = int(K_val)
					self.num_clusters = K

					if hasattr(self, "hand_clusterer"):
						if self.hand_clusterer is not None:
							if hasattr(self.hand_clusterer, "set_num_clusters"):
								self.hand_clusterer.set_num_clusters(K)
							else:
								setattr(self.hand_clusterer, "num_clusters", K)

		if "cluster_mapping" in meta:
			mapping = dict(meta.get("cluster_mapping", {}))

			if mapping:
				if hasattr(self, "hand_clusterer"):
					if (self.hand_clusterer is not None) and hasattr(self.hand_clusterer, "load_mapping"):
						self.hand_clusterer.load_mapping(mapping)
						self.clusters = {int(k): set(v) for k, v in mapping.items()}
						applied = True
					else:
						self.clusters = {int(k): set(v) for k, v in mapping.items()}
						self.num_clusters = len(self.clusters)
						applied = True
				else:
					self.clusters = {int(k): set(v) for k, v in mapping.items()}
					self.num_clusters = len(self.clusters)
					applied = True

		return bool(applied)

