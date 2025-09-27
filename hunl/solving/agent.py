"""
I wrap CFRSolver into a simple acting interface suitable for interactive play or
integration tests. I own a solver, ensure devices are consistent, keep a HandClusterer
reference, and expose convenience methods to act, observe opponent/chance, adjust
latency profile, and load model bundles.

Key class: Agent. Key methods: act — bucketize our private hand, build uniform opponent
priors, run one re-solve, and return the chosen Action;
observe_opponent_action/observe_chance — advance internal trackers and range gadgets;
set_device — move solver nets; set_latency_profile — per-round iteration overrides;
load_bundle — load a CFV bundle (models + mapping) via model_io and apply to the solver.

Inputs: number of clusters, depth limit, optional iterations/device/profile/config; for
act: a PublicState and our private cards. Outputs: chosen Action plus updated internal
state; query/utility methods return small dicts or booleans.

Internal dependencies: CFRSolver, HandClusterer, ResolveConfig,
hunl.nets.model_io.load_cfv_bundle, engine primitives (GameNode/Action/ActionType).
External dependencies: torch (for device selection).

Invariants: num_clusters in Agent matches solver/clusterer; device consistency is
maintained across stage nets; last_public_key caches the public signature after each
act/observe call. Performance: per-decision compute is governed by round iteration
schedule; preflop cache inside solver reduces repeated resolves on identical public
states.
"""

import torch
from hunl.engine.game_node import GameNode
from hunl.engine.action import Action
from hunl.engine.action_type import ActionType
from hunl.solving.cfr_solver import CFRSolver
from hunl.ranges.hand_clusterer import HandClusterer
from hunl.resolve_config import ResolveConfig
from hunl.nets.model_io import load_cfv_bundle

class Agent:
	def __init__(self, num_clusters=1000, depth_limit=1, iterations=None, device=None, profile=None, config=None):
		if config is None:
			if iterations is not None:
				_total_iterations = int(iterations)
			else:
				_total_iterations = 1000

			cfg = ResolveConfig.from_env({
				"num_clusters": int(num_clusters),
				"depth_limit": int(depth_limit),
				"total_iterations": _total_iterations,
			})

			if profile is not None:
				cfg.profile = str(profile)

			self._config = cfg
		else:
			self._config = config

		self.solver = CFRSolver(config=self._config)
		self.solver.num_clusters = int(self._config.num_clusters)

		if device is not None:
			self.device = torch.device(device)
		else:
			_prefer_gpu = getattr(self._config, "prefer_gpu", True)
			if torch.cuda.is_available() and _prefer_gpu:
				_backend = "cuda"
			else:
				_backend = "cpu"
			self.device = torch.device(_backend)

		for k in list(self.solver.models.keys()):
			self.solver.models[k] = self.solver.models[k].to(self.device)

		if hasattr(self.solver, "hand_clusterer"):
			self.clusterer = self.solver.hand_clusterer
		else:
			self.clusterer = HandClusterer(
				self.solver,
				num_clusters=int(self._config.num_clusters),
				profile=self._config.profile
			)

		self.num_clusters = int(self._config.num_clusters)
		self.last_public_key = None


	def set_device(self, device):
		self.device = torch.device(device)
		for k in list(self.solver.models.keys()):
			self.solver.models[k] = self.solver.models[k].to(self.device)
		return str(self.device)

	def set_latency_profile(self, round_iters=None):
		if isinstance(round_iters, dict):
			if not hasattr(self.solver, "_round_iters"):
				self.solver._round_iters = {}
			for r, it in round_iters.items():
				self.solver._round_iters[int(r)] = int(it)
		return dict(getattr(self.solver, "_round_iters", {}))

	def _uniform_range(self):
		if self.num_clusters > 0:
			u = 1.0 / float(self.num_clusters)
		else:
			u = 0.0

		result = {}
		for i in range(self.num_clusters):
			result[i] = u

		return result


	def _range_on_bucket(self, cid):
		r = {i: 0.0 for i in range(self.num_clusters)}
		if 0 <= int(cid) < self.num_clusters:
			r[int(cid)] = 1.0
		return r

	def _bucketize_own_hand(self, cards, board):
		if isinstance(cards, str):
			h = cards
		else:
			h = " ".join(cards)
		return int(self.clusterer.hand_to_bucket(h))

	def _public_key(self, ps):
		cb = getattr(ps, "current_bets", (0.0, 0.0))

		_cp = getattr(ps, "current_player", None)
		if _cp is not None:
			current_player_val = int(_cp)
		else:
			current_player_val = -1

		_players = getattr(ps, "players_in_hand", [True, True])[0:2]
		players_in_hand_bools = []
		for x in _players:
			players_in_hand_bools.append(bool(x))
		players_in_hand_tuple = tuple(players_in_hand_bools)

		return (
			tuple(getattr(ps, "board_cards", [])),
			int(getattr(ps, "current_round", 0)),
			(int(cb[0]), int(cb[1])),
			int(getattr(ps, "pot_size", 0.0)),
			current_player_val,
			int(getattr(ps, "dealer", 0)),
			bool(getattr(ps, "is_terminal", False)),
			bool(getattr(ps, "is_showdown", False)),
			players_in_hand_tuple,
		)


	def act(self, public_state, our_private_cards):
		node = GameNode(public_state)
		if isinstance(our_private_cards, str):
			cards = our_private_cards.split()
		elif isinstance(our_private_cards, (list, tuple)):
			cards = list(our_private_cards)
		else:
			cards = []
		cid = self._bucketize_own_hand(cards, list(public_state.board_cards))
		r_self = self._range_on_bucket(cid)
		r_opp = self._uniform_range()
		node.player_ranges[public_state.current_player] = r_self
		node.player_ranges[(public_state.current_player + 1) % 2] = r_opp
		self.solver.total_iterations = int(getattr(self.solver, "_round_iters", {}).get(int(public_state.current_round), 
					getattr(self._config, "total_iterations", 1000)))

		res = self.solver.run_cfr(node)
		self.last_public_key = self._public_key(node.public_state)
		return res

	def observe_opponent_action(self, prev_public_state, new_public_state, observed_action_type):
		prev_node = GameNode(prev_public_state)
		next_node = GameNode(new_public_state)
		self.solver.apply_opponent_action_update(prev_node, next_node, observed_action_type)
		self.last_public_key = self._public_key(new_public_state)
		return True

	def observe_chance(self, new_public_state):
		node = GameNode(new_public_state)
		self.solver.lift_ranges_after_chance(node)
		self.last_public_key = self._public_key(new_public_state)
		return True

	def load_bundle(self, path):
		b = load_cfv_bundle(path, device=self.device)
		applied = self.solver.apply_cfv_bundle(b, device=self.device)
		return {"loaded_models": list(b.get("models", {}).keys()), "applied": applied}

