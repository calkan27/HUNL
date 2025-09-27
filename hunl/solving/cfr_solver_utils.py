import itertools
from typing import Dict, Any

from hunl.engine.poker_utils import DECK, best_hand, hand_rank
from hunl.engine.game_node import GameNode
from hunl.engine.action_type import ActionType
from hunl.solving.range_gadget import RangeGadget


class CFRSolverUtilsMixin:
	def generate_all_possible_hands(self):
		hands = []
		for combo in itertools.combinations(DECK, 2):
			hands.append(" ".join(combo))
		return hands

	def _stable_seed(self, entity: Any, board: list[str]) -> int:
		s = f"{str(entity)}|{','.join(board)}"
		h = 0
		i = 0
		while i < len(s):
			h = (h * 131 + ord(s[i])) & 0xFFFFFFFF
			i += 1
		return h

	def _player_wins(self, player_hand: list[str], opponent_hand: list[str], board: list[str]) -> int:
		ph = list(player_hand) + list(board)
		oh = list(opponent_hand) + list(board)
		rp = hand_rank(best_hand(ph))
		ro = hand_rank(best_hand(oh))
		if rp > ro:
			return 1
		if ro > rp:
			return -1
		return 0

	def get_stage(self, node: GameNode) -> str:
		cr = int(node.public_state.current_round)
		if cr == 0:
			return "preflop"
		elif cr == 1:
			return "flop"
		elif cr == 2:
			return "turn"
		elif cr == 3:
			return "river"
		return "none"

	def _count_compatible_hands(self, hset, board_set: set) -> int:
		if not hset:
			return 0
		c = 0
		for h in hset:
			if isinstance(h, str):
				parts = h.split()
				if len(parts) >= 2:
					a = parts[0]
					b = parts[1]
				else:
					a = ""
					b = ""
			else:
				pair = list(h)
				if len(pair) >= 2:
					a = pair[0]
					b = pair[1]
				else:
					a = ""
					b = ""
			if a == b:
				continue
			if (a in board_set) or (b in board_set):
				continue
			c += 1
		return c

	def _renormalize_range_vs_board(self, r: dict, clusters: dict, board_set: set) -> dict:
		num = {}
		for cid, p in dict(r).items():
			cid_i = int(cid)
			hset = clusters.get(cid_i, set())
			cnum = self._count_compatible_hands(hset, board_set)
			if cnum > 0:
				num[cid_i] = float(p) * float(cnum)
		if not num:
			avail = {}
			for k, v in clusters.items():
				k_i = int(k)
				cnt = self._count_compatible_hands(v, board_set)
				if cnt > 0:
					avail[k_i] = cnt
			if not avail:
				return r
			u = 1.0 / float(len(avail))
			out = {}
			for k in avail.keys():
				out[int(k)] = u
			return out
		s = 0.0
		for v in num.values():
			s += float(v)
		if s > 0.0:
			for k in list(num.keys()):
				num[k] = float(num[k]) / float(s)
		return num

	def lift_ranges_after_chance(self, node: GameNode) -> dict:
		ps = node.public_state
		board = list(getattr(ps, "board_cards", []))
		board_set = set(board)
		cl = getattr(self, "clusters", {}) or {}
		for pl in (0, 1):
			if isinstance(node.player_ranges[pl], dict):
				node.player_ranges[pl] = self._renormalize_range_vs_board(
				 node.player_ranges[pl],
				 cl,
				 board_set
				)
		if not hasattr(self, "own_range_tracking"):
			self.own_range_tracking = {}
		get_key = getattr(self, "_state_key", None)
		if callable(get_key):
			key = get_key(node)
			cp = int(getattr(node.public_state, "current_player", 0))
			self.own_range_tracking[key] = {}
			for k, v in dict(node.player_ranges[cp]).items():
				self.own_range_tracking[key][int(k)] = float(v)
		return {0: dict(node.player_ranges[0]), 1: dict(node.player_ranges[1])}

	def apply_opponent_action_update(self, prev_node: GameNode, new_node: GameNode, observed_action_type: ActionType):
		if not hasattr(self, "opponent_cfv_upper_tracking"):
			self.opponent_cfv_upper_tracking = {}
		if not hasattr(self, "own_range_tracking"):
			self.own_range_tracking = {}
		get_key = getattr(self, "_state_key", None)
		if not callable(get_key):
			return
		prev_key = get_key(prev_node)
		next_key = get_key(new_node)
		prev_u = dict(self.opponent_cfv_upper_tracking.get(prev_key, {}))
		next_u = dict(self.opponent_cfv_upper_tracking.get(next_key, {}))
		merged = {}
		for k in set(list(prev_u.keys()) + list(next_u.keys())):
			a = float(prev_u.get(int(k), float("-inf")))
			b = float(next_u.get(int(k), float("-inf")))
			merged[int(k)] = a if (a >= b) else b
		self.opponent_cfv_upper_tracking[next_key] = merged
		prev_own = dict(self.own_range_tracking.get(prev_key, {}))
		if prev_own:
			self.own_range_tracking[next_key] = {int(k): float(v) for k, v in prev_own.items()}

	def update_tracking_on_own_action(self, node: GameNode, agent_player: int = 0, counterfactual_values: Dict[int, Dict[int, float]] = None):
		if not hasattr(self, "own_range_tracking"):
			self.own_range_tracking = {}
		if not hasattr(self, "opponent_cfv_upper_tracking"):
			self.opponent_cfv_upper_tracking = {}
		key = self._state_key(node) if hasattr(self, "_state_key") else None
		if key is None:
			return
		if hasattr(self, "cfr_values"):
			if node in self.cfr_values:
				values = self.cfr_values[node]
			else:
				values = None
		else:
			values = None
		if hasattr(node, "player_ranges"):
			priors = dict(node.player_ranges[agent_player])
		else:
			priors = {}
		if hasattr(self, "_allowed_actions_agent"):
			allowed = self._allowed_actions_agent(node.public_state)
		else:
			allowed = []
		a_idx = None
		last = getattr(node.public_state, "last_action", None)
		if last is not None:
			if hasattr(last, "action_type"):
				a_idx = int(last.action_type.value)
		post = {}
		norm = 0.0
		for cid, prior in priors.items():
			w = float(prior)
			if (values is not None) and (a_idx is not None):
				base = values.get_average_strategy(int(cid))
				m = self._mask_strategy(base, allowed)
				if (0 <= a_idx) and (a_idx < len(m)):
					w *= float(m[a_idx])
				else:
					w = 0.0
			post[int(cid)] = w
			norm += w
		if norm > 0.0:
			for cid in list(post.keys()):
				post[cid] = post[cid] / norm
		else:
			if len(priors) > 0:
				u = 1.0 / float(len(priors))
				for cid in list(post.keys()):
					post[cid] = u
		self.own_range_tracking[key] = post
		opp = (agent_player + 1) % 2
		if isinstance(counterfactual_values, dict):
			avg_src = counterfactual_values.get(opp, {})
		else:
			avg_src = {}
		avg = {}
		for cid, vec in avg_src.items():
			if isinstance(vec, (list, tuple)):
				if len(vec) > 0:
					s = 0.0
					n = 0
					for x in vec:
						s += float(x)
						n += 1
					if n > 0:
						mu = s / float(n)
					else:
						mu = 0.0
				else:
					mu = 0.0
			else:
				if isinstance(vec, (int, float)):
					mu = float(vec)
				else:
					mu = 0.0
			avg[int(cid)] = mu
		self.opponent_cfv_upper_tracking[key] = avg

	def _range_gadget_for(self, node: GameNode):
		if not hasattr(self, "_range_gadgets"):
			self._range_gadgets = {}
		key = self._state_key(node) if hasattr(self, "_state_key") else None
		if key is None:
			return None
		if key not in self._range_gadgets:
			self._range_gadgets[key] = RangeGadget()
		return self._range_gadgets[key]

	def _range_gadget_commit(self, node: GameNode, opp_upper_dict: Dict[int, float]):
		g = self._range_gadget_for(node)
		if g is None:
			return {}
		u = g.update(dict(opp_upper_dict))
		if not hasattr(self, "opponent_cfv_upper_tracking"):
			self.opponent_cfv_upper_tracking = {}
		self.opponent_cfv_upper_tracking[self._state_key(node)] = dict(u)
		return dict(u)

	def _range_gadget_begin(self, node: GameNode):
		g = self._range_gadget_for(node)
		if g is None:
			return {}
		prev = getattr(self, "opponent_cfv_upper_tracking", {})
		u = g.begin(prev.get(self._state_key(node), {}))
		if not hasattr(self, "opponent_cfv_upper_tracking"):
			self.opponent_cfv_upper_tracking = {}
		self.opponent_cfv_upper_tracking[self._state_key(node)] = dict(u)
		return dict(u)

	def _ensure_sparse_schedule(self):
		if hasattr(self, "_round_iters") and hasattr(self, "_omit_prefix_iters") and hasattr(self, "_round_actions"):
			return
		self._round_iters = {0: 0, 1: 1000, 2: 2000, 3: 500}
		self._round_iters_on_cache = {0: 0}
		self._omit_prefix_iters = {"preflop": 980, "flop": 500, "turn": 500, "river": 1000}
		pf = bool(getattr(getattr(self, "_config", None), "paper_faithful", False))
		if pf:
			self._round_actions = {0: {"half_pot": False, "two_pot": False}, 1: {"half_pot": False, "two_pot": False}, 2: {"half_pot": False, "two_pot": False}, 3: {"half_pot": False, "two_pot": False}}
		else:
			self._round_actions = {0: {"half_pot": True, "two_pot": True}, 1: {"half_pot": False, "two_pot": False}, 2: {"half_pot": False, "two_pot": False}, 3: {"half_pot": False, "two_pot": False}}

	def _as_int(self, x, default_val: int) -> int:
		if isinstance(x, bool):
			return int(x)
		if isinstance(x, int):
			return x
		if isinstance(x, float):
			return int(x)
		if isinstance(x, str):
			s = x.strip()
			sign_ok = (s.startswith("-") and s[1:].isdigit()) or s.isdigit()
			if sign_ok:
				return int(s)
			else:
				return default_val
		if hasattr(x, "__int__"):
			v = x.__int__()
			if isinstance(v, int):
				return v
			else:
				return default_val
		return default_val

	def apply_round_iteration_schedule(self, round_index: int) -> int:
		self._ensure_sparse_schedule()
		ri = self._as_int(round_index, 0)
		default_iters = self._as_int(getattr(self, "total_iterations", 0), 0)
		val = self._round_iters.get(ri, default_iters)
		it = self._as_int(val, default_iters)
		self.total_iterations = it
		return it

	def _extract_hand_ab(self, h):
		if isinstance(h, str):
			parts = h.split()
			if len(parts) >= 2:
				return parts[0], parts[1]
			return "", ""
		else:
			pair = list(h)
			if len(pair) >= 2:
				return pair[0], pair[1]
			return "", ""

	def _hand_usable(self, a: str, b: str, board_set: set) -> bool:
		if a == b:
			return False
		if (a in board_set) or (b in board_set):
			return False
		return True

	def _collect_usable_hands(self, base_clusters: dict, board_set: set) -> set:
		hands = set()
		for _, hs in base_clusters.items():
			for h in hs:
				a, b = self._extract_hand_ab(h)
				if self._hand_usable(a, b, board_set):
					hands.add(f"{a} {b}")
		return hands

	def _spread_range_over_compatible(self, old_r: dict, base_clusters: dict, board_set: set) -> dict:
		out = {}
		for cid, p in dict(old_r).items():
			hset = base_clusters.get(int(cid), set())
			comp = []
			for h in hset:
				a, b = self._extract_hand_ab(h)
				if self._hand_usable(a, b, board_set):
					comp.append(f"{a} {b}")
			if not comp:
				continue
			w = float(p) / float(len(comp))
			for h in comp:
				prev = out.get(h, 0.0)
				out[h] = prev + w
		return out

	def _push_full_hand_expansion(self, node: GameNode):
		ps = node.public_state
		board_set = set(list(getattr(ps, "board_cards", [])))
		prev = {
		 "clusters": getattr(self, "clusters", None),
		 "num_clusters": int(getattr(self, "num_clusters", 0)),
		 "ranges": [dict(node.player_ranges[0]), dict(node.player_ranges[1])],
		}
		base_clusters = dict(getattr(self, "clusters", {}) or {})
		hands = self._collect_usable_hands(base_clusters, board_set)
		if not hands:
			all_h = []
			for a, b in itertools.combinations([c for c in DECK if c not in board_set], 2):
				all_h.append(f"{a} {b}")
			hands = set(all_h)
		order = sorted(list(hands))
		new_clusters = {}
		i = 0
		while i < len(order):
			new_clusters[i] = {order[i]}
			i += 1
		old_ranges = [dict(node.player_ranges[0]), dict(node.player_ranges[1])]
		new_ranges = [{}, {}]
		sp0 = self._spread_range_over_compatible(old_ranges[0], base_clusters, board_set)
		sp1 = self._spread_range_over_compatible(old_ranges[1], base_clusters, board_set)
		if not sp0:
			u0 = 1.0 / float(len(order)) if len(order) > 0 else 0.0
			i = 0
			while i < len(order):
				new_ranges[0][i] = u0
				i += 1
		else:
			s0 = 0.0
			for v in sp0.values():
				s0 += float(v)
			if s0 > 0.0:
				i = 0
				while i < len(order):
					h = order[i]
					new_ranges[0][i] = float(sp0.get(h, 0.0)) / s0
					i += 1
		if not sp1:
			u1 = 1.0 / float(len(order)) if len(order) > 0 else 0.0
			i = 0
			while i < len(order):
				new_ranges[1][i] = u1
				i += 1
		else:
			s1 = 0.0
			for v in sp1.values():
				s1 += float(v)
			if s1 > 0.0:
				i = 0
				while i < len(order):
					h = order[i]
					new_ranges[1][i] = float(sp1.get(h, 0.0)) / s1
					i += 1
		self.clusters = new_clusters
		self.num_clusters = len(order)
		node.player_ranges[0] = dict(new_ranges[0])
		node.player_ranges[1] = dict(new_ranges[1])
		return prev

	def _pop_full_hand_expansion(self, snap, node: GameNode):
		if not isinstance(snap, dict):
			return False
		self.clusters = snap.get("clusters", self.clusters)
		self.num_clusters = int(snap.get("num_clusters", getattr(self, "num_clusters", 0)))
		if "ranges" in snap:
			if isinstance(snap["ranges"], list):
				if len(snap["ranges"]) >= 2:
					node.player_ranges[0] = dict(snap["ranges"][0])
					node.player_ranges[1] = dict(snap["ranges"][1])
		return True
