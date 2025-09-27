from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from hunl.engine.action_type import ActionType

class PublicChanceCFR:
	def __init__(
	 self,
	 depth_limit: int = 1,
	 bet_fractions: List[float] = None,
	 include_all_in: bool = True,
	 regret_matching_plus: bool = True,
	 importance_weighting: bool = True,
	):
		self.depth_limit = int(depth_limit)
		self.bet_fractions = list(bet_fractions or [0.5, 1.0])
		self.include_all_in = bool(include_all_in)
		self.rm_plus = bool(regret_matching_plus)
		self.use_iw = bool(importance_weighting)
		self.regret: Dict[Any, List[float]] = {}
		self.strat_sum: Dict[Any, List[float]] = {}

	def _infoset_key(self, node: Dict[str, Any], idx: int, pov_player: int) -> Tuple:
		ps = node["nodes"][idx].public_state

		if hasattr(ps, "current_bets"):
			cb0 = int(getattr(ps, "current_bets", (0, 0))[0])
		else:
			cb0 = 0

		if hasattr(ps, "current_bets"):
			cb1 = int(getattr(ps, "current_bets", (0, 0))[1])
		else:
			cb1 = 0

		return (
		 tuple(getattr(ps, "board_cards", [])),
		 int(getattr(ps, "current_round", 0)),
		 (cb0, cb1),
		 int(getattr(ps, "current_player", 0)),
		 int(pov_player),
		)

	def _policy_from_regret(self, key: Any, A: int, mask: List[bool]) -> List[float]:
		if key not in self.regret:
			self.regret[key] = [0.0] * A
			self.strat_sum[key] = [0.0] * A

		r = self.regret[key]

		if self.rm_plus:
			rp = []
			for i in range(A):
				if mask[i]:
					rp.append(max(0.0, r[i]))
				else:
					rp.append(0.0)
		else:
			rp = []
			for i in range(A):
				if (r[i] > 0.0) and mask[i]:
					rp.append(r[i])
				else:
					rp.append(0.0)

		s = sum(rp)

		if s <= 0.0:
			k = 0
			for b in mask:
				if b:
					k += 1
			if k < 1:
				k = 1
			result = []
			for i in range(A):
				if mask[i]:
					result.append(1.0 / float(k))
				else:
					result.append(0.0)
			return result
		else:
			result = []
			for i in range(A):
				if mask[i]:
					result.append(rp[i] / s)
				else:
					result.append(0.0)
			return result

	def _chance_children(self, tree: Dict[str, Any], idx: int) -> List[int]:
		result = []
		for i, p in enumerate(tree["parents"]):
			if p == idx:
				result.append(i)
		return result

	def _menu_mask(self, menu: List[Any], A: int) -> List[bool]:
		mask = [False] * A
		for a in menu:
			mask[int(getattr(a, "value", 0))] = True
		return mask

	def _opponent_cfv_upper_bound_value(self, opp_cfv_upper_vec: List[float], r_opp: List[float]) -> float:
		if (not opp_cfv_upper_vec) or (not r_opp):
			return 0.0
		n = min(len(opp_cfv_upper_vec), len(r_opp))
		s = 0.0
		i = 0
		while i < n:
			s += float(r_opp[i]) * float(opp_cfv_upper_vec[i])
			i += 1
		return -float(s)

	def _evaluate_leaf(
	 self,
	 leaf_value_fn,
	 ps,
	 pov_player: int,
	 r_us: List[float],
	 r_opp: List[float],
	) -> float:
		v = leaf_value_fn(ps, pov_player, r_us, r_opp)
		arr = np.asarray(v, dtype=float).reshape(-1)

		if arr.size == len(r_us):
			return float(np.dot(arr, np.asarray(r_us, dtype=float)))
		else:
			if arr.size == 1:
				return float(arr[0])
			else:
				return 0.0

	def solve_subgame(
	 self,
	 root_node,
	 r_us,
	 r_opp,
	 opp_cfv_constraints,
	 T,
	 leaf_value_fn,
	):
		self.regret.clear()
		self.strat_sum.clear()

		root_idx = 0
		pov_player = int(getattr(root_node["nodes"][root_idx].public_state, "current_player", 0))

		if int(T) > 0:
			iters = int(T)
		else:
			iters = 1

		k = 0
		while k < iters:
			self._external_sample_traverse(
			 tree=root_node,
			 idx=root_idx,
			 pov_player=pov_player,
			 r_us=list(r_us),
			 r_opp=list(r_opp),
			 opp_cfv_upper_vec=list(opp_cfv_constraints or []),
			 leaf_value_fn=leaf_value_fn,
			 iw_c=1.0,
			 root_idx=root_idx,
			 apply_root_gadget=True,
			)
			k += 1

		menu0 = root_node["menus"][root_idx]
		if not menu0:
			menu0 = [ActionType.CALL]

		_max_val = 0
		has_any = False
		for a in menu0:
			val = int(getattr(a, "value", 0))
			has_any = True
			if val > _max_val:
				_max_val = val

		if has_any:
			Aall = _max_val + 1
		else:
			Aall = 1
		if Aall < 1:
			Aall = 1

		key0 = self._infoset_key(root_node, root_idx, pov_player)

		if key0 not in self.strat_sum:
			n = 0
			for _ in menu0:
				n += 1
			if n < 1:
				n = 1
			root_policy = {}
			for a in menu0:
				root_policy[a] = 1.0 / float(n)
		else:
			ss = self.strat_sum[key0]
			s = 0.0
			for a in menu0:
				s += ss[int(getattr(a, "value", 0))]
			if s <= 0.0:
				n = 0
				for _ in menu0:
					n += 1
				if n < 1:
					n = 1
				root_policy = {}
				for a in menu0:
					root_policy[a] = 1.0 / float(n)
			else:
				root_policy = {}
				for a in menu0:
					root_policy[a] = ss[int(getattr(a, "value", 0))] / s

		node_values = {}

		opp_cfv = {}
		if opp_cfv_constraints is not None:
			nc = len(opp_cfv_constraints)
		else:
			nc = 0
		i = 0
		while i < nc:
			opp_cfv[i] = float(opp_cfv_constraints[i])
			i += 1

		return root_policy, node_values, opp_cfv

	def set_warm_start(self, warm_start: Optional[Dict[Any, List[float]]] = None) -> None:
		self._warm_start = {}
		if not isinstance(warm_start, dict):
			return
		for k, v in warm_start.items():
			if isinstance(v, (list, tuple)):
				vec = []
				i = 0
				while i < len(v):
					x = v[i]
					if isinstance(x, (int, float)):
						vec.append(float(x))
					else:
						if isinstance(x, bool):
							vec.append(float(int(x)))
						else:
							vec = []
							break
					i += 1
				if vec:
					self._warm_start[k] = vec
			else:
				if isinstance(v, dict):
					keys = list(v.keys())
					int_keys = []
					j = 0
					while j < len(keys):
						tk = keys[j]
						if isinstance(tk, int):
							int_keys.append(tk)
						else:
							if isinstance(tk, str):
								s = tk.strip()
								if s.startswith("-") or s.startswith("+"):
									ns = s[1:]
								else:
									ns = s
								if ns.isdigit():
									int_keys.append(int(s))
						j += 1
					int_keys.sort()
					vec2 = []
					m = 0
					while m < len(int_keys):
						x = v.get(int_keys[m], None)
						if isinstance(x, (int, float)):
							vec2.append(float(x))
						else:
							if isinstance(x, bool):
								vec2.append(float(int(x)))
							else:
								vec2 = []
								break
						m += 1
					if vec2:
						self._warm_start[k] = vec2
				else:
					continue

	def _traverse_terminal_or_leaf(
	 self,
	 tree,
	 idx,
	 pov_player,
	 r_us,
	 r_opp,
	 leaf_value_fn,
	 iw_c,
	):
		ps = tree["nodes"][idx].public_state
		base = self._evaluate_leaf(leaf_value_fn, ps, pov_player, r_us, r_opp)

		if self.use_iw:
			factor = iw_c
		else:
			factor = 1.0

		return base * factor

	def _traverse_chance(
	 self,
	 tree,
	 idx,
	 pov_player,
	 r_us,
	 r_opp,
	 opp_cfv_upper_vec,
	 leaf_value_fn,
	 iw_c,
	 root_idx,
	 apply_root_gadget,
	):
		children = self._chance_children(tree, idx)

		if not children:
			return self._traverse_terminal_or_leaf(tree, idx, pov_player, r_us, r_opp, leaf_value_fn, iw_c)

		p = 1.0 / float(len(children))
		ci = children[np.random.randint(0, len(children))]

		if self.use_iw:
			iw_factor = (1.0 / p)
		else:
			iw_factor = 1.0

		return self._external_sample_traverse(
		 tree, ci, pov_player, r_us, r_opp, opp_cfv_upper_vec, leaf_value_fn, iw_c * iw_factor, root_idx, apply_root_gadget
		)

	def _action_to_child_map(self, tree, idx):
		children = self._chance_children(tree, idx)
		action_to_child = {}

		for ci in children:
			edge = tree["edges"][ci]
			if edge is None:
				continue
			aid = int(getattr(edge, "value", 0))
			action_to_child[aid] = ci

		return action_to_child

	def _traverse_our(
	 self,
	 tree,
	 idx,
	 pov_player,
	 r_us,
	 r_opp,
	 opp_cfv_upper_vec,
	 leaf_value_fn,
	 iw_c,
	 root_idx,
	 apply_root_gadget,
	 menu,
	 Aall,
	):
		key = self._infoset_key(tree, idx, pov_player)
		mask = self._menu_mask(menu, Aall)
		policy = self._policy_from_regret(key, Aall, mask)

		action_to_child = self._action_to_child_map(tree, idx)
		q_vals = [0.0] * Aall

		for aid in range(Aall):
			if not mask[aid]:
				continue
			ci = action_to_child.get(aid, None)
			if ci is None:
				continue
			q_vals[aid] = self._external_sample_traverse(
			 tree, ci, pov_player, r_us, r_opp, opp_cfv_upper_vec, leaf_value_fn, iw_c, root_idx, apply_root_gadget
			)

		ev = 0.0
		for aid in range(Aall):
			if mask[aid]:
				ev += policy[aid] * q_vals[aid]

		for aid in range(Aall):
			if mask[aid]:
				self.regret[key][aid] += (q_vals[aid] - ev)
				self.strat_sum[key][aid] += policy[aid]

		return ev

	def _traverse_opp(
	 self,
	 tree,
	 idx,
	 pov_player,
	 r_us,
	 r_opp,
	 opp_cfv_upper_vec,
	 leaf_value_fn,
	 iw_c,
	 root_idx,
	 apply_root_gadget,
	 menu,
	 Aall,
	):
		action_to_child = self._action_to_child_map(tree, idx)
		key_opp = self._infoset_key(tree, idx, (pov_player + 1) % 2)
		mask = self._menu_mask(menu, Aall)
		pol_opp = self._policy_from_regret(key_opp, Aall, mask)

		q_vals = [0.0] * Aall

		for aid in range(Aall):
			if not mask[aid]:
				continue
			ci = action_to_child.get(aid, None)
			if ci is None:
				continue
			q_vals[aid] = self._external_sample_traverse(
			 tree, ci, pov_player, r_us, r_opp, opp_cfv_upper_vec, leaf_value_fn, iw_c, root_idx, apply_root_gadget
			)

		ev_hero = 0.0
		for aid in range(Aall):
			if mask[aid]:
				ev_hero += pol_opp[aid] * q_vals[aid]

		for aid in range(Aall):
			if mask[aid]:
				v_opp_a = -q_vals[aid]
				ev_opp = -ev_hero
				self.regret[key_opp][aid] += (v_opp_a - ev_opp)
				self.strat_sum[key_opp][aid] += pol_opp[aid]

		if apply_root_gadget:
			if idx == root_idx:
				val_terminate_hero = self._opponent_cfv_upper_bound_value(list(opp_cfv_upper_vec or []), list(r_opp or []))
				if ev_hero < val_terminate_hero:
					return ev_hero
				else:
					return val_terminate_hero
			else:
				return ev_hero
		else:
			return ev_hero

	def _external_sample_traverse(
	 self,
	 tree,
	 idx,
	 pov_player,
	 r_us,
	 r_opp,
	 opp_cfv_upper_vec,
	 leaf_value_fn,
	 iw_c,
	 root_idx,
	 apply_root_gadget,
	):
		kind = tree["kinds"][idx]

		if kind in ("terminal", "leaf"):
			return self._traverse_terminal_or_leaf(tree, idx, pov_player, r_us, r_opp, leaf_value_fn, iw_c)
		else:
			if kind == "chance":
				return self._traverse_chance(
				 tree, idx, pov_player, r_us, r_opp, opp_cfv_upper_vec, leaf_value_fn, iw_c, root_idx, apply_root_gadget
				)
			else:
				menu = tree["menus"][idx]
				_max_val = 0
				has_any = False

				for a in menu:
					val = int(getattr(a, "value", 0))
					has_any = True
					if val > _max_val:
						_max_val = val

				if has_any:
					Aall = _max_val + 1
				else:
					Aall = 1

				if Aall < 1:
					Aall = 1

				if kind == "our":
					return self._traverse_our(
					 tree, idx, pov_player, r_us, r_opp, opp_cfv_upper_vec, leaf_value_fn, iw_c, root_idx, apply_root_gadget, menu, Aall
					)
				else:
					if kind == "opp":
						return self._traverse_opp(
						 tree, idx, pov_player, r_us, r_opp, opp_cfv_upper_vec, leaf_value_fn, iw_c, root_idx, apply_root_gadget, menu, Aall
						)
					else:
						return self._traverse_terminal_or_leaf(
						 tree, idx, pov_player, r_us, r_opp, leaf_value_fn, iw_c
						)
