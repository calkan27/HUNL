from hunl.constants import EPS_SUM
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np

from hunl.engine.game_node import GameNode
from hunl.engine.public_state import PublicState
from hunl.engine.action import Action
from hunl.engine.action_type import ActionType
from hunl.engine.poker_utils import DECK


class LookaheadTreeBuilder:
	def __init__(
	 self,
	 depth_limit: int = 1,
	 bet_fractions: List[float] = None,
	 include_all_in: bool = True,
	 max_actions_per_branch: Optional[int] = None,
	):
		self.depth_limit = int(depth_limit)
		self.bet_fractions = list(bet_fractions or [0.5, 1.0])
		self.include_all_in = bool(include_all_in)
		if max_actions_per_branch is None:
			self.max_actions_per_branch = None
		else:
			self.max_actions_per_branch = int(max_actions_per_branch)
		self.leaf_callback: Optional[
		 Callable[[PublicState, int, List[float], List[float]], np.ndarray]
		] = None

	def set_leaf_callback(
	 self,
	 fn: Callable[[PublicState, int, List[float], List[float]], np.ndarray]
	) -> None:
		self.leaf_callback = fn

	def _action_menu(
	 self,
	 ps: PublicState,
	 for_player: bool,
	 pot_fracs: Tuple[float, ...],
	 is_root: bool,
	) -> List[ActionType]:
		if hasattr(ps, "legal_actions"):
			legal = ps.legal_actions()
		else:
			legal = []
		out: List[ActionType] = []
		if ActionType.FOLD in legal:
			out.append(ActionType.FOLD)
		if ActionType.CALL in legal:
			out.append(ActionType.CALL)
		want_half = False
		want_pot = False
		want_2pot = False
		i = 0
		while i < len(pot_fracs):
			f = float(pot_fracs[i])
			if abs(f - 0.5) < EPS_SUM:
				want_half = True
			if abs(f - 1.0) < EPS_SUM:
				want_pot = True
			if abs(f - 2.0) < EPS_SUM:
				want_2pot = True
			i += 1
		if want_half:
			if ActionType.HALF_POT_BET in legal:
				out.append(ActionType.HALF_POT_BET)
		if want_pot:
			if ActionType.POT_SIZED_BET in legal:
				out.append(ActionType.POT_SIZED_BET)
		if want_2pot:
			if ActionType.TWO_POT_BET in legal:
				out.append(ActionType.TWO_POT_BET)
		if self.include_all_in:
			if ActionType.ALL_IN in legal:
				out.append(ActionType.ALL_IN)
		if self.max_actions_per_branch is not None:
			if len(out) > self.max_actions_per_branch:
				out = out[: int(self.max_actions_per_branch)]
		return out

	def _deal_next_card(
	 self,
	 ps: PublicState
	) -> List[str]:
		board = list(getattr(ps, "board_cards", []))
		h0 = list(getattr(ps, "hole_cards", [[], []])[0])
		h1 = list(getattr(ps, "hole_cards", [[], []])[1])
		used = set(board + h0 + h1)
		avail = [c for c in DECK if c not in used]
		if int(ps.current_round) == 1:
			return [c for c in avail]
		else:
			if int(ps.current_round) == 2:
				return [c for c in avail]
			else:
				return []

	def _init_build_state(
	 self,
	 public_state: PublicState
	) -> Dict[str, Any]:
		root = GameNode(public_state)
		nodes: List[GameNode] = [root]
		parents: List[int] = [-1]
		edges: List[Any] = [None]
		if int(public_state.current_player) == int(public_state.dealer):
			k0 = "our"
		else:
			k0 = "opp"
		kinds: List[str] = [k0]
		depth_actions: List[int] = [0]
		menus: List[List[ActionType]] = [[]]
		stack: List[int] = [0]
		stage_start = int(public_state.current_round)
		return {
		 "nodes": nodes,
		 "parents": parents,
		 "edges": edges,
		 "kinds": kinds,
		 "depth_actions": depth_actions,
		 "menus": menus,
		 "stack": stack,
		 "stage_start": stage_start,
		}

	def _is_chance_node(
	 self,
	 ps: PublicState,
	 depth_actions: List[int],
	 idx: int,
	 stage_start: int
	) -> bool:
		bets = tuple(getattr(ps, "current_bets", (0, 0))[:2])
		at_root = (int(depth_actions[idx]) == 0)
		if bets[0] == bets[1]:
			if int(ps.current_round) < 3:
				if not (at_root and (stage_start <= 1)):
					return True
				else:
					return False
			else:
				return False
		else:
			return False

	def _expand_chance_children(
	 self,
	 state: Dict[str, Any],
	 ni: int
	) -> None:
		cur = state["nodes"][ni]
		ps = cur.public_state
		card_list = self._deal_next_card(ps)
		i = 0
		while i < len(card_list):
			card = card_list[i]
			if hasattr(ps, "clone"):
				ps2 = ps.clone()
			else:
				ps2 = ps
			if int(ps2.current_round) == 1:
				if card not in ps2.board_cards:
					ps2.board_cards.append(card)
					ps2.current_round = 2
					ps2.current_bets = [0, 0]
					ps2.last_raiser = None
					ps2.current_player = (ps2.dealer + 1) % 2
			else:
				if int(ps2.current_round) == 2:
					if card not in ps2.board_cards:
						ps2.board_cards.append(card)
						ps2.current_round = 3
						ps2.current_bets = [0, 0]
						ps2.last_raiser = None
						ps2.current_player = (ps2.dealer + 1) % 2
			child = GameNode(ps2)
			child.player_ranges = [
			 dict(cur.player_ranges[0]),
			 dict(cur.player_ranges[1]),
			]
			state["nodes"].append(child)
			state["parents"].append(ni)
			state["edges"].append(card)
			if int(state["stage_start"]) <= 1:
				state["kinds"].append("leaf")
				state["depth_actions"].append(state["depth_actions"][ni])
				state["menus"].append([])
			else:
				if int(ps2.current_player) == int(ps2.dealer):
					state["kinds"].append("our")
				else:
					state["kinds"].append("opp")
				state["depth_actions"].append(state["depth_actions"][ni])
				state["menus"].append([])
				state["stack"].append(len(state["nodes"]) - 1)
			i += 1

	def _advance_to_action_children(
	 self,
	 state: Dict[str, Any],
	 ni: int
	) -> None:
		cur = state["nodes"][ni]
		ps = cur.public_state
		actor = int(getattr(ps, "current_player", 0))
		is_our = (actor == int(getattr(ps, "dealer", 0)))
		menu = self._action_menu(
		 ps=ps,
		 for_player=is_our,
		 pot_fracs=tuple(self.bet_fractions),
		 is_root=bool(state["depth_actions"][ni] == 0),
		)
		state["menus"][ni] = menu
		if is_our:
			state["kinds"][ni] = "our"
		else:
			state["kinds"][ni] = "opp"
		i = 0
		while i < len(menu):
			a = menu[i]
			ps2 = ps.update_state(cur, Action(a))
			child = GameNode(ps2)
			child.player_ranges = [
			 dict(cur.player_ranges[0]),
			 dict(cur.player_ranges[1]),
			]
			state["nodes"].append(child)
			state["parents"].append(ni)
			state["edges"].append(a)
			if bool(getattr(ps2, "is_terminal", False)):
				state["kinds"].append("terminal")
				state["depth_actions"].append(state["depth_actions"][ni] + 1)
				state["menus"].append([])
				i += 1
				continue
			if int(state["stage_start"]) <= 1:
				if int(ps2.current_round) != int(state["stage_start"]):
					state["kinds"].append("leaf")
					state["depth_actions"].append(state["depth_actions"][ni] + 1)
					state["menus"].append([])
					i += 1
					continue
			if int(ps2.current_player) == int(ps2.dealer):
				state["kinds"].append("our")
			else:
				state["kinds"].append("opp")
			state["depth_actions"].append(state["depth_actions"][ni] + 1)
			state["menus"].append([])
			state["stack"].append(len(state["nodes"]) - 1)
			i += 1

	def _finalize_tree(
	 self,
	 state: Dict[str, Any]
	) -> Dict[str, Any]:
		return {
		 "nodes": state["nodes"],
		 "parents": state["parents"],
		 "edges": state["edges"],
		 "kinds": state["kinds"],
		 "depth_actions": state["depth_actions"],
		 "menus": state["menus"],
		 "stage_start": int(state["stage_start"]),
		}

	def build(
	 self,
	 public_state: PublicState
	) -> Dict[str, Any]:
		state = self._init_build_state(public_state)
		while state["stack"]:
			ni = state["stack"].pop()
			cur = state["nodes"][ni]
			ps = cur.public_state
			if bool(getattr(ps, "is_terminal", False)):
				state["kinds"][ni] = "terminal"
				continue
			if int(state["stage_start"]) <= 1:
				if int(ps.current_round) != int(state["stage_start"]):
					state["kinds"][ni] = "leaf"
					continue
			if int(state["depth_actions"][ni]) >= int(self.depth_limit):
				state["kinds"][ni] = "leaf"
				continue
			if self._is_chance_node(ps, state["depth_actions"], ni, int(state["stage_start"])):
				state["kinds"][ni] = "chance"
				self._expand_chance_children(state, ni)
				continue
			self._advance_to_action_children(state, ni)
		return self._finalize_tree(state)

	def propagate(
	 self,
	 tree: Dict[str, Any],
	 r_us: List[float],
	 r_opp: List[float],
	 pov_player: int,
	) -> Dict[str, Any]:
		K = int(len(r_us))
		N = int(len(tree["nodes"]))
		reach_us = np.zeros((N, K), dtype=float)
		reach_opp = np.zeros((N, K), dtype=float)
		root_idx = 0
		if K > 0:
			reach_us[root_idx, :] = np.asarray(r_us, dtype=float)
			reach_opp[root_idx, :] = np.asarray(r_opp, dtype=float)
		else:
			reach_us[root_idx, :] = np.zeros((0,), dtype=float)
			reach_opp[root_idx, :] = np.zeros((0,), dtype=float)
		children_by_parent: Dict[int, List[int]] = {}
		i = 0
		while i < N:
			p = int(tree["parents"][i])
			if p in children_by_parent:
				children_by_parent[p].append(i)
			else:
				children_by_parent[p] = [i]
			i += 1
		p = 0
		while p < N:
			if p < 0:
				p += 1
				continue
			idx = children_by_parent.get(int(p), [])
			if not idx:
				p += 1
				continue
			kind = tree["kinds"][p]
			if (kind == "our") or (kind == "opp"):
				d = len(idx)
				if d < 1:
					d = 1
				w = 1.0 / float(d)
				j = 0
				while j < len(idx):
					ci = int(idx[j])
					reach_us[ci, :] += reach_us[p, :] * w
					reach_opp[ci, :] += reach_opp[p, :] * w
					j += 1
			else:
				if kind == "chance":
					j = 0
					while j < len(idx):
						ci = int(idx[j])
						reach_us[ci, :] += reach_us[p, :]
						reach_opp[ci, :] += reach_opp[p, :]
						j += 1
			p += 1
		values: List[Optional[np.ndarray]] = [None] * N
		if self.leaf_callback is not None:
			i = 0
			while i < N:
				k = tree["kinds"][i]
				if (k == "leaf") or (k == "terminal"):
					ps = tree["nodes"][i].public_state
					v = self.leaf_callback(
					 ps,
					 int(pov_player),
					 list(reach_us[i, :]),
					 list(reach_opp[i, :]),
					)
					values[i] = np.asarray(v, dtype=float)
				i += 1
		return {
		 "reach_us": reach_us,
		 "reach_opp": reach_opp,
		 "values": values,
		}

