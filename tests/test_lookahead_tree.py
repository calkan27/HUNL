"""
Test suite for LookaheadTreeBuilder action menus, next-card dealing, tree building under depth limits, and reach propagation with leaf callbacks.
"""

import hunl.engine.poker_utils as poker_utils
import numpy as np
import pytest

from hunl.engine.action_type import ActionType
from hunl.engine.action import Action
from hunl.solving.lookahead_tree import LookaheadTreeBuilder
from hunl.engine.public_state import PublicState
from hunl.engine.game_node import GameNode
from hunl.engine.poker_utils import DECK


class _FakePSForMenu:
	"""
	Minimal public-state stand-in exposing a legal_actions() method to test action menu construction.
	"""
	def __init__(self, legal):
		"""
		Initialize with a fixed list of legal actions returned by legal_actions().
		"""
		self._legal = list(legal)

	def legal_actions(self):
		"""
		Return the fixed list of legal actions supplied at construction time.
		"""
		return list(self._legal)


def test_action_menu_sparse_root_and_all_in_toggle():
	"""
	Validate that _action_menu respects configured bet_fractions, includes or excludes ALL_IN by flag, and 
	filters to the environment's legal action set.
	"""
	b = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=True)
	legal = [ActionType.CALL, ActionType.HALF_POT_BET, ActionType.POT_SIZED_BET, ActionType.ALL_IN]
	ps = _FakePSForMenu(legal)
	menu = b._action_menu(ps, for_player=True, pot_fracs=(0.5, 1.0), is_root=True)
	assert ActionType.CALL in menu
	assert ActionType.HALF_POT_BET in menu
	assert ActionType.POT_SIZED_BET in menu
	assert ActionType.ALL_IN in menu
	assert ActionType.TWO_POT_BET not in menu
	b2 = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[2.0], include_all_in=False)
	legal2 = [ActionType.CALL, ActionType.TWO_POT_BET, ActionType.ALL_IN]
	ps2 = _FakePSForMenu(legal2)
	menu2 = b2._action_menu(ps2, for_player=True, pot_fracs=(2.0,), is_root=True)
	assert ActionType.CALL in menu2
	assert ActionType.TWO_POT_BET in menu2
	assert ActionType.ALL_IN not in menu2


def test_deal_next_card_excludes_used_cards():
	"""
	Ensure _deal_next_card enumerates only unused cards by excluding current board and both players' hole cards, 
	with no duplicates in the result set.
	"""
	deck = list(DECK)
	board = deck[:3]
	hole0 = deck[3:5]
	hole1 = deck[5:7]
	ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
	ps.hole_cards[0] = list(hole0)
	ps.hole_cards[1] = list(hole1)
	ps.current_round = 1
	b = LookaheadTreeBuilder(depth_limit=0)
	nxt = b._deal_next_card(ps)
	used = set(board + hole0 + hole1)
	assert all(c not in used for c in nxt)
	assert len(set(nxt)) == len(nxt)


def test_build_depth_limit_and_leaf_kind(monkeypatch):
	"""
	Confirm that build respects depth_limit by yielding a leaf or terminal at the root and records the correct
	stage_start for a flop state.
	"""
	deck = list(DECK)
	board = deck[:3]
	ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
	ps.current_round = 1
	ps.current_bets = [10, 0]
	ps.last_raiser = 0
	ps.current_player = 1
	b = LookaheadTreeBuilder(depth_limit=0, bet_fractions=[0.5, 1.0], include_all_in=True)
	tree = b.build(ps)
	assert tree["kinds"][0] in ("leaf", "terminal")
	assert tree["stage_start"] == 1


def test_propagate_reach_and_leaf_callback():
	"""
	Verify that propagate computes reach probabilities, invokes the registered leaf callback, and produces at 
	least one leaf/terminal value entry.
	"""
	deck = list(DECK)
	board = deck[:3]
	ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
	ps.current_round = 1
	ps.current_bets = [10, 0]
	ps.last_raiser = 0
	ps.current_player = 1
	b = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=False)
	tree = b.build(ps)
	r_us = [0.5, 0.5]
	r_opp = [0.5, 0.5]

	def cb(pstate, pov, r1, r2):
		"""
		Return a per-bucket value vector equal to the acting player's range to test propagation of reach into leaf evaluation.
		"""
		return np.array(r1, dtype=float)

	b.set_leaf_callback(cb)
	out = b.propagate(tree, r_us, r_opp, pov_player=ps.current_player)
	assert "reach_us" in out and "reach_opp" in out and "values" in out
	assert any(v is not None for v in out["values"])

