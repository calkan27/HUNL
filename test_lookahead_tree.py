import numpy as np
import pytest

from action_type import ActionType
from action import Action
from lookahead_tree import LookaheadTreeBuilder
from public_state import PublicState
from game_node import GameNode
from poker_utils import DECK


class _FakePSForMenu:
    def __init__(self, legal):
        self._legal = list(legal)
    def legal_actions(self):
        return list(self._legal)


def test_action_menu_sparse_root_and_all_in_toggle():
    # Verify sparse menu {F,C, 0.5P, 1P, 2P, ALL_IN} filtered by legal set and bet_fractions
    b = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=True)
    legal = [ActionType.CALL, ActionType.HALF_POT_BET, ActionType.POT_SIZED_BET, ActionType.ALL_IN]
    ps = _FakePSForMenu(legal)
    menu = b._action_menu(ps, for_player=True, pot_fracs=(0.5, 1.0), is_root=True)
    assert ActionType.CALL in menu
    assert ActionType.HALF_POT_BET in menu
    assert ActionType.POT_SIZED_BET in menu
    assert ActionType.ALL_IN in menu
    assert ActionType.TWO_POT_BET not in menu  # not requested in bet_fractions

    b2 = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[2.0], include_all_in=False)
    legal2 = [ActionType.CALL, ActionType.TWO_POT_BET, ActionType.ALL_IN]
    ps2 = _FakePSForMenu(legal2)
    menu2 = b2._action_menu(ps2, for_player=True, pot_fracs=(2.0,), is_root=True)
    assert ActionType.CALL in menu2
    assert ActionType.TWO_POT_BET in menu2
    assert ActionType.ALL_IN not in menu2  # disabled


def test_deal_next_card_excludes_used_cards():
    # Build a real PublicState with 3 board cards + 4 hole cards -> next-card list excludes these 7 cards
    deck = list(DECK)
    board = deck[:3]
    hole0 = deck[3:5]
    hole1 = deck[5:7]
    ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
    ps.hole_cards[0] = list(hole0)
    ps.hole_cards[1] = list(hole1)
    ps.current_round = 1  # flop
    b = LookaheadTreeBuilder(depth_limit=0)
    nxt = b._deal_next_card(ps)
    used = set(board + hole0 + hole1)
    assert all(c not in used for c in nxt)
    assert len(set(nxt)) == len(nxt)


def test_build_depth_limit_and_leaf_kind(monkeypatch):
    # Avoid huge expansion by making current bets unequal (so no chance at root)
    deck = list(DECK)
    board = deck[:3]
    ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
    ps.current_round = 1  # flop
    ps.current_bets = [10, 0]  # avoid immediate chance node
    ps.last_raiser = 0
    ps.current_player = 1
    b = LookaheadTreeBuilder(depth_limit=0, bet_fractions=[0.5, 1.0], include_all_in=True)
    tree = b.build(ps)
    # With limit 0, the root should be a leaf (not chance)
    assert tree["kinds"][0] in ("leaf", "terminal")
    assert tree["stage_start"] == 1


def test_propagate_reach_and_leaf_callback():
    # Small tree with 2 actions at root (ensure legal CALL and POT exist).
    deck = list(DECK)
    board = deck[:3]
    ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
    ps.current_round = 1
    ps.current_bets = [10, 0]   # ensure betting node
    ps.last_raiser = 0
    ps.current_player = 1       # non-dealer acts first postflop; root will be opp/our depending on implementation
    b = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=False)
    tree = b.build(ps)

    # Two-bucket dummy ranges -> equal mass per bucket.
    r_us = [0.5, 0.5]
    r_opp = [0.5, 0.5]

    # Leaf callback returns a per-bucket vector equal to r_us (so dot = 1.0 * 0.5 + 0.5 = 1)
    def cb(pstate, pov, r1, r2):
        return np.array(r1, dtype=float)

    b.set_leaf_callback(cb)
    out = b.propagate(tree, r_us, r_opp, pov_player=ps.current_player)
    assert "reach_us" in out and "reach_opp" in out and "values" in out
    # There must be at least one leaf/terminal value computed
    assert any(v is not None for v in out["values"])

