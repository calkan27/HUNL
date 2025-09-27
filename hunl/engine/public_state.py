"""
I represent the public game state for Heads-Up No-Limit Texas Hold’em and compose street
logic, action semantics, and terminal resolution through mixins. I track stacks, bets,
pot, dealer, current player, street index, board and hole cards, actions, and
bookkeeping for raises, refunds, and terminal flags.

Key class: PublicState (mixes PublicStateActionsMixin, PublicStateStreetsMixin,
PublicStateUtilsMixin). Key methods: clone — deep copy for hypothetical lookahead;
legal_actions — enumerate {FOLD, CALL/CHECK, HALF_POT_BET, POT_SIZED_BET, TWO_POT_BET,
ALL_IN}; terminal_utility — chips won/lost per player at fold/showdown;
to_canonical/public_summary — features for models.

Inputs: initial stacks, blinds, dealer, optional pre-dealt board; calls to
update_state(Action) to advance. Outputs: new PublicState snapshots, consistent
pot/stacks/bets, terminal flags and utilities. Invariants: pot monotonicity modulo
explicit refund, stack non-negativity, distinct cards, street transitions gated by
equalized bets or all-in lock.

Dependencies: hunl.engine.action_type, action, poker_utils. Side effects: none; I return
new objects for updates.

Edge cases: short all-in below minimum raise may set last_raise_was_allin_below_min to
preserve legality; I fast-forward to showdown when all-in and equalized. Performance:
designed for frequent cloning in tree search; internal checks are O(1) per update.
"""

from hunl.constants import EPS_MASS
import random
import copy
from typing import List, Optional, Dict, Any

from hunl.engine.action_type import ActionType
from hunl.engine.poker_utils import DECK, board_one_hot as _b, hand_rank, best_hand

from hunl.engine.public_state_actions import PublicStateActionsMixin
from hunl.engine.public_state_streets import PublicStateStreetsMixin
from hunl.engine.public_state_utils import PublicStateUtilsMixin
from hunl.engine.poker_utils import DECK, board_one_hot as _board_one_hot, hand_rank, best_hand

class PublicState(PublicStateActionsMixin, PublicStateStreetsMixin, PublicStateUtilsMixin):
	def __init__(
	 self,
	 initial_stacks: List[int] = [200, 200],
	 small_blind: int = 1,
	 big_blind: int = 2,
	 board_cards: Optional[List[str]] = None,
	 dealer: int = 0,
	):
		self.board_cards = list(board_cards) if board_cards is not None else []

		self.pot_size = 0
		self.current_bets = [0, 0]
		self.current_round = 0

		self.is_terminal = False
		self.is_showdown = False
		self.last_action = None

		self.DECK = []
		i = 0
		while i < len(DECK):
			c = DECK[i]
			if c not in self.board_cards:
				self.DECK.append(c)
			i += 1

		random.shuffle(self.DECK)

		self.stacks = list(initial_stacks)
		self.initial_stacks = list(initial_stacks)

		self.small_blind = int(small_blind)
		self.big_blind = int(big_blind)

		self.dealer = int(dealer)
		self.current_player = None

		self.actions = []
		self.players_in_hand = [True, True]

		self.last_raiser = None
		self.last_raise_increment = int(self.big_blind)
		self.last_raise_was_allin_below_min = None
		self.consecutive_checks = 0
		self.last_refund_amount = 0.0

		self.hole_cards = [[], []]
		self.hole_cards[0] = [self.DECK.pop(), self.DECK.pop()]
		self.hole_cards[1] = [self.DECK.pop(), self.DECK.pop()]

		sb_player = int(self.dealer)
		bb_player = (int(self.dealer) + 1) % 2

		self.current_bets[sb_player] = int(self.small_blind)
		self.current_bets[bb_player] = int(self.big_blind)

		self.stacks[sb_player] -= int(self.small_blind)
		self.stacks[bb_player] -= int(self.big_blind)

		self.pot_size = int(self.small_blind) + int(self.big_blind)

		self.total_contrib = [0, 0]
		self.total_contrib[sb_player] += int(self.small_blind)
		self.total_contrib[bb_player] += int(self.big_blind)

		self.last_raiser = int(bb_player)
		self.current_player = int(self.dealer)

	def clone(self):
		new_state = PublicState.__new__(PublicState)

		new_state.pot_size = int(self.pot_size)
		new_state.current_bets = list(self.current_bets)
		new_state.current_round = int(self.current_round)

		new_state.is_terminal = bool(self.is_terminal)
		new_state.is_showdown = bool(self.is_showdown)
		new_state.last_action = None

		new_state.DECK = list(self.DECK)
		new_state.stacks = list(self.stacks)
		new_state.initial_stacks = list(self.initial_stacks)

		new_state.small_blind = int(self.small_blind)
		new_state.big_blind = int(self.big_blind)

		new_state.dealer = int(self.dealer)
		if self.current_player is None:
			new_state.current_player = None
		else:
			new_state.current_player = int(self.current_player)

		new_state.actions = [tuple(x) for x in self.actions]
		new_state.players_in_hand = list(self.players_in_hand)

		new_state.board_cards = list(self.board_cards)
		new_state.hole_cards = [
		 list(self.hole_cards[0]),
		 list(self.hole_cards[1]),
		]

		if self.last_raiser is None:
			new_state.last_raiser = None
		else:
			new_state.last_raiser = int(self.last_raiser)

		new_state.last_raise_increment = int(self.last_raise_increment)

		if self.last_raise_was_allin_below_min is None:
			new_state.last_raise_was_allin_below_min = None
		else:
			new_state.last_raise_was_allin_below_min = int(self.last_raise_was_allin_below_min)

		new_state.total_contrib = list(self.total_contrib)
		new_state.consecutive_checks = int(self.consecutive_checks)
		new_state.last_refund_amount = float(self.last_refund_amount)

		return new_state

	def _assert_invariants(self, prev_pot=None):
		ok = True
		if prev_pot is not None:
			allowed_refund = float(getattr(self, "last_refund_amount", 0.0))
			min_pot = float(prev_pot) - allowed_refund
			if float(self.pot_size) + EPS_MASS < min_pot:
				self.pot_size = min_pot if min_pot > 0.0 else 0.0
				ok = False
		if self.current_player is not None:
			cp = int(self.current_player)
			if cp not in (0, 1):
				self.current_player = int(self.dealer)
				ok = False
		seen = set()
		fixed_board = []
		i = 0
		while i < len(self.board_cards):
			c = self.board_cards[i]
			if c in seen:
				ok = False
			else:
				seen.add(c)
				fixed_board.append(c)
			i += 1
		if len(fixed_board) != len(self.board_cards):
			self.board_cards = fixed_board
		i = 0
		while i < len(self.hole_cards[0]):
			c = self.hole_cards[0][i]
			if c in seen:
				ok = False
			else:
				seen.add(c)
			i += 1
		i = 0
		while i < len(self.hole_cards[1]):
			c = self.hole_cards[1][i]
			if c in seen:
				ok = False
			else:
				seen.add(c)
			i += 1
		i = 0
		while i < 2:
			if int(self.stacks[i]) < 0:
				self.stacks[i] = 0
				ok = False
			if int(self.stacks[i]) > int(self.initial_stacks[i]):
				self.stacks[i] = int(self.initial_stacks[i])
				ok = False
			if int(self.current_bets[i]) < 0:
				self.current_bets[i] = 0
				ok = False
			i += 1
		return bool(ok)

	def _non_dealer(self) -> int:
		return (int(self.dealer) + 1) % 2

	def copy(self):
		return self.clone()

	def legal_actions(self):
		return PublicStateUtilsMixin.legal_actions(self)

	def to_canonical(self) -> Dict[str, Any]:
		if getattr(self, "initial_stacks", None) is not None:
			total_initial = float(sum(self.initial_stacks))
		else:
			total_initial = 1.0

		if total_initial <= 0.0:
			total_initial = 1.0

		pot_norm = float(self.pot_size) / float(total_initial)

		if getattr(self, "last_raise_increment", None) is not None:
			last_bet = int(self.last_raise_increment)
		else:
			last_bet = 0

		eff_stack = int(self.stacks[0]) if int(self.stacks[0]) < int(self.stacks[1]) else int(self.stacks[1])

		if float(self.pot_size) > 0.0:
			den = float(self.pot_size)
		else:
			den = 1.0

		spr = float(eff_stack) / float(den)

		if self.current_player is None:
			cp = 0
		else:
			cp = int(self.current_player)

		parity = (cp - int(self.dealer)) & 1

		return {
		 "street": int(self.current_round),
		 "board": tuple(self.board_cards),
		 "board_one_hot": list(self.board_one_hot()),
		 "pot_norm": float(pot_norm),
		 "last_bet": int(last_bet),
		 "spr": float(spr),
		 "parity": int(parity),
		}
