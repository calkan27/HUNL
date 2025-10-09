"""
Test suite for the PublicState engine in HUNL: validates blinds initialization, legal-action gating, 
	 street advancement, raise/min-raise logic, all-in/runout handling, exogenous actions, refund accounting, 
	 randomized invariants, terminal utilities, and canonical summaries.
"""

import hunl.engine.poker_utils as poker_utils
import math
import random
import itertools
import pytest

from hunl.engine.action_type import ActionType
from hunl.engine.action import Action
from hunl.engine.game_node import GameNode
from hunl.engine.poker_utils import DECK, board_one_hot, best_hand, hand_rank, RANKS, SUITS
from hunl.engine.public_state import PublicState


def step(state: PublicState, act: ActionType) -> PublicState:
	"""
	Apply a single legal action by the current player to advance the PublicState and return the successor state.
	"""
	return state.update_state(GameNode(state), Action(act))


def legal(state: PublicState):
	"""
	Return the set of legal ActionType options for the acting player in the given PublicState.
	"""
	return set(state.legal_actions())


def mass_check(state: PublicState):
	"""
	Assert conservation of chips by checking that the pot size equals the sum of total contributions.
	"""
	assert int(state.pot_size) == int(sum(state.total_contrib))


def uniq_cards_ok(state: PublicState):
	"""
	Verify that all cards across board and both players’ hole cards are unique and non-overlapping.
	"""
	seen = set()
	for c in state.board_cards:
		assert c not in seen
		seen.add(c)
	for c in state.hole_cards[0]:
		assert c not in seen
		seen.add(c)
	for c in state.hole_cards[1]:
		assert c not in seen
		seen.add(c)


def actor_indices(state: PublicState):
	"""
	Return a tuple of (small blind index, big blind index) based on the dealer location.
	"""
	sb = state.dealer
	bb = (state.dealer + 1) % 2
	return sb, bb


def to_call_for(state: PublicState, p: int) -> int:
	"""
	Compute the outstanding chips required for player p to call given current street bets.
	"""
	o = (p + 1) % 2
	return max(0, state.current_bets[o] - state.current_bets[p])


def set_current_player(state: PublicState, p: int):
	"""
	Override the current player index in the provided PublicState for test control.
	"""
	state.current_player = p


@pytest.fixture(autouse=True)
def _seed_each_test():
	"""
	Seed Python’s RNG before each test to stabilize randomized scenarios.
	"""
	random.seed(12345)


@pytest.fixture
def fresh_state() -> PublicState:
	"""
	Construct a fresh PublicState with standard stacks and blinds and return it for tests.
	"""
	s = PublicState(initial_stacks=[200, 200], small_blind=1, big_blind=2, dealer=0)
	assert s.current_round == 0
	return s


def test_initial_blinds_setup_and_invariants(fresh_state):
	"""
	Verify initial blinds posting, pot size, stack bounds, last-raiser bookkeeping,
	mass conservation, unique cards, and initial actor.
	"""
	s = fresh_state
	sb, bb = actor_indices(s)
	assert s.current_bets[sb] == s.small_blind
	assert s.current_bets[bb] == s.big_blind
	assert s.pot_size == s.small_blind + s.big_blind == 3
	for i in (0, 1):
		assert 0 <= s.stacks[i] <= s.initial_stacks[i]
	assert s.last_raiser == bb
	assert s.last_raise_increment == s.big_blind
	mass_check(s)
	uniq_cards_ok(s)
	assert s.current_player == s.dealer


def test_game_node_signature_and_equality_hash(fresh_state):
	"""
	Ensure GameNode equality and hashing are deterministic for identical states and
	diverge after different action sequences.
	"""
	random.seed(777)
	s1 = PublicState()
	random.seed(777)
	s2 = PublicState()
	n1, n2 = GameNode(s1), GameNode(s2)
	assert n1 == n2
	assert hash(n1) == hash(n2)
	s1 = step(s1, ActionType.CALL)
	s2 = step(s2, ActionType.CALL)
	assert GameNode(s1) == GameNode(s2)
	s1 = step(s1, ActionType.CALL)
	s2 = step(s2, ActionType.CALL)
	assert GameNode(s1) == GameNode(s2)
	s1 = step(s1, ActionType.POT_SIZED_BET)
	assert GameNode(s1) != GameNode(s2)


def test_action_repr():
	"""
	Confirm Action __repr__ outputs the enum name for readability and debugging.
	"""
	assert repr(Action(ActionType.CALL)) == "CALL"
	assert repr(Action(ActionType.ALL_IN)) == "ALL_IN"


def test_legal_actions_preflop_initial_sb(fresh_state):
	"""
	Check the legal action menu for the small blind facing one chip to call on the first preflop decision.
	"""
	s = fresh_state
	sb = s.current_player
	to_call = to_call_for(s, sb)
	assert to_call == 1
	L = legal(s)
	assert ActionType.CALL in L
	assert ActionType.FOLD in L
	assert ActionType.HALF_POT_BET in L
	assert ActionType.POT_SIZED_BET in L
	assert ActionType.TWO_POT_BET in L
	assert ActionType.ALL_IN in L


def test_illegal_fold_when_to_call_zero_and_enforced(fresh_state):
	"""
	Assert that folding is disallowed when there is nothing to call and that attempting it raises an error.
	"""
	s = fresh_state
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	assert s.current_round == 1
	assert to_call_for(s, s.current_player) == 0
	L = legal(s)
	assert ActionType.FOLD not in L
	with pytest.raises(ValueError, match="IllegalAction"):
		s.update_state(GameNode(s), Action(ActionType.FOLD))


def test_board_one_hot_and_public_summary_fields(fresh_state):
	"""
	Validate board one-hot length and mass, and confirm public_summary and 
	to_canonical provide consistent normalized fields.
	"""
	s = fresh_state
	vec = s.board_one_hot()
	assert isinstance(vec, list)
	assert len(vec) == 52
	assert sum(vec) == len(s.board_cards)
	ps = s.public_summary()
	assert "pot_norm" in ps and isinstance(ps["pot_norm"], float)
	assert 0 <= ps["pot_norm"] <= 1
	assert ps["street"] == s.current_round
	assert isinstance(ps["board_one_hot"], list) and len(ps["board_one_hot"]) == 52
	canon = s.to_canonical()
	assert canon["street"] == s.current_round
	assert canon["spr"] >= 0
	assert canon["parity"] in (0, 1)


def test_bet_call_advances_street_and_actor_order(fresh_state):
	"""
	Confirm that bet–call closes the street, non-dealer acts first postflop, and the board
	length updates appropriately.
	"""
	s = fresh_state
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	assert s.current_round == 1
	sb, bb = actor_indices(s)
	assert s.current_player == bb
	s = step(s, ActionType.POT_SIZED_BET)
	assert s.current_round == 1
	s = step(s, ActionType.CALL)
	assert s.current_round == 2
	assert s.current_player == bb
	assert len(s.board_cards) == 4


def test_check_check_advances_street(fresh_state):
	"""
	Verify that consecutive checks at to_call=0 advance the street.
	"""
	s = fresh_state
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	assert s.current_round == 1
	assert to_call_for(s, s.current_player) == 0
	s = step(s, ActionType.CALL)
	assert s.current_round == 1
	s = step(s, ActionType.CALL)
	assert s.current_round == 2


def test_min_raise_enforced_after_call_then_pot_raise(fresh_state):
	"""
	Ensure min-raise increment bookkeeping updates after a pot bet and that a subsequent pot raise exceeds call-only delta.
	"""
	s = fresh_state
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	assert s.pot_size == 4
	s = step(s, ActionType.POT_SIZED_BET)
	assert s.last_raise_increment == 4
	my_before = s.current_bets[s.current_player]
	to_call = to_call_for(s, s.current_player)
	assert to_call == 4
	s2 = step(s, ActionType.POT_SIZED_BET)
	delta_bet = s2.current_bets[(s.current_player)] - my_before
	assert delta_bet > to_call
	assert s2.last_raiser == (s.current_player)


def test_min_raise_resets_each_street(fresh_state):
	"""
	Check that last_raise_increment resets at the start of a new street to the big blind.
	"""
	s = fresh_state
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	assert s.last_raise_increment == s.big_blind


def test_short_all_in_does_not_reopen_raising_rights_after_call_raise(fresh_state):
	"""
	Validate that a short all-in below the minimum raise does not reopen raising rights for the opponent.
	"""
	s = fresh_state
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	bb = (s.dealer + 1) % 2
	sb = s.dealer
	s = step(s, ActionType.POT_SIZED_BET)
	assert s.last_raise_increment == 4
	to_call = to_call_for(s, sb)
	assert to_call == 4
	s.stacks[sb] = to_call + 2
	s = step(s, ActionType.ALL_IN)
	assert s.last_raise_was_allin_below_min == sb
	set_current_player(s, bb)
	L = legal(s)
	assert ActionType.CALL in L
	assert ActionType.FOLD in L
	assert ActionType.HALF_POT_BET not in L
	assert ActionType.POT_SIZED_BET not in L
	assert ActionType.TWO_POT_BET not in L
	assert (ActionType.ALL_IN not in L) or (s.stacks[bb] <= to_call_for(s, bb))


def test_all_in_lock_and_runout_both_all_in_equalized(fresh_state):
	"""
	Ensure that matched all-ins lead to a fast-forward runout to showdown with five community cards.
	"""
	s = fresh_state
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	assert s.current_round == 1
	s = step(s, ActionType.ALL_IN)
	s = step(s, ActionType.CALL)
	assert s.is_terminal
	assert s.is_showdown
	assert len(s.board_cards) == 5


def test_fast_forward_requires_equalized_not_after_unmatched_all_in(fresh_state):
	"""
	Confirm that fast-forward runout does not occur if all-in bets are not yet equalized during the node.
	"""
	s = fresh_state
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	p = s.current_player
	s.stacks[p] = 5
	s = step(s, ActionType.ALL_IN)
	assert not s.is_terminal


def test_exogenous_check_only_when_to_call_zero_and_in_turn(fresh_state):
	"""
	Test that exogenous checks are ignored out of turn or when facing a bet, and
	advance the street only after consecutive checks in turn.
	"""
	s = fresh_state
	p = s.current_player
	s2 = s.apply_exogenous_opponent_check(p)
	assert s2 is s
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	p = s.current_player
	o = (p + 1) % 2
	prev_round = s.current_round
	s = s.apply_exogenous_opponent_check(p)
	assert s.current_round == prev_round
	s = s.apply_exogenous_opponent_check(o)
	assert s.current_round == prev_round + 1


def test_exogenous_bet_and_all_in_enforce_legality_and_in_turn(fresh_state):
	"""
	Verify exogenous bet/all-in handlers ignore out-of-turn or illegal actions and
	uphold mass and uniqueness invariants when applied legally.
	"""
	s = fresh_state
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	out_of_turn = (s.current_player + 1) % 2
	s2 = s.apply_exogenous_opponent_bet(out_of_turn, ActionType.POT_SIZED_BET)
	assert s2 is s
	s2 = s.apply_exogenous_opponent_bet(s.current_player, ActionType.FOLD)
	assert s2 is s
	prev_pot = s.pot_size
	s = s.apply_exogenous_opponent_bet(s.current_player, ActionType.POT_SIZED_BET)
	assert s.pot_size > prev_pot
	assert s.last_raiser == (s.current_player,)
	mass_check(s)
	s = s.apply_exogenous_opponent_all_in(s.current_player)
	mass_check(s)
	uniq_cards_ok(s)


def test_refund_on_short_call_and_pot_monotonicity_exception(fresh_state):
	"""
	Exercise the refund path when a short caller cannot fully match a large bet and assert pot monotonicity within allowed refund slack.
	"""
	s = fresh_state
	sb, bb = actor_indices(s)
	p = sb
	o = bb
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	assert s.current_round == 1
	s.current_bets[o] = 50
	s.total_contrib[o] += 50
	s.stacks[o] -= 50
	s.pot_size += 50
	mass_check(s)
	set_current_player(s, p)
	s.stacks[p] = 10
	prev_pot = float(s.pot_size)
	s2 = step(s, ActionType.CALL)
	assert s2.last_refund_amount == pytest.approx(40.0)
	assert s2.pot_size + 1e-12 >= prev_pot - s2.last_refund_amount
	mass_check(s2)


def test_random_sequences_invariants_and_no_street_jumps():
	"""
	Run randomized legal-action sequences and assert no illegal street jumps, 
	mass conservation, unique cards, stack bounds, and pot monotonicity up to refund slack.
	"""
	for seed in [1, 7, 11]:
		random.seed(seed)
		s = PublicState()
		prev_round = s.current_round
		prev_pot = float(s.pot_size)
		steps = 0
		while not s.is_terminal and steps < 200:
			L = list(legal(s))
			act = random.choice(L)
			s = step(s, act)
			assert s.current_round in (prev_round, prev_round + 1)
			if s.current_round == prev_round + 1:
				prev_round = s.current_round
			assert s.pot_size + 1e-12 >= prev_pot - getattr(s, "last_refund_amount", 0.0)
			prev_pot = float(s.pot_size)
			mass_check(s)
			uniq_cards_ok(s)
			for i in (0, 1):
				assert 0 <= s.stacks[i] <= s.initial_stacks[i]
			steps += 1


def test_terminal_utility_fold_preflop(fresh_state):
	"""
	Confirm terminal utilities on a preflop fold equal winner’s pot minus own
	contribution and loser’s negative contribution, summing to zero.
	"""
	s = fresh_state
	winner = (s.dealer + 1) % 2
	s = step(s, ActionType.FOLD)
	assert s.is_terminal
	u = s.terminal_utility()
	assert u[winner] == pytest.approx(s.pot_size - s.total_contrib[winner])
	loser = (winner + 1) % 2
	assert u[loser] == pytest.approx(-s.total_contrib[loser])
	assert u[winner] == -u[loser]


def test_terminal_utility_showdown_equal_contrib_win_and_tie():
	"""
	Drive to a river showdown with equal contributions to verify zero-sum utilities,
	then craft an uneven-contribution showdown to validate side-pot accounting.
	"""
	random.seed(99)
	s = PublicState()
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	assert s.is_terminal and s.is_showdown
	assert s.pot_size == 4
	assert sum(s.total_contrib) == 4
	u = s.terminal_utility()
	assert abs(sum(u)) < 1e-9
	assert all(isinstance(x, float) for x in u)
	s2 = PublicState()
	s2.is_terminal = True
	s2.is_showdown = True
	s2.total_contrib = [60.0, 40.0]
	s2.pot_size = sum(s2.total_contrib)
	board = ["AC", "KC", "QC", "2D", "3H"]
	s2.board_cards = board
	s2.hole_cards[0] = ["9C", "8C"]
	s2.hole_cards[1] = ["7D", "6H"]
	uniq_cards_ok(s2)
	u2 = s2.terminal_utility()
	assert u2[0] == pytest.approx(40.0)
	assert u2[1] == pytest.approx(-40.0)
	assert abs(sum(u2)) < 1e-9


def test_legal_actions_when_raises_blocked_to_call_zero():
	"""
	Ensure that when raises are blocked and to_call is zero, only checking is
	offered and all raise actions are excluded.
	"""
	s = PublicState()
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	s.last_raise_was_allin_below_min = 0
	set_current_player(s, (s.dealer + 1) % 2)
	L = legal(s)
	assert ActionType.CALL in L
	assert ActionType.FOLD not in L
	assert ActionType.HALF_POT_BET not in L
	assert ActionType.POT_SIZED_BET not in L
	assert ActionType.TWO_POT_BET not in L
	assert ActionType.ALL_IN not in L


def test_advance_street_requires_equalization_not_after_bet_only():
	"""
	Check that a bet alone does not advance the street and that a subsequent
	call closes the betting and advances the street.
	"""
	s = PublicState()
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	cr = s.current_round
	s = step(s, ActionType.POT_SIZED_BET)
	assert s.current_round == cr
	s = step(s, ActionType.CALL)
	assert s.current_round == cr + 1


def test_minimum_bet_at_least_big_blind_when_pot_small():
	"""
	Validate that when the pot is artificially small, the effective minimum 
	bet is floored to at least the big blind size.
	"""
	s = PublicState()
	s = step(s, ActionType.CALL)
	s = step(s, ActionType.CALL)
	s.total_contrib = [0, 1]
	s.current_bets = [0, 1]
	s.stacks[0] = s.initial_stacks[0]
	s.stacks[1] = s.initial_stacks[1] - 1
	s.pot_size = 1
	set_current_player(s, 0)
	assert ActionType.HALF_POT_BET in legal(s)
	prev = s.pot_size
	s = s.apply_exogenous_opponent_bet(0, ActionType.HALF_POT_BET)
	assert s.pot_size >= prev + s.big_blind
	mass_check(s)


def test_to_canonical_parity_and_spr_progression():
	"""
	Ensure to_canonical parity remains well-defined and SPR remains nonnegative 
	while parity can change as actors and streets progress.
	"""
	s = PublicState()
	c0 = s.to_canonical()
	s = step(s, ActionType.CALL)
	c1 = s.to_canonical()
	s = step(s, ActionType.CALL)
	c2 = s.to_canonical()
	assert c0["parity"] in (0, 1)
	assert c1["parity"] in (0, 1)
	assert c2["parity"] in (0, 1)
	assert c0["spr"] >= 0 and c1["spr"] >= 0 and c2["spr"] >= 0

