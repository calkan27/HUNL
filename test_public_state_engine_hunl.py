# test_public_state_engine_hunl.py
import math
import random
import itertools
import pytest

from action_type import ActionType
from action import Action
from game_node import GameNode
from poker_utils import DECK, board_one_hot, best_hand, hand_rank, RANKS, SUITS
from public_state import PublicState


# ---------- Helpers ----------

def step(state: PublicState, act: ActionType) -> PublicState:
    """Apply an action by the current player."""
    return state.update_state(GameNode(state), Action(act))

def legal(state: PublicState):
    return set(state.legal_actions())

def mass_check(state: PublicState):
    """Check pot equals sum of total contributions."""
    assert int(state.pot_size) == int(sum(state.total_contrib))

def uniq_cards_ok(state: PublicState):
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
    sb = state.dealer
    bb = (state.dealer + 1) % 2
    return sb, bb

def to_call_for(state: PublicState, p: int) -> int:
    o = (p + 1) % 2
    return max(0, state.current_bets[o] - state.current_bets[p])

def set_current_player(state: PublicState, p: int):
    state.current_player = p


# ---------- Fixtures ----------

@pytest.fixture(autouse=True)
def _seed_each_test():
    random.seed(12345)


@pytest.fixture
def fresh_state() -> PublicState:
    s = PublicState(initial_stacks=[200, 200], small_blind=1, big_blind=2, dealer=0)
    # basic init sanity
    assert s.current_round == 0  # preflop
    return s


# ---------- Basic structure / invariants ----------

def test_initial_blinds_setup_and_invariants(fresh_state):
    s = fresh_state
    sb, bb = actor_indices(s)

    # blinds posted
    assert s.current_bets[sb] == s.small_blind
    assert s.current_bets[bb] == s.big_blind
    assert s.pot_size == s.small_blind + s.big_blind == 3

    # stack bounds
    for i in (0, 1):
        assert 0 <= s.stacks[i] <= s.initial_stacks[i]

    # initial last_raiser must be BB per your constructor
    assert s.last_raiser == bb
    assert s.last_raise_increment == s.big_blind

    # pot equals contributions
    mass_check(s)

    # unique cards across hands/board
    uniq_cards_ok(s)

    # actor sanity
    assert s.current_player == s.dealer  # SB acts first preflop


def test_game_node_signature_and_equality_hash(fresh_state):
    # Two identical states should hash equal; then diverge after a move
    random.seed(777)
    s1 = PublicState()
    random.seed(777)
    s2 = PublicState()
    n1, n2 = GameNode(s1), GameNode(s2)
    assert n1 == n2
    assert hash(n1) == hash(n2)

    s1 = step(s1, ActionType.CALL)     # SB completes
    s2 = step(s2, ActionType.CALL)
    assert GameNode(s1) == GameNode(s2)

    s1 = step(s1, ActionType.CALL)     # BB checks -> flop
    s2 = step(s2, ActionType.CALL)
    assert GameNode(s1) == GameNode(s2)

    # Now diverge
    s1 = step(s1, ActionType.POT_SIZED_BET)
    assert GameNode(s1) != GameNode(s2)


def test_action_repr():
    assert repr(Action(ActionType.CALL)) == "CALL"
    assert repr(Action(ActionType.ALL_IN)) == "ALL_IN"


# ---------- Legal actions filtering ----------

def test_legal_actions_preflop_initial_sb(fresh_state):
    s = fresh_state
    sb = s.current_player
    to_call = to_call_for(s, sb)
    assert to_call == 1

    L = legal(s)
    assert ActionType.CALL in L
    assert ActionType.FOLD in L
    # When facing a bet, raises may be available:
    assert ActionType.HALF_POT_BET in L
    assert ActionType.POT_SIZED_BET in L
    assert ActionType.TWO_POT_BET in L
    assert ActionType.ALL_IN in L


def test_illegal_fold_when_to_call_zero_and_enforced(fresh_state):
    s = fresh_state
    # SB completes, BB checks -> flop, to_call == 0 for next actor
    s = step(s, ActionType.CALL)
    s = step(s, ActionType.CALL)
    assert s.current_round == 1
    assert to_call_for(s, s.current_player) == 0

    L = legal(s)
    assert ActionType.FOLD not in L
    with pytest.raises(ValueError, match="IllegalAction"):
        s.update_state(GameNode(s), Action(ActionType.FOLD))


# ---------- Board encoding & canonical summary ----------

def test_board_one_hot_and_public_summary_fields(fresh_state):
    s = fresh_state
    vec = s.board_one_hot()
    assert isinstance(vec, list)
    assert len(vec) == 52
    assert sum(vec) == len(s.board_cards)  # no board yet -> 0

    ps = s.public_summary()
    assert "pot_norm" in ps and isinstance(ps["pot_norm"], float)
    assert 0 <= ps["pot_norm"] <= 1
    assert ps["street"] == s.current_round
    assert isinstance(ps["board_one_hot"], list) and len(ps["board_one_hot"]) == 52

    canon = s.to_canonical()
    assert canon["street"] == s.current_round
    assert canon["spr"] >= 0
    assert canon["parity"] in (0, 1)


# ---------- Street advancement (bet+call and check–check) & actor order ----------

def test_bet_call_advances_street_and_actor_order(fresh_state):
    s = fresh_state
    # Preflop: SB completes, BB checks -> flop
    s = step(s, ActionType.CALL)   # SB
    s = step(s, ActionType.CALL)   # BB
    assert s.current_round == 1  # flop
    # Non-dealer acts first postflop
    sb, bb = actor_indices(s)
    assert s.current_player == bb

    # Flop: BB pot bets; SB calls -> advance to turn
    s = step(s, ActionType.POT_SIZED_BET)
    assert s.current_round == 1
    s = step(s, ActionType.CALL)
    assert s.current_round == 2  # turn
    # Postflop: non-dealer acts first again
    assert s.current_player == bb
    # Board deals to 4 cards at turn
    assert len(s.board_cards) == 4


def test_check_check_advances_street(fresh_state):
    s = fresh_state
    # Preflop equalize to get to flop
    s = step(s, ActionType.CALL)
    s = step(s, ActionType.CALL)
    assert s.current_round == 1
    assert to_call_for(s, s.current_player) == 0

    # Flop: check–check should close
    s = step(s, ActionType.CALL)  # check
    assert s.current_round == 1
    s = step(s, ActionType.CALL)  # check
    assert s.current_round == 2   # turn now


# ---------- Min-raise logic ----------

def test_min_raise_enforced_after_call_then_pot_raise(fresh_state):
    s = fresh_state
    # To flop
    s = step(s, ActionType.CALL)  # SB completes
    s = step(s, ActionType.CALL)  # BB checks -> flop, pot=4
    assert s.pot_size == 4
    # BB pot-bets (4), last_raise_increment becomes 4
    s = step(s, ActionType.POT_SIZED_BET)
    assert s.last_raise_increment == 4
    # SB now pot-raises facing 4 to call:
    my_before = s.current_bets[s.current_player]
    to_call = to_call_for(s, s.current_player)
    assert to_call == 4
    s2 = step(s, ActionType.POT_SIZED_BET)
    # In your implementation, raise_inc = max(min_raise_inc, pot_after_call)
    # pot_after_call = s.pot_size (which already included matched)
    # We can infer the raiser's total bet increase:
    delta_bet = s2.current_bets[(s.current_player)] - my_before
    # This delta includes both matched (to_call) and raise_inc
    assert delta_bet > to_call  # must include a raise
    assert s2.last_raiser == (s.current_player)  # raiser recorded


def test_min_raise_resets_each_street(fresh_state):
    s = fresh_state
    # Preflop equalize -> flop
    s = step(s, ActionType.CALL)
    s = step(s, ActionType.CALL)
    # Street bookkeeping reset at flop start
    assert s.last_raise_increment == s.big_blind


# ---------- Short all-in does NOT reopen raising rights ----------

def test_short_all_in_does_not_reopen_raising_rights_after_call_raise(fresh_state):
    s = fresh_state
    # To flop
    s = step(s, ActionType.CALL)  # SB completes
    s = step(s, ActionType.CALL)  # BB checks -> flop (pot=4)
    bb = (s.dealer + 1) % 2
    sb = s.dealer

    # BB pot-bets 4; min raise inc becomes 4
    s = step(s, ActionType.POT_SIZED_BET)
    assert s.last_raise_increment == 4
    # Make SB short such that after calling 4, only +2 remain (below min raise)
    to_call = to_call_for(s, sb)
    assert to_call == 4
    s.stacks[sb] = to_call + 2  # 6 chips total
    # SB goes ALL-IN (call 4 + raise 2 < min-raise 4) -> no-reopen flag set
    s = step(s, ActionType.ALL_IN)
    assert s.last_raise_was_allin_below_min == sb

    # Now BB faces +2 to call; raising rights must NOT be reopened
    set_current_player(s, bb)
    L = legal(s)
    # Only FOLD/CALL (no raises allowed)
    assert ActionType.CALL in L
    assert ActionType.FOLD in L
    assert ActionType.HALF_POT_BET not in L
    assert ActionType.POT_SIZED_BET not in L
    assert ActionType.TWO_POT_BET not in L
    assert (ActionType.ALL_IN not in L) or (s.stacks[bb] <= to_call_for(s, bb))


# ---------- All-in lock and runout ----------

def test_all_in_lock_and_runout_both_all_in_equalized(fresh_state):
    s = fresh_state
    # To flop
    s = step(s, ActionType.CALL)
    s = step(s, ActionType.CALL)
    assert s.current_round == 1
    # Actor (BB) open-shoves; opponent calls -> both all-in and equalized
    s = step(s, ActionType.ALL_IN)
    # Next actor should have to_call > 0 and have full stack; call to equalize
    s = step(s, ActionType.CALL)
    # Fast-forward to showdown with 5 community cards
    assert s.is_terminal
    assert s.is_showdown
    assert len(s.board_cards) == 5


def test_fast_forward_requires_equalized_not_after_unmatched_all_in(fresh_state):
    s = fresh_state
    # To flop
    s = step(s, ActionType.CALL)
    s = step(s, ActionType.CALL)
    # Make actor short and all-in far below opponent; not equalized yet mid-node
    p = s.current_player
    s.stacks[p] = 5
    s = step(s, ActionType.ALL_IN)
    # If bets not equalized yet, should not be terminal immediately
    # (opponent must act)
    assert not s.is_terminal


# ---------- Exogenous actions ----------

def test_exogenous_check_only_when_to_call_zero_and_in_turn(fresh_state):
    s = fresh_state
    # Preflop, to_call>0 -> exogenous check is ignored
    p = s.current_player
    s2 = s.apply_exogenous_opponent_check(p)
    assert s2 is s  # unchanged

    # To flop -> to_call 0 for current player
    s = step(s, ActionType.CALL)  # SB completes
    s = step(s, ActionType.CALL)  # BB checks -> flop

    p = s.current_player
    o = (p + 1) % 2
    # First check
    prev_round = s.current_round
    s = s.apply_exogenous_opponent_check(p)
    assert s.current_round == prev_round
    # Second check closes the street
    s = s.apply_exogenous_opponent_check(o)
    assert s.current_round == prev_round + 1


def test_exogenous_bet_and_all_in_enforce_legality_and_in_turn(fresh_state):
    s = fresh_state
    # To flop
    s = step(s, ActionType.CALL)
    s = step(s, ActionType.CALL)

    # Out-of-turn bettor ignored
    out_of_turn = (s.current_player + 1) % 2
    s2 = s.apply_exogenous_opponent_bet(out_of_turn, ActionType.POT_SIZED_BET)
    assert s2 is s  # unchanged

    # Illegal kind ignored
    s2 = s.apply_exogenous_opponent_bet(s.current_player, ActionType.FOLD)
    assert s2 is s

    # Legal exogenous pot bet by current actor
    prev_pot = s.pot_size
    s = s.apply_exogenous_opponent_bet(s.current_player, ActionType.POT_SIZED_BET)
    assert s.pot_size > prev_pot
    assert s.last_raiser == (s.current_player,)
    mass_check(s)

    # Now exogenous ALL-IN by the next current player
    s = s.apply_exogenous_opponent_all_in(s.current_player)
    # Either matched or partially matched + refund; but invariants must hold
    mass_check(s)
    uniq_cards_ok(s)


# ---------- Refund accounting & pot monotonicity (with allowed_refund) ----------

def test_refund_on_short_call_and_pot_monotonicity_exception(fresh_state):
    s = fresh_state
    # Build a controlled state: one player has made a large bet,
    # other has too few chips to fully call -> refund required.
    # We do this by directly setting state, keeping invariants consistent.
    sb, bb = actor_indices(s)
    p = sb
    o = bb

    # Set street to flop, bets closed and reset
    s = step(s, ActionType.CALL)  # SB completes
    s = step(s, ActionType.CALL)  # BB checks -> flop (pot=4)
    assert s.current_round == 1
    # Let opponent place a large bet by manual tweak (legalizing the situation):
    # We'll say opponent has already bet 50 this street.
    s.current_bets[o] = 50
    s.total_contrib[o] += 50
    s.stacks[o] -= 50
    s.pot_size += 50
    mass_check(s)

    # Now current player is p; to_call is 50
    set_current_player(s, p)
    s.stacks[p] = 10  # short
    prev_pot = float(s.pot_size)
    s2 = step(s, ActionType.CALL)  # triggers refund of 40
    assert s2.last_refund_amount == pytest.approx(40.0)
    # Pot may decrease by up to last_refund_amount relative to prev_pot
    assert s2.pot_size + 1e-12 >= prev_pot - s2.last_refund_amount
    mass_check(s2)


# ---------- Randomized sequences: invariants, no street jumps, mass conservation ----------

def test_random_sequences_invariants_and_no_street_jumps():
    for seed in [1, 7, 11]:
        random.seed(seed)
        s = PublicState()
        prev_round = s.current_round
        prev_pot = float(s.pot_size)
        steps = 0
        while not s.is_terminal and steps < 200:
            L = list(legal(s))
            # Choose a random legal action
            act = random.choice(L)
            s = step(s, act)
            # No illegal street jumps
            assert s.current_round in (prev_round, prev_round + 1)
            if s.current_round == prev_round + 1:
                prev_round = s.current_round
            # Pot monotonicity with refund exception
            assert s.pot_size + 1e-12 >= prev_pot - getattr(s, "last_refund_amount", 0.0)
            prev_pot = float(s.pot_size)
            # Mass and uniqueness
            mass_check(s)
            uniq_cards_ok(s)
            # Stack bounds
            for i in (0, 1):
                assert 0 <= s.stacks[i] <= s.initial_stacks[i]
            steps += 1


# ---------- Terminal utilities ----------

def test_terminal_utility_fold_preflop(fresh_state):
    s = fresh_state
    # SB folds immediately
    winner = (s.dealer + 1) % 2
    s = step(s, ActionType.FOLD)
    assert s.is_terminal
    u = s.terminal_utility()
    # Winner gets pot - their own contrib; loser loses their contrib
    assert u[winner] == pytest.approx(s.pot_size - s.total_contrib[winner])
    loser = (winner + 1) % 2
    assert u[loser] == pytest.approx(-s.total_contrib[loser])
    assert u[winner] == -u[loser]


def test_terminal_utility_showdown_equal_contrib_win_and_tie():
    # Deterministic setup by reseeding
    random.seed(99)
    s = PublicState()
    # To river with no further betting: check all streets
    s = step(s, ActionType.CALL)  # SB completes -> pot=4 at flop
    s = step(s, ActionType.CALL)  # BB checks -> flop
    # check–check flop
    s = step(s, ActionType.CALL)
    s = step(s, ActionType.CALL)
    # check–check turn
    s = step(s, ActionType.CALL)
    s = step(s, ActionType.CALL)
    # check–check river closes to showdown
    s = step(s, ActionType.CALL)
    s = step(s, ActionType.CALL)

    assert s.is_terminal and s.is_showdown
    # Equal contributions (blinds only); pot should be 4
    assert s.pot_size == 4
    assert sum(s.total_contrib) == 4

    u = s.terminal_utility()
    # Winner gets +2, loser -2 OR tie gives both 0
    assert abs(sum(u)) < 1e-9
    assert all(isinstance(x, float) for x in u)

    # Now directly test extras path by crafting uneven contrib and showdown.
    # Force terminal showdown with custom contrib and known winner.
    s2 = PublicState()
    # Freeze terminal showdown state
    s2.is_terminal = True
    s2.is_showdown = True
    # Create uneven contributions: player 0 > player 1
    s2.total_contrib = [60.0, 40.0]
    s2.pot_size = sum(s2.total_contrib)
    # Give player 0 a clear winning hand over player 1 on a fixed board:
    # We'll provide a board & hole cards that do not collide.
    board = ["AS", "KD", "QH", "JC", "TC"]  # Broadway straight on board
    # Give P0 a higher flush kicker scenario by modifying to another board:
    board = ["AC", "KC", "QC", "2D", "3H"]  # clubs 3-flush; hands decide
    s2.board_cards = board
    # P0: clubs to make a strong flush; P1: off-suit rags
    s2.hole_cards[0] = ["9C", "8C"]
    s2.hole_cards[1] = ["7D", "6H"]
    uniq_cards_ok(s2)
    u2 = s2.terminal_utility()
    # With c0=60, c1=40 -> main 80, extra0=20, extra1=0 -> winner 100, loser 0
    # Utilities: u0 = 100 - 60 = +40; u1 = 0 - 40 = -40
    assert u2[0] == pytest.approx(40.0)
    assert u2[1] == pytest.approx(-40.0)
    assert abs(sum(u2)) < 1e-9


# ---------- Legal menu restrictions when raises are blocked ----------

def test_legal_actions_when_raises_blocked_to_call_zero():
    s = PublicState()
    # Take to flop
    s = step(s, ActionType.CALL)
    s = step(s, ActionType.CALL)
    # Force "no reopen" flag (simulating prior short all-in on the street)
    s.last_raise_was_allin_below_min = 0
    set_current_player(s, (s.dealer + 1) % 2)  # any actor; to_call zero
    L = legal(s)
    # Only CHECK (CALL) should be offered; no raises, no ALL_IN
    assert ActionType.CALL in L
    assert ActionType.FOLD not in L
    assert ActionType.HALF_POT_BET not in L
    assert ActionType.POT_SIZED_BET not in L
    assert ActionType.TWO_POT_BET not in L
    assert ActionType.ALL_IN not in L


# ---------- Advance street requires equalization ----------

def test_advance_street_requires_equalization_not_after_bet_only():
    s = PublicState()
    # Preflop: SB completes -> flop after BB checks
    s = step(s, ActionType.CALL)
    s = step(s, ActionType.CALL)
    cr = s.current_round
    # Actor bets; street must not advance until called/equalized
    s = step(s, ActionType.POT_SIZED_BET)
    assert s.current_round == cr
    # Now call to close
    s = step(s, ActionType.CALL)
    assert s.current_round == cr + 1


# ---------- Pot minimum on small pot (min bet at least BB) via exogenous bet ----------

def test_minimum_bet_at_least_big_blind_when_pot_small():
    s = PublicState()
    # Move to flop and then artificially reduce pot to 1 chip (keeping mass consistency)
    s = step(s, ActionType.CALL)
    s = step(s, ActionType.CALL)
    # reset bets & make pot artificially tiny (keep consistency)
    s.total_contrib = [0, 1]
    s.current_bets = [0, 1]
    s.stacks[0] = s.initial_stacks[0]
    s.stacks[1] = s.initial_stacks[1] - 1
    s.pot_size = 1
    set_current_player(s, 0)
    # Legal half-pot bet should floor to big blind (2)
    assert ActionType.HALF_POT_BET in legal(s)
    prev = s.pot_size
    s = s.apply_exogenous_opponent_bet(0, ActionType.HALF_POT_BET)
    assert s.pot_size >= prev + s.big_blind
    mass_check(s)


# ---------- to_canonical parity & spr sanity progression ----------

def test_to_canonical_parity_and_spr_progression():
    s = PublicState()
    c0 = s.to_canonical()
    # SB completes (actor changes), BB checks (street changes, parity flips)
    s = step(s, ActionType.CALL)
    c1 = s.to_canonical()
    s = step(s, ActionType.CALL)
    c2 = s.to_canonical()
    assert c0["parity"] in (0, 1)
    assert c1["parity"] in (0, 1)
    assert c2["parity"] in (0, 1)
    # SPR is finite and non-negative
    assert c0["spr"] >= 0 and c1["spr"] >= 0 and c2["spr"] >= 0

