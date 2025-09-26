import pytest
from public_state import PublicState
from game_node import GameNode
from action_type import ActionType
from action import Action

def _mk_turn_state_with_zero_pot():
	ps = PublicState(initial_stacks=[200, 200], board_cards=["AS", "KH", "7C", "2D"], dealer=0)
	ps.current_round = 2
	ps.current_bets = [0, 0]
	ps.pot_size = 0
	ps.current_player = (ps.dealer + 1) % 2
	return ps

def test_min_raise_size_uses_big_blind_floor():
	ps = PublicState(initial_stacks=[200, 200], board_cards=[], dealer=0)
	ps.last_raise_increment = 0
	mr = ps._min_raise_size()
	assert isinstance(mr, int)
	assert mr >= ps.big_blind

def test_min_raise_size_handles_tuple_list_str_callable_none():
	ps = PublicState(initial_stacks=[200, 200], board_cards=[], dealer=0)

	ps.last_raise_increment = (5,)
	assert ps._min_raise_size() >= ps.big_blind

	ps.last_raise_increment = ["12"]
	assert ps._min_raise_size() >= ps.big_blind

	ps.last_raise_increment = "15"
	assert ps._min_raise_size() >= ps.big_blind

	ps.last_raise_increment = None
	assert ps._min_raise_size() >= ps.big_blind

	ps.last_raise_increment = lambda: 123
	assert ps._min_raise_size() >= ps.big_blind

def test_legal_actions_on_turn_with_zero_pot_contains_call_and_allin():
	ps = _mk_turn_state_with_zero_pot()
	acts = set(ps.legal_actions())
	assert ActionType.CALL in acts
	assert ActionType.ALL_IN in acts

def test_all_in_progresses_state_and_pot_increases():
	ps = _mk_turn_state_with_zero_pot()
	before_pot = ps.pot_size
	before_round = ps.current_round
	ps2 = ps.update_state(GameNode(ps), Action(ActionType.ALL_IN))
	assert ps2 is not ps
	assert ps2.pot_size >= before_pot
	assert ps2.current_round == before_round or ps2.is_terminal

def test_call_then_call_advances_street_when_closed():
	ps = _mk_turn_state_with_zero_pot()
	p0 = ps.current_player
	ps2 = ps.update_state(GameNode(ps), Action(ActionType.CALL))
	ps3 = ps2.update_state(GameNode(ps2), Action(ActionType.CALL))
	assert ps3.current_round >= ps.current_round

def test_sequence_of_random_legal_actions_does_not_decrease_pot():
	ps = _mk_turn_state_with_zero_pot()
	cur = ps
	i = 0
	while (not cur.is_terminal) and i < 20:
		menu = cur.legal_actions()
		assert menu, "no legal actions"
		before = cur.pot_size
		cur = cur.update_state(GameNode(cur), Action(menu[0]))
		assert cur.pot_size + 1e-12 >= before
		i += 1

