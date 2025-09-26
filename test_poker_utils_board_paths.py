import pytest
from poker_utils import card_to_index, board_one_hot

def test_card_to_index_canonical_paths():
	assert card_to_index("JH") == card_to_index(("J","H"))
	assert card_to_index("10h") == card_to_index(("10","H"))
	assert card_to_index("h10") == card_to_index(("10","H"))  # suit-first tolerated

def test_board_one_hot_uses_normalizer():
	v1 = board_one_hot(["JH", "TD", "AS"])
	v2 = board_one_hot([("J","H"), ("10","d"), ("a","s")])
	assert v1 == v2
	assert sum(v1) == 3

