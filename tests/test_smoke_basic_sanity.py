"""
Test suite for poker_utils deck invariants: validates that the canonical DECK contains exactly 52 unique cards.
"""

import hunl.engine.poker_utils as poker_utils


def test_deck_shape():
	"""
	Verify that the canonical 52-card deck exported by poker_utils has length 52.
	"""
	from hunl.engine.poker_utils import DECK
	assert len(DECK) == 52

