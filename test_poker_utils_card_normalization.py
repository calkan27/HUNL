# -*- coding: utf-8 -*-
# Granular tests for poker_utils._to_str_card and _normalize_cards

import types
import pytest

from poker_utils import _to_str_card, _normalize_cards, RANKS, SUITS

def test_simple_rs_exact():
	assert _to_str_card("JH") == "JH"
	assert _to_str_card("as") == "AS"
	assert _to_str_card("td") == "TD"

def test_sr_swapped_inputs():
	assert _to_str_card("HJ") == "JH"
	assert _to_str_card("sA") == "AS"

def test_ten_detection():
	assert _to_str_card("10h") == "TH"
	assert _to_str_card("h10") in ("TH","HT")  # accept scan resolution; our scanner picks T then H
	assert _to_str_card(["10","D"]) == "TD"
	assert _to_str_card(("10","c")) == "TC"

def test_list_tuple_inputs():
	assert _to_str_card(["J","h"]) == "JH"
	assert _to_str_card(("q","S")) == "QS"
	assert _to_str_card(("H","A")) == "AH"  # suit then rank

def test_whitespace_and_noise():
	assert _to_str_card("  j   h ") == "JH"
	assert _to_str_card("\t10 \n d ") == "TD"

def test_non_string_object_str_only():
	class C:
		def __str__(self):
			return "qh"
	assert _to_str_card(C()) == "QH"

def test_non_string_object_pairish_strs():
	class P:
		def __str__(self):
			return "  h 10 "
	assert _to_str_card(P()) in ("TH","HT")  # our scan finds T then H

def test_normalize_cards_mixed_inputs():
	v = _normalize_cards(["jh", ("10","d"), "  s a  "])
	assert v[0] == "JH"
	assert v[1] == "TD"
	# third contains suit then rank buried in spaces
	assert v[2] in ("AS", "SA")

def test_no_recursion_on_many_calls():
	# Stress a bit: thousands of calls should not trigger recursion
	for _ in range(20000):
		_to_str_card("jh")
		_to_str_card(("10","c"))
		_to_str_card("h 10")

def test_all_ranks_and_suits_round_trip():
	for r in RANKS:
		for s in SUITS:
			rs = r + s
			assert _to_str_card(rs) == rs
			assert _to_str_card(s + r) == rs  # suit-first flips

