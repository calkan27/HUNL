"""
I provide a tiny smoke check for card normalization utilities to ensure text/tuple/list
inputs normalize to canonical two-character codes and that board normalization preserves
order. I print a couple of examples and expose a single assertion-style helper that is
compatible with pytest.

Key content: module-level prints to showcase normalization;
test_card_normalization_smoke — asserts basic behavior using _to_str_card and
_normalize_cards from the engine.

Inputs: raw card tokens such as "JH", ("J","H"), ["10","H"], and a short board tuple.
Outputs: printed normalized forms and a single assertion helper.

Internal dependencies: hunl.engine.poker_utils._to_str_card and ._normalize_cards.
External dependencies: none.

Invariants: T stands for Ten ("10" → "T"); suits/ranks are uppercased; no invalid tokens
are classified as legal cards. Performance: trivial.
"""

from hunl.engine.poker_utils import _to_str_card, _normalize_cards
print(_to_str_card("JH"), _to_str_card(("J","H")), _to_str_card(["10","H"]))
print(_normalize_cards(("JH","9D","AS","KD","2C")))

def test_card_normalization_smoke():
    assert _to_str_card("JH") == "JH"
    assert _to_str_card(("J","H")) == "JH"
    assert _to_str_card(["10","H"]) == "TH"
    assert _normalize_cards(("JH","9D","AS","KD","2C")) == ["JH","9D","AS","KD","2C"]
