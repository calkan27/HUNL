from poker_utils import _to_str_card, _normalize_cards
print(_to_str_card("JH"), _to_str_card(("J","H")), _to_str_card(["10","H"]))
print(_normalize_cards(("JH","9D","AS","KD","2C")))

def test_card_normalization_smoke():
    assert _to_str_card("JH") == "JH"
    assert _to_str_card(("J","H")) == "JH"
    assert _to_str_card(["10","H"]) == "TH"
    assert _normalize_cards(("JH","9D","AS","KD","2C")) == ["JH","9D","AS","KD","2C"]
