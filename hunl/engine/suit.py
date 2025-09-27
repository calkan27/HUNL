"""
I define Suit as an IntEnum with four values and a symbol helper. I use C, D, H, S for
Clubs, Diamonds, Hearts, and Spades to match the rest of the engine’s two-character
codes.

Key class: Suit; method symbol returns one of C/D/H/S.

Inputs: enum usage. Outputs: canonical suit characters. Invariants: total mapping; no
localization. Dependencies: stdlib enum. Performance: trivial.
"""

from enum import IntEnum

class Suit(IntEnum):
    Clubs   = 0
    Diamonds= 1
    Hearts  = 2
    Spades  = 3

    def symbol(self) -> str:
        return {
            Suit.Clubs: "C",
            Suit.Diamonds: "D",
            Suit.Hearts: "H",
            Suit.Spades: "S",
        }[self]
