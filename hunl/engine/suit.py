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
