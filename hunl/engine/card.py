"""
I implement a typed Card with rank and suit enums and two helpers: str for canonical
two-character code and from_string to parse external tokens. I make instances hashable
and immutable so they can be used in sets and maps if desired.

Key class: Card (dataclass frozen). Key methods: str — render like AH or 9d depending on
suit symbol mapping; from_string — robust parser that uppercases tokens and maps T to
Ten.

Inputs: Rank and Suit enums or raw two-character strings. Outputs: typed Card objects or
canonical strings. Invariants: input must be well-formed; I raise via enum lookup if a
character is invalid. Dependencies: engine.rank and engine.suit. Performance: constant
time.
"""

from dataclasses import dataclass
from hunl.engine.rank import Rank
from hunl.engine.suit import Suit

@dataclass(frozen=True)
class Card:
	rank: Rank
	suit: Suit

	def __str__(self) -> str:
		return f"{self.rank.symbol()}{self.suit.symbol()}"

	@staticmethod
	def from_string(code: str) -> "Card":
		code = str(code).strip().upper()
		rmap = {
			"2": Rank.Two,
			"3": Rank.Three,
			"4": Rank.Four,
			"5": Rank.Five,
			"6": Rank.Six,
			"7": Rank.Seven,
			"8": Rank.Eight,
			"9": Rank.Nine,
			"T": Rank.Ten,
			"J": Rank.Jack,
			"Q": Rank.Queen,
			"K": Rank.King,
			"A": Rank.Ace,
		}
		smap = {"C": Suit.Clubs, "D": Suit.Diamonds, "H": Suit.Hearts, "S": Suit.Spades}
		r = rmap[code[0]]
		s = smap[code[1]]
		return Card(r, s)


