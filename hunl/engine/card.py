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


