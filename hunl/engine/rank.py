"""
I define Rank as an IntEnum with symbols and an accessor that returns canonical
single-character representations for logging and parsing helpers. I keep numeric
ordering consistent with poker strength (Two low, Ace high).

Key class: Rank with values Two..Ace; method symbol to map to 2..9,T,J,Q,K,A.

Inputs: enum usage. Outputs: stable integer ordering and symbol strings. Invariants:
mapping is total and fixed. Dependencies: stdlib enum. Performance: trivial.
"""

from enum import IntEnum


class Rank(IntEnum):
	Two = 2
	Three = 3
	Four = 4
	Five = 5
	Six = 6
	Seven = 7
	Eight = 8
	Nine = 9
	Ten = 10
	Jack = 11
	Queen = 12
	King = 13
	Ace = 14

	def symbol(self) -> str:
		m = {
			2: "2",
			3: "3",
			4: "4",
			5: "5",
			6: "6",
			7: "7",
			8: "8",
			9: "9",
			10: "T",
			11: "J",
			12: "Q",
			13: "K",
			14: "A",
		}

		return m[int(self)]

