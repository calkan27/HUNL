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

