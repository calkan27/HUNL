from enum import Enum

class ActionType(Enum):
	FOLD = 0
	CALL = 1
	HALF_POT_BET = 2
	POT_SIZED_BET = 3
	TWO_POT_BET = 4
	ALL_IN = 5

