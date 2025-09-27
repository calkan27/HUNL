"""
I define the public action menu as an enum usable across engine, solver, and diagnostic
code. I include fold, call/check, half-pot bet, pot-sized bet, two-pot bet, and all-in.
I intentionally align numeric values with array indices in solver structures.

Key class: ActionType with members FOLD, CALL, HALF_POT_BET, POT_SIZED_BET, TWO_POT_BET,
ALL_IN.

Inputs: none beyond enum usage. Outputs: stable integer indices and names for menus and
strategies. Invariants: values are contiguous and start at zero to simplify indexing.
Performance: enum is zero-cost in hot loops when only .value is read.
"""

from enum import Enum

class ActionType(Enum):
	FOLD = 0
	CALL = 1
	HALF_POT_BET = 2
	POT_SIZED_BET = 3
	TWO_POT_BET = 4
	ALL_IN = 5

