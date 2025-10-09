"""
I represent a single public action as a thin wrapper around ActionType. I keep value
semantics simple for logging, equality, and display, and I avoid state beyond the type
itself.

Key class: Action. Key properties/methods: action_type — underlying enum; kind — alias;
repr — friendly name for logs.

Inputs: an ActionType value such as CALL, POT_SIZED_BET, or ALL_IN. Outputs: immutable
value objects that engine and solver pass to update_state and tree builders.

Dependencies: ActionType from the same package. Invariants: I never encode sizing here;
sizes are handled inside PublicState transitions. Performance: negligible.
"""

from hunl.engine.action_type import ActionType

class Action:
	def __init__(self, action_type):
		self.action_type = action_type

	@property
	def kind(self):
		return self.action_type
	def __repr__(self):
		return f"{self.action_type.name}"
