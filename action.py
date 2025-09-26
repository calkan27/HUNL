from action_type import ActionType

class Action:
	def __init__(self, action_type):
		self.action_type = action_type

	@property
	def kind(self):
		return self.action_type
	def __repr__(self):
		return f"{self.action_type.name}"
