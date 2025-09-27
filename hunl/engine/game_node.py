class GameNode:
	def __init__(self, public_state):
		self.public_state = public_state

		pih = getattr(public_state, "players_in_hand", None)
		if isinstance(pih, list):
			if len(pih) >= 2:
				self.players_in_hand = pih.copy()
			else:
				self.players_in_hand = [True, True]
		else:
			self.players_in_hand = [True, True]

		cp = getattr(public_state, "current_player", None)
		if cp is not None:
			self.current_player = int(cp)
		else:
			self.current_player = 0

		self.player_ranges = [{}, {}]

	def _serialize_actions(self):
		out = []

		alist = getattr(self.public_state, "actions", None)
		if isinstance(alist, list):
			i = 0
			while i < len(alist):
				pl, act = alist[i]

				at = getattr(act, "action_type", None)
				av = getattr(at, "value", at)

				if isinstance(av, bool) is False:
					if av is not None:
						val = int(av)
					else:
						val = 0
				else:
					val = 0

				out.append((int(pl), val))
				i += 1

		return out

	def _public_signature(self):
		actions_serialized = self._serialize_actions()

		if hasattr(self.public_state, "current_round"):
			cr = int(getattr(self.public_state, "current_round", 0))
		else:
			cr = int(getattr(self.public_state, "round_idx", 0))

		cb = tuple(getattr(self.public_state, "current_bets", (0, 0))[:2])

		if getattr(self.public_state, "current_player", None) is not None:
			curp = int(getattr(self.public_state, "current_player", -1))
		else:
			curp = -1

		is_term = bool(getattr(self.public_state, "is_terminal", False))
		is_show = bool(getattr(self.public_state, "is_showdown", False))

		pi = getattr(self.public_state, "players_in_hand", [True, True])[:2]
		pi_tuple = tuple(bool(x) for x in pi)

		sig = (
			tuple(getattr(self.public_state, "board_cards", [])),
			cr,
			cb,
			getattr(self.public_state, "pot_size", 0),
			curp,
			int(getattr(self.public_state, "dealer", 0)),
			is_term,
			is_show,
			pi_tuple,
			tuple(actions_serialized),
		)
		return sig

	def __hash__(self):
		if not hasattr(self, "_frozen_sig"):
			self._frozen_sig = self._public_signature()
		return hash(self._frozen_sig)

	def __eq__(self, other):
		if not isinstance(other, GameNode):
			return False

		if not hasattr(self, "_frozen_sig"):
			self._frozen_sig = self._public_signature()

		if not hasattr(other, "_frozen_sig"):
			other._frozen_sig = other._public_signature()

		return self._frozen_sig == other._frozen_sig

