from typing import Optional
from hunl.engine.poker_utils import DECK


class PublicStateStreetsMixin:
	def _reset_street_bookkeeping(self) -> None:
		self.last_raiser = None
		self.last_raise_increment = int(getattr(self, "big_blind", 2))
		self.last_raise_was_allin_below_min = None
		self.consecutive_checks = 0
		self._last_action_was_call_on_bet = False

	def _min_raise_size(self) -> int:
		val = getattr(self, "last_raise_increment", None)

		inc = 0
		ok = False

		try:
			inc = int(val)
			ok = True
		except Exception:
			pass

		if not ok:
			first = None
			try:
				it = iter(val)
				first = next(it)
			except Exception:
				first = None

			if first is not None:
				try:
					inc = int(first)
					ok = True
				except Exception:
					pass

			if not ok:
				try:
					inc = int(str(val).strip())
					ok = True
				except Exception:
					inc = 0

		bb = 2
		try:
			bb = int(getattr(self, "big_blind", 2))
		except Exception:
			try:
				bb = int(str(getattr(self, "big_blind", 2)).strip())
			except Exception:
				bb = 2

		if inc < bb:
			inc = bb
		if inc < 0:
			inc = 0

		return int(inc)

	def _deal_for_new_street(self) -> None:
		used = set(list(getattr(self, "board_cards", [])))
		h0 = list(getattr(self, "hole_cards", [[], []])[0])
		h1 = list(getattr(self, "hole_cards", [[], []])[1])
		i = 0
		while i < len(h0):
			used.add(h0[i])
			i += 1
		j = 0
		while j < len(h1):
			used.add(h1[j])
			j += 1
		r = int(getattr(self, "current_round", 0))
		if r == 1:
			target = 3
		else:
			if r == 2:
				target = 4
			else:
				target = 5
		k = 0
		while (len(self.board_cards) < target) and (k < len(DECK)):
			c = DECK[k]
			if c not in used:
				self.board_cards.append(c)
				used.add(c)
			k += 1

	def _fast_forward_to_showdown_if_allin_locked(self) -> None:
		if self.players_in_hand[0] and self.players_in_hand[1]:
			any_all_in = (int(self.stacks[0]) == 0) or (int(self.stacks[1]) == 0)
			bets_equalized = (int(self.current_bets[0]) == int(self.current_bets[1]))
			if any_all_in and bets_equalized:
				used = set(list(getattr(self, "board_cards", [])))
				h0 = list(getattr(self, "hole_cards", [[], []])[0])
				h1 = list(getattr(self, "hole_cards", [[], []])[1])
				i = 0
				while i < len(h0):
					used.add(h0[i])
					i += 1
				j = 0
				while j < len(h1):
					used.add(h1[j])
					j += 1
				k = 0
				while (len(self.board_cards) < 5) and (k < len(DECK)):
					c = DECK[k]
					if c not in used:
						self.board_cards.append(c)
						used.add(c)
					k += 1
				self.is_terminal = True
				self.is_showdown = True

	@staticmethod
	def _lr_norm(v):
		if v is None:
			return None
		if isinstance(v, (tuple, list)):
			if v:
				return int(v[0])
			else:
				return None
		else:
			return int(v)

	def _advance_street_if_closed(self, actor: int) -> None:
		opp = (int(actor) + 1) % 2
		any_all_in = (int(self.stacks[0]) == 0) or (int(self.stacks[1]) == 0)

		if self.players_in_hand[0] and self.players_in_hand[1]:
			if int(self.current_bets[actor]) == int(self.current_bets[opp]):
				if any_all_in:
					self.is_terminal = True
					self.is_showdown = True
					return

				lr = self._lr_norm(self.last_raiser)

				if lr is not None:
					if bool(getattr(self, "_last_action_was_call_on_bet", False)):
						if int(self.current_round) == 0:
							if (int(actor) == int(self.dealer)) and (int(lr) == (int(self.dealer) + 1) % 2):
								return
						if int(self.current_round) < 3:
							self.current_round = int(self.current_round) + 1
							self.current_bets = [0, 0]
							self._reset_street_bookkeeping()
							self._deal_for_new_street()
							if int(self.current_round) >= 1:
								self.current_player = (int(self.dealer) + 1) % 2
							else:
								self.current_player = int(self.dealer)
						else:
							self.is_terminal = True
							self.is_showdown = True
						return

					if int(actor) == int(lr):
						if int(self.current_round) < 3:
							self.current_round = int(self.current_round) + 1
							self.current_bets = [0, 0]
							self._reset_street_bookkeeping()
							self._deal_for_new_street()
							if int(self.current_round) >= 1:
								self.current_player = (int(self.dealer) + 1) % 2
							else:
								self.current_player = int(self.dealer)
						else:
							self.is_terminal = True
							self.is_showdown = True
				else:
					if bool(getattr(self, "_last_action_was_call_on_bet", False)) or (int(self.consecutive_checks) >= 2):
						if int(self.current_round) < 3:
							self.current_round = int(self.current_round) + 1
							self.current_bets = [0, 0]
							self._reset_street_bookkeeping()
							self._deal_for_new_street()
							if int(self.current_round) >= 1:
								self.current_player = (int(self.dealer) + 1) % 2
							else:
								self.current_player = int(self.dealer)
						else:
							self.is_terminal = True
							self.is_showdown = True
