from typing import Dict, Any, List, Tuple
from hunl.engine.poker_utils import board_one_hot as _board_one_hot
from hunl.engine.poker_utils import hand_rank, best_hand
from hunl.engine.action_type import ActionType


class PublicStateUtilsMixin:
	def terminal_utility(
		self
	) -> List[float]:
		u = [0.0, 0.0]

		if not bool(getattr(self, "is_terminal", False)):
			return u

		if int(self.players_in_hand.count(True)) == 1:
			if self.players_in_hand[0]:
				winner = 0
			else:
				winner = 1

			loser = 1 - winner

			u[winner] = float(self.pot_size - self.total_contrib[winner])
			u[loser] = float(-self.total_contrib[loser])

			return u

		if bool(getattr(self, "is_showdown", False)):
			h0 = list(self.hole_cards[0]) + list(self.board_cards)
			h1 = list(self.hole_cards[1]) + list(self.board_cards)

			r0 = hand_rank(best_hand(h0))
			r1 = hand_rank(best_hand(h1))

			c0 = float(self.total_contrib[0])
			c1 = float(self.total_contrib[1])

			if c0 < c1:
				m = c0
			else:
				m = c1

			main_pot = 2.0 * m

			if c0 > m:
				extra0 = c0 - m
			else:
				extra0 = 0.0

			if c1 > m:
				extra1 = c1 - m
			else:
				extra1 = 0.0

			if r0 > r1:
				win0 = main_pot + extra0
				win1 = extra1

				u[0] = win0 - c0
				u[1] = win1 - c1
			else:
				if r1 > r0:
					win1 = main_pot + extra1
					win0 = extra0

					u[1] = win1 - c1
					u[0] = win0 - c0
				else:
					win0 = 0.5 * main_pot + extra0
					win1 = 0.5 * main_pot + extra1

					u[0] = win0 - c0
					u[1] = win1 - c1

			return u

		return u

	def street_index(
		self
	) -> int:
		return int(getattr(self, "current_round", 0))

	def board_one_hot(
		self
	) -> List[int]:
		return _board_one_hot(list(self.board_cards))

	def public_summary(
		self
	) -> Dict[str, Any]:
		return {
			"pot_norm": float(self.pot_normalized()),
			"initial_stacks": tuple(int(x) for x in list(self.initial_stacks)),
			"street": int(self.street_index()),
			"board_one_hot": list(self.board_one_hot()),
		}

	def pot_normalized(
		self
	) -> float:
		if getattr(self, "initial_stacks", None) is not None:
			total_initial = float(sum(self.initial_stacks))
		else:
			total_initial = 1.0

		if total_initial <= 0.0:
			total_initial = 1.0

		return float(self.pot_size) / float(total_initial)

	def to_canonical(
		self
	) -> Dict[str, Any]:
		if getattr(self, "initial_stacks", None) is not None:
			total_initial = float(sum(self.initial_stacks))
		else:
			total_initial = 1.0

		if total_initial <= 0.0:
			total_initial = 1.0

		pot_norm = float(self.pot_size) / float(total_initial)

		if getattr(self, "last_raise_increment", None) is not None:
			last_bet = int(self.last_raise_increment)
		else:
			last_bet = 0

		if int(self.stacks[0]) < int(self.stacks[1]):
			eff_stack = int(self.stacks[0])
		else:
			eff_stack = int(self.stacks[1])

		if float(self.pot_size) > 0.0:
			den = float(self.pot_size)
		else:
			den = 1.0

		spr = float(eff_stack) / float(den)

		if getattr(self, "current_player", None) is None:
			cp = 0
		else:
			cp = int(self.current_player)

		parity = (cp - int(self.dealer)) & 1

		return {
			"street": int(self.current_round),
			"board": tuple(self.board_cards),
			"board_one_hot": list(self.board_one_hot()),
			"pot_norm": float(pot_norm),
			"last_bet": int(last_bet),
			"spr": float(spr),
			"parity": int(parity),
		}

	def legal_actions(self):
		out: List[ActionType] = []
		p = int(getattr(self, "current_player", 0))
		o = (p + 1) % 2

		cb = list(getattr(self, "current_bets", [0, 0]))
		if len(cb) < 2:
			cb = [0, 0]

		my_bet = int(cb[p])
		opp_bet = int(cb[o])

		to_call = opp_bet - my_bet
		if to_call < 0:
			to_call = 0

		min_raise_inc = int(self._min_raise_size())
		allow_raises = (
			getattr(self, "last_raise_was_allin_below_min", None) is None
		)

		if to_call > 0:
			out.append(ActionType.FOLD)

		out.append(ActionType.CALL)

		if not (self.players_in_hand[0] and self.players_in_hand[1]):
			seen_vals = set()
			filt: List[ActionType] = []
			i = 0
			while i < len(out):
				a = out[i]
				vi = int(a.value)
				if vi not in seen_vals:
					seen_vals.add(vi)
					filt.append(a)
				i += 1
			return filt

		if to_call == 0:
			if int(self.stacks[p]) > 0:
				if allow_raises:
					hp = max(min_raise_inc, int(float(self.pot_size) * 0.5))
					if (hp > 0) and (int(self.stacks[p]) >= hp):
						out.append(ActionType.HALF_POT_BET)

					pt = max(min_raise_inc, int(float(self.pot_size)))
					if (pt > 0) and (int(self.stacks[p]) >= pt):
						out.append(ActionType.POT_SIZED_BET)

					tp = max(min_raise_inc, int(float(self.pot_size) * 2.0))
					if (tp > 0) and (int(self.stacks[p]) >= tp):
						out.append(ActionType.TWO_POT_BET)

					out.append(ActionType.ALL_IN)
		else:
			if int(self.stacks[p]) > 0:
				if int(self.stacks[p]) <= int(to_call):
					out.append(ActionType.ALL_IN)
				else:
					if allow_raises:
						rem = int(self.stacks[p]) - int(to_call)
						if rem > 0:
							pac = float(self.pot_size) + float(to_call)

							hp = max(min_raise_inc, int(pac * 0.5))
						if (hp > 0) and (rem >= hp):
							out.append(ActionType.HALF_POT_BET)

						pt = max(min_raise_inc, int(pac))
						if (pt > 0) and (rem >= pt):
							out.append(ActionType.POT_SIZED_BET)

						tp = max(min_raise_inc, int(pac * 2.0))
						if (tp > 0) and (rem >= tp):
							out.append(ActionType.TWO_POT_BET)

						out.append(ActionType.ALL_IN)

		seen_vals = set()
		filt: List[ActionType] = []
		i = 0
		while i < len(out):
			a = out[i]
			vi = int(a.value)
			if vi not in seen_vals:
				seen_vals.add(vi)
				filt.append(a)
			i += 1
		return filt
