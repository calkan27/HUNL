from action_type import ActionType


class PublicStateActionsMixin:
	def update_state(self, node, action):
		legal = set(self.legal_actions()) if hasattr(self, "legal_actions") else set()
		if action.action_type not in legal:
			return self
		prev_pot = float(self.pot_size)
		prev_round = int(self.current_round)
		new_state = self.clone()
		player = new_state.current_player
		opponent = (player + 1) % 2
		if new_state.is_terminal:
			return new_state
		new_state.last_action = action
		to_call = new_state.current_bets[opponent] - new_state.current_bets[player]
		if to_call < 0:
			to_call = 0
		new_state._last_action_was_call_on_bet = False
		if action.action_type == ActionType.FOLD:
			self._apply_fold(new_state, player, prev_pot)
			ok_inv = bool(new_state._assert_invariants(prev_pot=prev_pot))
			if not ok_inv:
				return self
			new_state.actions.append((player, action))
			return new_state
		if action.action_type == ActionType.CALL:
			self._apply_call(new_state, player, to_call)
		elif action.action_type == ActionType.HALF_POT_BET:
			self._apply_half_pot_bet(new_state, player, to_call)
		elif action.action_type == ActionType.POT_SIZED_BET:
			self._apply_pot_sized_bet(new_state, player, to_call)
		elif action.action_type == ActionType.TWO_POT_BET:
			self._apply_two_pot_bet(new_state, player, to_call)
		elif action.action_type == ActionType.ALL_IN:
			self._apply_all_in(new_state, player, to_call)
		if not new_state.is_terminal:
			if int(new_state.current_round) == prev_round:
				new_state.current_player = (player + 1) % 2
		if new_state.players_in_hand.count(True) == 1:
			new_state.is_terminal = True
		if new_state.pot_size < 0:
			new_state.pot_size = 0
		for i in (0, 1):
			if new_state.stacks[i] > new_state.initial_stacks[i]:
				new_state.stacks[i] = new_state.initial_stacks[i]
			if new_state.current_bets[i] < 0:
				new_state.current_bets[i] = 0
		ok_inv = bool(new_state._assert_invariants(prev_pot=prev_pot))
		if not ok_inv:
			return self
		new_state.actions.append((player, action))
		return new_state

	def _apply_fold(self, st, player, prev_pot):
		st.players_in_hand[player] = False
		st.is_terminal = True
		st._assert_invariants(prev_pot=prev_pot)

	def _apply_call(self, st, player, to_call):
		opponent = (player + 1) % 2

		if to_call == 0:
			st.consecutive_checks = st.consecutive_checks + 1
		else:
			pass

		caller_stack_before = st.stacks[player]
		if caller_stack_before < to_call:
			matched = caller_stack_before
		else:
			matched = to_call

		st.stacks[player] = st.stacks[player] - matched
		st.current_bets[player] = st.current_bets[player] + matched
		st.pot_size = st.pot_size + matched

		if matched > 0:
			st.total_contrib[player] += matched
		else:
			pass

		if matched < to_call:
			excess = to_call - matched
			st.last_refund_amount = float(excess)

			st.current_bets[opponent] = (
				st.current_bets[opponent] - excess
			)
			st.stacks[opponent] = st.stacks[opponent] + excess
			st.pot_size = st.pot_size - excess
			st.total_contrib[opponent] -= excess
		else:
			pass

		if to_call > 0:
			st.consecutive_checks = 0
			st._last_action_was_call_on_bet = True
		else:
			pass

		if st.current_bets[player] == st.current_bets[opponent]:
			if (st.stacks[0] == 0) or (st.stacks[1] == 0):
				st._fast_forward_to_showdown_if_allin_locked()
			else:
				st._advance_street_if_closed(actor=player)
		else:
			pass

	def _apply_half_pot_bet(self, st, player, to_call):
		opponent = (player + 1) % 2

		if to_call == 0:
			target = st.pot_size * 0.5

			if target < st.big_blind:
				target = st.big_blind
			else:
				pass

			if target > st.stacks[player]:
				target = st.stacks[player]
			else:
				pass

			if isinstance(target, float):
				bet_amt = int(target)
			else:
				bet_amt = target

			if bet_amt > 0:
				st.stacks[player] = st.stacks[player] - bet_amt
				st.current_bets[player] = (
					st.current_bets[player] + bet_amt
				)
				st.pot_size = st.pot_size + bet_amt
				st.total_contrib[player] += bet_amt

				st.last_raiser = player
				st.last_raise_increment = bet_amt

				min_raise_inc = st._min_raise_size()

				if (st.stacks[player] == 0) and (bet_amt < min_raise_inc):
					st.last_raise_was_allin_below_min = player
				else:
					st.last_raise_was_allin_below_min = None

				st.consecutive_checks = 0
			else:
				pass
		else:
			caller_stack_before = st.stacks[player]
			if caller_stack_before < to_call:
				matched = caller_stack_before
			else:
				matched = to_call

			st.stacks[player] = st.stacks[player] - matched
			st.current_bets[player] = (
				st.current_bets[player] + matched
			)
			st.pot_size = st.pot_size + matched
			st.total_contrib[player] += matched

			if matched < to_call:
				excess = to_call - matched
				st.last_refund_amount = float(excess)

				st.current_bets[opponent] = (
					st.current_bets[opponent] - excess
				)
				st.stacks[opponent] = st.stacks[opponent] + excess
				st.pot_size = st.pot_size - excess
				st.total_contrib[opponent] -= excess

				if st.current_bets[player] == st.current_bets[opponent]:
					if (st.stacks[0] == 0) or (st.stacks[1] == 0):
						st._fast_forward_to_showdown_if_allin_locked()
					else:
						st._advance_street_if_closed(actor=player)
				else:
					pass

				return
			else:
				pass

			pot_after_call = st.pot_size
			min_raise_inc = st._min_raise_size()
			half_pot_inc = pot_after_call * 0.5

			if half_pot_inc > min_raise_inc:
				raise_inc = half_pot_inc
			else:
				raise_inc = min_raise_inc

			if raise_inc > st.stacks[player]:
				raise_inc = st.stacks[player]
			else:
				pass

			if isinstance(raise_inc, float):
				raise_inc = int(raise_inc)
			else:
				pass

			if raise_inc > 0:
				st.stacks[player] = st.stacks[player] - raise_inc
				st.current_bets[player] = (
					st.current_bets[player] + raise_inc
				)
				st.pot_size = st.pot_size + raise_inc
				st.total_contrib[player] += raise_inc

				if raise_inc >= min_raise_inc:
					st.last_raiser = player
					st.last_raise_increment = raise_inc
					st.last_raise_was_allin_below_min = None
				else:
					if st.stacks[player] == 0:
						st.last_raise_was_allin_below_min = player
					else:
						pass

				st.consecutive_checks = 0
			else:
				pass

		if st.current_bets[player] == st.current_bets[opponent]:
			if (st.stacks[0] == 0) or (st.stacks[1] == 0):
				st._fast_forward_to_showdown_if_allin_locked()
			else:
				st._advance_street_if_closed(actor=player)
		else:
			pass

	def _apply_pot_sized_bet(self, st, player, to_call):
		opponent = (player + 1) % 2

		if to_call == 0:
			target = st.pot_size

			if target < st.big_blind:
				target = st.big_blind
			else:
				pass

			if target > st.stacks[player]:
				target = st.stacks[player]
			else:
				pass

			bet_amt = target

			if bet_amt > 0:
				st.stacks[player] = st.stacks[player] - bet_amt
				st.current_bets[player] = (
					st.current_bets[player] + bet_amt
				)
				st.pot_size = st.pot_size + bet_amt
				st.total_contrib[player] += bet_amt

				st.last_raiser = player
				st.last_raise_increment = bet_amt

				min_raise_inc = st._min_raise_size()

				if (st.stacks[player] == 0) and (bet_amt < min_raise_inc):
					st.last_raise_was_allin_below_min = player
				else:
					st.last_raise_was_allin_below_min = None

				st.consecutive_checks = 0
			else:
				pass
		else:
			caller_stack_before = st.stacks[player]
			if caller_stack_before < to_call:
				matched = caller_stack_before
			else:
				matched = to_call

			st.stacks[player] = st.stacks[player] - matched
			st.current_bets[player] = (
				st.current_bets[player] + matched
			)
			st.pot_size = st.pot_size + matched
			st.total_contrib[player] += matched

			if matched < to_call:
				excess = to_call - matched
				st.last_refund_amount = float(excess)

				st.current_bets[opponent] = (
					st.current_bets[opponent] - excess
				)
				st.stacks[opponent] = st.stacks[opponent] + excess
				st.pot_size = st.pot_size - excess
				st.total_contrib[opponent] -= excess

				if st.current_bets[player] == st.current_bets[opponent]:
					if (st.stacks[0] == 0) or (st.stacks[1] == 0):
						st._fast_forward_to_showdown_if_allin_locked()
					else:
						st._advance_street_if_closed(actor=player)
				else:
					pass

				return
			else:
				pass

			pot_after_call = st.pot_size
			min_raise_inc = st._min_raise_size()

			if pot_after_call > min_raise_inc:
				raise_inc = pot_after_call
			else:
				raise_inc = min_raise_inc

			if raise_inc > st.stacks[player]:
				raise_inc = st.stacks[player]
			else:
				pass

			if raise_inc > 0:
				st.stacks[player] = st.stacks[player] - raise_inc
				st.current_bets[player] = (
					st.current_bets[player] + raise_inc
				)
				st.pot_size = st.pot_size + raise_inc
				st.total_contrib[player] += raise_inc

				if raise_inc >= min_raise_inc:
					st.last_raiser = player
					st.last_raise_increment = raise_inc
					st.last_raise_was_allin_below_min = None
				else:
					if st.stacks[player] == 0:
						st.last_raise_was_allin_below_min = player
					else:
						pass

				st.consecutive_checks = 0
			else:
				pass

		if st.current_bets[player] == st.current_bets[opponent]:
			if (st.stacks[0] == 0) or (st.stacks[1] == 0):
				st._fast_forward_to_showdown_if_allin_locked()
			else:
				st._advance_street_if_closed(actor=player)
		else:
			pass

	def _apply_two_pot_bet(self, st, player, to_call):
		opponent = (player + 1) % 2

		if to_call == 0:
			target = st.pot_size * 2.0

			if target < st.big_blind:
				target = st.big_blind
			else:
				pass

			if target > st.stacks[player]:
				target = st.stacks[player]
			else:
				pass

			if isinstance(target, float):
				bet_amt = int(target)
			else:
				bet_amt = target

			if bet_amt > 0:
				st.stacks[player] = st.stacks[player] - bet_amt
				st.current_bets[player] = (
					st.current_bets[player] + bet_amt
				)
				st.pot_size = st.pot_size + bet_amt
				st.total_contrib[player] += bet_amt

				st.last_raiser = player
				st.last_raise_increment = bet_amt

				min_raise_inc = st._min_raise_size()

				if (st.stacks[player] == 0) and (bet_amt < min_raise_inc):
					st.last_raise_was_allin_below_min = player
				else:
					st.last_raise_was_allin_below_min = None

				st.consecutive_checks = 0
			else:
				pass
		else:
			caller_stack_before = st.stacks[player]
			if caller_stack_before < to_call:
				matched = caller_stack_before
			else:
				matched = to_call

			st.stacks[player] = st.stacks[player] - matched
			st.current_bets[player] = (
				st.current_bets[player] + matched
			)
			st.pot_size = st.pot_size + matched
			st.total_contrib[player] += matched

			if matched < to_call:
				excess = to_call - matched
				st.last_refund_amount = float(excess)

				st.current_bets[opponent] = (
					st.current_bets[opponent] - excess
				)
				st.stacks[opponent] = st.stacks[opponent] + excess
				st.pot_size = st.pot_size - excess
				st.total_contrib[opponent] -= excess

				if st.current_bets[player] == st.current_bets[opponent]:
					if (st.stacks[0] == 0) or (st.stacks[1] == 0):
						st._fast_forward_to_showdown_if_allin_locked()
					else:
						st._advance_street_if_closed(actor=player)
				else:
					pass

				return
			else:
				pass

			pot_after_call = st.pot_size
			min_raise_inc = st._min_raise_size()
			two_pot_inc = pot_after_call * 2.0

			if two_pot_inc > min_raise_inc:
				raise_inc = two_pot_inc
			else:
				raise_inc = min_raise_inc

			if raise_inc > st.stacks[player]:
				raise_inc = st.stacks[player]
			else:
				pass

			if isinstance(raise_inc, float):
				raise_inc = int(raise_inc)
			else:
				pass

			if raise_inc > 0:
				st.stacks[player] = st.stacks[player] - raise_inc
				st.current_bets[player] = (
					st.current_bets[player] + raise_inc
				)
				st.pot_size = st.pot_size + raise_inc
				st.total_contrib[player] += raise_inc

				if raise_inc >= min_raise_inc:
					st.last_raiser = player
					st.last_raise_increment = raise_inc
					st.last_raise_was_allin_below_min = None
				else:
					if st.stacks[player] == 0:
						st.last_raise_was_allin_below_min = player
					else:
						pass

				st.consecutive_checks = 0
			else:
				pass

		if st.current_bets[player] == st.current_bets[opponent]:
			if (st.stacks[0] == 0) or (st.stacks[1] == 0):
				st._fast_forward_to_showdown_if_allin_locked()
			else:
				st._advance_street_if_closed(actor=player)
		else:
			pass

	def _apply_all_in(self, st, player, to_call):
		opponent = (player + 1) % 2

		st._last_action_was_call_on_bet = False
		st.last_refund_amount = 0.0

		min_raise_inc = st._min_raise_size()

		if to_call > 0:
			caller_stack_before = st.stacks[player]
			if caller_stack_before < to_call:
				matched = caller_stack_before
			else:
				matched = to_call

			st.stacks[player] = st.stacks[player] - matched
			st.current_bets[player] = (
				st.current_bets[player] + matched
			)
			st.pot_size = st.pot_size + matched

			if matched > 0:
				st.total_contrib[player] += matched
			else:
				pass

			if matched < to_call:
				excess = to_call - matched
				st.last_refund_amount = float(excess)

				st.current_bets[opponent] = (
					st.current_bets[opponent] - excess
				)
				st.stacks[opponent] = st.stacks[opponent] + excess
				st.pot_size = st.pot_size - excess
				st.total_contrib[opponent] -= excess

				if st.current_bets[player] == st.current_bets[opponent]:
					if (st.stacks[0] == 0) or (st.stacks[1] == 0):
						st._fast_forward_to_showdown_if_allin_locked()
					else:
						st._advance_street_if_closed(actor=player)
				else:
					pass

				return
			else:
				pass

			rest = st.stacks[player]

			if rest < 0:
				rest = 0
			else:
				pass

			if rest > 0:
				st.current_bets[player] = (
					st.current_bets[player] + rest
				)
				st.stacks[player] = 0
				st.pot_size = st.pot_size + rest
				st.total_contrib[player] += rest

				increment_above_call = rest

				if increment_above_call >= min_raise_inc:
					st.last_raiser = player
					st.last_raise_increment = increment_above_call
					st.last_raise_was_allin_below_min = None
				else:
					st.last_raise_was_allin_below_min = player
			else:
				pass

			st.consecutive_checks = 0
		else:
			rest = st.stacks[player]

			if rest < 0:
				rest = 0
			else:
				pass

			if rest > 0:
				st.current_bets[player] = (
					st.current_bets[player] + rest
				)
				st.stacks[player] = 0
				st.pot_size = st.pot_size + rest
				st.total_contrib[player] += rest

				if rest >= min_raise_inc:
					st.last_raiser = player
					st.last_raise_increment = rest
					st.last_raise_was_allin_below_min = None
				else:
					st.last_raise_was_allin_below_min = player
			else:
				pass

			st.consecutive_checks = 0

		if st.current_bets[player] == st.current_bets[opponent]:
			if (st.stacks[0] == 0) or (st.stacks[1] == 0):
				st._fast_forward_to_showdown_if_allin_locked()
			else:
				st._advance_street_if_closed(actor=player)
		else:
			pass

	def apply_exogenous_opponent_check(self, actor):
		if self.is_terminal:
			return self
		else:
			pass

		if actor in (0, 1):
			pass
		else:
			return self

		if self.current_player is None:
			return self
		else:
			if int(self.current_player) != int(actor):
				return self
			else:
				pass

		if hasattr(self, "legal_actions"):
			legal = set(self.legal_actions())
		else:
			legal = set()

		if ActionType.CALL in legal:
			pass
		else:
			return self

		p = int(actor)
		o = (p + 1) % 2

		my_bet = self.current_bets[p]
		opp_bet = self.current_bets[o]
		to_call = opp_bet - my_bet

		if to_call < 0:
			to_call = 0
		else:
			pass

		if to_call == 0:
			pass
		else:
			return self

		prev_pot = float(self.pot_size)
		prev_round = int(self.current_round)

		self.last_refund_amount = 0.0
		self.consecutive_checks = self.consecutive_checks + 1
		self._last_action_was_call_on_bet = False

		if self.current_bets[p] == self.current_bets[o]:
			self._advance_street_if_closed(actor=p)
			self._fast_forward_to_showdown_if_allin_locked()
		else:
			pass

		if not self.is_terminal:
			if int(self.current_round) == int(prev_round):
				self.current_player = o
			else:
				pass
		else:
			pass

		self.last_action = ("EXOGENOUS_CHECK", p)
		self._assert_invariants(prev_pot=prev_pot)
		return self

	def apply_exogenous_opponent_bet(self, bettor, kind):
		if self.is_terminal:
			return self
		else:
			pass

		if bettor in (0, 1):
			pass
		else:
			return self

		if kind == ActionType.ALL_IN:
			return self.apply_exogenous_opponent_all_in(bettor)
		else:
			pass

		if kind in (
			ActionType.HALF_POT_BET,
			ActionType.POT_SIZED_BET,
			ActionType.TWO_POT_BET,
		):
			pass
		else:
			return self

		if self.current_player is None:
			return self
		else:
			if int(self.current_player) != int(bettor):
				return self
			else:
				pass

		if hasattr(self, "legal_actions"):
			legal = set(self.legal_actions())
		else:
			legal = set()

		if kind in legal:
			pass
		else:
			return self

		prev_pot = float(self.pot_size)
		self.last_refund_amount = 0.0

		p = int(bettor)
		o = (p + 1) % 2

		my_bet = self.current_bets[p]
		opp_bet = self.current_bets[o]
		to_call = opp_bet - my_bet

		if to_call < 0:
			to_call = 0
		else:
			pass

		if kind == ActionType.HALF_POT_BET:
			self._apply_half_pot_bet(self, p, to_call)
		else:
			if kind == ActionType.POT_SIZED_BET:
				self._apply_pot_sized_bet(self, p, to_call)
			else:
				if kind == ActionType.TWO_POT_BET:
					self._apply_two_pot_bet(self, p, to_call)
				else:
					pass

		self.last_raiser = (p,)
		self.last_action = ("EXOGENOUS_BET", int(kind.value))
		self._assert_invariants(prev_pot=prev_pot)
		return self

	def apply_exogenous_opponent_all_in(self, bettor):
		if self.is_terminal:
			return self
		else:
			pass

		if bettor in (0, 1):
			pass
		else:
			return self

		if self.current_player is None:
			return self
		else:
			if int(self.current_player) != int(bettor):
				return self
			else:
				pass

		prev_pot = float(self.pot_size)
		self.last_refund_amount = 0.0

		p = int(bettor)
		o = (p + 1) % 2

		my_bet = self.current_bets[p]
		opp_bet = self.current_bets[o]
		to_call = opp_bet - my_bet

		if to_call < 0:
			to_call = 0
		else:
			pass

		caller_stack_before = self.stacks[p]
		if caller_stack_before < to_call:
			matched = caller_stack_before
		else:
			matched = to_call

		self.stacks[p] = self.stacks[p] - matched
		self.current_bets[p] = self.current_bets[p] + matched
		self.pot_size = self.pot_size + matched

		if matched > 0:
			self.total_contrib[p] += matched
		else:
			pass

		if matched < to_call:
			excess = to_call - matched
			self.last_refund_amount = float(excess)

			self.current_bets[o] = self.current_bets[o] - excess
			self.stacks[o] = self.stacks[o] + excess
			self.pot_size = self.pot_size - excess
			self.total_contrib[o] -= excess
		else:
			rest = self.stacks[p]

			if rest < 0:
				rest = 0
			else:
				pass

			if rest > 0:
				self.current_bets[p] = self.current_bets[p] + rest
				self.stacks[p] = 0
				self.pot_size = self.pot_size + rest
				self.total_contrib[p] += rest

				if to_call > 0:
					increment_above_call = rest
				else:
					increment_above_call = rest

				min_raise_inc = self._min_raise_size()

				if to_call == 0:
					if rest >= min_raise_inc:
						self.last_raiser = (p,)
						self.last_raise_increment = rest
						self.last_raise_was_allin_below_min = None
					else:
						self.last_raise_was_allin_below_min = p
				else:
					if increment_above_call >= min_raise_inc:
						self.last_raiser = (p,)
						self.last_raise_increment = increment_above_call
						self.last_raise_was_allin_below_min = None
					else:
						self.last_raise_was_allin_below_min = p
			else:
				pass

			self.consecutive_checks = 0

		if self.current_bets[p] == self.current_bets[o]:
			if (self.stacks[0] == 0) or (self.stacks[1] == 0):
				self._fast_forward_to_showdown_if_allin_locked()
			else:
				self._advance_street_if_closed(actor=p)
		else:
			pass

		self.last_action = ("EXOGENOUS_ALL_IN", p)
		self._assert_invariants(prev_pot=prev_pot)
		return self
