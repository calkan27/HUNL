import math
from action_type import ActionType
from action import Action
from game_node import GameNode

class AIVATEvaluator:
	def __init__(self, value_fn, policy_fn, chance_policy_fn=None, agent_player=0):
		self.value_fn = value_fn
		self.policy_fn = policy_fn
		self.chance_policy_fn = chance_policy_fn
		self.agent_player = int(agent_player)

	def evaluate(self, episode):
		node = episode["initial_node"]
		ev = 0.0
		events = episode.get("events", [])
		i = 0
		while i < len(events):
			ev += self._cv_term(node, events[i])
			node = self._advance(node, events[i])
			i += 1
		ev += self._terminal_payoff(node)
		return {"aivat": float(ev), "num_terms": int(len(events))}

	def _cv_term(self, node, event):
		t = event.get("type")
		if t == "chance":
			pol = self._action_dist(node, player=None, chance=True)
			obs = event.get("action")
			o = self._q_after(node, None, obs)
			e = 0.0
			for a, pa in pol.items():
				e += float(pa) * float(self._q_after(node, None, a))
			return float(o - e)
		if t == "agent":
			p = self.agent_player
			pol = self._action_dist(node, player=p, chance=False)
			obs = event.get("action")
			o = self._q_after(node, p, obs)
			e = 0.0
			for a, pa in pol.items():
				e += float(pa) * float(self._q_after(node, p, a))
			return float(o - e)
		return 0.0

	def _advance(self, node, event):
		ps = node.public_state
		t = event.get("type")
		if t == "chance":
			card = event.get("action")
			new_ps = ps.clone()
			if card not in new_ps.board_cards and len(new_ps.board_cards) < 5:
				new_ps.board_cards.append(card)
				if len(new_ps.board_cards) == 3:
					new_ps.current_round = 1
					new_ps.current_bets = [0, 0]
					new_ps.last_raiser = None
					new_ps.current_player = (new_ps.dealer + 1) % 2
				elif len(new_ps.board_cards) == 4:
					new_ps.current_round = 2
					new_ps.current_bets = [0, 0]
					new_ps.last_raiser = None
					new_ps.current_player = (new_ps.dealer + 1) % 2
				elif len(new_ps.board_cards) == 5:
					new_ps.current_round = 3
					new_ps.current_bets = [0, 0]
					new_ps.last_raiser = None
					new_ps.current_player = (new_ps.dealer + 1) % 2
			new_node = GameNode(new_ps)
			new_node.player_ranges = [dict(node.player_ranges[0]), dict(node.player_ranges[1])]
			return new_node
		if t in ("agent","opponent"):
			a = event.get("action")
			new_ps = ps.update_state(node, Action(a))
			new_node = GameNode(new_ps)
			new_node.player_ranges = [dict(node.player_ranges[0]), dict(node.player_ranges[1])]
			return new_node
		return node

	def _terminal_payoff(self, node):
		ps = node.public_state
		if not getattr(ps, "is_terminal", False):
			return 0.0
		cfv = self.value_fn(node, self.agent_player)
		return float(self._range_expect(node, self.agent_player, cfv))

	def _q_after(self, node, player, action_type):
		ps = node.public_state

		if player is None:
			card = action_type
			new_ps = ps.clone()
			if card not in new_ps.board_cards and len(new_ps.board_cards) < 5:
				new_ps.board_cards.append(card)
				if len(new_ps.board_cards) == 3:
					new_ps.current_round = 1
					new_ps.current_bets = [0, 0]
					new_ps.last_raiser = None
					new_ps.current_player = (new_ps.dealer + 1) % 2
				elif len(new_ps.board_cards) == 4:
					new_ps.current_round = 2
					new_ps.current_bets = [0, 0]
					new_ps.last_raiser = None
					new_ps.current_player = (new_ps.dealer + 1) % 2
				elif len(new_ps.board_cards) == 5:
					new_ps.current_round = 3
					new_ps.current_bets = [0, 0]
					new_ps.last_raiser = None
					new_ps.current_player = (new_ps.dealer + 1) % 2
		else:
			new_ps = ps.update_state(node, Action(action_type))

		child = GameNode(new_ps)

		child.player_ranges = []
		for pr in node.player_ranges:
			child.player_ranges.append(dict(pr))

		if player is None:
			target_player = self.agent_player
		else:
			target_player = player

		cfv = self.value_fn(child, target_player)
		return float(self._range_expect(child, target_player, cfv))


	def _range_expect(self, node, player, cfv_by_cluster):
		r = node.player_ranges[player]
		s = 0.0
		t = 0.0
		for cid, p in r.items():
			t += float(p)
		if t <= 0.0:
			return 0.0
		for cid, p in r.items():
			v = cfv_by_cluster.get(int(cid), [0.0])
			if isinstance(v, (list, tuple)):
				val = float(v[0])
			else:
				val = float(v)
			s += (float(p) / t) * val
		return s

	def _action_dist(self, node, player=None, chance=False):
		if chance:
			if self.chance_policy_fn is not None:
				items = self.chance_policy_fn(node)
				if isinstance(items, dict):
					return dict(items)
				else:
					return dict(items)
			else:
				return {}
		else:
			target = player if player is not None else self.agent_player
			items = self.policy_fn(node, target)
			if isinstance(items, dict):
				return dict(items)
			else:
				return dict(items)
