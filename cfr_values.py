import math
from collections import defaultdict
from action_type import ActionType


class CFRValues:
	def __init__(self):
		num_actions = len(ActionType)
		self.cumulative_regret = defaultdict(lambda: [0.0] * num_actions)
		self.cumulative_positive_regret = defaultdict(lambda: [0.0] * num_actions)
		self.cumulative_strategy = defaultdict(lambda: [0.0] * num_actions)
		self.strategy = defaultdict(lambda: [1.0 / num_actions] * num_actions)
		self.pruned_actions = defaultdict(set)
		self.regret_squared_sums = defaultdict(lambda: [0.0] * num_actions)

	@staticmethod
	def _normal_cdf(x: float) -> float:
		return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

	@staticmethod
	def _safe_std_err(iteration: int, mean_regret: float, sq_sum: float) -> float:
		variance = (sq_sum / float(iteration)) - (mean_regret ** 2)
		if variance < 1e-12:
			variance = 1e-12
		std_err = math.sqrt(variance) / math.sqrt(float(iteration))
		if std_err <= 0.0:
			std_err = 1e-6
		return std_err

	def compute_strategy(self, cluster_id):
		num_actions = len(ActionType)
		regrets = self.cumulative_positive_regret[cluster_id]
		pruned = self.pruned_actions[cluster_id]

		strategy = [0.0] * num_actions
		total_pos = 0.0

		i = 0
		while i < num_actions:
			if i not in pruned:
				total_pos += regrets[i]
			i += 1

		if total_pos > 0.0:
			i = 0
			while i < num_actions:
				if i not in pruned:
					strategy[i] = regrets[i] / total_pos
				i += 1
		else:
			avail = num_actions - len(pruned)
			if avail > 0:
				u = 1.0 / avail
				i = 0
				while i < num_actions:
					if i not in pruned:
						strategy[i] = u
					i += 1

		self.strategy[cluster_id] = strategy
		return strategy

	def update_strategy(self, cluster_id, strategy):
		i = 0
		while i < len(strategy):
			self.cumulative_strategy[cluster_id][i] += strategy[i]
			i += 1

	def get_average_strategy(self, cluster_id):
		total = sum(self.cumulative_strategy[cluster_id])
		num_actions = len(ActionType)
		avg = [0.0] * num_actions

		if total > 0.0:
			i = 0
			while i < num_actions:
				avg[i] = self.cumulative_strategy[cluster_id][i] / total
				i += 1
		else:
			avail = num_actions - len(self.pruned_actions[cluster_id])
			if avail > 0:
				u = 1.0 / avail
				i = 0
				while i < num_actions:
					if i not in self.pruned_actions[cluster_id]:
						avg[i] = u
					i += 1

		return avg

	def prune_actions(
		self,
		cluster_id,
		iteration,
		total_iterations,
		min_iterations: int = 100,
		alpha: float = 0.05,
	):
		num_actions = len(ActionType)

		if iteration < min_iterations:
			return

		i = 0
		while i < num_actions:
			if i in self.pruned_actions[cluster_id]:
				i += 1
				continue

			mean_regret = self.cumulative_regret[cluster_id][i] / float(iteration)
			sq_sum = self.regret_squared_sums[cluster_id][i]
			std_err = self._safe_std_err(iteration, mean_regret, sq_sum)

			t_stat = mean_regret / std_err
			p_value = self._normal_cdf(t_stat)

			if p_value < alpha:
				self.pruned_actions[cluster_id].add(i)

			i += 1

	def reassess_pruned_actions(
		self,
		cluster_id,
		iteration,
		alpha: float = 0.05,
	):
		to_check = list(self.pruned_actions[cluster_id])

		j = 0
		while j < len(to_check):
			i = to_check[j]

			if iteration <= 1:
				j += 1
				continue

			mean_regret = self.cumulative_regret[cluster_id][i] / float(iteration)
			sq_sum = self.regret_squared_sums[cluster_id][i]
			std_err = self._safe_std_err(iteration, mean_regret, sq_sum)

			t_stat = mean_regret / std_err
			p_value = 1.0 - self._normal_cdf(t_stat)

			if p_value < alpha:
				self.pruned_actions[cluster_id].discard(i)

			j += 1

