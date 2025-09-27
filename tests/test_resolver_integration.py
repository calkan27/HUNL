"""
Test suite for resolver integration and ValueServer wiring: verifies bet-size modes, 
opponent-upper-bound monotonic updates, 
 stage-appropriate CFV net usage, diagnostic acceptance on flop, terminal solving on turn,
 and constraint-mode behavior (self-play vs best-response).
"""

import hunl.nets.value_server as value_server
import hunl.engine.poker_utils as poker_utils
import hunl.solving.resolver_integration as resolver_integration
import numpy as np
import torch
import pytest

from hunl.solving.resolver_integration import resolve_at_with_diag, resolve_at, _bet_fraction_schedule_for_mode, _tighten_cfv_upper_bounds
from hunl.nets.value_server import ValueServer
from hunl.engine.public_state import PublicState
from hunl.engine.poker_utils import DECK
from hunl.engine.action_type import ActionType


class _DummyCFVNet(torch.nn.Module):
	"""
	Minimal CFV net stub returning constant per-bucket outputs and supporting zero-sum enforcement to exercise resolver plumbing without learning dynamics.
	"""
	def __init__(self, K, board_dim=52):
		"""
		Initialize with derived input size matching pot-normalization, board one-hot, and two range vectors for K clusters.
		"""
		super().__init__()
		self.num_clusters = int(K)
		self.input_size = 1 + board_dim + 2 * K

	def forward(self, x: torch.Tensor):
		"""
		Produce deterministic CFV tensors (p1 positive, p2 negative) shaped [N, K] to enable zero-sum checks and query counting.
		"""
		N = x.shape[0]; K = self.num_clusters
		p1 = torch.full((N, K), 0.1, dtype=torch.float32, device=x.device)
		p2 = torch.full((N, K), -0.05, dtype=torch.float32, device=x.device)
		return p1, p2

	@torch.no_grad()
	def enforce_zero_sum(self, r1, r2, p1, p2):
		"""
		Shift predictions so that the range-weighted expectations sum to zero per row for both players.
		"""
		sum_r1 = torch.clamp(torch.sum(r1, dim=1, keepdim=True), min=1e-9)
		sum_r2 = torch.clamp(torch.sum(r2, dim=1, keepdim=True), min=1e-9)
		s = torch.sum(r1 * p1, dim=1, keepdim=True) + torch.sum(r2 * p2, dim=1, keepdim=True)
		a = -0.5 * s / sum_r1
		b = -0.5 * s / sum_r2
		return p1 + a, p2 + b


def _uniform_range(K):
	"""
	Build a uniform probability dictionary over K clusters for use as ranges or bounds.
	"""
	u = 1.0 / float(K) if K > 0 else 0.0
	return {i: u for i in range(K)}


@pytest.fixture
def flop_state():
	"""
	Construct a flop PublicState with neutral bets to exercise flop-stage resolve behavior and diagnostic checks.
	"""
	deck = list(DECK)
	board = deck[:3]
	ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
	ps.current_round = 1
	ps.pot_size = 8.0
	ps.current_bets = [0, 0]
	ps.last_raiser = None
	ps.current_player = (ps.dealer + 1) % 2
	return ps


@pytest.fixture
def turn_state():
	"""
	Construct a turn PublicState to verify that turn leaves are solved to terminal without invoking value nets.
	"""
	deck = list(DECK)
	board = deck[:4]
	ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
	ps.current_round = 2
	ps.pot_size = 20.0
	ps.current_bets = [0, 0]
	ps.last_raiser = None
	ps.current_player = (ps.dealer + 1) % 2
	return ps


def test_bet_fracs_modes():
	"""
	Validate that _bet_fracs_from_mode returns the expected bet-fraction presets across known modes 
	and sensible defaults for unknown modes per stage.
	"""
	assert _bet_fraction_schedule_for_mode("sparse_2", "flop") == [0.5, 1.0]
	assert _bet_fraction_schedule_for_mode("sparse_3", "flop") == [0.5, 1.0, 2.0]
	assert _bet_fraction_schedule_for_mode("full", "flop") == [0.5, 1.0, 2.0]
	assert _bet_fraction_schedule_for_mode("unknown_mode", "flop") == [0.5, 1.0]
	assert _bet_fraction_schedule_for_mode("unknown_mode", "turn") == [0.5, 1.0, 2.0]


def test_update_opp_upper_monotone():
	"""
	Check that proposed opponent CFV upper bounds are merged via coordinatewise minimum, ensuring 
	monotone tightening of constraints.
	"""
	prev = {0: 1.0, 1: 2.0}
	prop = {0: 1.5, 1: 1.5, 2: 3.0}
	out = _tighten_cfv_upper_bounds(prev, prop)
	assert out[0] == 1.0 and out[1] == 1.5 and out[2] == 3.0


def test_resolve_at_with_diag_flop_uses_flop_net_and_acceptance(flop_state):
	K = 6
	net_flop = _DummyCFVNet(K)
	net_turn = _DummyCFVNet(K)
	vs = ValueServer(models={"flop": net_flop, "turn": net_turn}, max_wait_ms=1)
	r_us = _uniform_range(K)
	w_opp = _uniform_range(K)
	cfg = {
	 "iterations": 2,
	 "depth_limit": 1,
	 "bet_size_mode": "sparse_2",
	 "constraint_mode": "sp",
	 "value_server": vs,
	}

	pol, w_next, our_cfv, diag = resolve_at_with_diag(
	 flop_state, r_us, w_opp, config=cfg, value_server=None
	)

	assert diag["stage"] == "flop"
	assert diag["range_mass_ok"] is True
	assert diag["policy_actions_ok"] is True
	assert diag["flop_net_queries"] >= 1
	assert diag["turn_net_queries"] == 0
	assert set(pol.keys()) <= {
	 ActionType.FOLD,
	 ActionType.CALL,
	 ActionType.HALF_POT_BET,
	 ActionType.POT_SIZED_BET,
	 ActionType.ALL_IN,
	}

	vs.stop(join=True)

