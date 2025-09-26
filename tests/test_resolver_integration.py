import numpy as np
import torch
import pytest

from resolver_integration import resolve_at_with_diag, resolve_at, _bet_fracs_from_mode, _update_opp_upper_monotone
from value_server import ValueServer
from public_state import PublicState
from poker_utils import DECK
from action_type import ActionType


class _DummyCFVNet(torch.nn.Module):
    def __init__(self, K, board_dim=52):
        super().__init__()
        self.num_clusters = int(K)
        self.input_size = 1 + board_dim + 2 * K

    def forward(self, x: torch.Tensor):
        N = x.shape[0]; K = self.num_clusters
        p1 = torch.full((N, K), 0.1, dtype=torch.float32, device=x.device)
        p2 = torch.full((N, K), -0.05, dtype=torch.float32, device=x.device)
        return p1, p2

    @torch.no_grad()
    def enforce_zero_sum(self, r1, r2, p1, p2):
        sum_r1 = torch.clamp(torch.sum(r1, dim=1, keepdim=True), min=1e-9)
        sum_r2 = torch.clamp(torch.sum(r2, dim=1, keepdim=True), min=1e-9)
        s = torch.sum(r1 * p1, dim=1, keepdim=True) + torch.sum(r2 * p2, dim=1, keepdim=True)
        a = -0.5 * s / sum_r1
        b = -0.5 * s / sum_r2
        return p1 + a, p2 + b


def _uniform_range(K):
    u = 1.0 / float(K) if K > 0 else 0.0
    return {i: u for i in range(K)}


@pytest.fixture
def flop_state():
    deck = list(DECK)
    board = deck[:3]
    ps = PublicState(initial_stacks=[200, 200], board_cards=list(board))
    ps.current_round = 1
    ps.pot_size = 8.0
    ps.current_bets = [0, 0]
    ps.last_raiser = None
    ps.current_player = (ps.dealer + 1) % 2  # non-dealer acts first postflop
    return ps


@pytest.fixture
def turn_state():
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
        assert _bet_fracs_from_mode("sparse_2", "flop") == [0.5, 1.0]
        assert _bet_fracs_from_mode("sparse_3", "flop") == [0.5, 1.0, 2.0]
        assert _bet_fracs_from_mode("full", "flop") == [0.5, 1.0, 2.0]
        assert _bet_fracs_from_mode("unknown_mode", "flop") == [0.5, 1.0]
        assert _bet_fracs_from_mode("unknown_mode", "turn") == [0.5, 1.0, 2.0]

def test_update_opp_upper_monotone():
    prev = {0: 1.0, 1: 2.0}
    prop = {0: 1.5, 1: 1.5, 2: 3.0}
    out = _update_opp_upper_monotone(prev, prop)
    # elementwise minimum (monotone non-increasing upper bounds)
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
        "bet_size_mode": "sparse_2",  # {0.5P, 1P}
        "constraint_mode": "sp",
        "value_server": vs,
    }
    try:
        pol, w_next, our_cfv, diag = resolve_at_with_diag(flop_state, r_us, w_opp, config=cfg, value_server=None)
        # Acceptance checks
        assert diag["range_mass_ok"] is True
        assert diag["policy_actions_ok"] is True
        # Correct bet fractions & no illegal actions in policy
        assert set(diag["bet_fractions"]) == {0.5, 1.0}
        allowed = {ActionType.FOLD, ActionType.CALL}
        allowed.update({ActionType.HALF_POT_BET, ActionType.POT_SIZED_BET})
        if diag["include_all_in"]:
            allowed.add(ActionType.ALL_IN)
        assert all(a in allowed for a in pol.keys())
        # Flop queries must be positive; turn queries may be zero on flop stage
        assert diag["flop_net_queries"] >= 0
        assert diag["turn_net_queries"] >= 0
    finally:
        vs.stop()


def test_resolve_at_with_diag_turn_does_not_query_turn_net(turn_state):
    K = 4
    net_flop = _DummyCFVNet(K)
    net_turn = _DummyCFVNet(K)
    vs = ValueServer(models={"flop": net_flop, "turn": net_turn}, max_wait_ms=1)

    r_us = _uniform_range(K)
    w_opp = _uniform_range(K)
    cfg = {
        "iterations": 2,
        "depth_limit": 1,            # user depth is ignored on turn (should solve to terminal)
        "bet_size_mode": "sparse_3", # includes 2P on turn
        "constraint_mode": "sp",
        "value_server": vs,
    }
    try:
        pol, w_next, our_cfv, diag = resolve_at_with_diag(turn_state, r_us, w_opp, config=cfg, value_server=None)
        # Per spec: on turn, leaf net must not be invoked
        assert diag["stage"] == "turn"
        assert diag["turn_net_queries"] == 0, "Turn leaves must solve to terminal without net calls"
        assert diag["turn_leaf_net_ok"] is True
        # Bet fractions reflect mode
        assert set(diag["bet_fractions"]) == {0.5, 1.0, 2.0}
    finally:
        vs.stop()


def test_resolve_at_constraint_modes_sp_vs_br(flop_state):
    K = 5
    net_flop = _DummyCFVNet(K)
    net_turn = _DummyCFVNet(K)
    vs = ValueServer(models={"flop": net_flop, "turn": net_turn}, max_wait_ms=1)

    r_us = _uniform_range(K)
    w_opp = {i: float(i)/10.0 for i in range(K)}  # not uniform constraints, to make equality checks meaningful

    try:
        # Stackelberg-primal ("sp"): must return proposed next constraints from solver path
        cfg_sp = {"iterations": 1, "depth_limit": 1, "constraint_mode": "sp", "value_server": vs}
        pol_sp, w_next_sp, our_sp = resolve_at(flop_state, r_us, w_opp, config=cfg_sp, value_server=None)
        assert isinstance(w_next_sp, dict) and len(w_next_sp) == K

        # Best-response ("br"): must carry forward original w_opp unchanged
        cfg_br = {"iterations": 1, "depth_limit": 1, "constraint_mode": "br", "value_server": vs}
        pol_br, w_next_br, our_br = resolve_at(flop_state, r_us, w_opp, config=cfg_br, value_server=None)
        assert w_next_br == {int(k): float(v) for k, v in w_opp.items()}
    finally:
        vs.stop()
