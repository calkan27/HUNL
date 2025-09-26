# pytest + hypothesis test suite for:
# GROUP F — Re-solver wrapper (public chance CFR) & value server
#
# Covers:
# - value_server.ValueServer batching, stage routing, zero-sum wrapping, counters, fallbacks
# - result_handle.ResultHandle blocking and return type behavior
# - lookahead_tree.LookaheadTreeBuilder action menus, deal-next-card, propagate + leaf callback
# - cfr_core.PublicChanceCFR core traversal, root gadget (follow/terminate), warm-start
# - resolver_integration helpers and public API resolve_at_with_diag / resolve_at
#   (acceptance checks, action-menu restriction, turn leaves do not invoke nets)
# - model_io.save_cfv_bundle / load_cfv_bundle with a real CounterfactualValueNetwork

import math
import threading
import time
import os
import tempfile
from typing import List, Dict, Any

import numpy as np
import pytest
import torch
from torch import nn
from hypothesis import given, settings, strategies as st

from value_server import ValueServer
from result_handle import ResultHandle
from lookahead_tree import LookaheadTreeBuilder
from cfr_core import PublicChanceCFR
from resolver_integration import (
    resolve_at_with_diag,
    resolve_at,
    _stage_from_round,
    _to_vec,
    _ensure_value_server,
    _depth_and_bets,
    _bet_fracs_from_mode,
    _update_opp_upper_monotone,
)
from model_io import save_cfv_bundle, load_cfv_bundle
from cfv_network import CounterfactualValueNetwork  # real net to test model IO path
from action_type import ActionType
from poker_utils import DECK, board_one_hot


# ------------------------------
# Test helpers
# ------------------------------

def _norm(v: List[float]) -> List[float]:
    s = sum(v)
    if s > 0:
        return [x / s for x in v]
    return v


def make_input_batch(K: int, B: int, stage: str = "flop") -> torch.Tensor:
    """
    Build a batch of inputs with layout [pot_norm(1), board_one_hot(52), r1(K), r2(K)].
    For 'flop': 3 ones in board_one_hot; for 'turn': 4; preflop/river cases not used here.
    """
    num_board = 3 if stage == "flop" else 4
    X = []
    for _ in range(B):
        pot_norm = np.clip(np.random.uniform(1e-3, 1.0), 1e-3, 1.0)
        bvec = [0] * 52
        idxs = np.random.choice(52, size=num_board, replace=False)
        for j in idxs:
            bvec[int(j)] = 1
        r1 = np.random.rand(K).tolist()
        r2 = np.random.rand(K).tolist()
        r1 = _norm(r1) if sum(r1) > 0 else [1.0 / K] * K
        r2 = _norm(r2) if sum(r2) > 0 else [1.0 / K] * K
        X.append([pot_norm] + bvec + r1 + r2)
    return torch.tensor(X, dtype=torch.float32)


class DummyZeroSumCFVNet(nn.Module):
    """
    Minimal CFV-like model exposing:
      - attributes: num_clusters, input_size
      - forward(x) -> (p1, p2) with shape (B, K)
      - enforce_zero_sum(r1, r2, p1, p2) -> (f1, f2) s.t. range-weighted sums sum to ~0
    Used to probe ValueServer slicing, stage routing, zero-sum call, and counters.
    """
    def __init__(self, input_size: int, num_clusters: int):
        super().__init__()
        self.input_size = int(input_size)
        self.num_clusters = int(num_clusters)
        # simple linear projections to K values (shared for both players, just for shape)
        self.lin = nn.Linear(self.input_size, 2 * self.num_clusters, bias=False)
        # initialize deterministically
        torch.manual_seed(123)
        nn.init.uniform_(self.lin.weight, -0.01, 0.01)

    def forward(self, x: torch.Tensor):
        out = self.lin(x)
        B, _ = out.shape
        K = self.num_clusters
        p1 = out[:, :K]
        p2 = out[:, K:]
        return p1, p2

    @torch.no_grad()
    def enforce_zero_sum(self, r1: torch.Tensor, r2: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor):
        # subtract same delta so that <r1, f1> + <r2, f2> = 0
        # Let sum_r1 = sum(r1)=1, sum_r2=1 (inputs are ranges). Then:
        # s1 = <r1,p1>, s2 = <r2,p2>; set f1 = p1 - d, f2 = p2 - d with d = (s1+s2)/2.
        s1 = (r1 * p1).sum(dim=1, keepdim=True)
        s2 = (r2 * p2).sum(dim=1, keepdim=True)
        d = (s1 + s2) / 2.0
        return p1 - d, p2 - d


# Minimal fake node used by LookaheadTreeBuilder.propagate leaf-callback test
class _NodeWrap:
    def __init__(self, ps):
        self.public_state = ps


# Fake public state for menu + dealing tests (only what we need)
class _FakePS:
    def __init__(self, legal, board, hole0, hole1, cr=1, dealer=0, cur=0, terminal=False):
        self._legal = list(legal)
        self.board_cards = list(board)
        self.hole_cards = [list(hole0), list(hole1)]
        self.current_round = int(cr)           # 1=flop, 2=turn
        self.dealer = int(dealer)
        self.current_player = int(cur)
        self.is_terminal = bool(terminal)
        self.current_bets = [0, 0]
        self.pot_size = 100.0  # arbitrary

    def legal_actions(self):
        return list(self._legal)


# ------------------------------
# ValueServer + ResultHandle
# ------------------------------

@settings(deadline=None, max_examples=60)
@given(
    K=st.integers(min_value=1, max_value=8),
    B=st.integers(min_value=1, max_value=16),
)
def test_value_server_enforces_zero_sum_and_slices_ranges_correctly_hypothesis(K, B):
    insz = 1 + 52 + 2 * K
    model = DummyZeroSumCFVNet(insz, K)
    vs = ValueServer(models={"flop": model}, device=torch.device("cpu"), max_batch_size=1024, max_wait_ms=2)

    xb = make_input_batch(K, B, stage="flop")
    v1, v2 = vs.query("flop", xb, as_numpy=False)

    assert v1.shape == (B, K)
    assert v2.shape == (B, K)

    # Check outer zero-sum residual near zero (range-weighted)
    # Extract ranges as in the server slicing logic
    start_r1 = 1 + 52
    end_r1 = start_r1 + K
    start_r2 = end_r1
    end_r2 = start_r2 + K
    r1 = xb[:, start_r1:end_r1]
    r2 = xb[:, start_r2:end_r2]

    s1 = (r1 * v1).sum(dim=1)
    s2 = (r2 * v2).sum(dim=1)
    residual = torch.abs(s1 + s2).max().item()
    assert residual < 1e-5

    # Counters reflect the queries
    ctr = vs.get_counters()
    assert int(ctr.get("flop", 0)) >= B

    vs.stop(join=True)


def test_value_server_missing_model_returns_zeros_and_no_counter_increment():
    K = 4
    insz = 1 + 52 + 2 * K
    model = DummyZeroSumCFVNet(insz, K)
    # Only register a 'flop' model; query both flop and turn
    vs = ValueServer(models={"flop": model}, device=torch.device("cpu"))
    xb = make_input_batch(K, 3, stage="flop")

    # Existing stage
    v1, v2 = vs.query("flop", xb, as_numpy=False)
    assert v1.shape == (3, K) and v2.shape == (3, K)

    # Missing stage -> server replies with zeros (1, 0) per request item in its internal batching
    # However, query() passes entire xb in one handle; server groups per stage.
    # To assert fallback path, call stage "turn" which isn't registered.
    v1t, v2t = vs.query("turn", xb, as_numpy=False)
    # Fallback path uses handles[i].set((torch.zeros(1, 0), torch.zeros(1, 0))) per element;
    # But since we pass a single handle with full batch, we expect shape (1, 0).
    assert v1t.shape[1] == 0 and v2t.shape[1] == 0

    ctr = vs.get_counters()
    # Only flop counter should have increased
    assert ctr.get("turn", 0) == 0
    assert ctr.get("flop", 0) >= 3
    vs.stop(join=True)


def test_result_handle_blocking_and_types():
    h = ResultHandle()

    def _setter():
        time.sleep(0.05)
        v = (torch.ones(1, 2), torch.zeros(1, 2))
        h.set(v)

    t = threading.Thread(target=_setter)
    t.start()

    v1_np, v2_np = h.result(as_numpy=True)
    assert isinstance(v1_np, np.ndarray) and isinstance(v2_np, np.ndarray)
    assert v1_np.shape == (1, 2) and v2_np.shape == (1, 2)

    # call again as torch
    h2 = ResultHandle()
    h2.set((torch.full((1, 3), 7.0), torch.full((1, 3), -7.0)))
    v1_t, v2_t = h2.result(as_numpy=False)
    assert torch.allclose(v1_t, torch.full((1, 3), 7.0))
    assert torch.allclose(v2_t, torch.full((1, 3), -7.0))

    t.join(timeout=1.0)


# ------------------------------
# LookaheadTreeBuilder
# ------------------------------

def test_action_menu_respects_sparse_sizes_and_allin_flag():
    # Build a fake PS where all sparse actions are "legal"
    legal = [
        ActionType.FOLD,
        ActionType.CALL,
        ActionType.HALF_POT_BET,
        ActionType.POT_SIZED_BET,
        ActionType.TWO_POT_BET,
        ActionType.ALL_IN,
    ]
    ps = _FakePS(legal=legal, board=[], hole0=[], hole1=[], cr=1, dealer=0, cur=0)
    builder = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=True)

    # root
    menu = builder._action_menu(ps, for_player=True, pot_fracs=(0.5, 1.0), is_root=True)
    # Expect fold + call + half + pot + all-in (no two-pot because not configured on flop mode here)
    assert ActionType.FOLD in menu
    assert ActionType.CALL in menu
    assert ActionType.HALF_POT_BET in menu
    assert ActionType.POT_SIZED_BET in menu
    assert ActionType.ALL_IN in menu
    assert ActionType.TWO_POT_BET not in menu

    # If bet_fractions include 2.0, TWO_POT_BET appears when legal
    menu2 = builder._action_menu(ps, True, (0.5, 1.0, 2.0), True)
    assert ActionType.TWO_POT_BET in menu2

    # Cap by max_actions_per_branch
    builder2 = LookaheadTreeBuilder(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=True, max_actions_per_branch=3)
    menu3 = builder2._action_menu(ps, True, (0.5, 1.0), True)
    assert len(menu3) == 3


def test_deal_next_card_counts():
    # Use fake PS with hole cards so that used are known.
    # Flop: 3 board + 2 + 2 hole => 7 used -> 52-7 = 45 available turn cards.
    ps_flop = _FakePS(legal=[], board=["AS", "KD", "3c".upper()], hole0=["7H", "2D"], hole1=["9C", "9D"], cr=1)
    builder = LookaheadTreeBuilder(depth_limit=0)
    nxt = builder._deal_next_card(ps_flop)
    assert len(nxt) == 45
    # Turn: 4 board + 2 + 2 hole => 8 used -> 52-8 = 44 available river cards.
    ps_turn = _FakePS(legal=[], board=["AS", "KD", "3C", "5S"], hole0=["7H", "2D"], hole1=["9C", "9D"], cr=2)
    nxt2 = builder._deal_next_card(ps_turn)
    assert len(nxt2) == 44


def test_propagate_calls_leaf_callback_and_propagates_reach():
    K = 4
    builder = LookaheadTreeBuilder(depth_limit=1)

    # Construct a minimal tree with: root(kind our) -> two children(kind leaf)
    # Reach should split uniformly over actions (2 children).
    class _LeafPS:
        def __init__(self):
            self.initial_stacks = [200, 200]
            self.pot_size = 100.0
            self.board_cards = []
            self.current_round = 1
            self.is_terminal = False

    root_ps = _LeafPS()
    tree = {
        "nodes": [_NodeWrap(root_ps), _NodeWrap(_LeafPS()), _NodeWrap(_LeafPS())],
        "parents": [-1, 0, 0],
        "edges": [None, ActionType.CALL, ActionType.POT_SIZED_BET],
        "kinds": ["our", "leaf", "leaf"],
        "depth_actions": [0, 1, 1],
        "menus": [[ActionType.CALL, ActionType.POT_SIZED_BET], [], []],
        "stage_start": 1,
    }

    seen = []

    def leaf_cb(ps, pov, r1, r2):
        # record reach vectors at leaves; return per-bucket vector of zeros for shape
        seen.append((r1, r2))
        return np.zeros((K,), dtype=float)

    builder.set_leaf_callback(leaf_cb)

    r_us = [1.0 / K] * K
    r_opp = [1.0 / K] * K
    out = builder.propagate(tree, r_us, r_opp, pov_player=0)
    # Leaf callback called twice, and reach at leaves equals parent's reach divided by #actions (2)
    assert len(seen) == 2
    for (ru, ro) in seen:
        assert np.allclose(ru, np.array(r_us) * 0.5)
        assert np.allclose(ro, np.array(r_opp) * 0.5)
    # Values are np.ndarray
    assert isinstance(out["values"][1], np.ndarray) and isinstance(out["values"][2], np.ndarray)


# ------------------------------
# cfr_core.PublicChanceCFR
# ------------------------------

def test_public_chance_cfr_root_gadget_and_warm_start_setter(monkeypatch):
    # Build a small "tree" with root an opponent node (so root gadget applies),
    # two children leaves with values V1, V2 for hero (pov 0). The terminate
    # value from opponent CFV upper bounds should be min-ified at root.
    K = 3

    class _LeafPS:
        def __init__(self):
            self.initial_stacks = [200, 200]
            self.pot_size = 100.0
            self.board_cards = []
            self.current_round = 1
            self.is_terminal = False

    # Provide leaf values via leaf_value_fn: return per-bucket vector such that
    # <r_us, v> = +0.6 (better than terminate which we'll set to -0.1).
    r_us = [1.0 / K] * K
    good_v = np.full((K,), 0.6)
    def leaf_fn(ps, pov, ru, ro):
        return torch.tensor(good_v, dtype=torch.float32)

    # Tree: root is opp-kind with two leaves
    tree = {
        "nodes": [_NodeWrap(_LeafPS()), _NodeWrap(_LeafPS()), _NodeWrap(_LeafPS())],
        "parents": [-1, 0, 0],
        "edges": [None, ActionType.CALL, ActionType.POT_SIZED_BET],
        "kinds": ["opp", "leaf", "leaf"],
        "depth_actions": [0, 1, 1],
        "menus": [[ActionType.CALL, ActionType.POT_SIZED_BET], [], []],
        "stage_start": 1,
    }

    solver = PublicChanceCFR(depth_limit=1, bet_fractions=[0.5, 1.0], include_all_in=True)
    # Warm start should be accepted silently
    solver.set_warm_start({"dummy": [0.2, 0.8]})

    # Terminate value = - <r_opp, opp_cfv_upper> ; choose opp bounds so terminate (-0.1) < follow (+0.6),
    # thus the root gadget should pick -0.1 (min) for hero (pov 0).
    r_opp = [1.0 / K] * K
    opp_upper = [0.1] * K  # <r_opp, opp_upper> = 0.1 => terminate = -0.1

    root_policy, node_values, opp_cfv = solver.solve_subgame(
        root_node=tree,
        r_us=r_us,
        r_opp=r_opp,
        opp_cfv_constraints=opp_upper,
        T=10,
        leaf_value_fn=leaf_fn,
    )
    # Strategy must be a distribution over the root menu
    assert set(root_policy.keys()) == set(tree["menus"][0])
    assert abs(sum(root_policy.values()) - 1.0) < 1e-8
    # Returned opp_cfv should mirror constraints by index
    for i in range(len(opp_upper)):
        assert pytest.approx(opp_cfv[i], rel=0, abs=1e-12) == opp_upper[i]


# ------------------------------
# resolver_integration helpers
# ------------------------------

@settings(deadline=None, max_examples=80)
@given(
    K=st.integers(min_value=1, max_value=16),
    items=st.lists(
        st.tuples(
            st.integers(min_value=-5, max_value=25),  # indices, some out of range
            st.floats(min_value=0.0, max_value=10.0),
        ),
        min_size=0,
        max_size=40,
    ),
)
def test__to_vec_normalizes_and_clips_indices(K, items):
    d: Dict[int, float] = {}
    for i, p in items:
        if p < 0:
            p = 0.0
        d[int(i)] = d.get(int(i), 0.0) + float(p)

    v = _to_vec(d, K)
    assert len(v) == K
    s = sum(v)
    if any((0 <= i < K and p > 0) for i, p in d.items()):
        assert abs(s - 1.0) < 1e-9
        assert all(x >= 0.0 for x in v)
    else:
        # no mass fell into [0,K); expect all zeros
        assert s == 0.0


def test__stage_from_round_mapping():
    assert _stage_from_round(0) == "flop"
    assert _stage_from_round(1) == "flop"
    assert _stage_from_round(2) == "turn"
    assert _stage_from_round(3) == "river" or _stage_from_round(3) == "flop"  # resolved later as flop/turn only
    assert _stage_from_round(99) == "flop"


def test__bet_fracs_from_mode_and__depth_and_bets_turn_and_flop():
    # Mode mapping
    assert _bet_fracs_from_mode("sparse_2", "flop") == [0.5, 1.0]
    assert _bet_fracs_from_mode("sparse_3", "flop") == [0.5, 1.0, 2.0]
    assert _bet_fracs_from_mode("full", "flop") == [0.5, 1.0, 2.0]
    # Defaults per stage when mode not provided
    dl, bf, include_all, cm = _depth_and_bets("turn", 1, {})
    assert dl >= 90 and 2.0 in bf and include_all and cm == "sp"
    dl2, bf2, include_all2, cm2 = _depth_and_bets("flop", 2, {})
    assert dl2 == 2 and 2.0 not in bf2 and include_all2 and cm2 == "sp"


@settings(deadline=None, max_examples=60)
@given(
    prev=st.dictionaries(st.integers(0, 10), st.floats(min_value=-1e2, max_value=1e2)),
    prop=st.dictionaries(st.integers(0, 10), st.floats(min_value=-1e2, max_value=1e2)),
)
def test__update_opp_upper_monotone_is_coordinatewise_min(prev, prop):
    out = _update_opp_upper_monotone(prev, prop)
    for k in set(list(prev.keys()) + list(prop.keys())):
        a = float(prev.get(k, float("inf")))
        b = float(prop.get(k, float("inf")))
        assert out[k] == (a if a < b else b)


def test__ensure_value_server_from_models_and_bundle(tmp_path):
    K = 4
    insz = 1 + 52 + 2 * K
    # real network to test bundle save/load
    net = CounterfactualValueNetwork(input_size=insz, num_clusters=K)
    models = {"flop": net}
    # From models
    vs1 = _ensure_value_server({"models": models}, None)
    assert isinstance(vs1, ValueServer)
    vs1.stop(join=True)
    # From bundle path
    bundle_path = os.path.join(tmp_path, "cfv_bundle.pt")
    save_cfv_bundle(models=models, cluster_mapping={}, input_meta={"num_clusters": K}, path=bundle_path, seed=123)
    vs2 = _ensure_value_server({"bundle_path": bundle_path}, None)
    assert isinstance(vs2, ValueServer)
    vs2.stop(join=True)


# ------------------------------
# resolver_integration end-to-end with diagnostics
# ------------------------------

def _fake_tree_for_resolve(root_round: int, menu: List[Any], leaf_round: int):
    """
    Build a tiny fake tree with root(kind 'our') -> one child leaf,
    so cfr traversal is deterministic, and leaf_value_fn is invoked.
    """
    class _PS:
        def __init__(self, cr):
            self.initial_stacks = [200, 200]
            self.pot_size = 100.0
            self.board_cards = [] if cr == 0 else (["AS", "KD", "3C"] if cr == 1 else ["AS", "KD", "3C", "2D"])
            self.current_round = cr
            self.current_player = 0  # hero acts at root
            self.dealer = 0
            self.is_terminal = False
            self.current_bets = [0, 0]

        def terminal_utility(self):
            # Not a terminal in our setup
            return [0.0, 0.0]

    root_ps = _PS(root_round)
    leaf_ps = _PS(leaf_round)
    tree = {
        "nodes": [_NodeWrap(root_ps), _NodeWrap(leaf_ps)],
        "parents": [-1, 0],
        "edges": [None, menu[0]],
        "kinds": ["our", "leaf"],
        "depth_actions": [0, 1],
        "menus": [menu, []],
        "stage_start": root_round,
    }
    return tree


def test_resolve_at_with_diag_flop_queries_net_and_acceptance(monkeypatch):
    # Flop stage -> leaf callback should query the flop model via ValueServer; acceptance checks pass.
    K = 5
    insz = 1 + 52 + 2 * K
    flop_net = DummyZeroSumCFVNet(insz, K)
    vs = ValueServer(models={"flop": flop_net}, device=torch.device("cpu"))

    # Monkeypatch LookaheadTreeBuilder.build to return a tiny tree:
    fake_menu = [ActionType.CALL]  # simple menu
    fake_tree = _fake_tree_for_resolve(root_round=1, menu=fake_menu, leaf_round=1)

    def _fake_build(self, public_state):
        return fake_tree

    monkeypatch.setattr(LookaheadTreeBuilder, "build", _fake_build)

    # Prepare ranges and constraints
    r_us = {i: 1.0 / K for i in range(K)}
    w_opp = {i: 0.0 for i in range(K)}  # no constraint pressure
    ps = fake_tree["nodes"][0].public_state

    pol, w_next, our_cfv, diag = resolve_at_with_diag(
        public_state=ps,
        r_us=r_us,
        w_opp=w_opp,
        config={"depth_limit": 1, "bet_size_mode": "sparse_2"},
        value_server=vs,
    )

    # Acceptance checks:
    assert diag["stage"] == "flop"
    assert diag["range_mass_ok"] is True
    assert diag["policy_actions_ok"] is True
    # Flop net should be queried at least once
    assert diag["flop_net_queries"] >= 1
    assert diag["turn_net_queries"] == 0
    assert diag["turn_leaf_net_ok"] is True

    # Root policy over our fake menu
    assert set(pol.keys()) == set(fake_menu)
    assert abs(sum(pol.values()) - 1.0) < 1e-9

    vs.stop(join=True)


def test_resolve_at_with_diag_turn_does_not_query_net_and_acceptance(monkeypatch):
    # Turn stage -> leaf callback must NOT query any net (solve to terminal),
    # so ValueServer counters for 'turn' remain unchanged.
    K = 6
    insz = 1 + 52 + 2 * K
    turn_net = DummyZeroSumCFVNet(insz, K)
    vs = ValueServer(models={"turn": turn_net}, device=torch.device("cpu"))  # supplied but must not be used

    fake_menu = [ActionType.CALL]
    fake_tree = _fake_tree_for_resolve(root_round=2, menu=fake_menu, leaf_round=2)

    def _fake_build(self, public_state):
        return fake_tree

    # ensure builder returns our tiny tree
    monkeypatch.setattr(LookaheadTreeBuilder, "build", _fake_build)

    r_us = {i: 1.0 / K for i in range(K)}
    w_opp = {i: 0.0 for i in range(K)}
    ps = fake_tree["nodes"][0].public_state

    pol, w_next, our_cfv, diag = resolve_at_with_diag(
        public_state=ps,
        r_us=r_us,
        w_opp=w_opp,
        config={"depth_limit": 1, "bet_size_mode": "sparse_3"},
        value_server=vs,
    )

    # No turn net queries expected
    assert diag["stage"] == "turn"
    assert diag["turn_net_queries"] == 0
    assert diag["turn_leaf_net_ok"] is True
    # Menu contains only allowed actions (fold/call/pot/all-in plus configured sizes)
    assert diag["policy_actions_ok"] is True
    vs.stop(join=True)


def test_resolve_at_constraint_modes_sp_and_br(monkeypatch):
    # In 'sp' (self-play) mode, w_next is what the subgame solver returns (the bounds).
    # In 'br' mode, w_next must equal the input w_opp (pass-through).
    K = 4
    insz = 1 + 52 + 2 * K
    flop_net = DummyZeroSumCFVNet(insz, K)
    vs = ValueServer(models={"flop": flop_net}, device=torch.device("cpu"))

    fake_menu = [ActionType.CALL]
    fake_tree = _fake_tree_for_resolve(root_round=1, menu=fake_menu, leaf_round=1)
    monkeypatch.setattr(LookaheadTreeBuilder, "build", lambda self, ps: fake_tree)

    r_us = {i: 1.0 / K for i in range(K)}
    w_opp = {i: float(i) / K for i in range(K)}
    ps = fake_tree["nodes"][0].public_state

    # sp mode (default)
    pol_sp, w_next_sp, _, diag_sp = resolve_at_with_diag(ps, r_us, w_opp, config={"depth_limit": 1}, value_server=vs)
    # resolve_at wrapper selects w_next = returned in 'sp'
    pol_sp2, w_next_sp2, _ = resolve_at(ps, r_us, w_opp, config={"depth_limit": 1}, value_server=vs)
    assert set(w_next_sp.keys()) == set(range(K))
    assert w_next_sp == w_next_sp2

    # br mode: top-level resolve_at must return pass-through w_opp
    pol_br, w_next_br, _ = resolve_at(
        ps, r_us, w_opp, config={"depth_limit": 1, "constraint_mode": "br"}, value_server=vs
    )
    assert w_next_br == w_opp
    vs.stop(join=True)


# ------------------------------
# model_io save/load (with real network)
# ------------------------------

def test_model_io_bundle_roundtrip(tmp_path):
    K = 3
    insz = 1 + 52 + 2 * K
    net = CounterfactualValueNetwork(input_size=insz, num_clusters=K)
    models = {"flop": net}
    cluster_mapping = {i: i for i in range(K)}
    input_meta = {"num_clusters": K, "board_one_hot_dim": 52, "uses_pot_norm": True}

    path = os.path.join(tmp_path, "bundle.pt")
    out_path = save_cfv_bundle(models=models, cluster_mapping=cluster_mapping, input_meta=input_meta, path=path, seed=42)
    assert os.path.isfile(out_path)

    loaded = load_cfv_bundle(out_path, device=torch.device("cpu"))
    assert "models" in loaded and "meta" in loaded
    lm = loaded["models"]
    assert "flop" in lm
    ln = lm["flop"]
    assert getattr(ln, "num_clusters", None) == K
    assert getattr(ln, "input_size", None) == insz

    meta = loaded["meta"]
    assert meta["input_meta"]["num_clusters"] == K
    assert meta["input_meta"]["board_one_hot_dim"] == 52
    # ensure loaded model runs & outputs correct shape
    xb = make_input_batch(K, 2, stage="flop")
    with torch.no_grad():
        p1, p2 = ln(xb)
    assert p1.shape == (2, K) and p2.shape == (2, K)

