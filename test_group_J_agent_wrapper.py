# test_group_J_agent_wrapper.py
# ===========================================================
# GROUP J — Agent wrapper
#
# Exercises all named behavior in agent.Agent:
#   - __init__ (config wiring, device selection, model placement, clusterer choice)
#   - set_device
#   - set_latency_profile (and its effect on subsequent act())
#   - _bucketize_own_hand
#   - _uniform_range
#   - _range_on_bucket
#   - _public_key
#   - act (range setting, iteration budget selection, last_public_key update)
#   - observe_opponent_action (transition update + last_public_key)
#   - observe_chance (chance update + last_public_key)
#   - load_bundle (model I/O pass-through)
#
# Tests use light stubs for CFRSolver, GameNode, HandClusterer, and CFV-bundle I/O.
# The expectations are limited to the wrapper’s responsibilities while remaining
# consistent with continual re-solving and CFV-net use at depth limits as per the papers.
# ===========================================================

import types
import math
import random

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

import agent as agent_mod
from resolve_config import ResolveConfig


# -----------------------------
# Fakes used to isolate wrapper
# -----------------------------

class _FakeModule:
	def __init__(self, name=""):
		self.name = name
		self.moved_to = None
	def to(self, device):
		self.moved_to = str(device)
		return self

class _FakeGameNode:
	def __init__(self, public_state):
		self.public_state = public_state
		self.player_ranges = [{}, {}]

class _FakeClusterer:
	def __init__(self, solver, num_clusters, profile):
		self.solver = solver
		self.num_clusters = int(num_clusters)
		self.profile = str(profile)
		self.calls = []
	def hand_to_bucket(self, hand_str):
		self.calls.append(hand_str)
		if self.num_clusters <= 0:
			return 0
		return abs(hash(hand_str)) % self.num_clusters

class _FakeSolver:
	"""Captures all interactions performed by the Agent wrapper."""
	def __init__(self, config=None):
		self._config = config
		self.num_clusters = int(getattr(config, "num_clusters", 0)) if config else 0
		self.models = {"flop": _FakeModule("flop"), "turn": _FakeModule("turn")}
		self._round_iters = {}
		self.total_iterations = int(getattr(config, "total_iterations", 0)) if config else 0
		self.last_run_node = None
		self.last_update = None
		self.last_lift = None
		self.last_bundle = None
		self.last_bundle_device = None

	def run_cfr(self, node):
		self.last_run_node = node
		return "ACT"

	def apply_opponent_action_update(self, prev_node, next_node, observed_action_type):
		self.last_update = (prev_node, next_node, observed_action_type)

	def lift_ranges_after_chance(self, node):
		self.last_lift = node

	def apply_cfv_bundle(self, bundle, device=None):
		self.last_bundle = bundle
		self.last_bundle_device = str(device) if device is not None else None
		return {"ok": True}

# Public-state stub for constructing GameNodes
class _PS:
	def __init__(
		self,
		board_cards=None,
		current_round=0,
		current_bets=(0.0, 0.0),
		pot_size=0.0,
		current_player=0,
		dealer=0,
		is_terminal=False,
		is_showdown=False,
		players_in_hand=(True, True),
	):
		self.board_cards = list(board_cards or [])
		self.current_round = int(current_round)
		self.current_bets = [float(current_bets[0]), float(current_bets[1])]
		self.pot_size = float(pot_size)
		self.current_player = int(current_player)
		self.dealer = int(dealer)
		self.is_terminal = bool(is_terminal)
		self.is_showdown = bool(is_showdown)
		self.players_in_hand = [bool(players_in_hand[0]), bool(players_in_hand[1])]


# -----------------------------
# Fixtures: patch Agent deps
# -----------------------------

@pytest.fixture
def patch_agent_deps(monkeypatch):
	"""Patch heavy deps with fakes; return a fresh FakeSolver factory per test."""
	def _install(fake_solver=None, provide_hand_clusterer=False):
		# CFRSolver -> Fake
		fs = fake_solver or _FakeSolver
		def _mk_solver(config=None):
			s = fs(config=config)
			if provide_hand_clusterer and not hasattr(s, "hand_clusterer"):
				s.hand_clusterer = _FakeClusterer(s, s.num_clusters, getattr(config, "profile", "bot"))
			return s
		monkeypatch.setattr(agent_mod, "CFRSolver", lambda config=None: _mk_solver(config=config), raising=True)
		# GameNode -> Fake
		monkeypatch.setattr(agent_mod, "GameNode", _FakeGameNode, raising=True)
		# HandClusterer -> Fake
		monkeypatch.setattr(agent_mod, "HandClusterer", _FakeClusterer, raising=True)
		# CFV bundle loader -> simple dict passthrough
		def _load_bundle(path, device=None):
			return {"models": {"flop": object(), "turn": object()}, "meta": {"path": str(path)}}
		monkeypatch.setattr(agent_mod, "load_cfv_bundle", _load_bundle, raising=True)
		return _mk_solver
	return _install


# -----------------------------
# __init__: config & device
# -----------------------------

def test_init_uses_given_config_and_moves_models_to_device_cpu(patch_agent_deps):
	_ = patch_agent_deps()
	cfg = ResolveConfig.from_env({"num_clusters": 7, "depth_limit": 1, "total_iterations": 9, "prefer_gpu": False})
	cg = cfg
	ag = agent_mod.Agent(config=cg, device="cpu")
	assert ag.solver._config is cg
	assert ag.solver.num_clusters == cfg.num_clusters == ag.num_clusters == 7
	assert all(m.moved_to == "cpu" for m in ag.solver.models.values())
	assert isinstance(ag.clusterer, _FakeClusterer)
	assert ag.clusterer.num_clusters == 7

def test_init_prefers_existing_solver_clusterer_over_ctor(patch_agent_deps):
	_ = patch_agent_deps(provide_hand_clusterer=True)
	cfg = ResolveConfig.from_env({"num_clusters": 5, "depth_limit": 1, "total_iterations": 3})
	ag = agent_mod.Agent(config=cfg, device="cpu")
	assert ag.clusterer is ag.solver.hand_clusterer
	assert isinstance(ag.clusterer, _FakeClusterer)

def test_init_device_selection_respects_cuda_availability_and_prefer_gpu(monkeypatch, patch_agent_deps):
	_ = patch_agent_deps()
	monkeypatch.setattr(agent_mod.torch.cuda, "is_available", lambda: True, raising=False)
	cfgA = ResolveConfig.from_env({"num_clusters": 4, "prefer_gpu": True})
	agA = agent_mod.Agent(config=cfgA)
	assert str(agA.device) == "cuda"
	assert all(m.moved_to == "cuda" for m in agA.solver.models.values())
	monkeypatch.setattr(agent_mod.torch.cuda, "is_available", lambda: False, raising=False)
	cfgB = ResolveConfig.from_env({"num_clusters": 4, "prefer_gpu": True})
	agB = agent_mod.Agent(config=cfgB)
	assert str(agB.device) == "cpu"
	assert all(m.moved_to == "cpu" for m in agB.solver.models.values())


# -----------------------------
# set_device
# -----------------------------

def test_set_device_moves_all_models_and_returns_device_string(patch_agent_deps):
	_ = patch_agent_deps()
	cfg = ResolveConfig.from_env({"num_clusters": 3})
	ag = agent_mod.Agent(config=cfg, device="cpu")
	ag.solver.models["extra"] = _FakeModule("extra")
	out = ag.set_device("cpu")
	assert out == "cpu"
	assert all(m.moved_to == "cpu" for m in ag.solver.models.values())


# -----------------------------
# set_latency_profile
# -----------------------------

def test_set_latency_profile_records_int_keys_and_values(patch_agent_deps):
	_ = patch_agent_deps()
	cfg = ResolveConfig.from_env({"num_clusters": 3})
	ag = agent_mod.Agent(config=cfg, device="cpu")
	out = ag.set_latency_profile({0.0: 3.9, 1.2: 2})
	assert out == {0: 3, 1: 2}
	assert getattr(ag.solver, "_round_iters", {}) == {0: 3, 1: 2}

def test_set_latency_profile_no_dict_returns_current(monkeypatch, patch_agent_deps):
	_ = patch_agent_deps()
	cfg = ResolveConfig.from_env({"num_clusters": 3})
	ag = agent_mod.Agent(config=cfg, device="cpu")
	out = ag.set_latency_profile(round_iters=None)
	assert out == {}


# -----------------------------
# Private helpers: ranges & key
# -----------------------------

def test_bucketize_uniform_and_onehot_range_logic(patch_agent_deps):
	_ = patch_agent_deps()
	cfg = ResolveConfig.from_env({"num_clusters": 8})
	ag = agent_mod.Agent(config=cfg, device="cpu")
	cid = ag._bucketize_own_hand(["AS", "KD"], board=["2C","3D","4H"])
	assert 0 <= cid < ag.num_clusters
	uni = ag._uniform_range()
	assert len(uni) == ag.num_clusters
	assert math.isclose(sum(uni.values()), 1.0, abs_tol=1e-12)
	oh = ag._range_on_bucket(cid)
	assert len(oh) == ag.num_clusters
	assert math.isclose(sum(oh.values()), 1.0, abs_tol=1e-12)
	assert sum(1 for v in oh.values() if v > 0.0) == 1

@given(
	K=st.integers(min_value=0, max_value=12),
	cid=st.integers(min_value=-3, max_value=20),
)
@settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_range_on_bucket_and_uniform_mass_properties(K, cid, patch_agent_deps):
	_ = patch_agent_deps()
	cfg = ResolveConfig.from_env({"num_clusters": K})
	ag = agent_mod.Agent(config=cfg, device="cpu")
	uni = ag._uniform_range()
	if K == 0:
		assert uni == {}
	else:
		assert len(uni) == K and math.isclose(sum(uni.values()), 1.0, abs_tol=1e-12)
	oh = ag._range_on_bucket(cid)
	assert len(oh) == K
	s = sum(oh.values())
	if 0 <= int(cid) < K:
		assert math.isclose(s, 1.0, abs_tol=1e-12)
		assert sum(1 for v in oh.values() if v > 0.0) == 1
	else:
		assert math.isclose(s, 0.0, abs_tol=1e-12)

def test_public_key_encodes_fields_and_coercions(patch_agent_deps):
	_ = patch_agent_deps()
	cfg = ResolveConfig.from_env({"num_clusters": 2})
	ag = agent_mod.Agent(config=cfg, device="cpu")
	ps = _PS(
		board_cards=["AH","KD","2C"],
		current_round=2,
		current_bets=(10.0, 15.0),
		pot_size=42.9,
		current_player=1,
		dealer=0,
		is_terminal=True,
		is_showdown=False,
		players_in_hand=(True, False),
	)
	key = ag._public_key(ps)
	assert key[0] == tuple(ps.board_cards)
	assert key[1] == ps.current_round
	assert key[2] == (int(ps.current_bets[0]), int(ps.current_bets[1]))
	assert key[3] == int(ps.pot_size)
	assert key[4] == ps.current_player
	assert key[5] == ps.dealer
	assert key[6] is True and key[7] is False
	assert key[8] == (True, False)


# -----------------------------
# act: range setting & iters
# -----------------------------

def test_act_sets_ranges_uses_round_iters_and_updates_last_key(patch_agent_deps):
	_ = patch_agent_deps()
	cfg = ResolveConfig.from_env({"num_clusters": 6, "total_iterations": 11})
	ag = agent_mod.Agent(config=cfg, device="cpu")
	ag.set_latency_profile({2: 7})
	ps = _PS(board_cards=["AH","KD","2C"], current_round=2, current_bets=(5.0, 5.0),
			 pot_size=20.0, current_player=1, dealer=0, is_terminal=False,
			 players_in_hand=(True, True))
	ret = ag.act(ps, our_private_cards=["QS","JC"])
	assert ret == "ACT"
	node = ag.solver.last_run_node
	assert isinstance(node, _FakeGameNode)
	r_us = node.player_ranges[1]
	r_opp = node.player_ranges[0]
	assert math.isclose(sum(r_us.values()), 1.0, abs_tol=1e-12)
	assert sum(1 for v in r_us.values() if v > 0.0) == 1
	assert math.isclose(sum(r_opp.values()), 1.0, abs_tol=1e-12)
	assert ag.solver.total_iterations == 7
	assert ag.last_public_key == ag._public_key(ps)
	ps2 = _PS(board_cards=["AH","KD","2C"], current_round=3, current_bets=(5.0, 10.0),
			  pot_size=25.0, current_player=0, dealer=0)
	ag.act(ps2, our_private_cards=["2H","2D"])
	assert ag.solver.total_iterations == cfg.total_iterations

def test_act_accepts_string_cards_and_ignores_board_in_bucketize(patch_agent_deps):
	_ = patch_agent_deps()
	cfg = ResolveConfig.from_env({"num_clusters": 4})
	ag = agent_mod.Agent(config=cfg, device="cpu")
	ps = _PS(board_cards=["2C","3D","4H"], current_round=1, current_player=0)
	ag.act(ps, our_private_cards="AS KD")
	node = ag.solver.last_run_node
	assert math.isclose(sum(node.player_ranges[0].values()), 1.0, abs_tol=1e-12)


# -----------------------------
# observe_* transitions
# -----------------------------

def test_observe_opponent_action_updates_solver_and_last_key(patch_agent_deps):
	_ = patch_agent_deps()
	cfg = ResolveConfig.from_env({"num_clusters": 3})
	ag = agent_mod.Agent(config=cfg, device="cpu")
	prev = _PS(board_cards=["AH","KD","2C"], current_round=1, current_bets=(5.0, 5.0),
			   pot_size=10.0, current_player=0)
	nxt = _PS(board_cards=["AH","KD","2C","9H"], current_round=2, current_bets=(5.0, 10.0),
			  pot_size=20.0, current_player=1)
	ok = ag.observe_opponent_action(prev, nxt, observed_action_type="CALL")
	assert ok is True
	assert ag.solver.last_update is not None
	prev_node, next_node, a = ag.solver.last_update
	assert isinstance(prev_node, _FakeGameNode) and isinstance(next_node, _FakeGameNode)
	assert a == "CALL"
	assert ag.last_public_key == ag._public_key(nxt)

def test_observe_chance_lifts_ranges_and_updates_last_key(patch_agent_deps):
	_ = patch_agent_deps()
	cfg = ResolveConfig.from_env({"num_clusters": 3})
	ag = agent_mod.Agent(config=cfg, device="cpu")
	nxt = _PS(board_cards=["AH","KD","2C","9H"], current_round=2, current_bets=(5.0, 10.0),
			  pot_size=20.0, current_player=1)
	ok = ag.observe_chance(nxt)
	assert ok is True
	assert isinstance(ag.solver.last_lift, _FakeGameNode)
	assert ag.last_public_key == ag._public_key(nxt)


# -----------------------------
# CFV bundle I/O
# -----------------------------

def test_load_bundle_passes_through_and_returns_metadata(patch_agent_deps):
	_ = patch_agent_deps()
	cfg = ResolveConfig.from_env({"num_clusters": 3})
	ag = agent_mod.Agent(config=cfg, device="cpu")
	out = ag.load_bundle(path="/tmp/fake_bundle.pt")
	assert sorted(out["loaded_models"]) == ["flop", "turn"]
	assert out["applied"] == {"ok": True}
	assert isinstance(ag.solver.last_bundle, dict)
	assert ag.solver.last_bundle_device == "cpu"

