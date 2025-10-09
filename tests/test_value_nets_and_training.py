import os
import io
import json
import math
import types
import importlib
import sys
from pathlib import Path
from typing import List, Dict

import pytest
import torch
from torch import nn

def make_toy_samples(K: int, n: int) -> List[Dict]:
    """make_toy_samples: helper to synthesize well-formed CFV training records (pot_norm + board(52) + r1 + r2, with zero-sum targets)."""
    samples = []
    for i in range(n):
        pot_norm = float((i % 5) + 1) / 10.0
        board = [0.0] * 52
        r1_raw = torch.rand(K).tolist()
        r2_raw = torch.rand(K).tolist()
        s1 = sum(r1_raw) or 1.0
        s2 = sum(r2_raw) or 1.0
        r1 = [x / s1 for x in r1_raw]
        r2 = [x / s2 for x in r2_raw]
        t1 = torch.linspace(-0.5, 0.5, steps=K).tolist()
        t2 = [-x for x in t1]
        inp = [pot_norm] + board + r1 + r2
        assert len(inp) == 1 + 52 + 2 * K
        samples.append({
            "input_vector": inp,
            "target_v1": t1,
            "target_v2": t2,
        })
    return samples

def test_cfv_network_forward_and_attributes():
    """test_cfv_network_forward_and_attributes: CounterfactualValueNetwork builds with expected shapes and attributes; forward emits (B,K) tensors."""
    import hunl.nets.cfv_network as cn
    K = 8
    model = cn.CounterfactualValueNetwork(input_size=1+52+2*K, num_clusters=K,
                                          input_layout={"pot": (0,1), "board": (1,53), "r1": (53,53+K), "r2": (53+K,53+2*K)})
    assert model.num_clusters == K
    assert model.input_size == 1+52+2*K
    assert isinstance(model, nn.Module)

    x = torch.randn(5, 1+52+2*K)
    p1, p2 = model(x)
    assert p1.shape == (5, K)
    assert p2.shape == (5, K)

def test_enforce_zero_sum_residual_invariance_and_numerical_stability():
    """test_enforce_zero_sum_residual_invariance_and_numerical_stability: enforce_zero_sum returns finite outputs; residual equals ~0 for any ranges; invariant to constant shifts."""
    import torch
    import hunl.nets.cfv_network as cn

    K = 7
    model = cn.CounterfactualValueNetwork(input_size=1 + 52 + 2 * K, num_clusters=K)

    B = 4
    x = torch.randn(B, 1 + 52 + 2 * K, requires_grad=True)
    p1, p2 = model(x)

    r1 = torch.zeros(B, K)
    r2 = torch.rand(B, K)
    r2[0] = 0.0

    f1, f2 = model.enforce_zero_sum(r1, r2, p1, p2)
    assert torch.isfinite(f1).all()
    assert torch.isfinite(f2).all()

    eps = 1e-7
    def _norm(r):
        s = r.sum(dim=1, keepdim=True)
        return torch.where(s > 0, r / torch.clamp(s, min=eps), torch.full_like(r, 1.0 / r.shape[1]))

    s1 = _norm(r1)
    s2 = _norm(r2)
    res = (s1 * f1).sum(dim=1) + (s2 * f2).sum(dim=1)
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-6)

    c = 3.14
    f1p, f2p = model.enforce_zero_sum(r1, r2, p1 + c, p2 + c)
    assert torch.allclose(f1, f1p, atol=1e-6)
    assert torch.allclose(f2, f2p, atol=1e-6)

def test_predict_with_zero_sum_equivalence():
    """test_predict_with_zero_sum_equivalence: predict_with_zero_sum matches manual forward + enforce_zero_sum."""
    import hunl.nets.cfv_network as cn
    K = 5
    model = cn.CounterfactualValueNetwork(input_size=1+52+2*K, num_clusters=K)
    x = torch.randn(3, 1+52+2*K)
    r1 = torch.rand(3, K)
    r2 = torch.rand(3, K)
    f1a, f2a = model.predict_with_zero_sum(x, r1, r2)
    p1, p2 = model(x)
    f1b, f2b = model.enforce_zero_sum(r1, r2, p1, p2)
    assert torch.allclose(f1a, f1b, atol=1e-7)
    assert torch.allclose(f2a, f2b, atol=1e-7)

def test_builder_three_stage_and_make_network_device_and_shapes():
    """test_builder_three_stage_and_make_network_device_and_shapes: factory helpers create {preflop,flop,turn} models on a device with consistent K."""
    import hunl.nets.cfv_network as cn
    K = 6
    models = cn.build_three_stage_cfv(1+52+2*K, 1+52+2*K, 1+52+2*K, num_clusters=K, device=torch.device("cpu"))
    assert set(models.keys()) == {"preflop", "flop", "turn"}
    for m in models.values():
        assert isinstance(m, nn.Module)
        assert m.num_clusters == K
    m = cn.make_cfv_network(1+52+2*K, K)
    assert isinstance(m, nn.Module)
    assert m.num_clusters == K

def test_cfv_shard_dataset_schema_filter_and_missing(tmp_path: Path):
    """test_cfv_shard_dataset_schema_filter_and_missing: CFVShardDataset filters by schema, iterates all rows when verify_schema=False, and skips missing files."""
    import hunl.nets.cfv_shard_dataset as ds

    good = tmp_path / "shard_good.jsonl"
    bad = tmp_path / "shard_bad.jsonl"
    with good.open("w") as f:
        for i in range(3):
            f.write(json.dumps({"schema": "cfv.v1", "input_vector": [0.0], "target_v1": [0.0], "target_v2": [0.0]}) + "\n")
    with bad.open("w") as f:
        for i in range(2):
            f.write(json.dumps({"schema": "other", "input_vector": [0.0], "target_v1": [0.0], "target_v2": [0.0]}) + "\n")

    d = ds.CFVShardDataset([str(good), str(bad), str(tmp_path / "does_not_exist.jsonl")], schema_version="cfv.v1", verify_schema=True)
    rows = list(iter(d))
    assert len(rows) == 3

    d2 = ds.CFVShardDataset([str(good), str(bad)], schema_version="cfv.v1", verify_schema=False)
    rows2 = list(iter(d2))
    assert len(rows2) == 5

class _StubDG:
    def __init__(self, K):
        self.num_clusters = K
        self._seed = None

    def set_seed(self, s):
        self._seed = s

    def pot_sampler_spec(self):
        return {"name": "toy_pot_sampler", "params": {"low": 0.1, "high": 1.0}}

    def range_generator_spec(self):
        return {"name": "toy_range_gen", "params": {"K": self.num_clusters}}

    def generate_unique_boards(self, stage, num_boards):
        return [("dummy", stage, num_boards)]

    def generate_training_data(self, stage, progress=None):
        K = self.num_clusters
        for _ in range(10):
            yield {
                "input_vector": [0.2] + [0.0]*52 + ([1.0/K]*K) + ([1.0/K]*K),
                "target_v1": [0.0]*K,
                "target_v2": [0.0]*K,
            }

def test_cfv_stream_dataset_iter_and_spec():
    """test_cfv_stream_dataset_iter_and_spec: CFVStreamDataset exposes a complete spec and yields exactly num_samples rows with correct input/target shapes."""
    import hunl.nets.cfv_stream_dataset as sd
    K = 9
    dg = _StubDG(K)
    ds = sd.CFVStreamDataset(dg, stage="flop", num_samples=7, seed=1729, schema_version="cfv.v1",
                             shard_meta={"origin": "unit-test"})
    assert ds.spec["schema"] == "cfv.v1"
    assert ds.spec["seed"] == 1729
    assert ds.spec["stage"] == "flop"
    assert ds.spec["num_clusters"] == K
    assert ds.spec["pot_sampler"]["name"] == "toy_pot_sampler"
    assert ds.spec["range_generator"]["name"] == "toy_range_gen"
    assert ds.spec["origin"] == "unit-test"

    rows = list(iter(ds))
    assert len(rows) == 7
    for r in rows:
        assert r["schema"] == "cfv.v1"
        assert r["stage"] == "flop"
        assert len(r["input_vector"]) == 1 + 52 + 2*K
        assert len(r["target_v1"]) == K
        assert len(r["target_v2"]) == K

def test_train_cfv_network_zero_sum_before_loss_lr_drop_and_early_stop(monkeypatch):
    """test_train_cfv_network_zero_sum_before_loss_lr_drop_and_early_stop: trainer runs, enforces zero-sum each step, drops LR at scheduled epoch, and early-stops on plateau."""
    import torch
    import hunl.nets.cfv_trainer as tr
    import hunl.nets.cfv_network as cn
    K = 6
    model = cn.CounterfactualValueNetwork(input_size=1 + 52 + 2 * K, num_clusters=K)
    def make_toy_samples(K, n):
        samples = []
        for _ in range(n):
            x = [0.3] + [0.0] * 52 + ([1.0 / K] * K) + ([1.0 / K] * K)
            y1 = [0.0] * K
            y2 = [0.0] * K
            samples.append({"input_vector": x, "target_v1": y1, "target_v2": y2})
        return samples
    train = make_toy_samples(K, 8)
    val = make_toy_samples(K, 8)
    recorded_lrs = []
    orig_step = torch.optim.Adam.step
    def spy_step(self, *args, **kwargs):
        if self.param_groups:
            recorded_lrs.append(float(self.param_groups[0]["lr"]))
        return orig_step(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", spy_step, raising=True)
    called = {"train": 0}
    orig_ezs = model.enforce_zero_sum
    def spy_enforce(r1, r2, p1, p2):
        called["train"] += 1
        return orig_ezs(r1, r2, p1, p2)
    monkeypatch.setattr(model, "enforce_zero_sum", spy_enforce, raising=True)
    orig_eval = tr._eval_loss_cfv
    def plateau_eval(*args, **kwargs):
        th, tm, trm = orig_eval(*args, **kwargs)
        if kwargs.get("split") == "val":
            return 1.0, tm, trm
        return th, tm, trm
    monkeypatch.setattr(tr, "_eval_loss_cfv", plateau_eval, raising=True)
    out = tr.train_cfv_network(
        model=model,
        train_samples=train,
        val_samples=val,
        epochs=12,
        batch_size=4,
        lr=1e-3,
        lr_drop_epoch=1,
        lr_after=5e-4,
        weight_decay=0.0,
        device=torch.device("cpu"),
        seed=123,
        early_stop_patience=3,
        min_delta=0.0,
    )
    assert len(out["history"]["train_huber"]) < 12
    assert any(abs(lr - 5e-4) < 1e-12 for lr in recorded_lrs)
    assert called["train"] > 0

def test_trainer_ranges_from_inputs_slices_correctly():
    """test_trainer_ranges_from_inputs_slices_correctly: internal slice helper recovers r1/r2 blocks from concatenated inputs."""
    import hunl.nets.cfv_trainer as tr
    K = 5
    x = torch.zeros(2, 1+52+2*K)
    x[0, 1+52+0] = 0.1
    x[0, 1+52+K+0] = 0.2
    r1, r2 = tr._ranges_from_inputs(x, K)
    assert r1.shape == (2, K)
    assert r2.shape == (2, K)
    assert float(r1[0, 0]) == pytest.approx(0.1)
    assert float(r2[0, 0]) == pytest.approx(0.2)

def test_flop_trainer_with_custom_target_provider_lr_drop_ckpts_and_zero_sum(monkeypatch, tmp_path: Path):
    """test_flop_trainer_with_custom_target_provider_lr_drop_ckpts_and_zero_sum: flop trainer honors custom target provider, LR schedule, zero-sum enforcement, and writes checkpoints."""
    import hunl.nets.cfv_trainer_flop as tf
    import hunl.nets.cfv_network as cn

    K = 6
    model = cn.CounterfactualValueNetwork(1+52+2*K, num_clusters=K)

    def target_provider(xb, y1b, y2b, turn_model):
        n = xb.shape[0]
        device = xb.device
        return torch.zeros(n, K, device=device), torch.zeros(n, K, device=device)

    train = make_toy_samples(K, 12)
    val = make_toy_samples(K, 12)

    recorded_lrs = []
    orig_step = torch.optim.Adam.step
    def spy_step(self, *args, **kwargs):
        if self.param_groups:
            recorded_lrs.append(float(self.param_groups[0]["lr"]))
        return orig_step(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", spy_step, raising=True)

    called = {"count": 0}
    orig_ezs = model.enforce_zero_sum
    def spy_enforce(r1, r2, p1, p2):
        called["count"] += 1
        return orig_ezs(r1, r2, p1, p2)
    monkeypatch.setattr(model, "enforce_zero_sum", spy_enforce, raising=True)

    out = tf.train_flop_cfv(
        model=model,
        train_samples=train,
        val_samples=val,
        epochs=3,
        batch_size=4,
        lr=1e-3,
        lr_after=5e-4,
        lr_drop_epoch=1,
        weight_decay=0.0,
        device=torch.device("cpu"),
        seed=7,
        ckpt_dir=str(tmp_path),
        save_best=True,
        target_provider=target_provider,
        turn_model=None,
    )

    assert called["count"] > 0
    assert any(abs(l - 1e-3) < 1e-12 for l in recorded_lrs)
    assert any(abs(l - 5e-4) < 1e-12 for l in recorded_lrs)

    epoch_ckpts = list(tmp_path.glob("flop_cfv_epoch_*.pt"))
    assert len(epoch_ckpts) >= 1

def test_default_turn_leaf_target_provider_with_stubbed_solver(monkeypatch):
    """test_default_turn_leaf_target_provider_with_stubbed_solver: default provider queries a stub solver/turn net and returns deterministic per-cluster targets."""
    import hunl.nets.cfv_trainer_flop as tf

    K = 5

    class FakeSolver:
        def __init__(self, depth_limit, num_clusters):
            self.models = {"turn": None}
            self.depth_limit = depth_limit
            self.num_clusters = num_clusters
            self.total_iterations = None
        def flop_label_targets_using_turn_net(self, node):
            t1 = [float(i) for i in range(self.num_clusters)]
            t2 = [-float(i) for i in range(self.num_clusters)]
            return t1, t2

    class FakePS:
        def __init__(self, initial_stacks, board_cards, dealer):
            self.initial_stacks = list(initial_stacks)
            self.board_cards = list(board_cards)
            self.dealer = int(dealer)
            self.current_round = None
            self.current_bets = None
            self.pot_size = None
            self.last_raiser = None
            self.stacks = None
            self.current_player = None

    class FakeNode:
        def __init__(self, ps):
            self.public_state = ps
            self.player_ranges = [dict(), dict()]

    class FakeTurnModel(nn.Module):
        def __init__(self, K):
            super().__init__()
            self.num_clusters = K
            self.lin = nn.Linear(1+52+2*K, K, bias=False)
        def forward(self, x):
            return self.lin(x), -self.lin(x)
        def parameters(self):
            return super().parameters()

    monkeypatch.setattr(tf, "CFRSolver", FakeSolver, raising=True)
    monkeypatch.setattr(tf, "PublicState", FakePS, raising=True)
    monkeypatch.setattr(tf, "GameNode", FakeNode, raising=True)

    N = 3
    xb = torch.zeros(N, 1+52+2*K, dtype=torch.float32)
    xb[:, 0] = 0.25
    xb[:, 1+52:1+52+K] = 1.0 / K
    xb[:, 1+52+K:1+52+2*K] = 1.0 / K

    y1b = torch.zeros(N, K)
    y2b = torch.zeros(N, K)

    tmodel = FakeTurnModel(K)
    t1, t2 = tf.default_turn_leaf_target_provider(xb, y1b, y2b, tmodel)

    for i in range(N):
        assert t1[i].tolist() == [float(j) for j in range(K)]
        assert t2[i].tolist() == [-float(j) for j in range(K)]

def test_flop_trainer_inline_ranges_slicing():
    """test_flop_trainer_inline_ranges_slicing: inline slice helper for flop trainer indexes r1/r2 correctly."""
    import hunl.nets.cfv_trainer_flop as tf
    K = 4
    x = torch.zeros(2, 1+52+2*K)
    x[0, 1+52+1] = 0.7
    x[0, 1+52+K+2] = 0.9
    r1, r2 = tf._ranges_from_inputs_inline(x, K)
    assert float(r1[0, 1]) == pytest.approx(0.7)
    assert float(r2[0, 2]) == pytest.approx(0.9)

def test_train_turn_cfv_zero_sum_lr_drop_and_ckpts(monkeypatch, tmp_path: Path):
    """test_train_turn_cfv_zero_sum_lr_drop_and_ckpts: turn trainer enforces zero-sum, applies LR drop, and emits best/epoch checkpoints."""
    import hunl.nets.cfv_trainer_turn as tt
    import hunl.nets.cfv_network as cn

    K = 6
    model = cn.CounterfactualValueNetwork(1+52+2*K, num_clusters=K)
    train = make_toy_samples(K, 12)
    val = make_toy_samples(K, 12)

    recorded_lrs = []
    orig_step = torch.optim.Adam.step
    def spy_step(self, *args, **kwargs):
        if self.param_groups:
            recorded_lrs.append(float(self.param_groups[0]["lr"]))
        return orig_step(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", spy_step, raising=True)

    called = {"n": 0}
    orig_ezs = model.enforce_zero_sum
    def spy_enforce(r1, r2, p1, p2):
        called["n"] += 1
        return orig_ezs(r1, r2, p1, p2)
    monkeypatch.setattr(model, "enforce_zero_sum", spy_enforce, raising=True)

    out = tt.train_turn_cfv(
        model=model,
        train_samples=train,
        val_samples=val,
        epochs=3,
        batch_size=4,
        lr=1e-3,
        lr_after=5e-4,
        lr_drop_epoch=1,
        weight_decay=0.0,
        device=torch.device("cpu"),
        seed=99,
        ckpt_dir=str(tmp_path),
        save_best=True,
    )

    assert called["n"] > 0
    assert any(abs(l - 1e-3) < 1e-12 for l in recorded_lrs)
    assert any(abs(l - 5e-4) < 1e-12 for l in recorded_lrs)

    best = list(tmp_path.glob("turn_cfv_best.pt"))
    epochs = list(tmp_path.glob("turn_cfv_epoch_*.pt"))
    assert len(best) == 1
    assert len(epochs) >= 1

def test_train_turn_cfv_streaming_pipeline_and_ckpts(monkeypatch, tmp_path: Path):
    """test_train_turn_cfv_streaming_pipeline_and_ckpts: streaming variant consumes generator iterators and persists checkpoints."""
    import hunl.nets.cfv_trainer_turn as tt
    import hunl.nets.cfv_network as cn

    K = 5
    model = cn.CounterfactualValueNetwork(1+52+2*K, num_clusters=K)

    def make_iter():
        def _gen():
            for _ in range(7):
                yield {
                    "input_vector": [0.3] + [0.0]*52 + ([1.0/K]*K) + ([1.0/K]*K),
                    "target_v1": [0.0]*K,
                    "target_v2": [0.0]*K,
                }
        return _gen

    recorded_lrs = []
    orig_step = torch.optim.Adam.step
    def spy_step(self, *args, **kwargs):
        if self.param_groups:
            recorded_lrs.append(float(self.param_groups[0]["lr"]))
        return orig_step(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", spy_step, raising=True)

    out = tt.train_turn_cfv_streaming(
        model=model,
        train_iter=make_iter(),
        val_iter=make_iter(),
        epochs=2,
        batch_size=4,
        lr=1e-3,
        lr_after=5e-4,
        lr_drop_epoch=1,
        weight_decay=0.0,
        device=torch.device("cpu"),
        seed=1234,
        ckpt_dir=str(tmp_path),
        save_best=True,
    )

    assert any(abs(l - 1e-3) < 1e-12 for l in recorded_lrs)
    assert any(abs(l - 5e-4) < 1e-12 for l in recorded_lrs)

    best = list(tmp_path.glob("turn_cfv_best.pt"))
    epochs = list(tmp_path.glob("turn_cfv_epoch_*.pt"))
    assert len(best) == 1
    assert len(epochs) >= 1

def test_config_io_save_and_load_json_yaml(tmp_path: Path, monkeypatch):
    """test_config_io_save_and_load_json_yaml: save_config/load_config round-trip works for JSON and YAML; objects with attributes serialize to dicts."""
    import importlib, sys

    if "hunl.utils.config_io" in sys.modules:
        del sys.modules["hunl.utils.config_io"]
    cfg = importlib.import_module("hunl.utils.config_io")

    d = {"a": 1, "b": 2}
    json_path = tmp_path / "cfg.json"
    yml_path  = tmp_path / "cfg.yml"
    cfg.save_config(d, str(json_path))
    cfg.save_config(d, str(yml_path))
    dj = cfg.load_config(str(json_path))
    dy = cfg.load_config(str(yml_path))
    assert dj == d
    assert dy == d

    class Obj:
        def __init__(self):
            self.x = 10
            self.y = "z"

    o = Obj()
    cfg_path = tmp_path / "obj.json"
    cfg.save_config(o, str(cfg_path))
    loaded = cfg.load_config(str(cfg_path))
    assert loaded == {"x": 10, "y": "z"}

def test_compose_resolve_config_from_yaml_with_monkeypatched_resolve_config(tmp_path: Path, monkeypatch):
    """test_compose_resolve_config_from_yaml_with_monkeypatched_resolve_config: compose_resolve_config_from_yaml 
		merges three YAMLs into a ResolveConfig, applies overrides/seed, and returns runtime overrides and bet fractions."""
    import types, sys, importlib

    fake_mod = types.ModuleType("resolve_config")

    class FakeResolveConfig:
        def __init__(self):
            self.num_clusters = 1000
            self.tau_re = 0.0
            self.drift_sample_size = 0
            self.use_cfv_in_features = False
            self.enforce_zero_sum_outer = True
            self.mc_samples_win = 0
            self.mc_samples_potential = 0
            self.lr_initial = 0.0
            self.lr_after = 0.0
            self.lr_drop_epoch = 0
            self.batch_size = 0
            self.depth_limit = 0
            self.total_iterations = 0
            self.bet_size_mode = ""
            self.profile = ""
            self.bet_fractions = {}
        @classmethod
        def from_env(cls, overrides):
            inst = cls()
            for k, v in (overrides or {}).items():
                setattr(inst, k, v)
            return inst

    fake_mod.ResolveConfig = FakeResolveConfig
    sys.modules["resolve_config"] = fake_mod
    sys.modules["hunl.resolve_config"] = fake_mod

    if "hunl.utils.config_io" in sys.modules:
        del sys.modules["hunl.utils.config_io"]
    cfg = importlib.import_module("hunl.utils.config_io")

    used_seeds = []
    def fake_sgs(seed):
        used_seeds.append(int(seed))
    monkeypatch.setattr(cfg, "set_global_seed", fake_sgs, raising=True)

    abst = tmp_path / "abst.yml"
    vnets = tmp_path / "vnets.yml"
    solv  = tmp_path / "solv.yml"

    abst.write_text(
        "seed: 111\n"
        "bucket_counts:\n"
        "  turn: 77\n"
        "tau_re: 0.25\n"
        "drift_sample_size: 42\n"
        "use_cfv_in_features: true\n"
    )
    vnets.write_text(
        "seed: 222\n"
        "outer_zero_sum: true\n"
        "mc_samples_win: 13\n"
        "mc_samples_potential: 17\n"
        "lr_schedule:\n"
        "  initial: 0.003\n"
        "  after: 0.0005\n"
        "  drop_epoch: 5\n"
        "batch_size: 256\n"
    )
    solv.write_text(
        "seed: 333\n"
        "depth_limit: 3\n"
        "total_iterations: 999\n"
        "bet_size_mode: sparse\n"
        "profile: test\n"
        "iterations_per_round:\n"
        '  "1": 100\n'
        '  "3": 250\n'
        "round_actions:\n"
        '  "1": { half_pot: true, two_pot: false }\n'
        '  "3": { half_pot: true, two_pot: true }\n'
        "bet_fractions:\n"
        '  "1": [0.5, 1.0]\n'
        '  "2": [0.5, 1.0, 2.0]\n'
    )

    out = cfg.compose_resolve_config_from_yaml(str(abst), str(vnets), str(solv), overrides={"custom": 1})

    assert out["seed"] == 111
    assert used_seeds and used_seeds[-1] == 111

    rc = out["config"]
    assert rc.num_clusters == 77
    assert rc.tau_re == 0.25
    assert rc.drift_sample_size == 42
    assert rc.use_cfv_in_features is True

    assert rc.enforce_zero_sum_outer is True
    assert rc.lr_initial == 0.003
    assert rc.lr_after == 0.0005
    assert rc.lr_drop_epoch == 5
    assert rc.batch_size == 256

    assert rc.depth_limit == 3
    assert rc.total_iterations == 999
    assert rc.bet_size_mode == "sparse"
    assert rc.profile == "test"

    ro = out["runtime_overrides"]
    assert ro["_round_iters"] == {1: 100, 3: 250}
    assert ro["_round_actions"][1]["half_pot"] is True and ro["_round_actions"][1]["two_pot"] is False
    assert ro["_round_actions"][3]["half_pot"] is True and ro["_round_actions"][3]["two_pot"] is True

    assert rc.bet_fractions[1] == [0.5, 1.0]
    assert rc.bet_fractions[2] == [0.5, 1.0, 2.0]

