import os
from resolve_config import ResolveConfig


def test_from_env_defaults_and_test_profile_overrides(monkeypatch):
    # Explicit overrides: set profile to 'test' and check auto test settings
    cfg = ResolveConfig.from_env({"profile": "test", "num_clusters": 128})
    assert cfg.profile == "test"
    # Test-profile overrides
    assert cfg.depth_limit == 0
    assert cfg.total_iterations == 1
    assert cfg.mc_samples_win == 0
    assert cfg.mc_samples_potential == 0
    assert cfg.opp_sample_size == 0
    assert cfg.use_cfv_in_features is False
    assert cfg.tau_re == float("inf")
    # Always set num_clusters to 1000 if not overridden; here we provided override so keep 128
    assert cfg.num_clusters == 128

