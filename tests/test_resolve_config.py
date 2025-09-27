"""
Test suite for ResolveConfig environment loading: verifies default handling, 'test' profile overrides, 
	 and explicit override precedence.
"""

import os
from hunl.resolve_config import ResolveConfig


def test_from_env_defaults_and_test_profile_overrides(monkeypatch):
	"""
	Ensure ResolveConfig.from_env applies the 'test' profile overrides (depth_limit, iterations, 
			sampling flags, feature toggles, tau_re) and respects explicit num_clusters overrides.
	"""
	cfg = ResolveConfig.from_env({"profile": "test", "num_clusters": 128})
	assert cfg.profile == "test"
	assert cfg.depth_limit == 0
	assert cfg.total_iterations == 1
	assert cfg.mc_samples_win == 0
	assert cfg.mc_samples_potential == 0
	assert cfg.opp_sample_size == 0
	assert cfg.use_cfv_in_features is False
	assert cfg.tau_re == float("inf")
	assert cfg.num_clusters == 128

