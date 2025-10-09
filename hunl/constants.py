"""
I define seed and epsilon constants that keep numeric behavior stable across the whole
codebase. I use SEED_DEFAULT and SEED_RIVER for reproducible sampling and dataset
generation. I keep EPS_ZS, EPS_MASS, and EPS_SUM as defensive tolerances for zero-sum
enforcement, probability mass conservation, and vector sums in solvers and data
pipelines.

Key symbols: SEED_DEFAULT — default RNG seed; SEED_RIVER — river-specific seed; EPS_ZS —
zero-sum tolerance; EPS_MASS — probability-mass tolerance; EPS_SUM — sum-of-components
tolerance.

I serve no I/O myself. Callers import these names and use them in numerical checks, loss
functions, and range normalization. I impose the invariant that callers never compare
floats to zero directly; they compare against these epsilons. This reduces spurious
failures and keeps training and evaluation deterministic when seeds are set.

Internal dependencies: none. External dependencies: none. Configuration knobs:
downstream code may override seeds through environment variables before constructing
RNGs; I do not read env vars directly.

Edge cases: if a caller treats these as exact thresholds for hard constraints,
performance may degrade by over-clipping tiny residuals. I expect callers to scale
tolerances to their value domain (e.g., pot-fraction vs chips).
"""

SEED_DEFAULT = 1729
SEED_RIVER  = 2027
EPS_ZS   = 1e-6
EPS_MASS = 1e-12
EPS_SUM  = 1e-9

