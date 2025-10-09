# HUNL — Continual Re-Solving Poker Agent (DeepStack-style)

Compact, reproducible implementation of continual re-solving for Heads-Up No-Limit (HUNL) Texas Hold’em with sparse action sets, depth limits, and neural counterfactual-value (CFV) networks.

* **Core ideas:** end-of-street depth limits; flop/turn CFV nets (pot-fraction targets) with **outer zero-sum** adjustment; river exact endgame; follow/terminate **range-gadget**; preflop cache.
* **Results (30k hands, 200bb, blinds 1/2):** ~**3% cw/100** (95% CI ≈ [2.0, 4.2]), **1.56 bb/100**; AIVAT ≈ **1.52 bb/100**; LBR probe **loses ≥300 mbb/g**; zero-sum residual max ≤1e-6.
  See **`HUNL_RESULTS.pdf`** and **`HUNL_THEORY.pdf`**.

---

## Repository layout (high-level)

```
hunl/
  engine/         # Public-state engine (legality, streets, utilities)
  solving/        # CFR solver, range gadget, lookahead tree, AIVAT
  nets/           # CFV nets, trainers, streaming datasets, value server
  ranges/         # Clustering and bucket interfaces
  data/           # Data generator + NPZ writers + manifests
  cli/            # Simple CLIs (play/eval/LBR)
  utils/          # Config IO, RNG helpers, result handle
tests/            # Pytest suites (unit + integration)
HUNL_RESULTS.pdf  # Empirical results report
HUNL_THEORY.pdf   # Theory & implementation notes
```

---

## Quickstart

### 1) Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy pytest
# (Add CUDA build of torch if needed)
```

### 2) Sanity check

```bash
# Fast mode trims depth/iters for quick turnaround
export FAST_TESTS=1
pytest -q tests
```

### 3) Play / evaluate (CLI)

```bash
# Interactive continual re-solving (baseline/self/acpc-client)
python -m hunl.cli.play_cli --mode baseline --hands 3 --depth 1 --iters 400

# Agent-vs-agent or agent-vs-policy summary + AIVAT
python -m hunl.cli.eval_cli --mode agent-vs-agent --episodes 1000 --depth-limit 1 --iterations 400

# Limited Best Response (flop greedy, rollout to terminal)
python -m hunl.cli.eval_cli_lbr --episodes 10000
```

---

## Data & models

### Generate supervised CFV data

```python
from hunl.data.data_generator import DataGenerator
dg = DataGenerator(num_boards=5, num_samples_per_board=2, player_stack=200, num_clusters=1000)
samples = dg.generate_training_data(stage="flop")   # list of dicts
```

### Train CFV nets (in-memory examples)

```python
from hunl.nets.cfv_network import CounterfactualValueNetwork
from hunl.nets.cfv_trainer import train_cfv_network

K = 1000
model = CounterfactualValueNetwork(input_size=1+52+2*K, num_clusters=K)
out = train_cfv_network(model=model, train_samples=samples, val_samples=samples[:256],
                        epochs=10, batch_size=512, device="cpu")
```

### Bundle I/O

```python
from hunl.nets.model_io import save_cfv_bundle, load_cfv_bundle
# save_cfv_bundle({"flop": model_f, "turn": model_t}, cluster_mapping, input_meta, "bundle.pt")
# b = load_cfv_bundle("bundle.pt")
```

---

## Key invariants (enforced in tests)

* **Outer zero-sum:** for any ranges (r_1,r_2), adjusted predictions (f_1,f_2) satisfy
  (\langle r_1,f_1\rangle + \langle r_2,f_2\rangle \approx 0).
* **Range mass:** per player sums to 1 (to tolerance) after chance-lift and mapping.
* **Pot monotonicity:** public pot does not decrease except by explicit refunds.
* **Follow/Terminate gadget:** opponent CFV upper bounds carried forward via componentwise max.

Run all:

```bash
pytest -q tests
```

Useful toggles:

* `FAST_TESTS=1` — minimal depth/iters for CI speed.
* `FAST_TEST_SEED=<int>` — deterministic fast tests.
* `DEBUG_FAST_TESTS=1` — enables certain fast-path diagnostics.

---

## Re-solving API (programmatic)

```python
from hunl.solving.resolver_integration import resolve_at
pol, w_next, our_cfv = resolve_at(
    public_state=ps,        # hunl.engine.public_state.PublicState
    r_us={i:1.0/K for i in range(K)},
    w_opp={i:0.0 for i in range(K)},
    config={
      "depth_limit": 1,
      "iterations": 400,
      "bet_size_mode": "sparse_2",  # {sparse_2|sparse_3|full}
    },
    value_server=None       # optional batched inference
)
```

---

## Repro tips

* Set seeds via `FAST_TEST_SEED` or `ResolveConfig.from_env({...})`.
* Use the **preflop cache** for throughput (bit-identical signatures).
* Log zero-sum residuals from `solver.get_last_diagnostics()` and CLIs.
* For long matches, prefer CPU inference for engine + GPU for nets (via `value_server`).

---

## Citing / reading

* **Results:** `HUNL_RESULTS.pdf`
* **Theory / manual:** `HUNL_THEORY.pdf`
* Tests document expected behavior and serve as executable specs (`tests/`).

---

## License

Add a license file (e.g., MIT/Apache-2.0). If omitted, all rights reserved by default.

---

## Acknowledgments

DeepStack-style continual re-solving inspired the overall structure; this codebase packages a streamlined, testable reproduction with sparse actions and pot-fraction CFV targets.

