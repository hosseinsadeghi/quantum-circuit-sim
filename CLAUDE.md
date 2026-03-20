# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project

**Quantum Algorithm Explorer** — a local-first web app for step-by-step interactive quantum circuit simulation. FastAPI + NumPy backend, Express + React frontend.

## Commands

| Task | Command |
|---|---|
| Run both servers (dev) | `npm start` (root) |
| Backend only | `uv run uvicorn backend.main:app --port 8001` |
| Frontend dev server | `cd frontend && npm run dev` |
| Run tests | `uv run pytest tests/ -q` |
| Build frontend | `cd frontend && npm run build` |
| Docker (full stack) | `docker compose up --build` |

## Architecture

```
simulator/                # NumPy quantum simulation engine (GPU-ready via backend abstraction)
  gates.py                # Gate matrices (H, X, CNOT, Rx, Rz, CCX, CCZ, ...)
  state_vector.py         # StateVector class (statevector simulation)
  density_matrix.py       # DensityMatrix — tensor contraction (O(2^2n) per gate)
  circuit.py              # Circuit IR — algorithms emit ops; Executor runs them
  circuit_optimizer.py    # Gate fusion, identity cancellation, commutation reordering
  executor.py             # Runs Circuit; SnapshotConfig controls serialization
  sparse_state.py         # SparseStateVector — dict-based, auto-densify at 25%
  array_backend.py        # ArrayBackend ABC; NumpyBackend + CupyBackend (GPU)
  noise.py                # Kraus noise channels (depolarizing, amplitude damping, ...)
  observables.py          # O(n·2^n) Bloch/entropy from SV; O(2^2n) from DM
  tracer.py               # Legacy snapshot helper

algorithms/               # 15 algorithm implementations (all use Circuit IR + Executor)
  base.py                 # Abstract Algorithm class
  bell_state.py, ghz.py (→16q), grover.py (→12q), qft.py (→14q)
  deutsch_jozsa.py (→12q), bernstein_vazirani.py (→14q), phase_estimation.py (→10q)
  qaoa_maxcut.py (→14q), ma_qaoa.py (→14q), adapt_qaoa.py (→14q)
  teleportation.py, ising_evolution.py (→10q), rabi.py, vqe.py, error_correction.py

backend/                  # FastAPI micro-service (port 8001)
  main.py                 # App entry point, CORS, router mount
  api/simulation.py       # /simulate, /simulate/async, /simulate/stream (WS), /sweep
  models/                 # Pydantic v2 request/response models

benchmarks/               # Performance measurement suite
  bench_scaling.py        # Random/QFT/Grover/GHZ at varying qubit counts

frontend/           # React 19 + Vite app (served by Express on port 3000)
  src/
    context/SimulationContext.jsx  # Global state: algorithm, params, mode, noise
    api/client.js                  # fetchAlgorithms, runSimulation, runSweep
    components/                    # AlgorithmSelector, ParameterPanel, NoisePanel,
                                   # CircuitDiagram (D3), AmplitudeChart, MeasurementHistogram,
                                   # BlochSphere, ObservablesPanel, PhaseWheel, StepScrubber

server.js           # Express: serves React frontend, proxies /api/** → FastAPI
docker-compose.yml  # backend + frontend services; backend health-checked before frontend starts
```

## Key data contract

`POST /api/simulate` returns a `SimulationTrace`:
- `steps[]` — each step has `state_vector`, `probabilities`, `basis_labels`, and `observables` (Bloch vectors, entropy, purity)
- `circuit_layout` — qubit wires + gate columns for the D3 circuit diagram
- `measurement` — final measurement probabilities

## Simulation modes

- `statevector` (default) — fast, exact, pure states only (16-18 qubits comfortable)
- `density_matrix` — supports noise; auto-selected with `noise_config` (10-12 qubits)
- `representation`: `"dense"` (default), `"sparse"` (dict-based), `"auto"` (sparse for n≥14)
- `optimize: true` — run circuit optimization passes before execution
- `backend`: `"numpy"` (default) or `"cupy"` (GPU, requires `cupy-cuda12x`)

## Async & streaming

- `POST /api/simulate/async` → returns `job_id`, poll via `GET /api/simulate/job/{id}`
- `WS /api/simulate/stream` → streams steps over WebSocket

## Git workflow

**Commit after every completed feature or bug fix.** Do not batch unrelated work into one commit.

- Stage only files relevant to the change (avoid `git add .`)
- Write a concise conventional commit message: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- Always run `uv run pytest tests/ -q` before committing; do not commit with failing tests
- Include `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>` in the commit body

## Adding a new algorithm

1. Create `algorithms/<name>.py` — subclass `Algorithm`, set `algorithm_id`, `name`, `category`, `description`, `parameter_schema`
2. Build a `Circuit`, call `Executor.run()`, return `result.to_trace_dict()`
3. Register the instance in `backend/api/simulation.py` → `ALGORITHMS` dict
4. Run tests; commit

## Adding a new simulator feature

1. Write the feature in the relevant `simulator/` module
2. Add tests in `tests/`
3. Run `uv run pytest tests/ -q`; commit when green

## Testing & benchmarking

```
uv run pytest tests/ -q                           # all tests (121 tests)
uv run pytest tests/test_executor.py -q            # specific module
uv run pytest tests/test_scaling.py -q             # scaling roadmap tests
python benchmarks/bench_scaling.py --max-qubits 14 # performance benchmarks
```

All NumPy arrays must be serialized via `.tolist()` before leaving the backend — the Executor handles this automatically.
