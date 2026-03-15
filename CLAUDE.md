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
simulator/          # Pure NumPy quantum simulation engine
  gates.py          # Gate matrices (H, X, CNOT, Rx, Rz, ...)
  state_vector.py   # StateVector class (statevector simulation)
  circuit.py        # Circuit IR — algorithms emit ops here; Executor runs them
  executor.py       # Runs Circuit in statevector or density_matrix mode
  density_matrix.py # Density matrix simulation (noise, mixed states)
  noise.py          # Kraus noise channels (depolarizing, amplitude damping, ...)
  observables.py    # Per-step Bloch vectors, entropy, purity
  tracer.py         # Legacy snapshot helper (still used by executor internally)

algorithms/         # Algorithm implementations (all use Circuit IR + Executor)
  base.py           # Abstract Algorithm class
  bell_state.py, grover.py, qaoa_maxcut.py
  deutsch_jozsa.py, bernstein_vazirani.py, qft.py, phase_estimation.py
  ghz.py, teleportation.py, ising_evolution.py, rabi.py, vqe.py
  error_correction.py

backend/            # FastAPI micro-service (port 8001)
  main.py           # App entry point, CORS, router mount
  api/simulation.py # POST /api/simulate, GET /api/algorithms, POST /api/sweep
  models/           # Pydantic v2 request/response models

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

- `statevector` (default) — fast, exact, pure states only
- `density_matrix` — supports noise; automatically selected when a `noise_config` is provided

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

## Testing

```
uv run pytest tests/ -q       # all tests
uv run pytest tests/test_executor.py -q   # specific module
```

All NumPy arrays must be serialized via `.tolist()` before leaving the backend — the Executor handles this automatically.
