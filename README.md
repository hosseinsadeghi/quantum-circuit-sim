# Quantum Algorithm Explorer

An interactive, local-first web application for step-by-step quantum circuit simulation. Select from 15 built-in quantum algorithms, configure parameters, and visualize state evolution through an intuitive browser interface with circuit diagrams, amplitude charts, Bloch spheres, and more.

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Docker](#docker)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Algorithms](#algorithms)
- [Simulation Modes](#simulation-modes)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [GitHub Pages Demo](#github-pages-demo)
- [License](#license)
- [Contact](#contact)

## Features

- **15 quantum algorithms** -- Bell State, GHZ, Grover's Search, QFT, Deutsch-Jozsa, Bernstein-Vazirani, Phase Estimation, QAOA MaxCut, MA-QAOA, ADAPT-QAOA, VQE, Quantum Teleportation, Ising Evolution, Rabi Oscillation, and Error Correction.
- **Step-by-step simulation** -- scrub through each gate application and observe state changes at every stage.
- **Interactive visualizations** -- circuit diagrams (D3.js), amplitude bar charts, measurement histograms, Bloch sphere projections, phase wheels, and observables panels (Recharts).
- **Multiple simulation backends** -- statevector (exact, pure states), density matrix (mixed states with noise support), sparse representation (memory-efficient for large circuits).
- **Noise modeling** -- Kraus noise channels including depolarizing, amplitude damping, phase damping, and custom channels.
- **Circuit optimization** -- gate fusion, identity cancellation, and commutation-based reordering.
- **GPU acceleration** -- optional CuPy backend for CUDA-capable GPUs.
- **Parameter sweeps** -- run an algorithm over a range of parameter values and compare results.
- **Async execution and WebSocket streaming** -- submit long-running jobs asynchronously with polling, or stream simulation steps in real time over WebSocket.
- **Demo mode** -- the frontend gracefully falls back to pre-computed traces when no backend is running, suitable for static hosting on GitHub Pages.

## Tech Stack

**Backend**

| Component | Technology |
|---|---|
| API framework | FastAPI (Python 3.13) |
| Simulation engine | NumPy, TensorNetwork |
| Data validation | Pydantic v2 |
| ASGI server | Uvicorn |
| Package manager | uv |

**Frontend**

| Component | Technology |
|---|---|
| UI framework | React 19 |
| Build tool | Vite 6 |
| Circuit diagrams | D3.js v7 |
| Charts | Recharts v2 |
| Reverse proxy | Express + http-proxy-middleware |

**Infrastructure**

| Component | Technology |
|---|---|
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions (GitHub Pages deploy) |

## Prerequisites

- **Python** 3.12 or later (3.13 recommended)
- **Node.js** 20 or later
- **uv** -- Python package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **npm** -- included with Node.js
- **Docker and Docker Compose** (optional, for containerized deployment)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/hosseinsadeghi/quantum-circuit-sim.git
   cd quantum-circuit-sim
   ```

2. **Install Python dependencies:**

   ```bash
   uv sync
   ```

3. **Install Node.js dependencies (root proxy server and frontend):**

   ```bash
   npm run install:all
   ```

4. **Build the frontend (optional, for production):**

   ```bash
   npm run build
   ```

## Usage

### Development mode

Start both the FastAPI backend and the Vite dev server with hot reload:

```bash
npm run dev
```

This launches:
- FastAPI backend at `http://localhost:8001`
- Vite dev server at `http://localhost:5173` (with API proxy to the backend)

### Production mode

Build the frontend and start both servers:

```bash
npm run build
npm start
```

This launches:
- FastAPI backend at `http://localhost:8001`
- Express server at `http://localhost:3000` (serves the React app and proxies `/api` requests to the backend)

Open `http://localhost:3000` in your browser.

### Running servers individually

```bash
# Backend only
uv run uvicorn backend.main:app --port 8001

# Frontend dev server only
cd frontend && npm run dev
```

## Docker

Build and run the full stack with Docker Compose:

```bash
docker compose up --build
```

This starts two containers:
- **backend** -- Python FastAPI on port 8001 (internal), with a health check on `/api/health`.
- **frontend** -- Node.js Express on port 3000 (exposed), proxying API requests to the backend container.

The frontend container waits for the backend health check to pass before starting.

Access the application at `http://localhost:3000`.

## API Reference

The FastAPI backend exposes the following endpoints under the `/api` prefix:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/algorithms` | List all available algorithms with their parameter schemas |
| `POST` | `/api/simulate` | Run a simulation and return the full trace |
| `POST` | `/api/sweep` | Run a parameter sweep across multiple values |
| `POST` | `/api/simulate/async` | Submit an async simulation job, returns a `job_id` |
| `GET` | `/api/simulate/job/{job_id}` | Poll for async job status and result |
| `WS` | `/api/simulate/stream` | Stream simulation steps over WebSocket |
| `GET` | `/api/health` | Backend health check |

### Simulate request body

```json
{
  "algorithm_id": "grover",
  "parameters": { "n_qubits": 4, "target_state": 5 },
  "mode": "statevector",
  "noise_config": null,
  "optimize": false
}
```

### Simulate response

Returns a `SimulationTrace` containing:
- `steps[]` -- each step includes `state_vector`, `probabilities`, `basis_labels`, and `observables` (Bloch vectors, entropy, purity).
- `circuit_layout` -- qubit wires and gate columns for rendering the circuit diagram.
- `measurement` -- final measurement probabilities and most likely outcome.
- `n_qubits` -- number of qubits in the circuit.

## Project Structure

```
quantum-circuit-sim/
|-- simulator/                  # NumPy quantum simulation engine
|   |-- gates.py                # Gate matrices (H, X, Y, Z, CNOT, CZ, SWAP, Rx, Ry, Rz, CCX, CCZ, ...)
|   |-- state_vector.py         # StateVector class (exact statevector simulation)
|   |-- density_matrix.py       # DensityMatrix class (tensor contraction, noise support)
|   |-- sparse_state.py         # SparseStateVector (dict-based, auto-densifies at 25%)
|   |-- circuit.py              # Circuit IR -- algorithms emit ops, Executor runs them
|   |-- circuit_optimizer.py    # Gate fusion, identity cancellation, commutation reordering
|   |-- executor.py             # Runs a Circuit, SnapshotConfig controls step serialization
|   |-- array_backend.py        # ArrayBackend ABC with NumpyBackend and CupyBackend (GPU)
|   |-- noise.py                # Kraus noise channels (depolarizing, amplitude damping, ...)
|   |-- observables.py          # Bloch vectors, entropy, purity computation
|   |-- tracer.py               # Legacy snapshot helper
|
|-- algorithms/                 # 15 algorithm implementations (all use Circuit IR + Executor)
|   |-- base.py                 # Abstract Algorithm class
|   |-- bell_state.py           # Bell State preparation
|   |-- ghz.py                  # GHZ state (up to 16 qubits)
|   |-- grover.py               # Grover's search (up to 12 qubits)
|   |-- qft.py                  # Quantum Fourier Transform (up to 14 qubits)
|   |-- deutsch_jozsa.py        # Deutsch-Jozsa algorithm (up to 12 qubits)
|   |-- bernstein_vazirani.py   # Bernstein-Vazirani algorithm (up to 14 qubits)
|   |-- phase_estimation.py     # Quantum Phase Estimation (up to 10 qubits)
|   |-- qaoa_maxcut.py          # QAOA for MaxCut (up to 14 qubits)
|   |-- ma_qaoa.py              # Multi-Angle QAOA (up to 14 qubits)
|   |-- adapt_qaoa.py           # ADAPT-QAOA (up to 14 qubits)
|   |-- vqe.py                  # Variational Quantum Eigensolver
|   |-- teleportation.py        # Quantum Teleportation
|   |-- ising_evolution.py      # Ising model time evolution (up to 10 qubits)
|   |-- rabi.py                 # Rabi oscillation
|   |-- error_correction.py     # Quantum Error Correction
|
|-- backend/                    # FastAPI micro-service (port 8001)
|   |-- main.py                 # App entry point, CORS, router mount
|   |-- api/
|   |   |-- simulation.py       # /simulate, /simulate/async, /simulate/stream, /sweep
|   |-- models/
|       |-- requests.py         # Pydantic v2 request models
|       |-- responses.py        # Pydantic v2 response models
|
|-- frontend/                   # React 19 + Vite application
|   |-- src/
|   |   |-- App.jsx             # Main application layout
|   |   |-- context/
|   |   |   |-- SimulationContext.jsx   # Global state management
|   |   |-- api/
|   |   |   |-- client.js       # API client (fetchAlgorithms, runSimulation, runSweep)
|   |   |-- components/
|   |       |-- AlgorithmSelector.jsx   # Algorithm picker
|   |       |-- ParameterPanel.jsx      # Dynamic parameter controls
|   |       |-- NoisePanel.jsx          # Noise channel configuration
|   |       |-- CircuitDiagram.jsx      # D3-based circuit diagram
|   |       |-- AmplitudeChart.jsx      # Amplitude bar chart
|   |       |-- MeasurementHistogram.jsx # Measurement probability histogram
|   |       |-- BlochSphere.jsx         # Bloch sphere visualization
|   |       |-- PhaseWheel.jsx          # Complex amplitude phase wheels
|   |       |-- ObservablesPanel.jsx    # Entropy, purity, Bloch vectors
|   |       |-- StepScrubber.jsx        # Step-by-step navigation slider
|   |-- vite.config.js
|   |-- package.json
|
|-- benchmarks/
|   |-- bench_scaling.py        # Performance benchmarks at varying qubit counts
|
|-- tests/                      # Pytest test suite (121+ tests)
|   |-- test_gates.py
|   |-- test_state_vector.py
|   |-- test_density_matrix.py
|   |-- test_circuit.py
|   |-- test_executor.py
|   |-- test_algorithms.py
|   |-- test_consolidation.py
|   |-- test_scaling.py
|
|-- server.js                   # Express reverse proxy (serves frontend, proxies /api)
|-- docker-compose.yml          # Multi-container orchestration
|-- Dockerfile.backend          # Python backend container
|-- Dockerfile.frontend         # Node.js frontend container (multi-stage build)
|-- pyproject.toml              # Python project config and dependencies
|-- package.json                # Root Node.js config (start scripts, proxy deps)
|-- .github/workflows/
|   |-- deploy-pages.yml        # GitHub Actions: deploy frontend to GitHub Pages
```

## Algorithms

| Algorithm | Category | Max Qubits | Description |
|---|---|---|---|
| Bell State | Foundational | 2 | Prepares a maximally entangled Bell pair |
| GHZ | Foundational | 16 | Greenberger-Horne-Zeilinger entangled state |
| Teleportation | Foundational | 3 | Quantum state teleportation protocol |
| Grover's Search | Search | 12 | Amplitude amplification for unstructured search |
| Deutsch-Jozsa | Oracle | 12 | Determines if a function is constant or balanced |
| Bernstein-Vazirani | Oracle | 14 | Finds a hidden bit string with a single query |
| QFT | Transform | 14 | Quantum Fourier Transform |
| Phase Estimation | Transform | 10 | Estimates eigenvalues of a unitary operator |
| QAOA MaxCut | Optimization | 14 | Quantum Approximate Optimization for MaxCut |
| MA-QAOA | Optimization | 14 | Multi-Angle QAOA variant |
| ADAPT-QAOA | Optimization | 14 | Adaptive QAOA with operator pool |
| VQE | Optimization | -- | Variational Quantum Eigensolver |
| Ising Evolution | Simulation | 10 | Time evolution under the Ising Hamiltonian |
| Rabi Oscillation | Simulation | -- | Single-qubit Rabi oscillation dynamics |
| Error Correction | Error Correction | -- | Quantum error correction code demonstration |

## Simulation Modes

| Mode | Description | Typical Scale |
|---|---|---|
| `statevector` | Exact pure-state simulation (default) | 16--18 qubits |
| `density_matrix` | Mixed-state simulation with noise support; auto-selected when `noise_config` is provided | 10--12 qubits |

**Additional options:**

- `representation`: `"dense"` (default), `"sparse"` (dict-based, memory-efficient), or `"auto"` (switches to sparse for 14+ qubits).
- `optimize`: `true` to run circuit optimization passes (gate fusion, identity cancellation, commutation reordering) before execution.
- `backend`: `"numpy"` (default) or `"cupy"` (GPU acceleration, requires `cupy-cuda12x`).

## Testing

Run the full test suite (121+ tests):

```bash
uv run pytest tests/ -q
```

Run tests for a specific module:

```bash
uv run pytest tests/test_executor.py -q
uv run pytest tests/test_algorithms.py -q
uv run pytest tests/test_scaling.py -q
```

## Benchmarking

Measure execution time and peak memory for standard circuits at varying qubit counts:

```bash
python benchmarks/bench_scaling.py                       # default suite
python benchmarks/bench_scaling.py --max-qubits 14       # custom max qubit count
python benchmarks/bench_scaling.py --output results.json  # save results to JSON
```

The benchmark suite covers Random, QFT, Grover, and GHZ circuits and compares NumPy and CuPy backends when available.

## GitHub Pages Demo

The frontend is automatically deployed to GitHub Pages on every push to the `main` branch via GitHub Actions. In demo mode (when no backend is available), the application displays pre-computed simulation traces, allowing users to explore the interface without running any servers.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

**Hossein Sadeghi**
Email: hosseinsadeghiesfahani@gmail.com
GitHub: [hosseinsadeghi](https://github.com/hosseinsadeghi)
