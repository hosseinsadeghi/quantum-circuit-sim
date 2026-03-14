# Quantum Algorithm Explorer — MVP Plan

## Context

Building a local-first web application that lets users explore small quantum algorithms (Grover's search, QAOA, Bell states) step-by-step, with interactive visualizations of circuit diagrams, state vector amplitude evolution, and measurement probability distributions. The project is in very early stages — no code exists yet, but FastAPI + NumPy are already in `pyproject.toml`.

The core value proposition is pedagogical: a user picks an algorithm, adjusts parameters, and watches the quantum state evolve gate by gate.

---

## Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Quantum simulator | Pure NumPy (custom) | Full control over step capture; zero new deps; ~200 LoC; transparent |
| Python backend | FastAPI + Uvicorn | Already installed; async; Pydantic v2; simulation-only micro-service |
| Node.js server | Express | Main app entry point; serves React frontend; proxies to Python |
| Frontend | React 19 + Vite | Fast setup; good D3 integration; built by Vite, served by Express |
| Circuit viz | D3.js (direct) | Custom SVG layout needed; no chart library handles this |
| State/measurement charts | Recharts | React-native; zero D3 knowledge needed for bar charts |
| State management | React built-ins (useState + useContext) | Single JSON blob per run; no Redux needed |

**Launch model:** `npm start` in project root starts both the Express server and the Python FastAPI process (via `concurrently`). Express serves the React frontend and proxies `/api/*` requests to the Python simulation backend running on port 8001. Users interact only with the Node.js server at `http://localhost:3000`.

---

## File Structure

```
quantum_project/
├── package.json                     # Node.js root: scripts, concurrently dep
├── server.js                        # Express app — serves frontend, proxies /api to Python
├── pyproject.toml                   # Python deps (FastAPI, Uvicorn, NumPy)
│
├── backend/                         # Python simulation micro-service (FastAPI)
│   ├── main.py                      # FastAPI entry point (uvicorn backend.main:app --port 8001)
│   ├── api/
│   │   ├── router.py
│   │   └── simulation.py           # POST /api/simulate, GET /api/algorithms
│   └── models/
│       ├── requests.py             # SimulateRequest (Pydantic)
│       └── responses.py            # SimulationTrace, SimulationStep, etc. (Pydantic)
│
├── simulator/                       # Pure NumPy quantum simulation engine
│   ├── gates.py                    # Gate matrices (H, X, CNOT, Ry, Rz, ...)
│   ├── state_vector.py             # StateVector class with gate application methods
│   └── tracer.py                   # Snapshots state after each gate → SimulationStep list
│
├── algorithms/                      # Algorithm implementations
│   ├── base.py                     # Abstract Algorithm(run(params) → SimulationTrace)
│   ├── bell_state.py               # Bell state preparation (2 qubits, no params)
│   ├── grover.py                   # Grover's search (2–4 qubits, target bitstring)
│   └── qaoa_maxcut.py              # QAOA Max-Cut (4–6 qubits, p=1–3, graph preset)
│
└── frontend/                        # React + Vite app
    ├── package.json
    ├── vite.config.js              # Dev proxy: /api → localhost:8001 (Python)
    └── src/
        ├── context/SimulationContext.jsx
        ├── api/client.js            # fetchAlgorithms(), runSimulation()
        ├── hooks/
        │   ├── useSimulation.js
        │   └── useStepPlayer.js    # Auto-play timer logic
        └── components/
            ├── AlgorithmSelector.jsx
            ├── ParameterPanel.jsx  # Dynamic from algorithm schema
            ├── StepScrubber.jsx
            ├── CircuitDiagram.jsx  # D3 SVG (useRef + useEffect pattern)
            ├── AmplitudeChart.jsx  # Recharts BarChart
            ├── MeasurementHistogram.jsx
            └── PhaseWheel.jsx      # Optional; SVG unit circles per basis state
```

**Launch flow:**
- `npm start` (root `package.json`) → runs `concurrently`:
  1. `uvicorn backend.main:app --port 8001` — Python simulation service
  2. `node server.js` — Express server on port 3000 (serves built React frontend, proxies `/api/*` → `localhost:8001`)
- In dev: `npm run dev` runs Vite dev server (in `frontend/`) proxying `/api` to port 8001, alongside Uvicorn
- Production: `npm run build` compiles React → `frontend/dist/`, Express serves it as static files

---

## Core Data Contract

`POST /api/simulate` returns a `SimulationTrace` — the central JSON contract between Python and React.

```json
{
  "algorithm": "grover",
  "n_qubits": 3,
  "parameters": { "target_state": "101", "num_iterations": 2 },
  "steps": [
    {
      "step_index": 0,
      "label": "Initialize |000>",
      "gate": null,
      "qubits_affected": [],
      "state_vector": { "real": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "imag": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] },
      "probabilities": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "basis_labels": ["|000>", "|001>", "|010>", "|011>", "|100>", "|101>", "|110>", "|111>"]
    }
  ],
  "measurement": {
    "basis_labels": ["|000>", "|001>", "|010>", "|011>", "|100>", "|101>", "|110>", "|111>"],
    "probabilities": [0.03, 0.03, 0.03, 0.03, 0.03, 0.78, 0.03, 0.03],
    "most_likely_outcome": "|101>"
  },
  "circuit_layout": {
    "qubit_labels": ["q0", "q1", "q2"],
    "columns": [
      { "column_index": 0, "gates": [{ "qubit": 0, "name": "H", "step_index": 1 }] }
    ]
  }
}
```

**Key decisions:**
- `state_vector` stores `real` and `imag` as separate `list[float]` — JSON has no complex type
- `probabilities` is precomputed server-side (`|amplitude|²`) — saves client math
- `circuit_layout` is **hardcoded per algorithm** — circuit layout is a graph coloring problem; hardcoding is correct for MVP scope
- QAOA angles `(beta, gamma)` are **precomputed lookup tables** per `(topology, p)` — no variational optimizer needed

---

## API Endpoints

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/algorithms` | Catalog with parameter schemas (drives ParameterPanel dynamically) |
| POST | `/api/simulate` | Run algorithm, return full SimulationTrace |
| GET | `/api/health` | Frontend startup check — enables Run button once backend is confirmed alive |

Express (`server.js`) proxies all `/api/*` traffic to Python on port 8001.

---

## Development Milestones

### Week 1 — Simulator Core ✅ DONE (2026-03-13)
- `simulator/gates.py`: all gate matrices as NumPy complex128 constants; unit tests confirm unitarity
- `simulator/state_vector.py`: `StateVector` class — `apply_single_qubit_gate()`, `apply_two_qubit_gate()`, `probabilities()`, `basis_labels()`
- `simulator/tracer.py`: wraps StateVector, snapshots after every gate → list of `SimulationStep`
- `algorithms/bell_state.py` + `algorithms/grover.py` (2–4 qubits) + `algorithms/qaoa_maxcut.py` (p=1–3, cycle/complete/path)
- **Result:** 31/31 pytest tests pass; Grover peak verified; QAOA probabilities sum to 1.0

### Week 2 — FastAPI Backend + Full Frontend ✅ DONE (2026-03-13)
- `backend/main.py`, `backend/api/`, `backend/models/` — FastAPI wiring with Pydantic v2 models
- CORS enabled for `localhost:5173` (Vite dev) and `localhost:3000` (Express)
- Root `package.json` + `server.js` (Express with http-proxy-middleware to port 8001)
- Full React frontend in `frontend/` — `SimulationContext`, `api/client.js`, all components wired end-to-end
- NumPy serialization handled in `tracer.py` via `.tolist()`; `json.dumps` test validates no leaks

### Week 3 — Visualizations ✅ DONE (2026-03-13)
- `AmplitudeChart.jsx`: Recharts BarChart, color-encoded by amplitude sign (purple/red), most-likely state highlighted green; `useDeferredValue` on `currentStep`; states < 0.001 collapsed to "other" bucket
- `CircuitDiagram.jsx`: D3 SVG — qubit wires, gate boxes, CNOT connectors (filled circle + ⊕ target), playhead column highlight synced to `currentStep`; **auto-scrolls container to keep active column in view**
- `MeasurementHistogram.jsx`: static final-state Recharts chart
- `PhaseWheel.jsx`: SVG unit circles per basis state (default off, toggle in header)
- CSS layout: two-column sidebar + main content; chart grid below circuit

### Week 4 — Polish + Packaging (Days 22–28)
- `useStepPlayer.js`: auto-play with speed control (0.5×/1×/2×/4×); Play/Pause/Step buttons; keyboard shortcuts (Space, ←/→) ✅ DONE
- Current-step info panel in StepScrubber: gate name + label per step ✅ DONE
- Frontend input validation + backend `400` error display in UI ✅ DONE
- `npm run build` → `frontend/dist/` → Express serves as static; `npm start` verified end-to-end
- README with screenshot, install, and run instructions
- Full run-through of all algorithms at all parameter combinations; tag `v0.1.0`

---

## Key Technical Risks

| Risk | Status | Mitigation |
|---|---|---|
| NumPy types not JSON serializable | ✅ Resolved | `.tolist()` called in `tracer.py` only; `json.dumps` pytest fixture confirms no leaks |
| D3 + React DOM conflicts | ✅ Resolved | `useRef` + `useEffect` pattern — React renders `<svg>` wrapper, D3 renders inside |
| QAOA variational optimizer complexity | ✅ Resolved | Precomputed `(gamma, beta)` lookup tables per `(topology, p)` in `qaoa_maxcut.py` |
| Circuit layout complexity | ✅ Resolved | `circuit_layout` hardcoded inside each algorithm's `run()` method |
| 256-bar chart perf at 8 qubits | ✅ Resolved | `useDeferredValue` on `currentStep`; states < 0.001 collapsed to "other" bucket |
| Integer params sent as strings from `<select>` | ✅ Fixed | `ParameterPanel` coerces to `parseInt` when `def.type === 'integer'`; algorithms also cast with `int()` defensively |
| Circuit playhead scrolling off-screen for long algorithms (QAOA p>1) | ✅ Fixed | `CircuitDiagram` auto-scrolls the container with `scrollTo({ behavior: 'smooth' })` to keep the active column centered |
| Express proxy when Python isn't ready | Open | `server.js` returns 502 with a clear message if port 8001 is unreachable |

---

## Verification

1. **Unit tests:** `pytest simulator/ algorithms/` — gate unitarity, Bell state amplitudes, Grover peak probability
2. **API smoke test:** `curl -X POST localhost:8001/api/simulate -H 'Content-Type: application/json' -d '{"algorithm_id":"grover","parameters":{"n_qubits":3,"target_state":"101"}}'`
3. **End-to-end:** Open `http://localhost:3000`, run all 3 algorithms at all parameter combos, drag scrubber, verify playhead + chart sync
4. **Serialization:** `json.dumps(trace.dict())` in a pytest fixture — confirms no NumPy types leak

---

## Critical Files (priority order)

1. `simulator/state_vector.py` — all algorithm correctness depends on this
2. `simulator/tracer.py` — defines the data contract; serialization bugs live here
3. `backend/models/responses.py` — Pydantic models; changing this breaks API and frontend simultaneously
4. `backend/api/simulation.py` — algorithm dispatch and validation
5. `server.js` — Express entry point; proxy config must match Python port (8001)
6. `frontend/src/context/SimulationContext.jsx` — all visualization components read from here
