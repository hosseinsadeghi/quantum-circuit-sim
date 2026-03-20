"""
Executor — runs a Circuit object and returns a SimulationTrace-compatible dict.

Supports two modes:
  - "statevector"    : uses StateVector + Tracer (fast, exact, pure states only)
  - "density_matrix" : uses DensityMatrix (supports noise and mixed states)

Both modes compute per-step observables when include_observables=True.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from simulator.circuit import (
    Circuit, GateOp, TwoQubitGateOp, ThreeQubitGateOp, MeasureOp, ResetOp,
    BarrierOp, ClassicalControlOp, PhaseOracleOp, DiffusionOp, Op,
)
from simulator.state_vector import StateVector
from simulator.density_matrix import DensityMatrix
from simulator.noise import NoiseModel
from simulator.observables import compute_observables_from_sv, compute_observables_from_dm


@dataclass
class SnapshotConfig:
    """Controls what data is captured in each simulation step snapshot.

    Attributes:
        include_state_vector: Whether to serialize full state vector/DM data.
        include_observables: Whether to compute observables (Bloch, entropy, etc.).
        observable_interval: Compute observables every N steps (1 = every step).
    """
    include_state_vector: bool = True
    include_observables: bool = True
    observable_interval: int = 1


class ExecutionResult:
    """Holds the result of running a Circuit through the Executor."""

    def __init__(
        self,
        steps: List[Dict],
        circuit_layout: Dict,
        n_qubits: int,
        clbits: List[int],
    ):
        self.steps = steps
        self.circuit_layout = circuit_layout
        self.n_qubits = n_qubits
        self.clbits = clbits

    def final_probabilities(self) -> List[float]:
        return self.steps[-1]["probabilities"]

    def basis_labels(self) -> List[str]:
        return self.steps[-1]["basis_labels"]

    def most_likely_outcome(self) -> str:
        probs = self.final_probabilities()
        labels = self.basis_labels()
        return labels[int(np.argmax(probs))]

    def to_trace_dict(self, algorithm_id: str, parameters: Dict[str, Any]) -> Dict:
        """
        Convert to the wire format expected by the frontend (SimulationTrace).
        Compatible with existing backend/models/responses.py.
        """
        final_probs = self.final_probabilities()
        basis_lbls = self.basis_labels()
        return {
            "algorithm": algorithm_id,
            "n_qubits": self.n_qubits,
            "parameters": parameters,
            "steps": self.steps,
            "measurement": {
                "basis_labels": basis_lbls,
                "probabilities": final_probs,
                "most_likely_outcome": self.most_likely_outcome(),
            },
            "circuit_layout": self.circuit_layout,
        }


class Executor:
    """
    Runs a Circuit and produces a list of SimulationStep dicts.

    Args:
        mode: "statevector" (default) or "density_matrix"
        noise_model: NoiseModel applied after each gate (density_matrix mode only)
        include_observables: if True, each step gets an "observables" sub-dict
        rng_seed: random seed for mid-circuit measurements
        snapshot_config: fine-grained control over snapshot contents
    """

    def __init__(
        self,
        mode: str = "statevector",
        noise_model: Optional[NoiseModel] = None,
        include_observables: bool = True,
        rng_seed: Optional[int] = None,
        snapshot_config: Optional[SnapshotConfig] = None,
        optimize: bool = False,
    ):
        if mode not in ("statevector", "density_matrix"):
            raise ValueError(f"mode must be 'statevector' or 'density_matrix', got {mode!r}")
        self.mode = mode
        self.noise_model = noise_model
        self._rng = np.random.default_rng(rng_seed)

        # Snapshot configuration: if provided, it takes precedence
        if snapshot_config is not None:
            self.snapshot_config = snapshot_config
        else:
            self.snapshot_config = SnapshotConfig(
                include_state_vector=True,
                include_observables=include_observables,
            )
        # Backward-compat alias
        self.include_observables = self.snapshot_config.include_observables

        self.optimize = optimize

        # Force density_matrix mode when noise is enabled
        if noise_model is not None and noise_model.is_noisy():
            self.mode = "density_matrix"

    def run(self, circuit: Circuit, init_label: Optional[str] = None) -> ExecutionResult:
        """Execute a Circuit and return an ExecutionResult."""
        if self.optimize:
            from simulator.circuit_optimizer import CircuitOptimizer
            circuit = CircuitOptimizer().optimize(circuit)

        n = circuit.n_qubits
        n_clbits = circuit.n_clbits
        clbits: List[int] = [0] * max(n_clbits, 1)

        if self.mode == "statevector":
            return self._run_sv(circuit, clbits, init_label)
        else:
            return self._run_dm(circuit, clbits, init_label)

    # ------------------------------------------------------------------
    # Statevector path
    # ------------------------------------------------------------------

    def _run_sv(self, circuit: Circuit, clbits: List[int],
                init_label: Optional[str]) -> ExecutionResult:
        n = circuit.n_qubits
        if circuit.initial_state is not None:
            sv = StateVector.from_array(circuit.initial_state)
        else:
            sv = StateVector(n)
        steps: List[Dict] = []
        step_assignments: List[int] = []

        # Step 0: initial state
        step_0 = self._sv_snapshot(sv, 0, init_label or f"Initialize |{'0'*n}⟩",
                                   gate=None, qubits=[])
        steps.append(step_0)

        step_counter = 1
        for op in circuit.ops:
            if isinstance(op, BarrierOp):
                continue  # no-op in simulation

            if isinstance(op, ClassicalControlOp):
                if clbits[op.clbit] == op.value:
                    self._apply_sv_op(sv, op.op)
                    step = self._sv_snapshot(sv, step_counter, op.label,
                                             gate=_op_name(op.op),
                                             qubits=_op_qubit_list(op.op, n))
                else:
                    # Condition not met — record a no-op step
                    step = self._sv_snapshot(sv, step_counter,
                                             f"{op.label} (skipped)",
                                             gate=None, qubits=[])
                steps.append(step)
                step_assignments.append(step_counter)
                step_counter += 1
                continue

            if isinstance(op, PhaseOracleOp):
                target_idx = int(op.target_state, 2)
                sv._state[target_idx] *= -1
                step = self._sv_snapshot(sv, step_counter, op.label,
                                         gate="Oracle", qubits=list(range(n)))
                steps.append(step)
                step_assignments.append(step_counter)
                step_counter += 1
                continue

            if isinstance(op, DiffusionOp):
                mean_amp = np.mean(sv._state)
                sv._state = 2 * mean_amp - sv._state
                step = self._sv_snapshot(sv, step_counter, op.label,
                                         gate="Diffusion", qubits=list(range(n)))
                steps.append(step)
                step_assignments.append(step_counter)
                step_counter += 1
                continue

            if isinstance(op, MeasureOp):
                # Project and collapse
                p0, p1 = self._sv_measure_probs(sv, op.qubit)
                outcome = int(self._rng.choice([0, 1], p=[p0, 1.0 - p0]))
                sv = self._sv_collapse(sv, op.qubit, outcome)
                clbits[op.clbit] = outcome
                step = self._sv_snapshot(sv, step_counter,
                                         f"{op.label} → {outcome}",
                                         gate="Measure", qubits=[op.qubit])
                steps.append(step)
                step_assignments.append(step_counter)
                step_counter += 1
                continue

            if isinstance(op, ResetOp):
                # Measure and conditionally flip
                p0, p1 = self._sv_measure_probs(sv, op.qubit)
                outcome = int(self._rng.choice([0, 1], p=[p0, 1.0 - p0]))
                sv = self._sv_collapse(sv, op.qubit, outcome)
                if outcome == 1:
                    from simulator import gates as G
                    sv.apply_single_qubit_gate(G.X, op.qubit)
                step = self._sv_snapshot(sv, step_counter, op.label,
                                         gate="Reset", qubits=[op.qubit])
                steps.append(step)
                step_assignments.append(step_counter)
                step_counter += 1
                continue

            # Standard gate op
            self._apply_sv_op(sv, op)
            step = self._sv_snapshot(sv, step_counter, op.label,
                                     gate=_op_name(op), qubits=_op_qubit_list(op, n))
            steps.append(step)
            step_assignments.append(step_counter)
            step_counter += 1

        # Rebuild step_assignments for auto_layout
        # step_assignments[i] = step_counter value when op i was executed
        # We need to re-derive this from ops (skipping barriers)
        circuit_layout = self._build_layout(circuit, steps)

        return ExecutionResult(
            steps=steps,
            circuit_layout=circuit_layout,
            n_qubits=circuit.n_qubits,
            clbits=clbits,
        )

    def _apply_sv_op(self, sv: StateVector, op) -> None:
        if isinstance(op, GateOp):
            sv.apply_single_qubit_gate(op.matrix, op.qubit)
        elif isinstance(op, TwoQubitGateOp):
            sv.apply_two_qubit_gate(op.matrix, op.qubit1, op.qubit2)
        elif isinstance(op, ThreeQubitGateOp):
            sv.apply_three_qubit_gate(op.matrix, op.qubit1, op.qubit2, op.qubit3)

    def _sv_snapshot(self, sv: StateVector, step_index: int, label: str,
                     gate: Optional[str], qubits: List[int]) -> Dict:
        cfg = self.snapshot_config
        step: Dict[str, Any] = {
            "step_index": step_index,
            "label": label,
            "gate": gate,
            "qubits_affected": qubits,
            "probabilities": sv.probabilities_list(),
            "basis_labels": sv.basis_labels(),
        }
        if cfg.include_state_vector:
            step["state_vector"] = {
                "real": sv.state_real(),
                "imag": sv.state_imag(),
            }
        else:
            step["state_vector"] = {"real": [], "imag": []}
        if cfg.include_observables and (
            step_index % cfg.observable_interval == 0
            or step_index == 0
        ):
            step["observables"] = compute_observables_from_sv(sv)
        return step

    def _sv_measure_probs(self, sv: StateVector, qubit: int):
        """Return (p0, p1) for a qubit measurement without collapsing."""
        n = sv.n_qubits
        state = sv._state.reshape([2] * n)
        # Sum over all basis states where qubit==0
        idx_0 = [slice(None)] * n
        idx_0[qubit] = 0
        p0 = float(np.sum(np.abs(state[tuple(idx_0)]) ** 2))
        return p0, 1.0 - p0

    def _sv_collapse(self, sv: StateVector, qubit: int, outcome: int) -> StateVector:
        """Collapse and renormalize after measuring qubit = outcome."""
        n = sv.n_qubits
        state = sv._state.reshape([2] * n).copy()
        idx_zero = [slice(None)] * n
        idx_zero[qubit] = 1 - outcome  # kill the other outcome
        state[tuple(idx_zero)] = 0.0
        flat = state.reshape(-1)
        norm = np.linalg.norm(flat)
        if norm > 1e-12:
            flat = flat / norm
        new_sv = StateVector.from_array(flat)
        return new_sv

    # ------------------------------------------------------------------
    # Density-matrix path
    # ------------------------------------------------------------------

    def _run_dm(self, circuit: Circuit, clbits: List[int],
                init_label: Optional[str]) -> ExecutionResult:
        n = circuit.n_qubits
        if circuit.initial_state is not None:
            dm = DensityMatrix.from_statevector(circuit.initial_state)
        else:
            dm = DensityMatrix(n)
        steps: List[Dict] = []

        # Step 0: initial state
        step_0 = self._dm_snapshot(dm, 0, init_label or f"Initialize |{'0'*n}⟩",
                                   gate=None, qubits=[])
        steps.append(step_0)

        step_counter = 1
        for op in circuit.ops:
            if isinstance(op, BarrierOp):
                continue

            if isinstance(op, ClassicalControlOp):
                if clbits[op.clbit] == op.value:
                    self._apply_dm_op(dm, op.op)
                    self._maybe_apply_noise(dm, op.op)
                    step = self._dm_snapshot(dm, step_counter, op.label,
                                             gate=_op_name(op.op),
                                             qubits=_op_qubit_list(op.op, n))
                else:
                    step = self._dm_snapshot(dm, step_counter,
                                             f"{op.label} (skipped)",
                                             gate=None, qubits=[])
                steps.append(step)
                step_counter += 1
                continue

            if isinstance(op, PhaseOracleOp):
                # Phase oracle as unitary: flip sign of |target⟩⟨target| component
                target_idx = int(op.target_state, 2)
                # Build oracle unitary: I - 2|target⟩⟨target|
                U = np.eye(dm.dim, dtype=np.complex128)
                U[target_idx, target_idx] = -1.0
                dm._rho = U @ dm._rho @ U.conj().T
                step = self._dm_snapshot(dm, step_counter, op.label,
                                         gate="Oracle", qubits=list(range(n)))
                steps.append(step)
                step_counter += 1
                continue

            if isinstance(op, DiffusionOp):
                # Diffusion: 2|ψ⟩⟨ψ| - I where |ψ⟩ = H^⊗n|0⟩
                dim = 2 ** n
                psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
                D = 2 * np.outer(psi, psi.conj()) - np.eye(dim, dtype=np.complex128)
                dm._rho = D @ dm._rho @ D.conj().T
                step = self._dm_snapshot(dm, step_counter, op.label,
                                         gate="Diffusion", qubits=list(range(n)))
                steps.append(step)
                step_counter += 1
                continue

            if isinstance(op, MeasureOp):
                outcome = dm.measure_and_collapse(op.qubit, self._rng)
                clbits[op.clbit] = outcome
                step = self._dm_snapshot(dm, step_counter,
                                         f"{op.label} → {outcome}",
                                         gate="Measure", qubits=[op.qubit])
                steps.append(step)
                step_counter += 1
                continue

            if isinstance(op, ResetOp):
                outcome = dm.measure_and_collapse(op.qubit, self._rng)
                if outcome == 1:
                    from simulator import gates as G
                    dm.apply_single_qubit_gate(G.X, op.qubit)
                step = self._dm_snapshot(dm, step_counter, op.label,
                                         gate="Reset", qubits=[op.qubit])
                steps.append(step)
                step_counter += 1
                continue

            # Standard gate op
            self._apply_dm_op(dm, op)
            self._maybe_apply_noise(dm, op)
            step = self._dm_snapshot(dm, step_counter, op.label,
                                     gate=_op_name(op), qubits=_op_qubit_list(op, n))
            steps.append(step)
            step_counter += 1

        circuit_layout = self._build_layout(circuit, steps)

        return ExecutionResult(
            steps=steps,
            circuit_layout=circuit_layout,
            n_qubits=circuit.n_qubits,
            clbits=clbits,
        )

    def _apply_dm_op(self, dm: DensityMatrix, op) -> None:
        if isinstance(op, GateOp):
            dm.apply_single_qubit_gate(op.matrix, op.qubit)
        elif isinstance(op, TwoQubitGateOp):
            dm.apply_two_qubit_gate(op.matrix, op.qubit1, op.qubit2)
        elif isinstance(op, ThreeQubitGateOp):
            dm.apply_three_qubit_gate(op.matrix, op.qubit1, op.qubit2, op.qubit3)

    def _maybe_apply_noise(self, dm: DensityMatrix, op) -> None:
        if self.noise_model is None or not self.noise_model.is_noisy():
            return
        gate_name = _op_name(op)
        if gate_name:
            kraus = self.noise_model.kraus_for_gate(gate_name)
            qubits = _op_qubit_list_raw(op)
            if kraus:
                for q in qubits:
                    dm.apply_kraus(kraus, q)
            # Qubit-specific noise
            for q in qubits:
                q_kraus = self.noise_model.kraus_for_qubit(q)
                if q_kraus:
                    dm.apply_kraus(q_kraus, q)

    def _dm_snapshot(self, dm: DensityMatrix, step_index: int, label: str,
                     gate: Optional[str], qubits: List[int]) -> Dict:
        cfg = self.snapshot_config
        step: Dict[str, Any] = {
            "step_index": step_index,
            "label": label,
            "gate": gate,
            "qubits_affected": qubits,
            "probabilities": dm.probabilities(),
            "basis_labels": dm.basis_labels(),
        }
        if cfg.include_state_vector:
            step["state_vector"] = {
                "real": dm.probabilities(),   # use probs for SV compat
                "imag": [0.0] * dm.dim,
            }
        else:
            step["state_vector"] = {"real": [], "imag": []}
        if cfg.include_observables and (
            step_index % cfg.observable_interval == 0
            or step_index == 0
        ):
            step["observables"] = compute_observables_from_dm(dm)
        return step

    # ------------------------------------------------------------------
    # Layout builder
    # ------------------------------------------------------------------

    def _build_layout(self, circuit: Circuit, steps: List[Dict]) -> Dict:
        """
        Use circuit.auto_layout() with step_index assignments derived from executed steps.

        We skip BarrierOps when assigning step indices (they produce no steps).
        """
        non_barrier_ops = [op for op in circuit.ops if not isinstance(op, BarrierOp)]
        # steps[0] is init; steps[1..] correspond to ops in order (skipping barriers)
        if len(non_barrier_ops) != len(steps) - 1:
            # Fallback: auto_layout without explicit step assignments
            return circuit.auto_layout()

        step_assignments = [steps[i + 1]["step_index"] for i in range(len(non_barrier_ops))]

        # We need to rebuild with barriers included but mapped to dummy steps
        full_assignments: List[int] = []
        non_barrier_idx = 0
        for op in circuit.ops:
            if isinstance(op, BarrierOp):
                full_assignments.append(-1)  # not rendered
            else:
                full_assignments.append(step_assignments[non_barrier_idx])
                non_barrier_idx += 1

        # Build layout using a version of auto_layout that handles per-op step indices
        return _layout_from_ops(circuit, full_assignments)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _op_name(op) -> Optional[str]:
    if isinstance(op, GateOp):
        return op.name
    if isinstance(op, TwoQubitGateOp):
        return op.name
    if isinstance(op, ThreeQubitGateOp):
        return op.name
    if isinstance(op, MeasureOp):
        return "Measure"
    if isinstance(op, ResetOp):
        return "Reset"
    if isinstance(op, ClassicalControlOp):
        return _op_name(op.op)
    if isinstance(op, PhaseOracleOp):
        return "Oracle"
    if isinstance(op, DiffusionOp):
        return "Diffusion"
    return None


def _op_qubit_list(op, n_qubits: int) -> List[int]:
    if isinstance(op, GateOp):
        return [op.qubit]
    if isinstance(op, TwoQubitGateOp):
        return [op.qubit1, op.qubit2]
    if isinstance(op, ThreeQubitGateOp):
        return [op.qubit1, op.qubit2, op.qubit3]
    if isinstance(op, MeasureOp):
        return [op.qubit]
    if isinstance(op, ResetOp):
        return [op.qubit]
    if isinstance(op, (PhaseOracleOp, DiffusionOp)):
        return list(range(n_qubits))
    if isinstance(op, ClassicalControlOp):
        return _op_qubit_list(op.op, n_qubits)
    return []


def _op_qubit_list_raw(op) -> List[int]:
    if isinstance(op, GateOp):
        return [op.qubit]
    if isinstance(op, TwoQubitGateOp):
        return [op.qubit1, op.qubit2]
    if isinstance(op, ThreeQubitGateOp):
        return [op.qubit1, op.qubit2, op.qubit3]
    if isinstance(op, ClassicalControlOp):
        return _op_qubit_list_raw(op.op)
    return []


def _layout_from_ops(circuit: Circuit, step_assignments: List[int]) -> Dict:
    """
    Build circuit_layout from ops + step_assignments.
    Greedy column packing, ignoring ops with step_index == -1.
    """
    from simulator.circuit import _op_circuit_gates, _op_qubits, BarrierOp, PhaseOracleOp, DiffusionOp

    n = circuit.n_qubits
    columns: List[Dict] = []
    qubit_last_col: Dict[int, int] = {}

    for op_idx, (op, step_idx) in enumerate(zip(circuit.ops, step_assignments)):
        if step_idx == -1 or isinstance(op, BarrierOp):
            continue

        # For PhaseOracleOp / DiffusionOp, occupy all qubits
        if isinstance(op, (PhaseOracleOp, DiffusionOp)):
            occupied = list(range(n))
        else:
            occupied = _op_qubits(op)

        if occupied:
            min_col = max((qubit_last_col.get(q, -1) for q in occupied), default=-1) + 1
        else:
            min_col = max(qubit_last_col.values(), default=-1) + 1

        existing = next((c for c in columns if c["column_index"] == min_col), None)
        if existing is None:
            existing = {"column_index": min_col, "gates": []}
            columns.append(existing)
            columns.sort(key=lambda c: c["column_index"])

        # Build gate entries for this op
        if isinstance(op, PhaseOracleOp):
            gate_entries = [{"qubit": q, "name": "Oracle", "step_index": step_idx}
                            for q in range(n)]
        elif isinstance(op, DiffusionOp):
            gate_entries = [{"qubit": q, "name": "Diffusion", "step_index": step_idx}
                            for q in range(n)]
        else:
            gate_entries = _op_circuit_gates(op, step_idx)

        existing["gates"].extend(gate_entries)
        for q in occupied:
            qubit_last_col[q] = min_col

    return {
        "qubit_labels": [f"q{i}" for i in range(n)],
        "columns": sorted(columns, key=lambda c: c["column_index"]),
    }
