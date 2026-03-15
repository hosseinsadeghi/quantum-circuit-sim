"""
Circuit IR — algorithms emit operations into a Circuit; Executor runs them.

Operations are dataclasses stored in Circuit.ops. The circuit also carries
metadata needed for circuit_layout auto-generation.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from simulator import gates as G


# ---------------------------------------------------------------------------
# Operation types
# ---------------------------------------------------------------------------

@dataclass
class GateOp:
    """Single-qubit gate."""
    qubit: int
    matrix: np.ndarray
    name: str
    label: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TwoQubitGateOp:
    """Two-qubit gate."""
    qubit1: int
    qubit2: int
    matrix: np.ndarray
    name: str
    label: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasureOp:
    """Mid-circuit measurement of qubit → classical bit."""
    qubit: int
    clbit: int
    label: str


@dataclass
class ResetOp:
    """Reset qubit to |0⟩."""
    qubit: int
    label: str


@dataclass
class BarrierOp:
    """Visual barrier — no physical effect."""
    qubits: List[int]
    label: str


@dataclass
class ClassicalControlOp:
    """Apply `op` only if clbits[clbit] == value."""
    clbit: int
    value: int
    op: Any  # GateOp | TwoQubitGateOp
    label: str


@dataclass
class PhaseOracleOp:
    """Grover phase oracle — flips sign of target_state amplitude."""
    target_state: str
    label: str


@dataclass
class DiffusionOp:
    """Grover diffusion operator — inversion about the mean."""
    n_qubits: int
    label: str


# Union type alias for type hints
Op = (
    GateOp
    | TwoQubitGateOp
    | MeasureOp
    | ResetOp
    | BarrierOp
    | ClassicalControlOp
    | PhaseOracleOp
    | DiffusionOp
)


# ---------------------------------------------------------------------------
# Circuit container
# ---------------------------------------------------------------------------

class Circuit:
    """
    Quantum circuit IR.

    Algorithms append operations via fluent helper methods.
    The Executor iterates `circuit.ops` to drive simulation.
    `auto_layout()` produces a circuit_layout dict compatible with the
    frontend CircuitDiagram component.
    """

    def __init__(self, n_qubits: int, n_clbits: int = 0):
        self.n_qubits = n_qubits
        self.n_clbits = n_clbits
        self.ops: List[Op] = []

    # ------------------------------------------------------------------
    # Single-qubit gates
    # ------------------------------------------------------------------

    def _single(self, matrix: np.ndarray, name: str, qubit: int,
                label: Optional[str] = None, params: Optional[Dict] = None) -> "Circuit":
        self.ops.append(GateOp(
            qubit=qubit, matrix=matrix, name=name,
            label=label or f"{name} q{qubit}",
            params=params or {},
        ))
        return self

    def h(self, qubit: int, label: Optional[str] = None) -> "Circuit":
        return self._single(G.H, "H", qubit, label)

    def x(self, qubit: int, label: Optional[str] = None) -> "Circuit":
        return self._single(G.X, "X", qubit, label)

    def y(self, qubit: int, label: Optional[str] = None) -> "Circuit":
        return self._single(G.Y, "Y", qubit, label)

    def z(self, qubit: int, label: Optional[str] = None) -> "Circuit":
        return self._single(G.Z, "Z", qubit, label)

    def s(self, qubit: int, label: Optional[str] = None) -> "Circuit":
        return self._single(G.S, "S", qubit, label)

    def t(self, qubit: int, label: Optional[str] = None) -> "Circuit":
        return self._single(G.T, "T", qubit, label)

    def rx(self, theta: float, qubit: int, label: Optional[str] = None) -> "Circuit":
        return self._single(G.Rx(theta), "Rx", qubit, label, {"theta": theta})

    def ry(self, theta: float, qubit: int, label: Optional[str] = None) -> "Circuit":
        return self._single(G.Ry(theta), "Ry", qubit, label, {"theta": theta})

    def rz(self, theta: float, qubit: int, label: Optional[str] = None) -> "Circuit":
        return self._single(G.Rz(theta), "Rz", qubit, label, {"theta": theta})

    def phase(self, theta: float, qubit: int, label: Optional[str] = None) -> "Circuit":
        return self._single(G.phase(theta), "P", qubit, label, {"theta": theta})

    # ------------------------------------------------------------------
    # Two-qubit gates
    # ------------------------------------------------------------------

    def _two(self, matrix: np.ndarray, name: str, qubit1: int, qubit2: int,
             label: Optional[str] = None, params: Optional[Dict] = None) -> "Circuit":
        self.ops.append(TwoQubitGateOp(
            qubit1=qubit1, qubit2=qubit2, matrix=matrix, name=name,
            label=label or f"{name} q{qubit1},q{qubit2}",
            params=params or {},
        ))
        return self

    def cnot(self, control: int, target: int, label: Optional[str] = None) -> "Circuit":
        return self._two(G.CNOT, "CNOT", control, target, label)

    def cz(self, qubit1: int, qubit2: int, label: Optional[str] = None) -> "Circuit":
        return self._two(G.CZ, "CZ", qubit1, qubit2, label)

    def swap(self, qubit1: int, qubit2: int, label: Optional[str] = None) -> "Circuit":
        return self._two(G.SWAP, "SWAP", qubit1, qubit2, label)

    # ------------------------------------------------------------------
    # Classical / special ops
    # ------------------------------------------------------------------

    def measure(self, qubit: int, clbit: int, label: Optional[str] = None) -> "Circuit":
        self.ops.append(MeasureOp(
            qubit=qubit, clbit=clbit,
            label=label or f"Measure q{qubit}→c{clbit}",
        ))
        return self

    def reset(self, qubit: int, label: Optional[str] = None) -> "Circuit":
        self.ops.append(ResetOp(qubit=qubit, label=label or f"Reset q{qubit}"))
        return self

    def barrier(self, *qubits: int, label: str = "Barrier") -> "Circuit":
        q_list = list(qubits) if qubits else list(range(self.n_qubits))
        self.ops.append(BarrierOp(qubits=q_list, label=label))
        return self

    def classical_control(self, clbit: int, value: int, op: Op,
                          label: Optional[str] = None) -> "Circuit":
        self.ops.append(ClassicalControlOp(
            clbit=clbit, value=value, op=op,
            label=label or f"if c{clbit}=={value}: {op.name}",
        ))
        return self

    def phase_oracle(self, target_state: str, label: Optional[str] = None) -> "Circuit":
        self.ops.append(PhaseOracleOp(
            target_state=target_state,
            label=label or f"Oracle |{target_state}⟩",
        ))
        return self

    def diffusion(self, label: Optional[str] = None) -> "Circuit":
        self.ops.append(DiffusionOp(
            n_qubits=self.n_qubits,
            label=label or "Diffusion",
        ))
        return self

    # ------------------------------------------------------------------
    # Circuit layout auto-generation
    # ------------------------------------------------------------------

    def auto_layout(self, step_assignments: Optional[List[int]] = None) -> Dict:
        """
        Greedy column packing: each op goes in the earliest column where
        none of its qubits are already occupied.

        step_assignments[i] = step_index that op i maps to in the trace.
        If None, step_assignments[i] = i + 1 (assumes step 0 is init).

        Returns a circuit_layout dict for the frontend.
        """
        if step_assignments is None:
            step_assignments = list(range(1, len(self.ops) + 1))

        columns: List[Dict] = []
        # qubit_last_col[q] = last column index that used qubit q
        qubit_last_col: Dict[int, int] = {}

        for op_idx, op in enumerate(self.ops):
            step_idx = step_assignments[op_idx]
            occupied_qubits = _op_qubits(op)

            if not occupied_qubits:
                # Barrier / no-qubit ops: put in own column, don't block
                col_idx = max(qubit_last_col.values(), default=-1) + 1
                columns.append({
                    "column_index": col_idx,
                    "gates": _op_circuit_gates(op, step_idx),
                })
                continue

            # Earliest column where all required qubits are free
            min_col = max((qubit_last_col.get(q, -1) for q in occupied_qubits), default=-1) + 1

            # Find if there's already a column at min_col with no conflicts
            # (greedy: just use min_col, create new column entry if needed)
            # Find existing column at min_col
            existing = next((c for c in columns if c["column_index"] == min_col), None)
            if existing is None:
                existing = {"column_index": min_col, "gates": []}
                columns.append(existing)
                columns.sort(key=lambda c: c["column_index"])

            existing["gates"].extend(_op_circuit_gates(op, step_idx))
            for q in occupied_qubits:
                qubit_last_col[q] = min_col

        return {
            "qubit_labels": [f"q{i}" for i in range(self.n_qubits)],
            "columns": sorted(columns, key=lambda c: c["column_index"]),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _op_qubits(op: Op) -> List[int]:
    if isinstance(op, GateOp):
        return [op.qubit]
    if isinstance(op, TwoQubitGateOp):
        return [op.qubit1, op.qubit2]
    if isinstance(op, MeasureOp):
        return [op.qubit]
    if isinstance(op, ResetOp):
        return [op.qubit]
    if isinstance(op, BarrierOp):
        return op.qubits
    if isinstance(op, ClassicalControlOp):
        return _op_qubits(op.op)
    if isinstance(op, PhaseOracleOp):
        # Affects all qubits (n_qubits unknown here — caller handles)
        return []
    if isinstance(op, DiffusionOp):
        return []
    return []


def _op_circuit_gates(op: Op, step_idx: int) -> List[Dict]:
    """Return circuit_layout gate entries for this op."""
    if isinstance(op, GateOp):
        return [{"qubit": op.qubit, "name": op.name, "step_index": step_idx}]
    if isinstance(op, TwoQubitGateOp):
        if op.name in ("CNOT",):
            return [
                {"qubit": op.qubit1, "name": "CNOT_ctrl", "step_index": step_idx},
                {"qubit": op.qubit2, "name": "CNOT_tgt", "step_index": step_idx},
            ]
        if op.name in ("CZ",):
            return [
                {"qubit": op.qubit1, "name": "CZ_ctrl", "step_index": step_idx},
                {"qubit": op.qubit2, "name": "CZ_tgt", "step_index": step_idx},
            ]
        return [
            {"qubit": op.qubit1, "name": f"{op.name}_1", "step_index": step_idx},
            {"qubit": op.qubit2, "name": f"{op.name}_2", "step_index": step_idx},
        ]
    if isinstance(op, MeasureOp):
        return [{"qubit": op.qubit, "name": "Measure", "step_index": step_idx}]
    if isinstance(op, ResetOp):
        return [{"qubit": op.qubit, "name": "Reset", "step_index": step_idx}]
    if isinstance(op, BarrierOp):
        return [{"qubit": q, "name": "Barrier", "step_index": step_idx} for q in op.qubits]
    if isinstance(op, ClassicalControlOp):
        inner = _op_circuit_gates(op.op, step_idx)
        for g in inner:
            g["classical_control"] = {"clbit": op.clbit, "value": op.value}
        return inner
    if isinstance(op, PhaseOracleOp):
        return []  # caller fills in per-qubit oracle gates
    if isinstance(op, DiffusionOp):
        return []  # caller fills in per-qubit diffusion gates
    return []
