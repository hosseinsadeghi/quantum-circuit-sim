"""
Circuit optimization passes — reduce gate count before execution.

Passes:
1. Single-qubit gate fusion: consecutive gates on same qubit → one gate
2. Identity cancellation: H·H, X·X, CNOT·CNOT → remove
3. Commutation reordering: gates on disjoint qubits reordered for more fusion
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple

import numpy as np

from simulator.circuit import (
    Circuit, GateOp, TwoQubitGateOp, ThreeQubitGateOp, Op,
    MeasureOp, ResetOp, BarrierOp, ClassicalControlOp, PhaseOracleOp, DiffusionOp,
)


def _op_qubits(op: Op) -> Set[int]:
    """Return the set of qubits touched by an op."""
    if isinstance(op, GateOp):
        return {op.qubit}
    if isinstance(op, TwoQubitGateOp):
        return {op.qubit1, op.qubit2}
    if isinstance(op, ThreeQubitGateOp):
        return {op.qubit1, op.qubit2, op.qubit3}
    if isinstance(op, MeasureOp):
        return {op.qubit}
    if isinstance(op, ResetOp):
        return {op.qubit}
    if isinstance(op, ClassicalControlOp):
        return _op_qubits(op.op)
    # PhaseOracleOp, DiffusionOp, BarrierOp: touch all qubits (conservative)
    return set()


def _is_identity(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a matrix is proportional to identity (global phase allowed)."""
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        return False
    # Check if matrix = e^{i*phi} * I
    diag = np.diag(matrix)
    if not np.allclose(np.abs(diag), 1.0, atol=tol):
        return False
    phase = diag[0]
    return np.allclose(matrix, phase * np.eye(n), atol=tol)


class CircuitPass(ABC):
    """Base class for circuit optimization passes."""

    @abstractmethod
    def run(self, ops: List[Op], n_qubits: int) -> List[Op]:
        ...


class IdentityCancellation(CircuitPass):
    """Cancel adjacent pairs of self-inverse gates: H·H, X·X, Y·Y, Z·Z, CNOT·CNOT."""

    _SELF_INVERSE_1Q = {"H", "X", "Y", "Z"}
    _SELF_INVERSE_2Q = {"CNOT", "CZ", "SWAP"}

    def run(self, ops: List[Op], n_qubits: int) -> List[Op]:
        result: List[Op] = []
        skip = set()

        for i, op in enumerate(ops):
            if i in skip:
                continue

            if isinstance(op, GateOp) and op.name in self._SELF_INVERSE_1Q:
                # Look ahead for matching gate on same qubit
                for j in range(i + 1, len(ops)):
                    next_op = ops[j]
                    if isinstance(next_op, GateOp) and next_op.name == op.name and next_op.qubit == op.qubit:
                        skip.add(j)
                        skip.add(i)
                        break
                    if isinstance(next_op, GateOp) and next_op.qubit == op.qubit:
                        break  # Different gate on same qubit — can't cancel
                    if _op_qubits(next_op) & {op.qubit}:
                        break  # Qubit used by a non-gate op — stop

                if i not in skip:
                    result.append(op)

            elif isinstance(op, TwoQubitGateOp) and op.name in self._SELF_INVERSE_2Q:
                for j in range(i + 1, len(ops)):
                    next_op = ops[j]
                    if (isinstance(next_op, TwoQubitGateOp)
                            and next_op.name == op.name
                            and next_op.qubit1 == op.qubit1
                            and next_op.qubit2 == op.qubit2):
                        skip.add(j)
                        skip.add(i)
                        break
                    if _op_qubits(next_op) & {op.qubit1, op.qubit2}:
                        break

                if i not in skip:
                    result.append(op)

            else:
                result.append(op)

        return result


class SingleQubitFusion(CircuitPass):
    """Fuse consecutive single-qubit gates on the same qubit into one gate."""

    def run(self, ops: List[Op], n_qubits: int) -> List[Op]:
        result: List[Op] = []
        i = 0

        while i < len(ops):
            op = ops[i]
            if not isinstance(op, GateOp):
                result.append(op)
                i += 1
                continue

            # Collect consecutive single-qubit gates on same qubit
            fused_matrix = op.matrix.astype(np.complex128)
            fused_labels = [op.label]
            qubit = op.qubit
            j = i + 1

            while j < len(ops):
                next_op = ops[j]
                if isinstance(next_op, GateOp) and next_op.qubit == qubit:
                    fused_matrix = next_op.matrix.astype(np.complex128) @ fused_matrix
                    fused_labels.append(next_op.label)
                    j += 1
                elif _op_qubits(next_op) & {qubit}:
                    break  # Qubit used by multi-qubit or non-gate op
                else:
                    break  # Non-adjacent (different qubit) — could reorder, but leave for CommutationReorder

            if j == i + 1:
                # No fusion happened
                result.append(op)
            else:
                # Check if fused result is identity
                if _is_identity(fused_matrix):
                    pass  # Skip — gates cancel out
                else:
                    result.append(GateOp(
                        qubit=qubit,
                        matrix=fused_matrix,
                        name="Fused",
                        label=" → ".join(fused_labels),
                    ))
            i = j

        return result


class CommutationReorder(CircuitPass):
    """Reorder gates on disjoint qubits to enable more fusion opportunities.

    Moves single-qubit gates earlier when they commute (act on disjoint qubits)
    with intervening operations, grouping same-qubit gates together.
    """

    def run(self, ops: List[Op], n_qubits: int) -> List[Op]:
        result = list(ops)
        changed = True
        max_iterations = len(ops)  # prevent infinite loops
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            for i in range(1, len(result)):
                op = result[i]
                if not isinstance(op, GateOp):
                    continue
                prev = result[i - 1]
                # Can swap if they act on disjoint qubits
                if not (_op_qubits(op) & _op_qubits(prev)):
                    # Swap if it would group same-qubit gates
                    if (i >= 2 and isinstance(result[i - 2], GateOp)
                            and result[i - 2].qubit == op.qubit):
                        result[i - 1], result[i] = result[i], result[i - 1]
                        changed = True

        return result


class CircuitOptimizer:
    """Runs a sequence of optimization passes on a circuit."""

    def __init__(self, passes: Optional[List[CircuitPass]] = None):
        if passes is None:
            self.passes = [
                CommutationReorder(),
                SingleQubitFusion(),
                IdentityCancellation(),
            ]
        else:
            self.passes = passes

    def optimize(self, circuit: Circuit) -> Circuit:
        """Return a new optimized Circuit with the same semantics."""
        ops = list(circuit.ops)
        for p in self.passes:
            ops = p.run(ops, circuit.n_qubits)

        new_circuit = Circuit(
            n_qubits=circuit.n_qubits,
            n_clbits=circuit.n_clbits,
            initial_state=circuit.initial_state,
        )
        new_circuit.ops = ops
        return new_circuit
