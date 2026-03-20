"""
Sparse state vector representation for circuits that remain sparse.

Stores only non-zero amplitudes as a Dict[int, complex]. Supports the same
interface as StateVector (apply gates, probabilities, etc.). Automatically
falls back to dense representation when sparsity exceeds a threshold.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


_SPARSITY_THRESHOLD = 0.25  # Switch to dense when >25% of amplitudes are non-zero
_ZERO_TOL = 1e-15


class SparseStateVector:
    """Sparse n-qubit quantum state vector.

    Stores amplitudes as {basis_index: complex} dict.
    Falls back to dense when sparsity exceeds threshold.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self._amps: Dict[int, complex] = {0: 1.0 + 0j}  # |0...0>
        self._dense: Optional[np.ndarray] = None  # set when we fall back

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SparseStateVector":
        n_qubits = int(np.log2(len(arr)))
        sv = cls(n_qubits)
        sv._amps = {}
        for i, a in enumerate(arr):
            if abs(a) > _ZERO_TOL:
                sv._amps[i] = complex(a)
        sv._maybe_densify()
        return sv

    @property
    def is_dense(self) -> bool:
        return self._dense is not None

    @property
    def sparsity(self) -> float:
        """Fraction of non-zero amplitudes."""
        if self.is_dense:
            return 1.0
        return len(self._amps) / self.dim

    def _maybe_densify(self) -> None:
        """Switch to dense representation if too many non-zero entries."""
        if self._dense is not None:
            return
        if len(self._amps) > _SPARSITY_THRESHOLD * self.dim:
            self._densify()

    def _densify(self) -> None:
        """Convert sparse to dense array."""
        self._dense = np.zeros(self.dim, dtype=np.complex128)
        for idx, amp in self._amps.items():
            self._dense[idx] = amp
        self._amps = {}

    def to_dense(self) -> np.ndarray:
        """Return a dense numpy array (copy)."""
        if self._dense is not None:
            return self._dense.copy()
        arr = np.zeros(self.dim, dtype=np.complex128)
        for idx, amp in self._amps.items():
            arr[idx] = amp
        return arr

    def copy(self) -> "SparseStateVector":
        sv = SparseStateVector(self.n_qubits)
        if self.is_dense:
            sv._dense = self._dense.copy()
        else:
            sv._amps = dict(self._amps)
        return sv

    def apply_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> None:
        """Apply a single-qubit gate to the specified qubit."""
        if self.is_dense:
            self._apply_dense_single(gate, qubit)
            return

        n = self.n_qubits
        new_amps: Dict[int, complex] = {}

        # Group basis states by all bits except the target qubit
        bit_pos = n - 1 - qubit
        mask = 1 << bit_pos

        # Process pairs of basis states (qubit=0, qubit=1)
        processed = set()
        for idx in list(self._amps.keys()):
            partner = idx ^ mask
            key = min(idx, partner)
            if key in processed:
                continue
            processed.add(key)

            idx0 = idx & ~mask  # qubit=0 version
            idx1 = idx0 | mask   # qubit=1 version

            a0 = self._amps.get(idx0, 0.0)
            a1 = self._amps.get(idx1, 0.0)

            # Apply 2x2 gate
            b0 = gate[0, 0] * a0 + gate[0, 1] * a1
            b1 = gate[1, 0] * a0 + gate[1, 1] * a1

            if abs(b0) > _ZERO_TOL:
                new_amps[idx0] = b0
            if abs(b1) > _ZERO_TOL:
                new_amps[idx1] = b1

        self._amps = new_amps
        self._maybe_densify()

    def _apply_dense_single(self, gate: np.ndarray, qubit: int) -> None:
        state = self._dense.reshape([2] * self.n_qubits)
        state = np.tensordot(gate, state, axes=[[1], [qubit]])
        state = np.moveaxis(state, 0, qubit)
        self._dense = state.reshape(self.dim)

    def apply_two_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int) -> None:
        """Apply a two-qubit gate. Falls back to dense for simplicity."""
        if not self.is_dense:
            self._densify()

        state = self._dense.reshape([2] * self.n_qubits)
        gate_tensor = gate.reshape(2, 2, 2, 2)
        state = np.tensordot(gate_tensor, state, axes=[[2, 3], [qubit1, qubit2]])

        remaining = [i for i in range(self.n_qubits) if i not in (qubit1, qubit2)]
        target = [None] * self.n_qubits
        target[qubit1] = 0
        target[qubit2] = 1
        for j, r in enumerate(remaining):
            target[r] = j + 2
        inv = [0] * self.n_qubits
        for i, pos in enumerate(target):
            inv[pos] = i
        state = np.transpose(state, inv)
        self._dense = state.reshape(self.dim)

    def apply_three_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int, qubit3: int) -> None:
        """Apply a three-qubit gate. Falls back to dense."""
        if not self.is_dense:
            self._densify()

        state = self._dense.reshape([2] * self.n_qubits)
        gate_tensor = gate.reshape(2, 2, 2, 2, 2, 2)
        state = np.tensordot(gate_tensor, state, axes=[[3, 4, 5], [qubit1, qubit2, qubit3]])

        remaining = [i for i in range(self.n_qubits) if i not in (qubit1, qubit2, qubit3)]
        target = [None] * self.n_qubits
        target[qubit1] = 0
        target[qubit2] = 1
        target[qubit3] = 2
        for j, r in enumerate(remaining):
            target[r] = j + 3
        inv = [0] * self.n_qubits
        for i, pos in enumerate(target):
            inv[pos] = i
        state = np.transpose(state, inv)
        self._dense = state.reshape(self.dim)

    def probabilities(self) -> np.ndarray:
        if self.is_dense:
            return (np.abs(self._dense) ** 2).real
        probs = np.zeros(self.dim)
        for idx, amp in self._amps.items():
            probs[idx] = abs(amp) ** 2
        return probs

    def basis_labels(self) -> List[str]:
        return [f"|{i:0{self.n_qubits}b}>" for i in range(self.dim)]

    def state_real(self) -> List[float]:
        return self.to_dense().real.tolist()

    def state_imag(self) -> List[float]:
        return self.to_dense().imag.tolist()

    def probabilities_list(self) -> List[float]:
        return self.probabilities().tolist()

    @property
    def _state(self) -> np.ndarray:
        """Compatibility property for code that accesses _state directly."""
        return self.to_dense()
