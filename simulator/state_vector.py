import numpy as np
from typing import List


class StateVector:
    """Represents an n-qubit quantum state vector."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self._state = np.zeros(self.dim, dtype=np.complex128)
        self._state[0] = 1.0  # |000...0>

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "StateVector":
        n_qubits = int(np.log2(len(arr)))
        sv = cls(n_qubits)
        sv._state = arr.astype(np.complex128).copy()
        return sv

    def copy(self) -> "StateVector":
        sv = StateVector(self.n_qubits)
        sv._state = self._state.copy()
        return sv

    def apply_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> None:
        """Apply a single-qubit gate to the specified qubit (0-indexed, MSB first)."""
        state = self._state.reshape([2] * self.n_qubits)
        state = np.tensordot(gate, state, axes=[[1], [qubit]])
        state = np.moveaxis(state, 0, qubit)
        self._state = state.reshape(self.dim)

    def apply_two_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int) -> None:
        """Apply a two-qubit gate. gate is 4x4, qubit1=control/first, qubit2=target/second."""
        state = self._state.reshape([2] * self.n_qubits)
        gate_tensor = gate.reshape(2, 2, 2, 2)
        state = np.tensordot(gate_tensor, state, axes=[[2, 3], [qubit1, qubit2]])
        # After tensordot: axes [out1, out2, remaining...]
        remaining = [i for i in range(self.n_qubits) if i not in (qubit1, qubit2)]
        # Build inverse permutation to put axes back
        target_axes = [None] * self.n_qubits
        target_axes[qubit1] = 0
        target_axes[qubit2] = 1
        for j, r in enumerate(remaining):
            target_axes[r] = j + 2
        inverse_perm = [0] * self.n_qubits
        for i, pos in enumerate(target_axes):
            inverse_perm[pos] = i
        state = np.transpose(state, inverse_perm)
        self._state = state.reshape(self.dim)

    def probabilities(self) -> np.ndarray:
        return (np.abs(self._state) ** 2).real

    def basis_labels(self) -> List[str]:
        return [f"|{i:0{self.n_qubits}b}>" for i in range(self.dim)]

    def state_real(self) -> List[float]:
        return self._state.real.tolist()

    def state_imag(self) -> List[float]:
        return self._state.imag.tolist()

    def probabilities_list(self) -> List[float]:
        return self.probabilities().tolist()
