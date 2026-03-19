"""
Density-matrix quantum state representation.

Supports:
- Unitary evolution: U rho U† via tensor contraction (O(2^2n) per gate)
- Noise via Kraus operators: rho -> sum_k K_k rho K_k†
- Mid-circuit measurement with state collapse
- Partial trace
- Bloch vector per qubit
- Entanglement entropy (von Neumann)
- Purity Tr(rho^2)
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


class DensityMatrix:
    """n-qubit density matrix, stored as (2^n x 2^n) complex128 array."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self._rho = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self._rho[0, 0] = 1.0  # |0...0><0...0|

    @classmethod
    def from_statevector(cls, psi: np.ndarray) -> "DensityMatrix":
        """Build rho = |psi><psi| from a state vector."""
        n_qubits = int(np.round(np.log2(len(psi))))
        dm = cls(n_qubits)
        psi = psi.astype(np.complex128)
        dm._rho = np.outer(psi, psi.conj())
        return dm

    def copy(self) -> "DensityMatrix":
        dm = DensityMatrix(self.n_qubits)
        dm._rho = self._rho.copy()
        return dm

    # ------------------------------------------------------------------
    # Unitary evolution — tensor contraction approach
    # ------------------------------------------------------------------

    def apply_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> None:
        """Apply single-qubit unitary U to qubit: rho -> U rho U†.

        Uses tensor contraction: reshape rho to (2,)*2n, contract U on the
        row axis and U* on the col axis for the target qubit. O(2^2n).
        """
        n = self.n_qubits
        U = gate.astype(np.complex128)
        rho = self._rho.reshape([2] * (2 * n))

        # Contract U with row-axis (qubit)
        rho = np.tensordot(U, rho, axes=[[1], [qubit]])
        rho = np.moveaxis(rho, 0, qubit)

        # Contract U* with col-axis (qubit + n)
        rho = np.tensordot(U.conj(), rho, axes=[[1], [qubit + n]])
        rho = np.moveaxis(rho, 0, qubit + n)

        self._rho = rho.reshape(self.dim, self.dim)

    def apply_two_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int) -> None:
        """Apply two-qubit unitary to (qubit1, qubit2) via tensor contraction. O(2^2n)."""
        n = self.n_qubits
        G4 = gate.astype(np.complex128).reshape(2, 2, 2, 2)
        rho = self._rho.reshape([2] * (2 * n))

        # Contract gate with row axes
        rho = np.tensordot(G4, rho, axes=[[2, 3], [qubit1, qubit2]])
        # After tensordot: axes [out1, out2, ...remaining row..., ...col...]
        # Move out1, out2 back to qubit1, qubit2 positions
        remaining_row = [i for i in range(n) if i not in (qubit1, qubit2)]
        target = [None] * n
        target[qubit1] = 0
        target[qubit2] = 1
        for j, r in enumerate(remaining_row):
            target[r] = j + 2
        inv = [0] * n
        for i, pos in enumerate(target):
            inv[pos] = i
        # Current shape: (2, 2, <n-2 row axes>, <n col axes>)
        # We need to reorder first n axes according to inv
        full_perm = inv + list(range(n, 2 * n))
        rho = np.transpose(rho, full_perm)

        # Contract gate* with col axes
        rho = np.tensordot(G4.conj(), rho, axes=[[2, 3], [qubit1 + n, qubit2 + n]])
        remaining_col = [i for i in range(n) if i not in (qubit1, qubit2)]
        target2 = [None] * n
        target2[qubit1] = 0
        target2[qubit2] = 1
        for j, r in enumerate(remaining_col):
            target2[r] = j + 2
        inv2 = [0] * n
        for i, pos in enumerate(target2):
            inv2[pos] = i
        full_perm2 = list(range(n)) + [x + n for x in inv2]
        rho = np.transpose(rho, full_perm2)

        self._rho = rho.reshape(self.dim, self.dim)

    def apply_three_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int, qubit3: int) -> None:
        """Apply three-qubit unitary via tensor contraction. O(2^2n)."""
        n = self.n_qubits
        G8 = gate.astype(np.complex128).reshape(2, 2, 2, 2, 2, 2)
        rho = self._rho.reshape([2] * (2 * n))

        qubits = [qubit1, qubit2, qubit3]

        # Contract gate with row axes
        rho = np.tensordot(G8, rho, axes=[[3, 4, 5], qubits])
        remaining = [i for i in range(n) if i not in qubits]
        target = [None] * n
        for idx, q in enumerate(qubits):
            target[q] = idx
        for j, r in enumerate(remaining):
            target[r] = j + 3
        inv = [0] * n
        for i, pos in enumerate(target):
            inv[pos] = i
        full_perm = inv + list(range(n, 2 * n))
        rho = np.transpose(rho, full_perm)

        # Contract gate* with col axes
        col_axes = [q + n for q in qubits]
        rho = np.tensordot(G8.conj(), rho, axes=[[3, 4, 5], col_axes])
        remaining2 = [i for i in range(n) if i not in qubits]
        target2 = [None] * n
        for idx, q in enumerate(qubits):
            target2[q] = idx
        for j, r in enumerate(remaining2):
            target2[r] = j + 3
        inv2 = [0] * n
        for i, pos in enumerate(target2):
            inv2[pos] = i
        full_perm2 = list(range(n)) + [x + n for x in inv2]
        rho = np.transpose(rho, full_perm2)

        self._rho = rho.reshape(self.dim, self.dim)

    # ------------------------------------------------------------------
    # Kraus channel (noise) — tensor contraction
    # ------------------------------------------------------------------

    def apply_kraus(self, kraus_ops: List[np.ndarray], qubit: int) -> None:
        """Apply a single-qubit Kraus channel on `qubit`.

        rho -> sum_k K_k rho K_k† using tensor contraction per operator.
        """
        n = self.n_qubits
        new_rho = np.zeros((self.dim, self.dim), dtype=np.complex128)
        saved_rho = self._rho.copy()

        for K in kraus_ops:
            self._rho = saved_rho.copy()
            self.apply_single_qubit_gate(K, qubit)
            new_rho += self._rho

        self._rho = new_rho

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure_and_collapse(self, qubit: int, rng: Optional[np.random.Generator] = None
                             ) -> int:
        """
        Measure `qubit` in the computational basis.

        Collapses the density matrix and returns the measurement outcome (0 or 1).
        Uses tensor contraction for the projectors.
        """
        if rng is None:
            rng = np.random.default_rng()

        n = self.n_qubits

        # Compute p(0) by tracing over the qubit=0 subspace
        rho_t = self._rho.reshape([2] * (2 * n))
        # p(outcome) = sum of rho[..., outcome, ..., outcome, ...] where
        # both row and col axes for this qubit are fixed to outcome
        # Then sum over all other axes = trace of projected block
        p0 = float(np.real(np.trace(
            rho_t.take(0, axis=qubit).take(0, axis=qubit + n - 1)
            .reshape(2 ** (n - 1), 2 ** (n - 1))
        )))
        p0 = max(0.0, min(1.0, p0))

        outcome = int(rng.choice([0, 1], p=[p0, 1.0 - p0]))

        # Project: zero out the other outcome
        proj = np.array([[1, 0], [0, 0]], dtype=np.complex128) if outcome == 0 \
            else np.array([[0, 0], [0, 1]], dtype=np.complex128)
        self.apply_single_qubit_gate(proj, qubit)

        # Renormalize
        prob = p0 if outcome == 0 else (1.0 - p0)
        if prob > 1e-15:
            self._rho /= prob

        return outcome

    # ------------------------------------------------------------------
    # Partial trace
    # ------------------------------------------------------------------

    def partial_trace(self, keep_qubits: List[int]) -> np.ndarray:
        """Return the reduced density matrix for `keep_qubits`.

        Returns a (2^k x 2^k) array where k = len(keep_qubits).
        """
        n = self.n_qubits
        trace_out = [i for i in range(n) if i not in keep_qubits]
        k = len(keep_qubits)

        # Reshape rho to tensor of shape (2,)*2n with axes [row_q0,...,row_qn-1, col_q0,...]
        rho_t = self._rho.reshape([2] * (2 * n))

        # Trace over each unwanted qubit
        current_rho = rho_t
        current_n = n
        for q in sorted(trace_out, reverse=True):
            current_rho = np.trace(current_rho, axis1=q, axis2=q + current_n)
            current_n -= 1

        dim_k = 2 ** k
        return current_rho.reshape(dim_k, dim_k)

    def single_qubit_dm(self, qubit: int) -> np.ndarray:
        """2x2 reduced density matrix for a single qubit."""
        return self.partial_trace([qubit])

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def bloch_vector(self, qubit: int) -> Tuple[float, float, float]:
        """(x, y, z) Bloch vector for qubit's reduced state."""
        rho_q = self.single_qubit_dm(qubit)
        x = float(2 * np.real(rho_q[0, 1]))
        y = float(2 * np.imag(rho_q[1, 0]))
        z = float(np.real(rho_q[0, 0] - rho_q[1, 1]))
        return x, y, z

    def z_expectation(self, qubit: int) -> float:
        """<Z> for a single qubit."""
        rho_q = self.single_qubit_dm(qubit)
        return float(np.real(rho_q[0, 0] - rho_q[1, 1]))

    def purity(self) -> float:
        """Tr(rho^2)."""
        return float(np.real(np.trace(self._rho @ self._rho)))

    def entanglement_entropy(self, partition: Optional[List[int]] = None) -> float:
        """Von Neumann entropy S = -Tr(rho_A log rho_A) for bipartition A.

        partition defaults to the first half of qubits.
        """
        if partition is None:
            partition = list(range(self.n_qubits // 2))
        rho_a = self.partial_trace(partition)
        eigvals = np.linalg.eigvalsh(rho_a)
        eigvals = eigvals[eigvals > 1e-15]
        return float(-np.sum(eigvals * np.log2(eigvals)))

    def probabilities(self) -> List[float]:
        """Diagonal of rho — measurement probabilities in computational basis."""
        return np.real(np.diag(self._rho)).tolist()

    def basis_labels(self) -> List[str]:
        return [f"|{i:0{self.n_qubits}b}>" for i in range(self.dim)]

    def state_real(self) -> List[List[float]]:
        return np.real(self._rho).tolist()

    def state_imag(self) -> List[List[float]]:
        return np.imag(self._rho).tolist()
