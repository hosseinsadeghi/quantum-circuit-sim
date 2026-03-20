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


def _contract_gate_on_rho(gate_tensor: np.ndarray, rho: np.ndarray,
                          qubits: List[int], n: int, k: int) -> np.ndarray:
    """Apply k-qubit gate to density matrix via tensor contraction.

    gate_tensor: (2,)*2k shaped gate (first k axes are outputs, last k are inputs)
    rho: (2,)*2n shaped density matrix tensor
    qubits: list of k qubit indices the gate acts on
    n: total number of qubits
    k: number of qubits the gate acts on

    Returns rho after U rho U†.
    """
    in_axes = list(range(k, 2 * k))
    remaining = [i for i in range(n) if i not in qubits]

    # --- Contract gate with row axes ---
    rho = np.tensordot(gate_tensor, rho, axes=[in_axes, qubits])
    # Output: [gate_out_0..k-1, remaining_row..., col_0..col_{n-1}]
    # Build permutation to restore [row_0..row_{n-1}, col_0..col_{n-1}]
    perm1 = [0] * (2 * n)
    for idx, q in enumerate(qubits):
        perm1[q] = idx  # gate outputs go to their qubit positions
    for j, r in enumerate(remaining):
        perm1[r] = k + j  # remaining rows
    for c in range(n):
        perm1[n + c] = k + len(remaining) + c  # col axes unchanged
    rho = np.transpose(rho, perm1)

    # --- Contract gate* with col axes ---
    col_targets = [q + n for q in qubits]
    rho = np.tensordot(gate_tensor.conj(), rho, axes=[in_axes, col_targets])
    # Output: [gate_out_0..k-1, row_0..row_{n-1}, remaining_col...]
    # Build permutation: rows at positions k..k+n-1, gate outputs to col positions
    perm2 = [0] * (2 * n)
    for i in range(n):
        perm2[i] = k + i  # row axes
    for idx, q in enumerate(qubits):
        perm2[n + q] = idx  # gate outputs go to col qubit positions
    for j, r in enumerate(remaining):
        perm2[n + r] = k + n + j  # remaining col axes
    rho = np.transpose(rho, perm2)

    return rho


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
        qubits = [qubit1, qubit2]
        rho = _contract_gate_on_rho(G4, rho, qubits, n, k=2)
        self._rho = rho.reshape(self.dim, self.dim)

    def apply_three_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int, qubit3: int) -> None:
        """Apply three-qubit unitary via tensor contraction. O(2^2n)."""
        n = self.n_qubits
        G8 = gate.astype(np.complex128).reshape(2, 2, 2, 2, 2, 2)
        rho = self._rho.reshape([2] * (2 * n))
        qubits = [qubit1, qubit2, qubit3]
        rho = _contract_gate_on_rho(G8, rho, qubits, n, k=3)
        self._rho = rho.reshape(self.dim, self.dim)

    # ------------------------------------------------------------------
    # Kraus channel (noise) — tensor contraction
    # ------------------------------------------------------------------

    def apply_kraus(self, kraus_ops: List[np.ndarray], qubit: int) -> None:
        """Apply a single-qubit Kraus channel on `qubit`.

        rho -> sum_k K_k rho K_k† using batched tensor contraction.
        Stacks Kraus ops into (num_ops, 2, 2) tensor and applies all at once.
        """
        n = self.n_qubits
        num_ops = len(kraus_ops)

        if num_ops == 0:
            return

        # Stack Kraus ops: (num_ops, 2, 2)
        K_stack = np.stack([K.astype(np.complex128) for K in kraus_ops])
        rho = self._rho.reshape([2] * (2 * n))

        new_rho = np.zeros_like(self._rho)

        # Batched: for each K_k, compute K_k rho K_k† via tensor contraction
        for k in range(num_ops):
            K = K_stack[k]
            tmp = rho.copy()
            # Contract K with row-axis
            tmp = np.tensordot(K, tmp, axes=[[1], [qubit]])
            tmp = np.moveaxis(tmp, 0, qubit)
            # Contract K* with col-axis
            tmp = np.tensordot(K.conj(), tmp, axes=[[1], [qubit + n]])
            tmp = np.moveaxis(tmp, 0, qubit + n)
            new_rho += tmp.reshape(self.dim, self.dim)

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
