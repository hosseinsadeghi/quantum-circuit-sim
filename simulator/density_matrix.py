"""
Density-matrix quantum state representation.

Supports:
- Unitary evolution: U ρ U†
- Noise via Kraus operators: ρ → Σ_k K_k ρ K_k†
- Mid-circuit measurement with state collapse
- Partial trace
- Bloch vector per qubit
- Entanglement entropy (von Neumann)
- Purity Tr(ρ²)
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


class DensityMatrix:
    """n-qubit density matrix, stored as (2^n × 2^n) complex128 array."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self._rho = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self._rho[0, 0] = 1.0  # |0...0⟩⟨0...0|

    @classmethod
    def from_statevector(cls, psi: np.ndarray) -> "DensityMatrix":
        """Build ρ = |ψ⟩⟨ψ| from a state vector."""
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
    # Unitary evolution
    # ------------------------------------------------------------------

    def apply_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> None:
        """Apply single-qubit unitary U to qubit: ρ → (I⊗…⊗U⊗…⊗I) ρ (I⊗…⊗U†⊗…⊗I)."""
        U = self._embed_single(gate, qubit)
        self._rho = U @ self._rho @ U.conj().T

    def apply_two_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int) -> None:
        """Apply two-qubit unitary to (qubit1, qubit2)."""
        U = self._embed_two(gate, qubit1, qubit2)
        self._rho = U @ self._rho @ U.conj().T

    def _embed_single(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        """Build full 2^n × 2^n unitary for single-qubit gate on `qubit`."""
        # Use tensor product: I ⊗ ... ⊗ gate ⊗ ... ⊗ I
        # qubit 0 = MSB (leftmost in ket notation)
        ops = [np.eye(2, dtype=np.complex128)] * self.n_qubits
        ops[qubit] = gate.astype(np.complex128)
        U = ops[0]
        for op in ops[1:]:
            U = np.kron(U, op)
        return U

    def _embed_two(self, gate: np.ndarray, qubit1: int, qubit2: int) -> np.ndarray:
        """Build full unitary for two-qubit gate.

        We use a permutation approach:
        1. Permute qubits so qubit1, qubit2 are at positions 0, 1.
        2. Apply gate as kron(gate, I_{n-2}).
        3. Permute back.
        """
        n = self.n_qubits
        # Full permutation: bring qubit1→0, qubit2→1, rest in order
        others = [i for i in range(n) if i not in (qubit1, qubit2)]
        perm = [qubit1, qubit2] + others  # source axes → positions 0,1,...

        # Build permutation matrix P such that P|q⟩ = |permuted q⟩
        P = self._perm_matrix(perm, n)

        gate_full = np.kron(gate.astype(np.complex128),
                            np.eye(2 ** (n - 2), dtype=np.complex128))
        U = P.conj().T @ gate_full @ P
        return U

    @staticmethod
    def _perm_matrix(perm: List[int], n: int) -> np.ndarray:
        """Build 2^n × 2^n permutation matrix for qubit permutation `perm`.

        perm[i] = source qubit that goes to position i.
        """
        dim = 2 ** n
        P = np.zeros((dim, dim), dtype=np.complex128)
        for j in range(dim):
            # j in original basis → reorder bits according to perm
            bits = [(j >> (n - 1 - perm[i])) & 1 for i in range(n)]
            new_j = sum(b << (n - 1 - i) for i, b in enumerate(bits))
            P[new_j, j] = 1.0
        return P

    # ------------------------------------------------------------------
    # Kraus channel (noise)
    # ------------------------------------------------------------------

    def apply_kraus(self, kraus_ops: List[np.ndarray], qubit: int) -> None:
        """Apply a single-qubit Kraus channel on `qubit`.

        ρ → Σ_k (I⊗K_k) ρ (I⊗K_k†)
        """
        new_rho = np.zeros_like(self._rho)
        for K in kraus_ops:
            K_full = self._embed_single(K, qubit)
            new_rho += K_full @ self._rho @ K_full.conj().T
        self._rho = new_rho

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure_and_collapse(self, qubit: int, rng: Optional[np.random.Generator] = None
                             ) -> int:
        """
        Measure `qubit` in the computational basis.

        Collapses the density matrix and returns the measurement outcome (0 or 1).
        """
        if rng is None:
            rng = np.random.default_rng()

        # Projectors for |0⟩ and |1⟩ on this qubit
        P0 = self._embed_single(np.array([[1, 0], [0, 0]], dtype=np.complex128), qubit)
        P1 = self._embed_single(np.array([[0, 0], [0, 1]], dtype=np.complex128), qubit)

        p0 = float(np.real(np.trace(P0 @ self._rho)))
        p0 = max(0.0, min(1.0, p0))  # numerical safety

        outcome = int(rng.choice([0, 1], p=[p0, 1.0 - p0]))
        P = P0 if outcome == 0 else P1
        prob = p0 if outcome == 0 else (1.0 - p0)

        self._rho = P @ self._rho @ P / prob
        return outcome

    # ------------------------------------------------------------------
    # Partial trace
    # ------------------------------------------------------------------

    def partial_trace(self, keep_qubits: List[int]) -> np.ndarray:
        """Return the reduced density matrix for `keep_qubits`.

        Returns a (2^k × 2^k) array where k = len(keep_qubits).
        """
        n = self.n_qubits
        trace_out = [i for i in range(n) if i not in keep_qubits]
        k = len(keep_qubits)

        # Reshape rho to tensor of shape (2,)*2n with axes [row_q0,...,row_qn-1, col_q0,...]
        rho_t = self._rho.reshape([2] * (2 * n))

        # Trace over each unwanted qubit
        # After tracing qubit q: axes q and q+n contract
        # We process trace_out in reverse order to keep axis indices valid
        current_rho = rho_t
        current_n = n
        for q in sorted(trace_out, reverse=True):
            # Axes: row_axes = 0..current_n-1, col_axes = current_n..2*current_n-1
            # Trace over axis q (row) and axis q+current_n (col)
            current_rho = np.trace(current_rho, axis1=q, axis2=q + current_n)
            current_n -= 1

        dim_k = 2 ** k
        return current_rho.reshape(dim_k, dim_k)

    def single_qubit_dm(self, qubit: int) -> np.ndarray:
        """2×2 reduced density matrix for a single qubit."""
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
        """⟨Z⟩ for a single qubit."""
        rho_q = self.single_qubit_dm(qubit)
        return float(np.real(rho_q[0, 0] - rho_q[1, 1]))

    def purity(self) -> float:
        """Tr(ρ²)."""
        return float(np.real(np.trace(self._rho @ self._rho)))

    def entanglement_entropy(self, partition: Optional[List[int]] = None) -> float:
        """Von Neumann entropy S = -Tr(ρ_A log ρ_A) for bipartition A.

        partition defaults to the first half of qubits.
        """
        if partition is None:
            partition = list(range(self.n_qubits // 2))
        rho_a = self.partial_trace(partition)
        eigvals = np.linalg.eigvalsh(rho_a)
        eigvals = eigvals[eigvals > 1e-15]
        return float(-np.sum(eigvals * np.log2(eigvals)))

    def probabilities(self) -> List[float]:
        """Diagonal of ρ — measurement probabilities in computational basis."""
        return np.real(np.diag(self._rho)).tolist()

    def basis_labels(self) -> List[str]:
        return [f"|{i:0{self.n_qubits}b}⟩" for i in range(self.dim)]

    def state_real(self) -> List[List[float]]:
        return np.real(self._rho).tolist()

    def state_imag(self) -> List[List[float]]:
        return np.imag(self._rho).tolist()
