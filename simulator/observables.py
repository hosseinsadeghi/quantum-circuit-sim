"""
Computes per-step observables from a DensityMatrix or StateVector.

Used by the Executor to populate the `observables` field in each SimulationStep.
"""
from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulator.density_matrix import DensityMatrix
    from simulator.state_vector import StateVector


def compute_observables_from_dm(dm: "DensityMatrix") -> Dict:
    """
    Return observable dict from a DensityMatrix:

    {
      "bloch_vectors": [[x, y, z], ...],   # one per qubit
      "z_expectations": [float, ...],       # one per qubit
      "entanglement_entropy": float,         # half-half bipartition
      "purity": float,
    }
    """
    bloch_vectors = []
    z_expectations = []
    for q in range(dm.n_qubits):
        x, y, z = dm.bloch_vector(q)
        bloch_vectors.append([x, y, z])
        z_expectations.append(z)

    entropy = dm.entanglement_entropy() if dm.n_qubits >= 2 else 0.0
    purity = dm.purity()

    return {
        "bloch_vectors": bloch_vectors,
        "z_expectations": z_expectations,
        "entanglement_entropy": entropy,
        "purity": purity,
    }


def compute_observables_from_sv(sv: "StateVector") -> Dict:
    """
    Compute observables directly from the statevector — O(n * 2^n).

    Avoids the O(2^2n) cost of constructing the full density matrix.
    Bloch vectors and Z expectations are computed via reduced single-qubit
    density matrices obtained by reshaping and summing the statevector.
    Entanglement entropy uses Schmidt decomposition (SVD on reshaped state).
    Purity is always 1.0 for a pure state.
    """
    n = sv.n_qubits
    psi = sv._state

    bloch_vectors = []
    z_expectations = []

    for q in range(n):
        rho_q = _reduced_single_qubit_dm(psi, q, n)
        x = float(2 * np.real(rho_q[0, 1]))
        y = float(2 * np.imag(rho_q[1, 0]))
        z = float(np.real(rho_q[0, 0] - rho_q[1, 1]))
        bloch_vectors.append([x, y, z])
        z_expectations.append(z)

    entropy = _entanglement_entropy_sv(psi, n) if n >= 2 else 0.0

    return {
        "bloch_vectors": bloch_vectors,
        "z_expectations": z_expectations,
        "entanglement_entropy": entropy,
        "purity": 1.0,  # Pure state — always 1.0
    }


def _reduced_single_qubit_dm(psi: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """
    Compute the 2x2 reduced density matrix for a single qubit from a statevector.

    Reshapes psi to (2^q, 2, 2^(n-q-1)) and contracts over the environment
    indices to get rho_q[i,j] = sum_k psi[k,i,:].conj() @ psi[k,j,:].
    Cost: O(2^n) per qubit.
    """
    state = psi.reshape(2 ** qubit, 2, 2 ** (n_qubits - qubit - 1))
    # rho_q[i, j] = sum over environment of conj(state[:, i, :]) * state[:, j, :]
    rho_q = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            rho_q[i, j] = np.sum(state[:, i, :].conj() * state[:, j, :])
    return rho_q


def _entanglement_entropy_sv(psi: np.ndarray, n_qubits: int) -> float:
    """
    Entanglement entropy via Schmidt decomposition (SVD on reshaped state).

    Bipartition: first n//2 qubits vs rest.
    Cost: O(2^n * min(2^k, 2^(n-k))) where k = n//2.
    """
    k = n_qubits // 2
    if k == 0 or k == n_qubits:
        return 0.0
    mat = psi.reshape(2 ** k, 2 ** (n_qubits - k))
    s = np.linalg.svd(mat, compute_uv=False)
    # Schmidt coefficients squared = eigenvalues of reduced DM
    s2 = s ** 2
    s2 = s2[s2 > 1e-15]
    return float(-np.sum(s2 * np.log2(s2)))
