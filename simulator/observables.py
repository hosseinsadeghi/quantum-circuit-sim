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
    Compute observables from a statevector by building the density matrix on the fly.

    Avoids storing a full density matrix in the statevector path by computing
    reduced density matrices via partial trace from the statevector directly.
    """
    from simulator.density_matrix import DensityMatrix
    dm = DensityMatrix.from_statevector(sv._state)
    return compute_observables_from_dm(dm)
