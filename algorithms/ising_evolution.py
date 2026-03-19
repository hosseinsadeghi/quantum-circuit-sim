"""
Ising model Trotter evolution.

Simulates the transverse-field Ising Hamiltonian:
  H = -J Σ_{<i,j>} Z_i Z_j - h Σ_i X_i

using first-order Trotterization:
  e^{-iHt} ≈ [e^{-i(-J)Z_iZ_j·dt} · e^{-i(-h)X_i·dt}]^n_steps
"""
import numpy as np
from typing import Any, Dict, Optional
from algorithms.base import Algorithm
from simulator.circuit import Circuit
from simulator.executor import Executor
from simulator.noise import NoiseModel
from simulator import gates as G


def _zz_evolution(J: float, dt: float) -> np.ndarray:
    """
    e^{iJ·dt·Z⊗Z} = diag(e^{iJdt}, e^{-iJdt}, e^{-iJdt}, e^{iJdt})
    """
    a = np.exp(1j * J * dt)
    b = np.exp(-1j * J * dt)
    return np.diag([a, b, b, a]).astype(np.complex128)


class IsingEvolutionAlgorithm(Algorithm):
    algorithm_id = "ising_evolution"
    name = "Ising Trotter Evolution"
    category = "physics"
    description = (
        "Simulates the transverse-field Ising model H = -J Σ Z_iZ_j - h Σ X_i "
        "via first-order Trotterization. Each Trotter step applies ZZ coupling "
        "followed by transverse-field X rotations. Watch entanglement entropy grow "
        "under coherent dynamics."
    )
    parameter_schema = {
        "type": "object",
        "properties": {
            "n_qubits": {
                "type": "integer",
                "minimum": 2,
                "maximum": 10,
                "default": 3,
                "description": "Number of spins (2–10)",
            },
            "n_steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 6,
                "default": 3,
                "description": "Number of Trotter steps",
            },
            "J": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 2.0,
                "default": 1.0,
                "description": "Ising coupling strength J",
            },
            "h": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 2.0,
                "default": 0.5,
                "description": "Transverse field strength h",
            },
            "dt": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 1.0,
                "default": 0.3,
                "description": "Trotter step size dt",
            },
        },
        "required": ["n_qubits", "n_steps", "J", "h", "dt"],
    }

    def run(self, parameters: Dict[str, Any], mode: str = "statevector",
            noise_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        n: int = int(parameters["n_qubits"])
        n_steps: int = int(parameters["n_steps"])
        J: float = float(parameters["J"])
        h: float = float(parameters["h"])
        dt: float = float(parameters["dt"])

        circ = Circuit(n_qubits=n)

        # Start in uniform superposition (low transverse-field ground state)
        for q in range(n):
            circ.h(q, f"H on q{q} — initial superposition")

        # Trotter steps
        for step in range(n_steps):
            circ.barrier(*range(n), label=f"Trotter step {step+1}")

            # ZZ coupling on neighbouring pairs (1D chain)
            for q in range(n - 1):
                zz = _zz_evolution(J, dt)
                circ._two(zz, "ZZ", q, q + 1,
                          f"Step {step+1}: ZZ coupling q{q}-q{q+1}, J={J}, dt={dt}")

            # Transverse field: Rx rotations
            for q in range(n):
                circ.rx(2 * h * dt, q,
                        f"Step {step+1}: Transverse field Rx({2*h*dt:.3f}) on q{q}")

        noise_model = NoiseModel.from_config(noise_config) if noise_config else None
        executor = Executor(mode=mode, noise_model=noise_model)
        result = executor.run(circ, init_label=f"Initialize |{'0'*n}⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)
