"""
Variational Quantum Eigensolver (VQE) — demonstration circuit.

Prepares a parameterized ansatz and evaluates the expectation value
of H = Z⊗Z (two-qubit Ising coupling). The optimal angle minimizes ⟨H⟩.

This is a static demonstration at a user-chosen angle rather than a
full classical optimization loop, showing the variational principle.
"""
import numpy as np
from typing import Any, Dict, Optional
from algorithms.base import Algorithm
from simulator.circuit import Circuit
from simulator.executor import Executor
from simulator.noise import NoiseModel


class VQEAlgorithm(Algorithm):
    algorithm_id = "vqe"
    name = "VQE (Ising Z⊗Z)"
    category = "variational"
    description = (
        "Demonstrates the Variational Quantum Eigensolver for H = Z⊗Z. "
        "The ansatz Ry(θ)⊗Ry(θ) followed by CNOT prepares an entangled state whose "
        "energy ⟨Z⊗Z⟩ depends on θ. At θ=π the state has energy -1 (ground state). "
        "Explore how the energy landscape changes with the variational angle."
    )
    parameter_schema = {
        "type": "object",
        "properties": {
            "theta": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 6.283,
                "default": 3.1416,
                "description": "Ansatz rotation angle θ (radians). θ=π → ground state of Z⊗Z.",
            },
            "n_layers": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3,
                "default": 1,
                "description": "Number of ansatz layers",
            },
        },
        "required": ["theta", "n_layers"],
    }

    def run(self, parameters: Dict[str, Any], mode: str = "statevector",
            noise_config: Optional[Dict[str, Any]] = None,
            optimize: bool = False) -> Dict[str, Any]:
        theta: float = float(parameters["theta"])
        n_layers: int = int(parameters["n_layers"])

        circ = Circuit(n_qubits=2)

        for layer in range(n_layers):
            circ.ry(theta, 0, f"Layer {layer+1}: Ry({theta:.3f}) on q0")
            circ.ry(theta, 1, f"Layer {layer+1}: Ry({theta:.3f}) on q1")
            circ.cnot(0, 1, f"Layer {layer+1}: CNOT q0→q1 — entangle")

        noise_model = NoiseModel.from_config(noise_config) if noise_config else None
        executor = Executor(mode=mode, noise_model=noise_model, optimize=optimize)
        result = executor.run(circ, init_label="Initialize |00⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)
