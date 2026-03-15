"""
GHZ (Greenberger-Horne-Zeilinger) state preparation.

Creates the maximally entangled n-qubit GHZ state:
|GHZ⟩ = (|00...0⟩ + |11...1⟩) / √2
"""
from typing import Any, Dict, Optional
from algorithms.base import Algorithm
from simulator.circuit import Circuit
from simulator.executor import Executor
from simulator.noise import NoiseModel


class GHZAlgorithm(Algorithm):
    algorithm_id = "ghz"
    name = "GHZ State"
    category = "communication"
    description = (
        "Prepares the maximally entangled GHZ state (|00...0⟩ + |11...1⟩) / √2. "
        "Demonstrates multipartite entanglement: all n qubits become perfectly correlated. "
        "Used in quantum communication, cryptography, and tests of Bell inequalities."
    )
    parameter_schema = {
        "type": "object",
        "properties": {
            "n_qubits": {
                "type": "integer",
                "minimum": 2,
                "maximum": 6,
                "default": 3,
                "description": "Number of qubits (2–6)",
            },
        },
        "required": ["n_qubits"],
    }

    def run(self, parameters: Dict[str, Any], mode: str = "statevector",
            noise_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        n: int = int(parameters["n_qubits"])

        circ = Circuit(n_qubits=n)
        circ.h(0, "Apply H to q0 — create superposition")
        for q in range(1, n):
            circ.cnot(0, q, f"CNOT q0→q{q} — entangle qubit {q}")

        noise_model = NoiseModel.from_config(noise_config) if noise_config else None
        executor = Executor(mode=mode, noise_model=noise_model)
        result = executor.run(circ, init_label=f"Initialize |{'0'*n}⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)
