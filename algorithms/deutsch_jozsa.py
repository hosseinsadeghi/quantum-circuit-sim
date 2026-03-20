"""
Deutsch-Jozsa algorithm.

Determines in a single query whether a black-box function f: {0,1}^n → {0,1}
is constant (same output for all inputs) or balanced (0 for half, 1 for half).
"""
from typing import Any, Dict, Optional
from algorithms.base import Algorithm
from simulator.circuit import Circuit
from simulator.executor import Executor
from simulator.noise import NoiseModel
from simulator import gates as G


class DeutschJozsaAlgorithm(Algorithm):
    algorithm_id = "deutsch_jozsa"
    name = "Deutsch-Jozsa"
    category = "interference"
    description = (
        "Determines in one query whether a Boolean function is constant or balanced. "
        "A constant function always returns the same value; a balanced function returns "
        "0 for exactly half the inputs and 1 for the other half. "
        "Measurement of |0...0⟩ → constant; any other outcome → balanced."
    )
    parameter_schema = {
        "type": "object",
        "properties": {
            "n_qubits": {
                "type": "integer",
                "minimum": 1,
                "maximum": 12,
                "default": 3,
                "description": "Number of input qubits (1–12)",
            },
            "oracle_type": {
                "type": "string",
                "enum": ["constant_0", "constant_1", "balanced"],
                "default": "balanced",
                "description": "Oracle type: constant_0, constant_1, or balanced",
            },
        },
        "required": ["n_qubits", "oracle_type"],
    }

    def run(self, parameters: Dict[str, Any], mode: str = "statevector",
            noise_config: Optional[Dict[str, Any]] = None,
            optimize: bool = False) -> Dict[str, Any]:
        n: int = int(parameters["n_qubits"])
        oracle_type: str = parameters["oracle_type"]

        # n input qubits + 1 ancilla qubit (index n)
        total = n + 1
        circ = Circuit(n_qubits=total)

        # Initialize ancilla in |1⟩
        circ.x(n, "Flip ancilla to |1⟩")

        # Apply H to all qubits
        for q in range(total):
            circ.h(q, f"Apply H to q{q}")

        circ.barrier(*range(total), label="Oracle begins")

        # Oracle implementation
        if oracle_type == "constant_0":
            pass  # f=0 → identity, no gates
        elif oracle_type == "constant_1":
            # f=1 → flip ancilla: X on ancilla
            circ.x(n, "Constant-1 oracle: flip ancilla")
        elif oracle_type == "balanced":
            # Balanced oracle: CNOT from each input qubit to ancilla
            for q in range(n):
                circ.cnot(q, n, f"Balanced oracle: CNOT q{q}→ancilla")
        else:
            raise ValueError(f"Unknown oracle_type: {oracle_type!r}")

        circ.barrier(*range(total), label="Oracle ends")

        # Apply H to input qubits
        for q in range(n):
            circ.h(q, f"Apply H to q{q} — interference")

        noise_model = NoiseModel.from_config(noise_config) if noise_config else None
        executor = Executor(mode=mode, noise_model=noise_model, optimize=optimize)
        result = executor.run(circ, init_label=f"Initialize |{'0'*total}⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)
