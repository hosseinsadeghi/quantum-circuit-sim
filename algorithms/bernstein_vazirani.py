"""
Bernstein-Vazirani algorithm.

Finds a hidden bitstring s such that f(x) = s·x (mod 2) using a single query,
versus n queries classically.
"""
from typing import Any, Dict, Optional
from algorithms.base import Algorithm
from simulator.circuit import Circuit
from simulator.executor import Executor
from simulator.noise import NoiseModel


class BernsteinVaziraniAlgorithm(Algorithm):
    algorithm_id = "bernstein_vazirani"
    name = "Bernstein-Vazirani"
    category = "interference"
    description = (
        "Recovers a hidden bitstring s from the function f(x) = s·x (mod 2) in a single query. "
        "Classically this requires n queries. After the algorithm, measuring the input register "
        "directly reveals s."
    )
    parameter_schema = {
        "type": "object",
        "properties": {
            "secret_string": {
                "type": "string",
                "pattern": "^[01]+$",
                "default": "101",
                "description": "The hidden bitstring s (e.g. '101'). Length determines qubit count.",
            },
        },
        "required": ["secret_string"],
    }

    def run(self, parameters: Dict[str, Any], mode: str = "statevector",
            noise_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        s: str = str(parameters["secret_string"])
        if not all(c in "01" for c in s):
            raise ValueError("secret_string must contain only '0' and '1'")

        n = len(s)
        total = n + 1  # n input + 1 ancilla

        circ = Circuit(n_qubits=total)

        # Initialize ancilla in |1⟩
        circ.x(n, "Flip ancilla to |1⟩")

        # H on all
        for q in range(total):
            circ.h(q, f"Apply H to q{q}")

        circ.barrier(*range(total), label="BV oracle begins")

        # Oracle: CNOT from q_i to ancilla whenever s[i] == '1'
        for i, bit in enumerate(s):
            if bit == "1":
                circ.cnot(i, n, f"Oracle: CNOT q{i}→ancilla (s[{i}]=1)")

        circ.barrier(*range(total), label="BV oracle ends")

        # H on input register
        for q in range(n):
            circ.h(q, f"Apply H to q{q} — extract secret bit")

        noise_model = NoiseModel.from_config(noise_config) if noise_config else None
        executor = Executor(mode=mode, noise_model=noise_model)
        result = executor.run(circ, init_label=f"Initialize |{'0'*total}⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)
