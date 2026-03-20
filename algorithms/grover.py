from typing import Any, Dict, Optional
from algorithms.base import Algorithm
from simulator.circuit import Circuit
from simulator.executor import Executor
from simulator.noise import NoiseModel


class GroverAlgorithm(Algorithm):
    algorithm_id = "grover"
    name = "Grover's Search"
    category = "interference"
    description = (
        "Grover's quantum search algorithm finds a marked item in an unsorted database "
        "quadratically faster than classical search. It uses a phase oracle to mark the "
        "target state and a diffusion operator to amplify its probability."
    )
    parameter_schema = {
        "type": "object",
        "properties": {
            "n_qubits": {
                "type": "integer",
                "minimum": 2,
                "maximum": 12,
                "default": 3,
                "description": "Number of qubits (2–12)",
            },
            "target_state": {
                "type": "string",
                "pattern": "^[01]+$",
                "description": "Target bitstring to search for (e.g. '101')",
            },
            "num_iterations": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "default": 1,
                "description": "Number of Grover iterations",
            },
        },
        "required": ["n_qubits", "target_state", "num_iterations"],
    }

    def run(self, parameters: Dict[str, Any], mode: str = "statevector",
            noise_config: Optional[Dict[str, Any]] = None,
            optimize: bool = False) -> Dict[str, Any]:
        n_qubits: int = int(parameters["n_qubits"])
        target_state: str = str(parameters["target_state"])
        num_iterations: int = int(parameters["num_iterations"])

        if len(target_state) != n_qubits:
            raise ValueError(f"target_state length must equal n_qubits ({n_qubits})")
        if not all(c in "01" for c in target_state):
            raise ValueError("target_state must contain only '0' and '1'")

        circ = Circuit(n_qubits=n_qubits)

        # Uniform superposition
        for q in range(n_qubits):
            circ.h(q, f"Apply H to q{q} — uniform superposition")

        # Grover iterations
        for i in range(num_iterations):
            circ.phase_oracle(target_state,
                              f"Iteration {i+1}: Oracle marks |{target_state}⟩ with phase flip")
            circ.diffusion(f"Iteration {i+1}: Diffusion — inversion about the mean")

        noise_model = NoiseModel.from_config(noise_config) if noise_config else None
        executor = Executor(mode=mode, noise_model=noise_model, optimize=optimize)
        result = executor.run(circ, init_label=f"Initialize |{'0'*n_qubits}⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)
