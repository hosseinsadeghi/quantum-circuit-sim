import numpy as np
from typing import Any, Dict, List
from algorithms.base import Algorithm
from simulator.tracer import Tracer
from simulator import gates


class GroverAlgorithm(Algorithm):
    algorithm_id = "grover"
    name = "Grover's Search"
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
                "maximum": 4,
                "default": 3,
                "description": "Number of qubits (2–4)",
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

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        n_qubits: int = int(parameters["n_qubits"])
        target_state: str = str(parameters["target_state"])
        num_iterations: int = int(parameters["num_iterations"])

        if len(target_state) != n_qubits:
            raise ValueError(f"target_state length must equal n_qubits ({n_qubits})")
        if not all(c in "01" for c in target_state):
            raise ValueError("target_state must contain only '0' and '1'")

        tracer = Tracer(n_qubits)
        tracer.snapshot(f"Initialize |{'0' * n_qubits}⟩", gate=None, qubits_affected=[])

        # Step 1: Apply H to all qubits — uniform superposition
        for q in range(n_qubits):
            tracer.apply_single(gates.H, "H", q, f"Apply H to q{q} — uniform superposition")

        # Grover iterations
        for iteration in range(num_iterations):
            # Oracle: flip phase of target state
            target_idx = int(target_state, 2)
            tracer.sv._state[target_idx] *= -1
            tracer.snapshot(
                f"Iteration {iteration + 1}: Oracle marks |{target_state}⟩ with phase flip",
                gate="Oracle",
                qubits_affected=list(range(n_qubits)),
            )

            # Diffusion operator: 2|ψ⟩⟨ψ| - I (inversion about the mean)
            mean_amp = np.mean(tracer.sv._state)
            tracer.sv._state = 2 * mean_amp - tracer.sv._state
            tracer.snapshot(
                f"Iteration {iteration + 1}: Diffusion — inversion about the mean",
                gate="Diffusion",
                qubits_affected=list(range(n_qubits)),
            )

        final_probs = tracer.sv.probabilities_list()
        basis_labels = tracer.sv.basis_labels()
        most_likely = basis_labels[final_probs.index(max(final_probs))]

        # Build circuit layout
        columns = []
        col_idx = 0

        # H column(s)
        for q in range(n_qubits):
            columns.append({
                "column_index": col_idx,
                "gates": [{"qubit": q, "name": "H", "step_index": q + 1}],
            })
            col_idx += 1

        # Oracle + Diffusion columns per iteration
        step_offset = n_qubits + 1
        for it in range(num_iterations):
            columns.append({
                "column_index": col_idx,
                "gates": [{"qubit": q, "name": "Oracle", "step_index": step_offset} for q in range(n_qubits)],
            })
            col_idx += 1
            step_offset += 1
            columns.append({
                "column_index": col_idx,
                "gates": [{"qubit": q, "name": "Diffusion", "step_index": step_offset} for q in range(n_qubits)],
            })
            col_idx += 1
            step_offset += 1

        circuit_layout = {
            "qubit_labels": [f"q{i}" for i in range(n_qubits)],
            "columns": columns,
        }

        return {
            "algorithm": self.algorithm_id,
            "n_qubits": n_qubits,
            "parameters": parameters,
            "steps": [s.to_dict() for s in tracer.steps],
            "measurement": {
                "basis_labels": basis_labels,
                "probabilities": final_probs,
                "most_likely_outcome": most_likely,
            },
            "circuit_layout": circuit_layout,
        }
