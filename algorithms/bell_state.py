from typing import Any, Dict
from algorithms.base import Algorithm
from simulator.tracer import Tracer
from simulator import gates


class BellStateAlgorithm(Algorithm):
    algorithm_id = "bell_state"
    name = "Bell State"
    description = (
        "Prepares the maximally entangled Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2 "
        "using a Hadamard gate to create superposition and a CNOT gate to entangle the qubits."
    )
    parameter_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        n_qubits = 2
        tracer = Tracer(n_qubits)
        tracer.snapshot("Initialize |00⟩", gate=None, qubits_affected=[])
        tracer.apply_single(gates.H, "H", 0, "Apply H to q0 — create superposition (|0⟩+|1⟩)/√2")
        tracer.apply_two(gates.CNOT, "CNOT", 0, 1, "Apply CNOT — entangle q0 and q1")

        final_probs = tracer.sv.probabilities_list()
        basis_labels = tracer.sv.basis_labels()
        most_likely = basis_labels[final_probs.index(max(final_probs))]

        circuit_layout = {
            "qubit_labels": ["q0", "q1"],
            "columns": [
                {
                    "column_index": 0,
                    "gates": [{"qubit": 0, "name": "H", "step_index": 1}],
                },
                {
                    "column_index": 1,
                    "gates": [
                        {"qubit": 0, "name": "CNOT_ctrl", "step_index": 2},
                        {"qubit": 1, "name": "CNOT_tgt", "step_index": 2},
                    ],
                },
            ],
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
