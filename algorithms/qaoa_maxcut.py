import numpy as np
from typing import Any, Dict, List, Tuple
from algorithms.base import Algorithm
from simulator.tracer import Tracer
from simulator import gates

# Precomputed optimal (gamma, beta) angles for p=1,2,3 on small graph topologies
# These are known analytical or numerically optimized values for common small graphs
PRECOMPUTED_ANGLES: Dict[str, Dict[int, List[Tuple[float, float]]]] = {
    "cycle": {
        1: [(np.pi / 4, np.pi / 8)],
        2: [(0.3927, 0.1963), (1.1781, 0.5890)],
        3: [(0.2618, 0.1309), (0.7854, 0.3927), (1.3090, 0.6545)],
    },
    "complete": {
        1: [(np.pi / 4, np.pi / 8)],
        2: [(0.4712, 0.2356), (1.4137, 0.7069)],
        3: [(0.3142, 0.1571), (0.9425, 0.4712), (1.5708, 0.7854)],
    },
    "path": {
        1: [(np.pi / 4, np.pi / 8)],
        2: [(0.3927, 0.1963), (1.1781, 0.5890)],
        3: [(0.2618, 0.1309), (0.7854, 0.3927), (1.3090, 0.6545)],
    },
}

GRAPH_EDGES: Dict[str, Dict[int, List[Tuple[int, int]]]] = {
    "cycle": {
        4: [(0, 1), (1, 2), (2, 3), (3, 0)],
        6: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
    },
    "complete": {
        4: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        6: [(i, j) for i in range(6) for j in range(i + 1, 6)],
    },
    "path": {
        4: [(0, 1), (1, 2), (2, 3)],
        6: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    },
}


class QAOAMaxCutAlgorithm(Algorithm):
    algorithm_id = "qaoa_maxcut"
    name = "QAOA Max-Cut"
    description = (
        "Quantum Approximate Optimization Algorithm (QAOA) applied to the Max-Cut problem. "
        "Finds a partition of graph vertices that maximizes the number of edges between partitions. "
        "Uses alternating cost (phase) and mixer (Rx rotation) layers with precomputed optimal angles."
    )
    parameter_schema = {
        "type": "object",
        "properties": {
            "n_qubits": {
                "type": "integer",
                "enum": [4, 6],
                "default": 4,
                "description": "Number of qubits / graph vertices (4 or 6)",
            },
            "p_layers": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3,
                "default": 1,
                "description": "Number of QAOA layers (depth p)",
            },
            "topology": {
                "type": "string",
                "enum": ["cycle", "complete", "path"],
                "default": "cycle",
                "description": "Graph topology",
            },
        },
        "required": ["n_qubits", "p_layers", "topology"],
    }

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        n_qubits: int = int(parameters["n_qubits"])
        p_layers: int = int(parameters["p_layers"])
        topology: str = parameters["topology"]

        edges = GRAPH_EDGES[topology][n_qubits]
        angle_layers = PRECOMPUTED_ANGLES[topology][p_layers]
        tracer = Tracer(n_qubits)

        # Initialize: apply H to all qubits
        tracer.snapshot(f"Initialize |{'0' * n_qubits}⟩", gate=None, qubits_affected=[])
        for q in range(n_qubits):
            tracer.apply_single(gates.H, "H", q, f"Apply H to q{q} — uniform superposition")

        # QAOA layers
        for layer_idx, (gamma, beta) in enumerate(angle_layers):
            # Cost unitary: apply e^{-i*gamma*C} = product of CZ rotations on edges
            for (u, v) in edges:
                # e^{-i*gamma*(1 - Z_u Z_v)/2} decomposes to:
                # CNOT(u,v), Rz(2*gamma) on v, CNOT(u,v)
                tracer.apply_two(gates.CNOT, "CNOT", u, v,
                    f"Layer {layer_idx+1}: Cost — CNOT q{u}→q{v} (edge {u}-{v})")
                tracer.apply_single(gates.Rz(2 * gamma), "Rz", v,
                    f"Layer {layer_idx+1}: Cost — Rz({2*gamma:.3f}) on q{v}")
                tracer.apply_two(gates.CNOT, "CNOT", u, v,
                    f"Layer {layer_idx+1}: Cost — CNOT q{u}→q{v} (uncompute)")

            # Mixer unitary: apply e^{-i*beta*B} = product of Rx(2*beta) on all qubits
            for q in range(n_qubits):
                tracer.apply_single(gates.Rx(2 * beta), "Rx", q,
                    f"Layer {layer_idx+1}: Mixer — Rx({2*beta:.3f}) on q{q}")

        final_probs = tracer.sv.probabilities_list()
        basis_labels = tracer.sv.basis_labels()
        most_likely = basis_labels[final_probs.index(max(final_probs))]

        # Circuit layout
        col_idx = 0
        columns = []
        step_idx = 1
        # H initialization columns
        for q in range(n_qubits):
            columns.append({
                "column_index": col_idx,
                "gates": [{"qubit": q, "name": "H", "step_index": step_idx}],
            })
            col_idx += 1
            step_idx += 1

        # Per-layer columns
        for layer_idx in range(p_layers):
            for (u, v) in edges:
                columns.append({
                    "column_index": col_idx,
                    "gates": [
                        {"qubit": u, "name": "CNOT_ctrl", "step_index": step_idx},
                        {"qubit": v, "name": "CNOT_tgt", "step_index": step_idx},
                    ],
                })
                col_idx += 1
                step_idx += 1
                columns.append({
                    "column_index": col_idx,
                    "gates": [{"qubit": v, "name": "Rz", "step_index": step_idx}],
                })
                col_idx += 1
                step_idx += 1
                columns.append({
                    "column_index": col_idx,
                    "gates": [
                        {"qubit": u, "name": "CNOT_ctrl", "step_index": step_idx},
                        {"qubit": v, "name": "CNOT_tgt", "step_index": step_idx},
                    ],
                })
                col_idx += 1
                step_idx += 1

            for q in range(n_qubits):
                columns.append({
                    "column_index": col_idx,
                    "gates": [{"qubit": q, "name": "Rx", "step_index": step_idx}],
                })
                col_idx += 1
                step_idx += 1

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
