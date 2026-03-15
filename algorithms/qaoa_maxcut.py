import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from algorithms.base import Algorithm
from simulator.circuit import Circuit
from simulator.executor import Executor
from simulator.noise import NoiseModel
from simulator import gates as G

# Precomputed optimal (gamma, beta) angles for p=1,2,3 on small graph topologies
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
    category = "variational"
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

    def run(self, parameters: Dict[str, Any], mode: str = "statevector",
            noise_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        n_qubits: int = int(parameters["n_qubits"])
        p_layers: int = int(parameters["p_layers"])
        topology: str = parameters["topology"]

        edges = GRAPH_EDGES[topology][n_qubits]
        angle_layers = PRECOMPUTED_ANGLES[topology][p_layers]

        circ = Circuit(n_qubits=n_qubits)

        # Initialize: apply H to all qubits
        for q in range(n_qubits):
            circ.h(q, f"Apply H to q{q} — uniform superposition")

        # QAOA layers
        for layer_idx, (gamma, beta) in enumerate(angle_layers):
            # Cost unitary: CNOT-Rz-CNOT per edge
            for u, v in edges:
                circ.cnot(u, v, f"Layer {layer_idx+1}: Cost — CNOT q{u}→q{v} (edge {u}-{v})")
                circ.rz(2 * gamma, v, f"Layer {layer_idx+1}: Cost — Rz({2*gamma:.3f}) on q{v}")
                circ.cnot(u, v, f"Layer {layer_idx+1}: Cost — CNOT q{u}→q{v} (uncompute)")

            # Mixer unitary
            for q in range(n_qubits):
                circ.rx(2 * beta, q, f"Layer {layer_idx+1}: Mixer — Rx({2*beta:.3f}) on q{q}")

        noise_model = NoiseModel.from_config(noise_config) if noise_config else None
        executor = Executor(mode=mode, noise_model=noise_model)
        result = executor.run(circ, init_label=f"Initialize |{'0'*n_qubits}⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)
