import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from algorithms.base import Algorithm
from algorithms.qaoa_maxcut import generate_edges
from simulator.circuit import Circuit
from simulator.executor import Executor
from simulator.noise import NoiseModel

# Precomputed ADAPT-QAOA results: which operators were selected and their
# optimized parameters. Format: list of (gate_name, qubit_targets, angle).
# These represent the adaptively-grown ansatz for small Max-Cut instances.
PRECOMPUTED_ADAPT: Dict[Tuple[str, int, int], List[Tuple[str, List[int], float]]] = {
    # (topology, n_qubits, n_adapt_steps)
    ("cycle", 4, 3): [
        ("RZZ", [0, 1], 0.785),
        ("RZZ", [2, 3], 0.785),
        ("RX", [0], 0.393),
    ],
    ("cycle", 4, 5): [
        ("RZZ", [0, 1], 0.785),
        ("RZZ", [2, 3], 0.785),
        ("RX", [0], 0.393),
        ("RX", [2], 0.393),
        ("RZZ", [1, 2], 0.524),
    ],
    ("path", 4, 3): [
        ("RZZ", [1, 2], 0.785),
        ("RZZ", [0, 1], 0.654),
        ("RX", [1], 0.393),
    ],
    ("complete", 4, 3): [
        ("RZZ", [0, 1], 0.471),
        ("RZZ", [2, 3], 0.471),
        ("RX", [0], 0.236),
    ],
    ("cycle", 6, 3): [
        ("RZZ", [0, 1], 0.785),
        ("RZZ", [3, 4], 0.785),
        ("RX", [0], 0.393),
    ],
}


class ADAPTQAOAAlgorithm(Algorithm):
    algorithm_id = "adapt_qaoa"
    name = "ADAPT-QAOA Max-Cut"
    category = "variational"
    description = (
        "Adaptive Derivative-Assembled Pseudo-Trotter QAOA for Max-Cut. "
        "Instead of fixed alternating layers, ADAPT-QAOA iteratively grows the "
        "ansatz by selecting operators from a pool based on gradient magnitude. "
        "Visualizes the adaptively-constructed circuit with precomputed parameters."
    )
    parameter_schema = {
        "type": "object",
        "properties": {
            "n_qubits": {
                "type": "integer",
                "minimum": 4,
                "maximum": 14,
                "default": 4,
                "description": "Number of qubits / graph vertices (4–14)",
            },
            "n_adapt_steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "default": 3,
                "description": "Number of ADAPT steps (operators added to ansatz)",
            },
            "topology": {
                "type": "string",
                "enum": ["cycle", "complete", "path"],
                "default": "cycle",
                "description": "Graph topology",
            },
        },
        "required": ["n_qubits", "n_adapt_steps", "topology"],
    }

    def run(self, parameters: Dict[str, Any], mode: str = "statevector",
            noise_config: Optional[Dict[str, Any]] = None,
            optimize: bool = False) -> Dict[str, Any]:
        n_qubits: int = int(parameters["n_qubits"])
        n_adapt_steps: int = int(parameters["n_adapt_steps"])
        topology: str = parameters["topology"]

        edges = generate_edges(topology, n_qubits)
        key = (topology, n_qubits, n_adapt_steps)

        if key in PRECOMPUTED_ADAPT:
            adapt_ops = PRECOMPUTED_ADAPT[key]
        else:
            # Fallback: use RZZ on first n_adapt_steps edges with default angle
            adapt_ops = []
            for i in range(min(n_adapt_steps, len(edges))):
                u, v = edges[i]
                adapt_ops.append(("RZZ", [u, v], np.pi / 4))
            # Fill remaining steps with single-qubit RX
            for i in range(len(adapt_ops), n_adapt_steps):
                adapt_ops.append(("RX", [i % n_qubits], np.pi / 8))

        circ = Circuit(n_qubits=n_qubits)

        # Initialize: H on all qubits
        for q in range(n_qubits):
            circ.h(q, f"Apply H to q{q} — uniform superposition")

        # Adaptively selected operators
        for step_idx, (gate_name, targets, angle) in enumerate(adapt_ops):
            if gate_name == "RZZ":
                circ.rzz(2 * angle, targets[0], targets[1],
                         f"ADAPT step {step_idx+1}: RZZ({2*angle:.3f}) on q{targets[0]},q{targets[1]}")
            elif gate_name == "RX":
                circ.rx(2 * angle, targets[0],
                        f"ADAPT step {step_idx+1}: Rx({2*angle:.3f}) on q{targets[0]}")
            elif gate_name == "RZ":
                circ.rz(2 * angle, targets[0],
                        f"ADAPT step {step_idx+1}: Rz({2*angle:.3f}) on q{targets[0]}")

        # Final cost layer: RZZ on all edges (for measurement in cost basis)
        for u, v in edges:
            circ.rzz(np.pi / 2, u, v,
                     f"Cost — RZZ({np.pi/2:.3f}) on edge {u}-{v}")

        noise_model = NoiseModel.from_config(noise_config) if noise_config else None
        executor = Executor(mode=mode, noise_model=noise_model, optimize=optimize)
        result = executor.run(circ, init_label=f"Initialize |{'0'*n_qubits}⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)
