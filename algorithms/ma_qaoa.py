import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from algorithms.base import Algorithm
from algorithms.qaoa_maxcut import generate_edges
from simulator.circuit import Circuit
from simulator.executor import Executor
from simulator.noise import NoiseModel

# Precomputed per-edge gamma and per-qubit beta angles for small instances.
# Keys: (topology, n_qubits, p_layers) -> list of (gamma_per_edge, beta_per_qubit) per layer.
# These were obtained via classical optimization of the MA-QAOA cost function.
PRECOMPUTED_MA_ANGLES: Dict[Tuple[str, int, int], List[Tuple[List[float], List[float]]]] = {
    ("cycle", 4, 1): [
        ([0.392, 0.392, 0.392, 0.392], [0.196, 0.196, 0.196, 0.196]),
    ],
    ("cycle", 4, 2): [
        ([0.262, 0.262, 0.262, 0.262], [0.131, 0.131, 0.131, 0.131]),
        ([0.785, 0.785, 0.785, 0.785], [0.393, 0.393, 0.393, 0.393]),
    ],
    ("complete", 4, 1): [
        ([0.471, 0.471, 0.471, 0.471, 0.471, 0.471],
         [0.236, 0.236, 0.236, 0.236]),
    ],
    ("path", 4, 1): [
        ([0.393, 0.524, 0.393], [0.196, 0.262, 0.262, 0.196]),
    ],
    ("cycle", 6, 1): [
        ([0.392, 0.392, 0.392, 0.392, 0.392, 0.392],
         [0.196, 0.196, 0.196, 0.196, 0.196, 0.196]),
    ],
}


class MAQAOAAlgorithm(Algorithm):
    algorithm_id = "ma_qaoa"
    name = "MA-QAOA Max-Cut"
    category = "variational"
    description = (
        "Multi-Angle QAOA (MA-QAOA) for Max-Cut — each edge and each qubit "
        "gets its own independent angle parameter per layer, allowing finer-grained "
        "optimization than standard QAOA. Uses RZZ cost gates and Rx mixer gates."
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
            "p_layers": {
                "type": "integer",
                "minimum": 1,
                "maximum": 2,
                "default": 1,
                "description": "Number of MA-QAOA layers",
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
            noise_config: Optional[Dict[str, Any]] = None,
            optimize: bool = False) -> Dict[str, Any]:
        n_qubits: int = int(parameters["n_qubits"])
        p_layers: int = int(parameters["p_layers"])
        topology: str = parameters["topology"]

        edges = generate_edges(topology, n_qubits)
        key = (topology, n_qubits, p_layers)

        if key in PRECOMPUTED_MA_ANGLES:
            angle_layers = PRECOMPUTED_MA_ANGLES[key]
        else:
            # Fallback: uniform angles matching standard QAOA
            angle_layers = [
                ([np.pi / 4] * len(edges), [np.pi / 8] * n_qubits)
            ] * p_layers

        circ = Circuit(n_qubits=n_qubits)

        # Initialize: H on all qubits
        for q in range(n_qubits):
            circ.h(q, f"Apply H to q{q} — uniform superposition")

        # MA-QAOA layers
        for layer_idx, (gammas, betas) in enumerate(angle_layers):
            # Cost layer: RZZ per edge with individual gamma
            for edge_idx, (u, v) in enumerate(edges):
                gamma = gammas[edge_idx]
                circ.rzz(2 * gamma, u, v,
                         f"Layer {layer_idx+1}: Cost — RZZ({2*gamma:.3f}) on edge {u}-{v}")

            # Mixer layer: Rx per qubit with individual beta
            for q in range(n_qubits):
                beta = betas[q]
                circ.rx(2 * beta, q,
                        f"Layer {layer_idx+1}: Mixer — Rx({2*beta:.3f}) on q{q}")

        noise_model = NoiseModel.from_config(noise_config) if noise_config else None
        executor = Executor(mode=mode, noise_model=noise_model, optimize=optimize)
        result = executor.run(circ, init_label=f"Initialize |{'0'*n_qubits}⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)
