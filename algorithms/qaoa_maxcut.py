import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from algorithms.base import Algorithm
from simulator.circuit import Circuit
from simulator.executor import Executor
from simulator.noise import NoiseModel
from simulator import gates as G


def generate_edges(topology: str, n: int) -> List[Tuple[int, int]]:
    """Generate graph edges for a given topology and vertex count."""
    if topology == "cycle":
        return [(i, (i + 1) % n) for i in range(n)]
    elif topology == "complete":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    elif topology == "path":
        return [(i, i + 1) for i in range(n - 1)]
    else:
        raise ValueError(f"Unknown topology: {topology!r}")


def generate_angles(topology: str, p_layers: int, n_edges: int) -> List[Tuple[float, float]]:
    """Generate QAOA angles using analytical initialization.

    For p=1: uses optimal Farhi et al. angles.
    For higher p: uses INTERP strategy (linearly interpolated from p=1).
    """
    # Optimal p=1 angles per topology (Farhi et al.)
    base_angles = {
        "cycle": (np.pi / 4, np.pi / 8),
        "complete": (np.pi / 4, np.pi / 8),
        "path": (np.pi / 4, np.pi / 8),
    }
    gamma_1, beta_1 = base_angles.get(topology, (np.pi / 4, np.pi / 8))

    if p_layers == 1:
        return [(gamma_1, beta_1)]

    # INTERP strategy: linearly ramp angles across layers
    angles = []
    for k in range(p_layers):
        t = (k + 1) / p_layers
        angles.append((gamma_1 * t, beta_1 * t))
    return angles


# Legacy precomputed angles for backward compatibility at small sizes
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
                "minimum": 4,
                "maximum": 14,
                "default": 4,
                "description": "Number of qubits / graph vertices (4–14)",
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
            noise_config: Optional[Dict[str, Any]] = None,
            optimize: bool = False) -> Dict[str, Any]:
        n_qubits: int = int(parameters["n_qubits"])
        p_layers: int = int(parameters["p_layers"])
        topology: str = parameters["topology"]

        edges = generate_edges(topology, n_qubits)

        # Use precomputed angles if available, otherwise generate analytically
        if topology in PRECOMPUTED_ANGLES and p_layers in PRECOMPUTED_ANGLES[topology]:
            angle_layers = PRECOMPUTED_ANGLES[topology][p_layers]
        else:
            angle_layers = generate_angles(topology, p_layers, len(edges))

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
        executor = Executor(mode=mode, noise_model=noise_model, optimize=optimize)
        result = executor.run(circ, init_label=f"Initialize |{'0'*n_qubits}⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)
