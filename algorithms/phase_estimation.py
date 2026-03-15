"""
Quantum Phase Estimation (QPE).

Estimates the phase φ such that U|ψ⟩ = e^{2πiφ}|ψ⟩.
Uses n counting qubits + 1 eigenstate qubit. The eigenstate is |1⟩ for
the T gate (phase = 1/8) and the S gate (phase = 1/4).
"""
import numpy as np
from typing import Any, Dict, Optional
from algorithms.base import Algorithm
from simulator.circuit import Circuit
from simulator.executor import Executor
from simulator.noise import NoiseModel


def _controlled_phase(theta: float) -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * theta)],
    ], dtype=np.complex128)


class PhaseEstimationAlgorithm(Algorithm):
    algorithm_id = "phase_estimation"
    name = "Quantum Phase Estimation"
    category = "interference"
    description = (
        "Estimates the eigenphase φ of a unitary U acting on |ψ⟩ = |1⟩. "
        "Uses n counting qubits: more qubits → higher precision. "
        "Applies the inverse QFT to extract φ in binary."
    )
    parameter_schema = {
        "type": "object",
        "properties": {
            "n_counting": {
                "type": "integer",
                "minimum": 2,
                "maximum": 4,
                "default": 3,
                "description": "Number of counting (precision) qubits (2–4)",
            },
            "unitary": {
                "type": "string",
                "enum": ["T", "S", "Z"],
                "default": "T",
                "description": "Unitary gate whose phase to estimate: T (φ=1/8), S (φ=1/4), Z (φ=1/2)",
            },
        },
        "required": ["n_counting", "unitary"],
    }

    def run(self, parameters: Dict[str, Any], mode: str = "statevector",
            noise_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        n: int = int(parameters["n_counting"])
        unitary: str = parameters["unitary"]

        phases = {"T": np.pi / 4, "S": np.pi / 2, "Z": np.pi}
        if unitary not in phases:
            raise ValueError(f"Unknown unitary: {unitary!r}")
        phi = phases[unitary]  # phase of the gate: U|1⟩ = e^{iφ}|1⟩

        total = n + 1  # n counting + 1 eigenstate qubit (index n)
        circ = Circuit(n_qubits=total)

        # Prepare eigenstate |1⟩ on qubit n
        circ.x(n, f"Prepare eigenstate |1⟩ on q{n}")

        # Apply H to counting qubits
        for q in range(n):
            circ.h(q, f"H on counting qubit q{q}")

        circ.barrier(*range(total), label="Controlled-U applications")

        # Apply controlled-U^{2^j} for j=0..n-1
        for j in range(n):
            reps = 2 ** j
            angle = phi * reps
            cp = _controlled_phase(angle)
            circ._two(cp, f"C-U^{reps}", j, n,
                      f"Controlled-U^{reps} (q{j}→q{n}), angle={angle:.4f}")

        circ.barrier(*range(n), label="Inverse QFT begins")

        # Inverse QFT on counting register
        for j in range(n - 1, -1, -1):
            for k in range(j - 1, -1, -1):
                theta = -np.pi / (2 ** (j - k))
                cp = _controlled_phase(theta)
                circ._two(cp, "CP†", k, j,
                          f"Inv-QFT: C-P†(π/{2**(j-k)}) q{k}→q{j}")
            circ.h(j, f"Inv-QFT: H on q{j}")

        # Bit reversal
        for q in range(n // 2):
            circ.swap(q, n - 1 - q, f"SWAP q{q}↔q{n-1-q}")

        noise_model = NoiseModel.from_config(noise_config) if noise_config else None
        executor = Executor(mode=mode, noise_model=noise_model)
        result = executor.run(circ, init_label=f"Initialize |{'0'*total}⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)
