"""
Quantum Fourier Transform (QFT).

Implements the n-qubit QFT using the standard decomposition:
  H gate + controlled phase rotations + SWAP reversal.
"""
import numpy as np
from typing import Any, Dict, Optional
from algorithms.base import Algorithm
from simulator.circuit import Circuit
from simulator.executor import Executor
from simulator.noise import NoiseModel


class QFTAlgorithm(Algorithm):
    algorithm_id = "qft"
    name = "Quantum Fourier Transform"
    category = "interference"
    description = (
        "The Quantum Fourier Transform (QFT) is the quantum analogue of the discrete Fourier transform. "
        "It maps |j⟩ → (1/√N) Σ_k e^{2πijk/N} |k⟩ and is a key subroutine in Shor's algorithm "
        "and quantum phase estimation."
    )
    parameter_schema = {
        "type": "object",
        "properties": {
            "n_qubits": {
                "type": "integer",
                "minimum": 2,
                "maximum": 14,
                "default": 3,
                "description": "Number of qubits (2–14)",
            },
            "input_state": {
                "type": "string",
                "enum": ["uniform", "basis_1", "basis_3", "basis_5"],
                "default": "basis_1",
                "description": "Input state: uniform superposition or a specific basis state",
            },
        },
        "required": ["n_qubits", "input_state"],
    }

    def run(self, parameters: Dict[str, Any], mode: str = "statevector",
            noise_config: Optional[Dict[str, Any]] = None,
            optimize: bool = False) -> Dict[str, Any]:
        n: int = int(parameters["n_qubits"])
        input_state: str = parameters["input_state"]

        circ = Circuit(n_qubits=n)

        # Prepare input state
        if input_state == "uniform":
            for q in range(n):
                circ.h(q, f"H on q{q} — uniform superposition input")
        elif input_state.startswith("basis_"):
            idx = int(input_state.split("_")[1]) % (2 ** n)
            for bit_pos in range(n):
                if (idx >> (n - 1 - bit_pos)) & 1:
                    circ.x(bit_pos, f"X on q{bit_pos} — prepare |{idx:0{n}b}⟩")

        circ.barrier(*range(n), label="QFT begins")

        # QFT circuit
        for j in range(n):
            circ.h(j, f"H on q{j}")
            for k in range(j + 1, n):
                theta = np.pi / (2 ** (k - j))
                # Controlled phase rotation: control=k, target=j
                # Use CZ-based decomposition via phase gate on target
                # We implement controlled-P(theta) as a standard two-qubit gate
                cp = _controlled_phase(theta)
                circ._two(cp, "CP", k, j,
                          f"Controlled-P(π/{2**(k-j)}) q{k}→q{j}")

        # Swap register to correct bit ordering
        for q in range(n // 2):
            circ.swap(q, n - 1 - q, f"SWAP q{q}↔q{n-1-q} — bit reversal")

        circ.barrier(*range(n), label="QFT ends")

        noise_model = NoiseModel.from_config(noise_config) if noise_config else None
        executor = Executor(mode=mode, noise_model=noise_model, optimize=optimize)
        result = executor.run(circ, init_label=f"Initialize |{'0'*n}⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)


def _controlled_phase(theta: float) -> np.ndarray:
    """Controlled-phase gate: applies phase e^{iθ} when both qubits are |1⟩."""
    import numpy as np
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * theta)],
    ], dtype=np.complex128)
