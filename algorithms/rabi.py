"""
Rabi oscillations.

Demonstrates the coherent oscillation of a qubit driven by a resonant field.
Each step applies an Rx rotation by a fixed angle, showing |0⟩ ↔ |1⟩ cycling.
Optionally includes amplitude damping noise to show decoherence.
"""
import numpy as np
from typing import Any, Dict, Optional
from algorithms.base import Algorithm
from simulator.circuit import Circuit
from simulator.executor import Executor
from simulator.noise import NoiseModel


class RabiAlgorithm(Algorithm):
    algorithm_id = "rabi"
    name = "Rabi Oscillations"
    category = "physics"
    description = (
        "Simulates Rabi oscillations: a qubit driven by a resonant field oscillates "
        "between |0⟩ and |1⟩. Each step applies Rx(Ω·dt). "
        "With noise enabled (density_matrix mode), decoherence damps the oscillation amplitude."
    )
    parameter_schema = {
        "type": "object",
        "properties": {
            "n_steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
                "description": "Number of drive steps",
            },
            "omega_dt": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 1.0,
                "default": 0.314,
                "description": "Rabi frequency × time step Ω·dt (radians per step, π/10 ≈ 0.314)",
            },
        },
        "required": ["n_steps", "omega_dt"],
    }

    def run(self, parameters: Dict[str, Any], mode: str = "statevector",
            noise_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        n_steps: int = int(parameters["n_steps"])
        omega_dt: float = float(parameters["omega_dt"])

        circ = Circuit(n_qubits=1)

        for step in range(n_steps):
            circ.rx(omega_dt, 0,
                    f"Step {step+1}: Rx({omega_dt:.3f}) — Rabi drive")

        noise_model = NoiseModel.from_config(noise_config) if noise_config else None
        executor = Executor(mode=mode, noise_model=noise_model)
        result = executor.run(circ, init_label="Initialize |0⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)
