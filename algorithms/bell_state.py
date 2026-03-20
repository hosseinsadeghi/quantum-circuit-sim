from typing import Any, Dict, Optional
from algorithms.base import Algorithm
from simulator.circuit import Circuit
from simulator.executor import Executor
from simulator.noise import NoiseModel


class BellStateAlgorithm(Algorithm):
    algorithm_id = "bell_state"
    name = "Bell State"
    category = "communication"
    description = (
        "Prepares the maximally entangled Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2 "
        "using a Hadamard gate to create superposition and a CNOT gate to entangle the qubits."
    )
    parameter_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def run(self, parameters: Dict[str, Any], mode: str = "statevector",
            noise_config: Optional[Dict[str, Any]] = None,
            optimize: bool = False) -> Dict[str, Any]:
        circ = Circuit(n_qubits=2)
        circ.h(0, "Apply H to q0 — create superposition (|0⟩+|1⟩)/√2")
        circ.cnot(0, 1, "Apply CNOT — entangle q0 and q1")

        noise_model = NoiseModel.from_config(noise_config) if noise_config else None
        executor = Executor(mode=mode, noise_model=noise_model, optimize=optimize)
        result = executor.run(circ, init_label="Initialize |00⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)
