"""
Quantum Teleportation protocol.

Teleports an arbitrary qubit state |ψ⟩ from Alice (q0) to Bob (q2)
using a shared Bell pair (q1, q2) and two classical bits.

Steps:
  1. Prepare |ψ⟩ on q0 (via Ry rotation)
  2. Create Bell pair on (q1, q2)
  3. Alice performs Bell measurement on (q0, q1)
  4. Bob applies corrections based on classical bits

Uses mid-circuit measurement and classical control.
"""
import numpy as np
from typing import Any, Dict, Optional
from algorithms.base import Algorithm
from simulator.circuit import Circuit, GateOp, TwoQubitGateOp
from simulator.executor import Executor
from simulator.noise import NoiseModel
from simulator import gates as G


class TeleportationAlgorithm(Algorithm):
    algorithm_id = "teleportation"
    name = "Quantum Teleportation"
    category = "communication"
    description = (
        "Transfers an arbitrary qubit state |ψ⟩ from Alice to Bob using a Bell pair "
        "and two classical bits. Alice measures her two qubits and sends the results "
        "classically; Bob applies corrections to reconstruct |ψ⟩ exactly."
    )
    parameter_schema = {
        "type": "object",
        "properties": {
            "state_angle": {
                "type": "number",
                "minimum": 0,
                "maximum": 6.283,
                "default": 1.047,
                "description": "Ry angle (radians) to prepare Alice's qubit (0=|0⟩, π=|1⟩, π/3≈1.047)",
            },
        },
        "required": ["state_angle"],
    }

    def run(self, parameters: Dict[str, Any], mode: str = "statevector",
            noise_config: Optional[Dict[str, Any]] = None,
            optimize: bool = False) -> Dict[str, Any]:
        angle: float = float(parameters["state_angle"])

        # 3 qubits: q0=Alice's state, q1=Alice's Bell qubit, q2=Bob's qubit
        # 2 classical bits: c0=Alice's q0 measurement, c1=Alice's q1 measurement
        circ = Circuit(n_qubits=3, n_clbits=2)

        # Step 1: Prepare Alice's qubit |ψ⟩ = Ry(angle)|0⟩
        circ.ry(angle, 0, f"Prepare |ψ⟩ = Ry({angle:.3f})|0⟩ on q0")

        circ.barrier(0, 1, 2, label="Create Bell pair")

        # Step 2: Create Bell pair on (q1, q2)
        circ.h(1, "H on q1 — Bell pair creation")
        circ.cnot(1, 2, "CNOT q1→q2 — entangle Bell pair")

        circ.barrier(0, 1, 2, label="Alice's Bell measurement")

        # Step 3: Alice's Bell measurement
        circ.cnot(0, 1, "Alice: CNOT q0→q1")
        circ.h(0, "Alice: H on q0")
        circ.measure(0, 0, "Alice measures q0 → c0")
        circ.measure(1, 1, "Alice measures q1 → c1")

        circ.barrier(0, 1, 2, label="Bob's corrections")

        # Step 4: Bob's classical corrections
        # If c1==1: X on q2
        circ.classical_control(1, 1,
                               GateOp(qubit=2, matrix=G.X, name="X",
                                      label="Bob: X on q2 if c1=1"),
                               label="Bob: if c1==1 apply X to q2")
        # If c0==1: Z on q2
        circ.classical_control(0, 1,
                               GateOp(qubit=2, matrix=G.Z, name="Z",
                                      label="Bob: Z on q2 if c0=1"),
                               label="Bob: if c0==1 apply Z to q2")

        noise_model = NoiseModel.from_config(noise_config) if noise_config else None
        executor = Executor(mode=mode, noise_model=noise_model, optimize=optimize)
        result = executor.run(circ, init_label="Initialize |000⟩")
        return result.to_trace_dict(self.algorithm_id, parameters)
