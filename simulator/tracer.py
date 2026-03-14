from typing import List, Optional
from simulator.state_vector import StateVector
import numpy as np


class SimulationStep:
    """Snapshot of quantum state after a gate application."""

    def __init__(
        self,
        step_index: int,
        label: str,
        gate: Optional[str],
        qubits_affected: List[int],
        state_vector: StateVector,
    ):
        self.step_index = step_index
        self.label = label
        self.gate = gate
        self.qubits_affected = qubits_affected
        self.state_real = state_vector.state_real()
        self.state_imag = state_vector.state_imag()
        self.probabilities = state_vector.probabilities_list()
        self.basis_labels = state_vector.basis_labels()

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "label": self.label,
            "gate": self.gate,
            "qubits_affected": self.qubits_affected,
            "state_vector": {
                "real": self.state_real,
                "imag": self.state_imag,
            },
            "probabilities": self.probabilities,
            "basis_labels": self.basis_labels,
        }


class Tracer:
    """Wraps StateVector and records snapshots after every gate."""

    def __init__(self, n_qubits: int):
        self.sv = StateVector(n_qubits)
        self.steps: List[SimulationStep] = []
        self._step_counter = 0

    def snapshot(self, label: str, gate: Optional[str] = None, qubits_affected: Optional[List[int]] = None):
        step = SimulationStep(
            step_index=self._step_counter,
            label=label,
            gate=gate,
            qubits_affected=qubits_affected or [],
            state_vector=self.sv,
        )
        self.steps.append(step)
        self._step_counter += 1

    def apply_single(self, gate_matrix, gate_name: str, qubit: int, label: Optional[str] = None):
        self.sv.apply_single_qubit_gate(gate_matrix, qubit)
        self.snapshot(
            label or f"Apply {gate_name} to q{qubit}",
            gate=gate_name,
            qubits_affected=[qubit],
        )

    def apply_two(self, gate_matrix, gate_name: str, qubit1: int, qubit2: int, label: Optional[str] = None):
        self.sv.apply_two_qubit_gate(gate_matrix, qubit1, qubit2)
        self.snapshot(
            label or f"Apply {gate_name} to q{qubit1}, q{qubit2}",
            gate=gate_name,
            qubits_affected=[qubit1, qubit2],
        )
