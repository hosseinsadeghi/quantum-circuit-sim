"""Tests for simulator/circuit.py — IR data classes and Circuit container."""
import numpy as np
import pytest
from simulator.circuit import Circuit, GateOp, TwoQubitGateOp, PhaseOracleOp, DiffusionOp


def test_circuit_basic_ops():
    circ = Circuit(n_qubits=3)
    circ.h(0).cnot(0, 1).rz(0.5, 2)
    assert len(circ.ops) == 3
    assert isinstance(circ.ops[0], GateOp)
    assert isinstance(circ.ops[1], TwoQubitGateOp)
    assert isinstance(circ.ops[2], GateOp)


def test_circuit_phase_oracle():
    circ = Circuit(n_qubits=3)
    circ.phase_oracle("101")
    assert isinstance(circ.ops[0], PhaseOracleOp)
    assert circ.ops[0].target_state == "101"


def test_circuit_diffusion():
    circ = Circuit(n_qubits=3)
    circ.diffusion()
    assert isinstance(circ.ops[0], DiffusionOp)


def test_auto_layout_single_qubit():
    circ = Circuit(n_qubits=2)
    circ.h(0).h(1)
    layout = circ.auto_layout()
    assert layout["qubit_labels"] == ["q0", "q1"]
    # Two ops on different qubits can be packed into the same column
    cols = layout["columns"]
    assert len(cols) >= 1
    # Total gates = 2
    total_gates = sum(len(c["gates"]) for c in cols)
    assert total_gates == 2


def test_auto_layout_sequential():
    circ = Circuit(n_qubits=2)
    circ.h(0).cnot(0, 1).h(1)
    layout = circ.auto_layout()
    cols = layout["columns"]
    # CNOT uses both qubits; H on q1 must come after CNOT
    # So there must be at least 2 distinct columns
    assert len(cols) >= 2


def test_circuit_measure():
    circ = Circuit(n_qubits=2, n_clbits=1)
    circ.h(0).measure(0, 0)
    from simulator.circuit import MeasureOp
    assert isinstance(circ.ops[1], MeasureOp)
    assert circ.ops[1].clbit == 0


def test_gate_op_fields():
    circ = Circuit(n_qubits=2)
    circ.rx(1.23, 0, label="custom label")
    op = circ.ops[0]
    assert op.name == "Rx"
    assert op.qubit == 0
    assert op.label == "custom label"
    assert abs(op.params["theta"] - 1.23) < 1e-9
