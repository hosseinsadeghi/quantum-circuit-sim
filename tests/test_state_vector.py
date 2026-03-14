import numpy as np
import pytest
from simulator.state_vector import StateVector
from simulator import gates


def test_initial_state_is_zero():
    sv = StateVector(2)
    assert np.allclose(sv._state, [1, 0, 0, 0])


def test_probabilities_sum_to_one():
    sv = StateVector(3)
    sv.apply_single_qubit_gate(gates.H, 0)
    assert np.isclose(sum(sv.probabilities()), 1.0)


def test_H_creates_superposition():
    sv = StateVector(1)
    sv.apply_single_qubit_gate(gates.H, 0)
    probs = sv.probabilities()
    assert np.isclose(probs[0], 0.5)
    assert np.isclose(probs[1], 0.5)


def test_X_flips_qubit():
    sv = StateVector(1)
    sv.apply_single_qubit_gate(gates.X, 0)
    assert np.isclose(sv.probabilities()[1], 1.0)


def test_bell_state_amplitudes():
    sv = StateVector(2)
    sv.apply_single_qubit_gate(gates.H, 0)
    sv.apply_two_qubit_gate(gates.CNOT, 0, 1)
    probs = sv.probabilities()
    assert np.isclose(probs[0], 0.5)  # |00>
    assert np.isclose(probs[1], 0.0)  # |01>
    assert np.isclose(probs[2], 0.0)  # |10>
    assert np.isclose(probs[3], 0.5)  # |11>


def test_basis_labels():
    sv = StateVector(2)
    labels = sv.basis_labels()
    assert labels == ["|00>", "|01>", "|10>", "|11>"]


def test_state_vector_serialization():
    sv = StateVector(2)
    sv.apply_single_qubit_gate(gates.H, 0)
    real = sv.state_real()
    imag = sv.state_imag()
    assert all(isinstance(v, float) for v in real)
    assert all(isinstance(v, float) for v in imag)
