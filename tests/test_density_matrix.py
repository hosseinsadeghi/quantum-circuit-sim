"""Tests for simulator/density_matrix.py."""
import numpy as np
import pytest
from simulator.density_matrix import DensityMatrix


def test_initial_state():
    dm = DensityMatrix(2)
    assert dm._rho[0, 0] == pytest.approx(1.0)
    assert np.sum(np.abs(dm._rho)) == pytest.approx(1.0)


def test_from_statevector():
    psi = np.array([1, 0, 0, 0], dtype=np.complex128)
    dm = DensityMatrix.from_statevector(psi)
    assert dm._rho[0, 0] == pytest.approx(1.0)
    assert dm.purity() == pytest.approx(1.0)


def test_h_gate():
    dm = DensityMatrix(1)
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    dm.apply_single_qubit_gate(H, 0)
    probs = dm.probabilities()
    assert probs[0] == pytest.approx(0.5, abs=1e-10)
    assert probs[1] == pytest.approx(0.5, abs=1e-10)


def test_purity_pure_state():
    dm = DensityMatrix(2)
    assert dm.purity() == pytest.approx(1.0, abs=1e-10)


def test_bloch_vector_zero_state():
    dm = DensityMatrix(1)
    x, y, z = dm.bloch_vector(0)
    assert x == pytest.approx(0.0, abs=1e-10)
    assert y == pytest.approx(0.0, abs=1e-10)
    assert z == pytest.approx(1.0, abs=1e-10)


def test_bloch_vector_one_state():
    from simulator import gates as G
    dm = DensityMatrix(1)
    dm.apply_single_qubit_gate(G.X, 0)
    x, y, z = dm.bloch_vector(0)
    assert z == pytest.approx(-1.0, abs=1e-10)


def test_bloch_vector_superposition():
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    dm = DensityMatrix(1)
    dm.apply_single_qubit_gate(H, 0)
    x, y, z = dm.bloch_vector(0)
    assert x == pytest.approx(1.0, abs=1e-8)
    assert abs(y) < 1e-8
    assert abs(z) < 1e-8


def test_z_expectation():
    dm = DensityMatrix(1)
    assert dm.z_expectation(0) == pytest.approx(1.0, abs=1e-10)


def test_entanglement_entropy_product_state():
    dm = DensityMatrix(2)
    # |00⟩ — product state, entropy = 0
    assert dm.entanglement_entropy() == pytest.approx(0.0, abs=1e-10)


def test_entanglement_entropy_bell_state():
    from simulator import gates as G
    dm = DensityMatrix(2)
    dm.apply_single_qubit_gate(G.H, 0)
    dm.apply_two_qubit_gate(G.CNOT, 0, 1)
    # Bell state: entropy = 1 ebit
    entropy = dm.entanglement_entropy()
    assert entropy == pytest.approx(1.0, abs=1e-8)


def test_partial_trace_two_qubits():
    from simulator import gates as G
    dm = DensityMatrix(2)
    dm.apply_single_qubit_gate(G.H, 0)
    dm.apply_two_qubit_gate(G.CNOT, 0, 1)
    # Reduced state of q0 in Bell state is maximally mixed
    rho_0 = dm.partial_trace([0])
    assert rho_0[0, 0] == pytest.approx(0.5, abs=1e-8)
    assert rho_0[1, 1] == pytest.approx(0.5, abs=1e-8)


def test_kraus_depolarizing():
    from simulator.noise import depolarizing_kraus
    dm = DensityMatrix(1)
    kraus = depolarizing_kraus(0.1)
    dm.apply_kraus(kraus, 0)
    # State is slightly mixed; purity < 1
    p = dm.purity()
    assert p < 1.0
    # Trace preserved
    assert np.trace(dm._rho) == pytest.approx(1.0, abs=1e-10)


def test_measurement_collapses():
    dm = DensityMatrix(1)
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    dm.apply_single_qubit_gate(H, 0)
    rng = np.random.default_rng(42)
    outcome = dm.measure_and_collapse(0, rng)
    # After measurement, state is pure in |0⟩ or |1⟩
    assert dm.purity() == pytest.approx(1.0, abs=1e-8)
    probs = dm.probabilities()
    if outcome == 0:
        assert probs[0] == pytest.approx(1.0, abs=1e-8)
    else:
        assert probs[1] == pytest.approx(1.0, abs=1e-8)
