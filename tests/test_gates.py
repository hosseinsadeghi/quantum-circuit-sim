import numpy as np
import pytest
from simulator import gates


def is_unitary(m: np.ndarray) -> bool:
    return np.allclose(m @ m.conj().T, np.eye(len(m)))


def test_H_unitary():
    assert is_unitary(gates.H)


def test_X_unitary():
    assert is_unitary(gates.X)


def test_Y_unitary():
    assert is_unitary(gates.Y)


def test_Z_unitary():
    assert is_unitary(gates.Z)


def test_S_unitary():
    assert is_unitary(gates.S)


def test_T_unitary():
    assert is_unitary(gates.T)


def test_CNOT_unitary():
    assert is_unitary(gates.CNOT)


def test_CZ_unitary():
    assert is_unitary(gates.CZ)


def test_SWAP_unitary():
    assert is_unitary(gates.SWAP)


def test_Rx_unitary():
    for theta in [0, np.pi / 4, np.pi / 2, np.pi]:
        assert is_unitary(gates.Rx(theta))


def test_Ry_unitary():
    for theta in [0, np.pi / 4, np.pi / 2, np.pi]:
        assert is_unitary(gates.Ry(theta))


def test_Rz_unitary():
    for theta in [0, np.pi / 4, np.pi / 2, np.pi]:
        assert is_unitary(gates.Rz(theta))


def test_H_squared_is_identity():
    assert np.allclose(gates.H @ gates.H, np.eye(2))


def test_X_is_NOT():
    result = gates.X @ np.array([1, 0], dtype=complex)
    assert np.allclose(result, [0, 1])


def test_CNOT_flips_target_when_control_is_1():
    # |11> -> |10>
    state = np.array([0, 0, 0, 1], dtype=complex)  # |11>
    result = gates.CNOT @ state
    assert np.allclose(result, [0, 0, 1, 0])  # |10>
