import numpy as np

# Single-qubit gates
I = np.eye(2, dtype=np.complex128)

H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)

T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)

# Two-qubit gates
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=np.complex128)

CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1],
], dtype=np.complex128)

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.complex128)


def Rx(theta: float) -> np.ndarray:
    """Rotation around X axis."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def Ry(theta: float) -> np.ndarray:
    """Rotation around Y axis."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def Rz(theta: float) -> np.ndarray:
    """Rotation around Z axis."""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)],
    ], dtype=np.complex128)


def phase(theta: float) -> np.ndarray:
    """Phase gate P(theta)."""
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)
