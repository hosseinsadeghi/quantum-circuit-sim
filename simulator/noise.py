"""
Noise channels as Kraus operators.

Each function returns a list of (2×2) Kraus matrices K_k satisfying
Σ_k K_k† K_k = I.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


def depolarizing_kraus(p: float) -> List[np.ndarray]:
    """
    Depolarizing channel with error probability p.

    ρ → (1 - p)ρ + (p/3)(XρX + YρY + ZρZ)

    Kraus decomposition (4 operators):
      K0 = sqrt(1 - 3p/4) I
      K1 = sqrt(p/4) X
      K2 = sqrt(p/4) Y
      K3 = sqrt(p/4) Z
    """
    if not 0.0 <= p <= 3.0 / 4.0:
        raise ValueError(f"Depolarizing probability p={p} must be in [0, 3/4]")
    a = np.sqrt(1.0 - 3.0 * p / 4.0)
    b = np.sqrt(p / 4.0)
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return [a * I, b * X, b * Y, b * Z]


def amplitude_damping_kraus(gamma: float) -> List[np.ndarray]:
    """
    Amplitude damping (energy relaxation T1).

    Models spontaneous emission: |1⟩ → |0⟩ with probability gamma.

    K0 = [[1, 0], [0, sqrt(1-gamma)]]   (no decay)
    K1 = [[0, sqrt(gamma)], [0, 0]]     (decay)
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"Amplitude damping gamma={gamma} must be in [0, 1]")
    K0 = np.array([[1, 0], [0, np.sqrt(1.0 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
    return [K0, K1]


def phase_damping_kraus(gamma: float) -> List[np.ndarray]:
    """
    Phase damping (dephasing T2 without energy loss).

    K0 = [[1, 0], [0, sqrt(1-gamma)]]
    K1 = [[0, 0], [0, sqrt(gamma)]]
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"Phase damping gamma={gamma} must be in [0, 1]")
    K0 = np.array([[1, 0], [0, np.sqrt(1.0 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=np.complex128)
    return [K0, K1]


def bit_flip_kraus(p: float) -> List[np.ndarray]:
    """Bit-flip channel. K0 = sqrt(1-p) I, K1 = sqrt(p) X."""
    K0 = np.sqrt(1.0 - p) * np.eye(2, dtype=np.complex128)
    K1 = np.sqrt(p) * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    return [K0, K1]


def phase_flip_kraus(p: float) -> List[np.ndarray]:
    """Phase-flip (Z) channel. K0 = sqrt(1-p) I, K1 = sqrt(p) Z."""
    K0 = np.sqrt(1.0 - p) * np.eye(2, dtype=np.complex128)
    K1 = np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return [K0, K1]


# ---------------------------------------------------------------------------
# NoiseModel
# ---------------------------------------------------------------------------

_CHANNEL_BUILDERS = {
    "depolarizing": depolarizing_kraus,
    "amplitude_damping": amplitude_damping_kraus,
    "phase_damping": phase_damping_kraus,
    "bit_flip": bit_flip_kraus,
    "phase_flip": phase_flip_kraus,
}


class NoiseModel:
    """
    Attaches noise channels to gate types or specific qubits.

    Config schema (example)::

        {
          "gate_noise": {
            "default": {"type": "depolarizing", "p": 0.01},
            "Measure": {"type": "amplitude_damping", "gamma": 0.05}
          },
          "qubit_noise": {
            "0": {"type": "phase_damping", "gamma": 0.02}
          }
        }

    ``gate_noise["default"]`` applies after every gate that has no specific entry.
    ``qubit_noise`` applies per-qubit after every gate on that qubit (cumulative with gate_noise).
    """

    def __init__(self):
        self._gate_noise: Dict[str, List[np.ndarray]] = {}
        self._default_noise: Optional[List[np.ndarray]] = None
        self._qubit_noise: Dict[int, List[np.ndarray]] = {}

    @classmethod
    def from_config(cls, config: Dict) -> "NoiseModel":
        nm = cls()
        gate_noise = config.get("gate_noise", {})
        for gate_name, spec in gate_noise.items():
            kraus = _build_kraus(spec)
            if gate_name == "default":
                nm._default_noise = kraus
            else:
                nm._gate_noise[gate_name] = kraus

        for qubit_str, spec in config.get("qubit_noise", {}).items():
            nm._qubit_noise[int(qubit_str)] = _build_kraus(spec)

        return nm

    def kraus_for_gate(self, gate_name: str) -> Optional[List[np.ndarray]]:
        """Return Kraus ops for the given gate name (or default, or None)."""
        return self._gate_noise.get(gate_name, self._default_noise)

    def kraus_for_qubit(self, qubit: int) -> Optional[List[np.ndarray]]:
        return self._qubit_noise.get(qubit)

    def is_noisy(self) -> bool:
        return bool(self._gate_noise or self._default_noise or self._qubit_noise)


def _build_kraus(spec: Dict) -> List[np.ndarray]:
    channel_type = spec["type"]
    if channel_type not in _CHANNEL_BUILDERS:
        raise ValueError(f"Unknown noise channel type: {channel_type!r}")
    builder = _CHANNEL_BUILDERS[channel_type]
    # Pass remaining keys as kwargs
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    return builder(**kwargs)
