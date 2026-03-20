"""
Array backend abstraction — decouple simulation code from NumPy.

Provides a unified interface for array operations so that GPU backends
(CuPy, etc.) can be swapped in without changing simulation logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np


class ArrayBackend(ABC):
    """Abstract interface for array operations used by the simulator."""

    @abstractmethod
    def zeros(self, shape, dtype=np.complex128):
        ...

    @abstractmethod
    def eye(self, n, dtype=np.complex128):
        ...

    @abstractmethod
    def array(self, data, dtype=np.complex128):
        ...

    @abstractmethod
    def tensordot(self, a, b, axes):
        ...

    @abstractmethod
    def moveaxis(self, a, source, destination):
        ...

    @abstractmethod
    def transpose(self, a, axes=None):
        ...

    @abstractmethod
    def outer(self, a, b):
        ...

    @abstractmethod
    def conj(self, a):
        ...

    @abstractmethod
    def abs(self, a):
        ...

    @abstractmethod
    def sum(self, a, axis=None):
        ...

    @abstractmethod
    def svd(self, a, compute_uv=True):
        ...

    @abstractmethod
    def trace(self, a, axis1=0, axis2=1):
        ...

    @abstractmethod
    def diag(self, a):
        ...

    @abstractmethod
    def reshape(self, a, shape):
        ...

    @abstractmethod
    def real(self, a):
        ...

    @abstractmethod
    def to_numpy(self, a) -> np.ndarray:
        """Convert array to numpy (no-op for NumPy backend, device→host for GPU)."""
        ...

    @abstractmethod
    def from_numpy(self, a: np.ndarray):
        """Convert numpy array to backend array."""
        ...

    @abstractmethod
    def matmul(self, a, b):
        ...

    @abstractmethod
    def norm(self, a):
        ...

    @abstractmethod
    def argmax(self, a):
        ...

    @abstractmethod
    def log2(self, a):
        ...

    @abstractmethod
    def eigvalsh(self, a):
        ...


class NumpyBackend(ArrayBackend):
    """NumPy-based array backend (default, CPU)."""

    def zeros(self, shape, dtype=np.complex128):
        return np.zeros(shape, dtype=dtype)

    def eye(self, n, dtype=np.complex128):
        return np.eye(n, dtype=dtype)

    def array(self, data, dtype=np.complex128):
        return np.array(data, dtype=dtype)

    def tensordot(self, a, b, axes):
        return np.tensordot(a, b, axes=axes)

    def moveaxis(self, a, source, destination):
        return np.moveaxis(a, source, destination)

    def transpose(self, a, axes=None):
        return np.transpose(a, axes)

    def outer(self, a, b):
        return np.outer(a, b)

    def conj(self, a):
        return np.conj(a)

    def abs(self, a):
        return np.abs(a)

    def sum(self, a, axis=None):
        return np.sum(a, axis=axis)

    def svd(self, a, compute_uv=True):
        return np.linalg.svd(a, compute_uv=compute_uv)

    def trace(self, a, axis1=0, axis2=1):
        return np.trace(a, axis1=axis1, axis2=axis2)

    def diag(self, a):
        return np.diag(a)

    def reshape(self, a, shape):
        return np.reshape(a, shape)

    def real(self, a):
        return np.real(a)

    def to_numpy(self, a) -> np.ndarray:
        return np.asarray(a)

    def from_numpy(self, a: np.ndarray):
        return a

    def matmul(self, a, b):
        return a @ b

    def norm(self, a):
        return np.linalg.norm(a)

    def argmax(self, a):
        return np.argmax(a)

    def log2(self, a):
        return np.log2(a)

    def eigvalsh(self, a):
        return np.linalg.eigvalsh(a)


class CupyBackend(ArrayBackend):
    """CuPy GPU backend. Requires cupy to be installed.

    Gate matrices (2x2, 4x4) are cached on GPU after first upload.
    """

    def __init__(self):
        try:
            import cupy as cp
            self.cp = cp
        except ImportError:
            raise ImportError(
                "CuPy is required for GPU backend. Install with: "
                "pip install cupy-cuda12x"
            )
        self._gate_cache = {}

    def _cache_gate(self, gate: np.ndarray):
        """Upload a small gate matrix to GPU and cache it."""
        key = id(gate)
        if key not in self._gate_cache:
            self._gate_cache[key] = self.cp.asarray(gate)
        return self._gate_cache[key]

    def zeros(self, shape, dtype=np.complex128):
        return self.cp.zeros(shape, dtype=dtype)

    def eye(self, n, dtype=np.complex128):
        return self.cp.eye(n, dtype=dtype)

    def array(self, data, dtype=np.complex128):
        return self.cp.array(data, dtype=dtype)

    def tensordot(self, a, b, axes):
        return self.cp.tensordot(a, b, axes=axes)

    def moveaxis(self, a, source, destination):
        return self.cp.moveaxis(a, source, destination)

    def transpose(self, a, axes=None):
        return self.cp.transpose(a, axes)

    def outer(self, a, b):
        return self.cp.outer(a, b)

    def conj(self, a):
        return self.cp.conj(a)

    def abs(self, a):
        return self.cp.abs(a)

    def sum(self, a, axis=None):
        return self.cp.sum(a, axis=axis)

    def svd(self, a, compute_uv=True):
        return self.cp.linalg.svd(a, compute_uv=compute_uv)

    def trace(self, a, axis1=0, axis2=1):
        return self.cp.trace(a, axis1=axis1, axis2=axis2)

    def diag(self, a):
        return self.cp.diag(a)

    def reshape(self, a, shape):
        return self.cp.reshape(a, shape)

    def real(self, a):
        return self.cp.real(a)

    def to_numpy(self, a) -> np.ndarray:
        return self.cp.asnumpy(a)

    def from_numpy(self, a: np.ndarray):
        return self.cp.asarray(a)

    def matmul(self, a, b):
        return a @ b

    def norm(self, a):
        return self.cp.linalg.norm(a)

    def argmax(self, a):
        return self.cp.argmax(a)

    def log2(self, a):
        return self.cp.log2(a)

    def eigvalsh(self, a):
        return self.cp.linalg.eigvalsh(a)


# Singleton backends
_numpy_backend = None
_cupy_backend = None


def get_backend(name: str = "numpy") -> ArrayBackend:
    """Get a backend instance by name."""
    global _numpy_backend, _cupy_backend

    if name == "numpy":
        if _numpy_backend is None:
            _numpy_backend = NumpyBackend()
        return _numpy_backend
    elif name == "cupy":
        if _cupy_backend is None:
            _cupy_backend = CupyBackend()
        return _cupy_backend
    else:
        raise ValueError(f"Unknown backend: {name!r}. Available: 'numpy', 'cupy'")
