#!/usr/bin/env python3
"""
Benchmark suite for quantum circuit simulator scaling.

Measures execution time and peak memory for standard circuits at varying
qubit counts. Compares backends (numpy vs cupy when available).

Usage:
    python benchmarks/bench_scaling.py                    # default suite
    python benchmarks/bench_scaling.py --max-qubits 14    # custom max
    python benchmarks/bench_scaling.py --output results.json  # save JSON
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is on path
sys.path.insert(0, ".")

from simulator.circuit import Circuit
from simulator.executor import Executor, SnapshotConfig


@dataclass
class BenchmarkResult:
    circuit_type: str
    n_qubits: int
    mode: str
    backend: str
    n_gates: int
    time_seconds: float
    peak_memory_mb: float
    ops_per_second: float


def build_random_circuit(n: int, depth: int = 20, seed: int = 42) -> Circuit:
    """Build a random circuit with single and two-qubit gates."""
    rng = np.random.default_rng(seed)
    circ = Circuit(n_qubits=n)
    for _ in range(depth):
        q = int(rng.integers(0, n))
        gate_type = int(rng.integers(0, 3))
        if gate_type == 0:
            circ.h(q)
        elif gate_type == 1:
            circ.rx(rng.uniform(0, 2 * np.pi), q)
        else:
            if n >= 2:
                q2 = int(rng.integers(0, n))
                while q2 == q:
                    q2 = int(rng.integers(0, n))
                circ.cnot(q, q2)
    return circ


def build_qft_circuit(n: int) -> Circuit:
    """Build QFT circuit."""
    circ = Circuit(n_qubits=n)
    for j in range(n):
        circ.h(j)
        for k in range(j + 1, n):
            theta = np.pi / (2 ** (k - j))
            cp = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * theta)],
            ], dtype=np.complex128)
            circ._two(cp, "CP", k, j)
    for q in range(n // 2):
        circ.swap(q, n - 1 - q)
    return circ


def build_grover_circuit(n: int, iterations: int = 1) -> Circuit:
    """Build Grover's algorithm circuit."""
    circ = Circuit(n_qubits=n)
    for q in range(n):
        circ.h(q)
    target = "1" * n
    for _ in range(iterations):
        circ.phase_oracle(target)
        circ.diffusion()
    return circ


def build_ghz_circuit(n: int) -> Circuit:
    """Build GHZ state circuit."""
    circ = Circuit(n_qubits=n)
    circ.h(0)
    for q in range(1, n):
        circ.cnot(0, q)
    return circ


CIRCUIT_BUILDERS = {
    "random": lambda n: build_random_circuit(n, depth=max(20, n * 2)),
    "qft": build_qft_circuit,
    "grover": build_grover_circuit,
    "ghz": build_ghz_circuit,
}


def benchmark_circuit(
    circuit_type: str,
    n_qubits: int,
    mode: str = "statevector",
    backend: str = "numpy",
    optimize: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark."""
    builder = CIRCUIT_BUILDERS[circuit_type]
    circ = builder(n_qubits)
    n_gates = len(circ.ops)

    # Minimal snapshots for benchmarking
    snapshot_cfg = SnapshotConfig(
        include_state_vector=False,
        include_observables=False,
    )

    executor = Executor(
        mode=mode,
        snapshot_config=snapshot_cfg,
        optimize=optimize,
        backend=backend,
    )

    # Warm up
    if n_qubits <= 8:
        executor.run(circ)

    # Measure
    tracemalloc.start()
    t0 = time.perf_counter()
    executor.run(circ)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        circuit_type=circuit_type,
        n_qubits=n_qubits,
        mode=mode,
        backend=backend,
        n_gates=n_gates,
        time_seconds=round(elapsed, 6),
        peak_memory_mb=round(peak / 1024 / 1024, 2),
        ops_per_second=round(n_gates / elapsed, 1) if elapsed > 0 else 0,
    )


def run_suite(
    max_qubits: int = 14,
    modes: List[str] = None,
    circuit_types: List[str] = None,
    backend: str = "numpy",
) -> List[BenchmarkResult]:
    """Run the full benchmark suite."""
    if modes is None:
        modes = ["statevector"]
    if circuit_types is None:
        circuit_types = ["random", "qft", "grover", "ghz"]

    results: List[BenchmarkResult] = []

    # SV mode qubit ranges
    sv_range = list(range(4, min(max_qubits + 1, 19), 2))
    # DM mode is much slower — cap lower
    dm_range = list(range(4, min(max_qubits + 1, 11), 2))

    for ctype in circuit_types:
        for mode in modes:
            qubit_range = dm_range if mode == "density_matrix" else sv_range
            for n in qubit_range:
                print(f"  {ctype:8s} | {mode:15s} | n={n:2d} | {backend} ...", end="", flush=True)
                try:
                    r = benchmark_circuit(ctype, n, mode=mode, backend=backend)
                    results.append(r)
                    print(f"  {r.time_seconds:8.4f}s  {r.peak_memory_mb:8.2f}MB  {r.ops_per_second:10.1f} ops/s")
                except Exception as e:
                    print(f"  FAILED: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark quantum circuit simulator")
    parser.add_argument("--max-qubits", type=int, default=14, help="Maximum qubit count")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy"])
    parser.add_argument("--dm", action="store_true", help="Also benchmark density_matrix mode")
    parser.add_argument("--circuits", nargs="+", default=None,
                        choices=list(CIRCUIT_BUILDERS.keys()),
                        help="Circuit types to benchmark")
    args = parser.parse_args()

    modes = ["statevector"]
    if args.dm:
        modes.append("density_matrix")

    print(f"Quantum Circuit Simulator Benchmark")
    print(f"{'='*70}")
    print(f"Max qubits: {args.max_qubits}, Backend: {args.backend}")
    print(f"Modes: {', '.join(modes)}")
    print(f"{'='*70}")

    results = run_suite(
        max_qubits=args.max_qubits,
        modes=modes,
        circuit_types=args.circuits,
        backend=args.backend,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary: {len(results)} benchmarks completed")
    if results:
        fastest = min(results, key=lambda r: r.time_seconds)
        slowest = max(results, key=lambda r: r.time_seconds)
        print(f"Fastest: {fastest.circuit_type} n={fastest.n_qubits} {fastest.time_seconds:.4f}s")
        print(f"Slowest: {slowest.circuit_type} n={slowest.n_qubits} {slowest.time_seconds:.4f}s")


if __name__ == "__main__":
    main()
