"""Tests for simulator/executor.py."""
import json
import pytest
from simulator.circuit import Circuit
from simulator.executor import Executor


def test_bell_state_sv():
    circ = Circuit(n_qubits=2)
    circ.h(0).cnot(0, 1)
    result = Executor(mode="statevector").run(circ)

    # Step 0 = init, step 1 = H, step 2 = CNOT
    assert len(result.steps) == 3
    probs = result.final_probabilities()
    assert probs[0] == pytest.approx(0.5, abs=1e-10)
    assert probs[3] == pytest.approx(0.5, abs=1e-10)
    assert probs[1] == pytest.approx(0.0, abs=1e-10)


def test_bell_state_dm():
    circ = Circuit(n_qubits=2)
    circ.h(0).cnot(0, 1)
    result = Executor(mode="density_matrix").run(circ)

    probs = result.final_probabilities()
    assert probs[0] == pytest.approx(0.5, abs=1e-8)
    assert probs[3] == pytest.approx(0.5, abs=1e-8)


def test_observables_present():
    circ = Circuit(n_qubits=2)
    circ.h(0)
    result = Executor(include_observables=True).run(circ)
    for step in result.steps:
        assert "observables" in step
        obs = step["observables"]
        assert "bloch_vectors" in obs
        assert "entanglement_entropy" in obs
        assert "purity" in obs


def test_grover_phase_oracle():
    circ = Circuit(n_qubits=2)
    circ.h(0).h(1)
    circ.phase_oracle("11")
    circ.diffusion()
    result = Executor().run(circ)
    probs = result.final_probabilities()
    # |11⟩ should have highest probability after 1 Grover iteration on 2 qubits
    assert probs[3] == pytest.approx(1.0, abs=1e-10)


def test_circuit_layout_generated():
    circ = Circuit(n_qubits=2)
    circ.h(0).cnot(0, 1)
    result = Executor().run(circ)
    layout = result.circuit_layout
    assert "qubit_labels" in layout
    assert layout["qubit_labels"] == ["q0", "q1"]
    assert len(layout["columns"]) >= 1


def test_to_trace_dict_serializable():
    circ = Circuit(n_qubits=2)
    circ.h(0).cnot(0, 1)
    result = Executor().run(circ)
    d = result.to_trace_dict("bell_state", {})
    json.dumps(d)  # must not raise


def test_noise_forces_dm_mode():
    from simulator.noise import NoiseModel
    nm = NoiseModel.from_config({"gate_noise": {"default": {"type": "depolarizing", "p": 0.01}}})
    executor = Executor(mode="statevector", noise_model=nm)
    assert executor.mode == "density_matrix"


def test_noisy_bell_state():
    from simulator.noise import NoiseModel
    nm = NoiseModel.from_config({"gate_noise": {"default": {"type": "depolarizing", "p": 0.05}}})
    circ = Circuit(n_qubits=2)
    circ.h(0).cnot(0, 1)
    result = Executor(noise_model=nm).run(circ)
    probs = result.final_probabilities()
    # Probabilities should still be near 0.5 each for |00⟩ and |11⟩
    assert probs[0] > 0.4
    assert probs[3] > 0.4
    # Sum to 1
    assert sum(probs) == pytest.approx(1.0, abs=1e-8)


def test_classical_control():
    from simulator.circuit import GateOp
    from simulator import gates as G
    circ = Circuit(n_qubits=2, n_clbits=1)
    circ.x(0)               # q0 = |1⟩
    circ.measure(0, 0)      # c0 = 1
    # Apply X to q1 only if c0 == 1 → should always trigger
    circ.classical_control(0, 1,
                           GateOp(qubit=1, matrix=G.X, name="X", label="X on q1"),
                           label="if c0==1: X on q1")
    result = Executor(rng_seed=0).run(circ)
    probs = result.final_probabilities()
    # q0=|1⟩ (collapsed), q1=|1⟩ → state |11⟩
    assert probs[3] == pytest.approx(1.0, abs=1e-8)
