"""Tests for scaling roadmap features: observables, DM optimization,
circuit optimizer, sparse state, snapshot config, and QAOA scaling."""
import json
import numpy as np
import pytest
from simulator.circuit import Circuit
from simulator.executor import Executor, SnapshotConfig
from simulator.state_vector import StateVector
from simulator.density_matrix import DensityMatrix
from simulator.observables import compute_observables_from_sv, compute_observables_from_dm


# -----------------------------------------------------------------------
# Tier 1.1: Observables computed directly from statevector
# -----------------------------------------------------------------------

class TestObservablesFromSV:
    def test_purity_always_one(self):
        circ = Circuit(n_qubits=2)
        circ.h(0).cnot(0, 1)
        result = Executor(include_observables=True).run(circ)
        for step in result.steps:
            if "observables" in step:
                assert step["observables"]["purity"] == pytest.approx(1.0)

    def test_bloch_matches_dm(self):
        """SV and DM observables should agree for a pure state."""
        circ = Circuit(n_qubits=2)
        circ.h(0).cnot(0, 1)
        sv_result = Executor(mode="statevector").run(circ)
        dm_result = Executor(mode="density_matrix").run(circ)
        sv_obs = sv_result.steps[-1]["observables"]
        dm_obs = dm_result.steps[-1]["observables"]
        for q in range(2):
            for i in range(3):
                assert sv_obs["bloch_vectors"][q][i] == pytest.approx(
                    dm_obs["bloch_vectors"][q][i], abs=1e-8
                )

    def test_entropy_matches_dm(self):
        circ = Circuit(n_qubits=2)
        circ.h(0).cnot(0, 1)
        sv_result = Executor(mode="statevector").run(circ)
        dm_result = Executor(mode="density_matrix").run(circ)
        assert sv_result.steps[-1]["observables"]["entanglement_entropy"] == pytest.approx(
            dm_result.steps[-1]["observables"]["entanglement_entropy"], abs=1e-8
        )

    def test_z_expectations_product_state(self):
        sv = StateVector(2)
        obs = compute_observables_from_sv(sv)
        # |00> state: both qubits z=+1
        assert obs["z_expectations"][0] == pytest.approx(1.0)
        assert obs["z_expectations"][1] == pytest.approx(1.0)


# -----------------------------------------------------------------------
# Tier 1.2: DM tensor contraction matches Kronecker product
# -----------------------------------------------------------------------

class TestDMTensorContraction:
    def test_single_qubit_h(self):
        dm = DensityMatrix(1)
        from simulator import gates as G
        dm.apply_single_qubit_gate(G.H, 0)
        probs = dm.probabilities()
        assert probs[0] == pytest.approx(0.5, abs=1e-10)
        assert probs[1] == pytest.approx(0.5, abs=1e-10)

    def test_two_qubit_cnot(self):
        dm = DensityMatrix(2)
        from simulator import gates as G
        dm.apply_single_qubit_gate(G.H, 0)
        dm.apply_two_qubit_gate(G.CNOT, 0, 1)
        probs = dm.probabilities()
        assert probs[0] == pytest.approx(0.5, abs=1e-8)
        assert probs[3] == pytest.approx(0.5, abs=1e-8)

    def test_three_qubit_ccx(self):
        dm = DensityMatrix(3)
        from simulator import gates as G
        dm.apply_single_qubit_gate(G.X, 0)
        dm.apply_single_qubit_gate(G.X, 1)
        dm.apply_three_qubit_gate(G.CCX, 0, 1, 2)
        probs = dm.probabilities()
        assert probs[7] == pytest.approx(1.0, abs=1e-8)

    def test_sv_dm_agree(self):
        """SV and DM should produce identical probabilities."""
        circ = Circuit(n_qubits=3)
        circ.h(0).cnot(0, 1).h(2).cz(1, 2)
        sv = Executor(mode="statevector").run(circ)
        dm = Executor(mode="density_matrix").run(circ)
        assert np.allclose(sv.final_probabilities(), dm.final_probabilities(), atol=1e-8)


# -----------------------------------------------------------------------
# Tier 1.3: Snapshot config
# -----------------------------------------------------------------------

class TestSnapshotConfig:
    def test_no_state_vector(self):
        cfg = SnapshotConfig(include_state_vector=False)
        circ = Circuit(n_qubits=2)
        circ.h(0).cnot(0, 1)
        result = Executor(snapshot_config=cfg).run(circ)
        for step in result.steps:
            assert step["state_vector"]["real"] == []

    def test_no_observables(self):
        cfg = SnapshotConfig(include_observables=False)
        circ = Circuit(n_qubits=2)
        circ.h(0).cnot(0, 1)
        result = Executor(snapshot_config=cfg).run(circ)
        for step in result.steps:
            assert "observables" not in step

    def test_observable_interval(self):
        cfg = SnapshotConfig(include_observables=True, observable_interval=2)
        circ = Circuit(n_qubits=2)
        circ.h(0).h(1).cnot(0, 1)
        result = Executor(snapshot_config=cfg).run(circ)
        # Step 0 should have observables (always included)
        assert "observables" in result.steps[0]

    def test_checkpoint_interval(self):
        cfg = SnapshotConfig(checkpoint_interval=2, include_state_vector=True)
        circ = Circuit(n_qubits=2)
        circ.h(0).h(1).cnot(0, 1)
        result = Executor(snapshot_config=cfg).run(circ)
        # Step 0 should have state vector (checkpoint)
        assert len(result.steps[0]["state_vector"]["real"]) > 0
        # Step 1 may or may not (depends on step_index % 2)

    def test_top_k_amplitudes(self):
        cfg = SnapshotConfig(top_k_amplitudes=2)
        circ = Circuit(n_qubits=3)
        circ.h(0).cnot(0, 1)  # Bell state on q0,q1
        result = Executor(snapshot_config=cfg).run(circ)
        sv = result.steps[-1]["state_vector"]
        # Should have real/imag lists with mostly zeros
        non_zero = sum(1 for x in sv["real"] if abs(x) > 1e-10)
        # Bell state has 2 non-zero amplitudes
        assert non_zero <= 2


# -----------------------------------------------------------------------
# Tier 2.1: Circuit optimizer
# -----------------------------------------------------------------------

class TestCircuitOptimizer:
    def test_identity_cancellation_hh(self):
        from simulator.circuit_optimizer import IdentityCancellation
        circ = Circuit(n_qubits=1)
        circ.h(0).h(0)
        p = IdentityCancellation()
        ops = p.run(circ.ops, 1)
        assert len(ops) == 0

    def test_identity_cancellation_cnot(self):
        from simulator.circuit_optimizer import IdentityCancellation
        circ = Circuit(n_qubits=2)
        circ.cnot(0, 1).cnot(0, 1)
        p = IdentityCancellation()
        ops = p.run(circ.ops, 2)
        assert len(ops) == 0

    def test_single_qubit_fusion(self):
        from simulator.circuit_optimizer import SingleQubitFusion
        circ = Circuit(n_qubits=1)
        circ.h(0).x(0).h(0)
        p = SingleQubitFusion()
        ops = p.run(circ.ops, 1)
        assert len(ops) == 1  # Fused into one gate
        assert ops[0].name == "Fused"

    def test_optimizer_preserves_semantics(self):
        from simulator.circuit_optimizer import CircuitOptimizer
        circ = Circuit(n_qubits=2)
        circ.h(0).x(0).cnot(0, 1).h(1)
        orig = Executor().run(circ)
        opt = CircuitOptimizer().optimize(circ)
        opt_result = Executor().run(opt)
        assert np.allclose(orig.final_probabilities(), opt_result.final_probabilities(), atol=1e-10)

    def test_optimize_flag(self):
        circ = Circuit(n_qubits=2)
        circ.h(0).h(0)  # Should cancel
        result = Executor(optimize=True).run(circ)
        # With optimization, H·H cancels, so final state is |00>
        assert result.final_probabilities()[0] == pytest.approx(1.0, abs=1e-10)


# -----------------------------------------------------------------------
# Tier 2.2: Dynamic QAOA
# -----------------------------------------------------------------------

class TestDynamicQAOA:
    def test_generate_edges_cycle(self):
        from algorithms.qaoa_maxcut import generate_edges
        edges = generate_edges("cycle", 8)
        assert len(edges) == 8
        assert (7, 0) in edges

    def test_generate_edges_complete(self):
        from algorithms.qaoa_maxcut import generate_edges
        edges = generate_edges("complete", 5)
        assert len(edges) == 10  # C(5,2)

    def test_generate_edges_path(self):
        from algorithms.qaoa_maxcut import generate_edges
        edges = generate_edges("path", 5)
        assert len(edges) == 4

    def test_qaoa_8_qubits(self):
        from algorithms.qaoa_maxcut import QAOAMaxCutAlgorithm
        alg = QAOAMaxCutAlgorithm()
        result = alg.run({"n_qubits": 8, "p_layers": 1, "topology": "path"})
        assert result["n_qubits"] == 8
        assert sum(result["steps"][-1]["probabilities"]) == pytest.approx(1.0)

    def test_ma_qaoa_8_qubits(self):
        from algorithms.ma_qaoa import MAQAOAAlgorithm
        alg = MAQAOAAlgorithm()
        result = alg.run({"n_qubits": 8, "p_layers": 1, "topology": "cycle"})
        assert result["n_qubits"] == 8

    def test_adapt_qaoa_8_qubits(self):
        from algorithms.adapt_qaoa import ADAPTQAOAAlgorithm
        alg = ADAPTQAOAAlgorithm()
        result = alg.run({"n_qubits": 8, "n_adapt_steps": 3, "topology": "cycle"})
        assert result["n_qubits"] == 8


# -----------------------------------------------------------------------
# Tier 2.3: Sparse state vector
# -----------------------------------------------------------------------

class TestSparseStateVector:
    def test_basic_initialization(self):
        from simulator.sparse_state import SparseStateVector
        sv = SparseStateVector(4)
        probs = sv.probabilities()
        assert probs[0] == pytest.approx(1.0)
        assert sum(probs) == pytest.approx(1.0)

    def test_h_gate_sparse(self):
        from simulator.sparse_state import SparseStateVector
        from simulator import gates as G
        sv = SparseStateVector(1)
        sv.apply_single_qubit_gate(G.H, 0)
        probs = sv.probabilities()
        assert probs[0] == pytest.approx(0.5)
        assert probs[1] == pytest.approx(0.5)

    def test_from_array(self):
        from simulator.sparse_state import SparseStateVector
        arr = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        sv = SparseStateVector.from_array(arr)
        probs = sv.probabilities()
        assert probs[0] == pytest.approx(0.5)
        assert probs[3] == pytest.approx(0.5)

    def test_sparse_matches_dense(self):
        from simulator.sparse_state import SparseStateVector
        from simulator import gates as G
        sparse = SparseStateVector(3)
        dense = StateVector(3)
        # Apply same gates
        for q in range(3):
            sparse.apply_single_qubit_gate(G.H, q)
            dense.apply_single_qubit_gate(G.H, q)
        assert np.allclose(sparse.probabilities(), dense.probabilities(), atol=1e-10)

    def test_densify_on_threshold(self):
        from simulator.sparse_state import SparseStateVector
        from simulator import gates as G
        sv = SparseStateVector(3)  # 8 basis states
        sv.apply_single_qubit_gate(G.H, 0)
        sv.apply_single_qubit_gate(G.H, 1)
        sv.apply_single_qubit_gate(G.H, 2)
        # After H on all 3 qubits, all 8 amplitudes are non-zero -> densified
        assert sv.is_dense


# -----------------------------------------------------------------------
# Tier 3.1: Array backend
# -----------------------------------------------------------------------

class TestArrayBackend:
    def test_numpy_backend(self):
        from simulator.array_backend import get_backend
        be = get_backend("numpy")
        a = be.zeros((2, 2))
        assert a.shape == (2, 2)
        assert isinstance(be.to_numpy(a), np.ndarray)

    def test_unknown_backend_raises(self):
        from simulator.array_backend import get_backend
        with pytest.raises(ValueError):
            get_backend("nonexistent")


# -----------------------------------------------------------------------
# Tier 1.4: Raised qubit limits
# -----------------------------------------------------------------------

class TestRaisedLimits:
    def test_ghz_8_qubits(self):
        from algorithms.ghz import GHZAlgorithm
        alg = GHZAlgorithm()
        result = alg.run({"n_qubits": 8})
        assert result["n_qubits"] == 8
        probs = result["steps"][-1]["probabilities"]
        assert probs[0] == pytest.approx(0.5, abs=1e-8)
        assert probs[255] == pytest.approx(0.5, abs=1e-8)

    def test_grover_6_qubits(self):
        from algorithms.grover import GroverAlgorithm
        alg = GroverAlgorithm()
        result = alg.run({"n_qubits": 6, "target_state": "101010", "num_iterations": 3})
        probs = result["steps"][-1]["probabilities"]
        target_idx = int("101010", 2)
        assert probs[target_idx] == max(probs)
