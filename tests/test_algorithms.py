import json
import pytest
from algorithms.bell_state import BellStateAlgorithm
from algorithms.grover import GroverAlgorithm
from algorithms.qaoa_maxcut import QAOAMaxCutAlgorithm


def test_bell_state_runs():
    alg = BellStateAlgorithm()
    result = alg.run({})
    assert result["algorithm"] == "bell_state"
    assert result["n_qubits"] == 2
    assert len(result["steps"]) == 3  # init + H + CNOT


def test_bell_state_final_probabilities():
    alg = BellStateAlgorithm()
    result = alg.run({})
    final_step = result["steps"][-1]
    probs = final_step["probabilities"]
    import numpy as np
    assert np.isclose(probs[0], 0.5)  # |00>
    assert np.isclose(probs[3], 0.5)  # |11>


def test_bell_state_json_serializable():
    alg = BellStateAlgorithm()
    result = alg.run({})
    # Must not raise
    json.dumps(result)


def test_grover_finds_target():
    import numpy as np
    alg = GroverAlgorithm()
    result = alg.run({"n_qubits": 3, "target_state": "101", "num_iterations": 2})
    assert result["algorithm"] == "grover"
    # Target state |101> = index 5 should have highest probability
    final_step = result["steps"][-1]
    probs = final_step["probabilities"]
    assert probs[5] == max(probs), f"Expected |101> (idx 5) to have max probability, got {probs}"


def test_grover_json_serializable():
    alg = GroverAlgorithm()
    result = alg.run({"n_qubits": 2, "target_state": "11", "num_iterations": 1})
    json.dumps(result)


def test_grover_invalid_target_length():
    alg = GroverAlgorithm()
    with pytest.raises(ValueError):
        alg.run({"n_qubits": 3, "target_state": "11", "num_iterations": 1})


def test_qaoa_runs():
    alg = QAOAMaxCutAlgorithm()
    result = alg.run({"n_qubits": 4, "p_layers": 1, "topology": "cycle"})
    assert result["algorithm"] == "qaoa_maxcut"
    assert result["n_qubits"] == 4


def test_qaoa_json_serializable():
    alg = QAOAMaxCutAlgorithm()
    result = alg.run({"n_qubits": 4, "p_layers": 1, "topology": "cycle"})
    json.dumps(result)


def test_qaoa_probabilities_sum_to_one():
    import numpy as np
    alg = QAOAMaxCutAlgorithm()
    result = alg.run({"n_qubits": 4, "p_layers": 1, "topology": "complete"})
    final_probs = result["steps"][-1]["probabilities"]
    assert np.isclose(sum(final_probs), 1.0)
