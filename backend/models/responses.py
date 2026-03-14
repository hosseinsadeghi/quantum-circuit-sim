from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class StateVectorData(BaseModel):
    real: List[float]
    imag: List[float]


class SimulationStep(BaseModel):
    step_index: int
    label: str
    gate: Optional[str]
    qubits_affected: List[int]
    state_vector: StateVectorData
    probabilities: List[float]
    basis_labels: List[str]


class MeasurementResult(BaseModel):
    basis_labels: List[str]
    probabilities: List[float]
    most_likely_outcome: str


class CircuitGate(BaseModel):
    qubit: int
    name: str
    step_index: int


class CircuitColumn(BaseModel):
    column_index: int
    gates: List[CircuitGate]


class CircuitLayout(BaseModel):
    qubit_labels: List[str]
    columns: List[CircuitColumn]


class SimulationTrace(BaseModel):
    algorithm: str
    n_qubits: int
    parameters: Dict[str, Any]
    steps: List[SimulationStep]
    measurement: MeasurementResult
    circuit_layout: CircuitLayout


class AlgorithmInfo(BaseModel):
    algorithm_id: str
    name: str
    description: str
    parameter_schema: Dict[str, Any]


class AlgorithmsResponse(BaseModel):
    algorithms: List[AlgorithmInfo]


class HealthResponse(BaseModel):
    status: str
    version: str
