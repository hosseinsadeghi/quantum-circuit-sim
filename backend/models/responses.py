from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class StateVectorData(BaseModel):
    real: List[float]
    imag: List[float]


class ObservablesData(BaseModel):
    bloch_vectors: List[List[float]]   # [[x,y,z], ...] per qubit
    z_expectations: List[float]
    entanglement_entropy: float
    purity: float


class SimulationStep(BaseModel):
    step_index: int
    label: str
    gate: Optional[str]
    qubits_affected: List[int]
    state_vector: StateVectorData
    probabilities: List[float]
    basis_labels: List[str]
    observables: Optional[ObservablesData] = None


class MeasurementResult(BaseModel):
    basis_labels: List[str]
    probabilities: List[float]
    most_likely_outcome: str


class CircuitGate(BaseModel):
    qubit: int
    name: str
    step_index: int
    classical_control: Optional[Dict[str, Any]] = None


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
    category: str = "general"


class AlgorithmsResponse(BaseModel):
    algorithms: List[AlgorithmInfo]


class HealthResponse(BaseModel):
    status: str
    version: str


# Sweep response
class SweepPoint(BaseModel):
    parameter_value: Any
    most_likely_outcome: str
    probabilities: List[float]
    basis_labels: List[str]


class SweepResponse(BaseModel):
    algorithm: str
    sweep_parameter: str
    points: List[SweepPoint]


class JobSubmittedResponse(BaseModel):
    job_id: str
    status: str = "pending"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    result: Optional[SimulationTrace] = None
    error: Optional[str] = None
