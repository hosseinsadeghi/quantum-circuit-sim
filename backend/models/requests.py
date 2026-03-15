from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


class SimulateRequest(BaseModel):
    algorithm_id: str = Field(..., description="Algorithm identifier")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm parameters")
    mode: str = Field(default="statevector", description="Simulation mode: statevector or density_matrix")
    noise_config: Optional[Dict[str, Any]] = Field(default=None, description="Noise model configuration")


class SweepRequest(BaseModel):
    algorithm_id: str
    fixed_parameters: Dict[str, Any] = Field(default_factory=dict)
    sweep_parameter: str
    sweep_values: list
    mode: str = Field(default="statevector")
    noise_config: Optional[Dict[str, Any]] = None
