from pydantic import BaseModel, Field
from typing import Any, Dict


class SimulateRequest(BaseModel):
    algorithm_id: str = Field(..., description="Algorithm identifier")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm parameters")
