from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Algorithm(ABC):
    algorithm_id: str
    name: str
    description: str
    parameter_schema: Dict[str, Any]
    category: str = "general"

    @abstractmethod
    def run(self, parameters: Dict[str, Any], mode: str = "statevector",
            noise_config: Optional[Dict[str, Any]] = None,
            optimize: bool = False) -> Dict[str, Any]:
        ...
