from abc import ABC, abstractmethod
from typing import Any, Dict


class Algorithm(ABC):
    @property
    @abstractmethod
    def algorithm_id(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameter_schema(self) -> Dict[str, Any]: ...

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]: ...
