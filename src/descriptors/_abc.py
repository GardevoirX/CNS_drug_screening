from abc import ABC, abstractmethod
from typing import Union

class DescriptorsABC(ABC):
    @abstractmethod
    def __call__(self, SMILES: str) -> list[float]:
        pass