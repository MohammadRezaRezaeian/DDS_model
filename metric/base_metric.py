from abc import ABC, abstractmethod
import numpy as np

class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics."""
    vars = dict()
    def _store_vars(self, varName: str, varValue: np.ndarray):
        self.vars.setdefault(varName, []).append(np.mean(varValue))

    @abstractmethod
    def metric(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculates the error or distance between actual and predicted states."""
        pass
    
    @abstractmethod
    def loss(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """Calculates the error or distance between actual and predicted states."""
        pass