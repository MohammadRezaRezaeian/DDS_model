from abc import ABC, abstractmethod
import numpy as np

class BaseOptimizer(ABC):
    """Abstract base class for model weight-updating algorithms."""
    vars = dict()
    def _store_vars(self, varName: str, varValue: np.ndarray):
        self.vars.setdefault(varName, []).append(np.mean(varValue))
    
    @abstractmethod
    def update(self, model, data: np.ndarray) -> float:
        """
        Executes the backward learning pass.
        Modifies the model's internal weights and returns a learning trace/metric.
        """
        pass