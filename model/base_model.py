from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    Abstract Base Class for all time-series structural models.
    Enforces a strict contract for predicting and state management.
    """
    vars = dict()
    def _store_vars(self, varName: str, varValue: np.ndarray):
        self.vars.setdefault(varName, []).append(np.mean(varValue))
        
    @abstractmethod
    def warmup(self, data: np.ndarray):
        """Warm-up layer to instantiate initial OLS weight states."""
        pass

    @abstractmethod
    def update(self, data: np.ndarray) -> list:
        """Returns the internal state/weights of the model for saving."""
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> list:
        """Executes a forward pass to predict the next state."""
        pass
        
    