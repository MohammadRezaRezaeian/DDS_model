import numpy as np
from .base_metric import BaseMetric

class MSELoss(BaseMetric):
    """Mean Squared Error implementation."""
    
    def metric(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        metric = np.mean((actual - predicted) ** 2)
        self._store_vars("metric", metric)
        return metric

    def var(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        var = np.var(actual - predicted)
        self._store_vars("Var", var)
        return var
    
    def loss(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        return actual - predicted