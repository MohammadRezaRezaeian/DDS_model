import numpy as np
from .base_metric import BaseMetric

class CPFrobeniusLoss(BaseMetric):
    """
    Frobenius Norm Error implementation for CP-Decomposition Models.
    Calculates the instantaneous shock gradient for Online SGD updates.
    """
    
    def metric(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        # The true Frobenius objective: 1/2 * || y_t - y_hat ||_F^2
        if len(actual) == 0: return 0.0
        
        frob_loss = 0.5 * np.sum((actual - predicted) ** 2)
        self._store_vars("metric", frob_loss)
        return float(frob_loss)

    def var(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        if len(actual) == 0: return 0.0
        
        var = np.var(actual - predicted)
        self._store_vars("Var", var)
        return float(var)
    
    def loss(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """
        Returns the raw error vector e_t = y_t - y_hat
        This acts as the direct negative gradient multiplier for the CP-SGD Optimizer.
        """
        if len(actual) == 0:
            return np.array(0.0)
            
        # Average the error across the Horizon block to get a stable 1D shock vector (e_t)
        return np.mean(actual - predicted, axis=0)