import numpy as np
from .base_metric import BaseMetric

class MSELoss(BaseMetric):
    """Mean Squared Error implementation."""
    
    # Pre-computed 0-indexed triangular numbers (T_n - 1)
    # This list covers prediction horizons up to 1,275 steps into the future.
    _TRI_INDICES = np.array([
        0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 90, 104, 119, 135, 
        152, 170, 189, 209, 230, 252, 275, 299, 324, 350, 377, 405, 
        434, 464, 495, 527, 560, 594, 629, 665, 702, 740, 779, 819, 
        860, 902, 945, 989, 1034, 1080, 1127, 1175, 1224, 1274
    ])
    
    def metric(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        metric = np.mean((actual - predicted) ** 2)
        self._store_vars("metric", metric)
        return metric

    def var(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        var = np.var(actual - predicted)
        self._store_vars("Var", var)
        return var
    
    def loss(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        # Safety check for empty arrays
        if len(actual) == 0:
            return np.array(0.0)
            
        # 1. Instantly mask the pre-computed list for valid indices
        valid_mask = self._TRI_INDICES < len(actual)
        tri_indices = self._TRI_INDICES[valid_mask]
                
        # Fallback to the first element if the array is somehow shorter than 1
        if len(tri_indices) == 0:
            tri_indices = np.array([0])
            
        # 2. Slice the arrays instantly
        actual_tri = actual[tri_indices]
        predicted_tri = predicted[tri_indices]
        
        # 3. Create the 1/n weights array (1.0, 0.5, 0.333...)
        weights = 1.0 / np.arange(1, len(tri_indices) + 1)
        
        # 4. Dynamically reshape weights for perfect broadcasting 
        shape_expansion = [-1] + [1] * (actual_tri.ndim - 1)
        weights = weights.reshape(shape_expansion)
        
        # 5. Apply the weights to the differences and find the max
        weighted_diff = (actual_tri - predicted_tri) * weights
        
        return np.max(weighted_diff, axis=0)