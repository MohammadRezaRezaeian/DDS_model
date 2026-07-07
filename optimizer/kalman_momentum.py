import numpy as np
from .base_optimizer import BaseOptimizer

class KalmanMomentum(BaseOptimizer):
    """
    Classic Vectorized Kalman Momentum Engine
    Uses true Kalman Gain to optimally update the tensor weights, 
    combining momentum gradients with uncertainty-bounded steps.
    """
    
    def update(self, model, history: np.ndarray, metric: np.ndarray) -> float:
        if len(metric) <= 0:
            return 0.0 
            
        # Reverse history so index 0 is lag 1 (t-1), index 1 is lag 2 (t-2), etc.
        lag_history = history[::-1]
        
        # 1. Clean the incoming metric to prevent inherited overflows
        surprise = np.clip(np.nan_to_num(metric), -1e5, 1e5) if metric.ndim > 1 else metric
        
        model.s_norm_max = np.linalg.norm(history)

        # 2. Safe Shock & Error Calculation (Element-wise)
        s_max = np.max(np.abs(history), axis=0)
        actual_state = history[-1]
        
        shock = actual_state / (s_max + 1e-6)
        
        max_surprise = np.max(np.abs(surprise))
        error = surprise / (max_surprise + 1e-10)
        
        shock = np.clip(shock, -5.0, 5.0)
        error = np.clip(error, -5.0, 5.0)

        self._store_vars("surprise", surprise)
        self._store_vars("error", error)
        
        # ---------------------------------------------------------
        # 3. VECTORIZED MOMENTUM GRADIENT (Einstein Summation)
        # Calculates the momentum for ALL lags and predictors instantly.
        # ---------------------------------------------------------
        model.momentum_beta_1 = np.abs(shock)
        model.momentum_beta_2 = np.abs(error)
        
        # Reshape betas for 3D broadcasting: (N, 1, 1)
        beta_1 = model.momentum_beta_1[:, None, None]
        beta_2 = model.momentum_beta_2[:, None, None]
        
        raw_grad_tensor = np.einsum('i, lk -> ikl', surprise, lag_history)
        raw_shock_tensor = np.einsum('i, lk -> ikl', actual_state, lag_history)
        
        momentum_tensor = (beta_1 * raw_shock_tensor) + (beta_2 * raw_grad_tensor)
        model.gradient_momentum_tensor = momentum_tensor
        
        # ---------------------------------------------------------
        # 4. CLASSIC KALMAN FILTER UPDATES
        # ---------------------------------------------------------
        
        # A. Process Noise (Q) and Measurement Noise (R)
        # Q prevents the variance from completely collapsing to 0
        Q = 1e-7
        
        # R must be positive. We reshape to (N, 1, 1) so each target asset 
        # uses its own specific shock variance when updating its weights.
        R = np.abs(shock**2).reshape(model.N, 1, 1) + 1e-8
        
        # B. Prior Uncertainty
        sigma_prior = model.sigma_sq_tensor + Q 
        
        # C. Calculate Kalman Gain (K)
        # Shape: (N, N, L) - optimal learning rate per specific weight
        K_gain = sigma_prior / (sigma_prior + R)
        
        # D. CLASSIC STATE UPDATE (Mu)
        # Mu_new = Mu_old + (Kalman_Gain * Innovation_Gradient)
        model.mu_tensor = model.mu_tensor + (K_gain * momentum_tensor)
        model.mu_tensor = np.clip(model.mu_tensor, -5.0, 5.0)
        
        # E. CLASSIC COVARIANCE UPDATE (Sigma)
        # Sigma_new = (1 - K) * Sigma_prior
        model.sigma_sq_tensor = (1.0 - K_gain) * sigma_prior
        model.sigma_sq_tensor = np.clip(model.sigma_sq_tensor, 1e-9, 0.5)

        # Return trace for diagnostics
        k_trace_sum = float(np.sum(K_gain))

        return float(np.clip(k_trace_sum, -10, 10))