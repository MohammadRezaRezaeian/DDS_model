import numpy as np
from .base_optimizer import BaseOptimizer

class KalmanMomentum(BaseOptimizer):
    """
    Horizon-Aware Adaptive Kalman Filter Engine
    Optimizes weights by comparing the mean of N future predictions 
    against the mean of N real future states.
    """
    mu_avg = 0
    
    def update(self, model, history: np.ndarray, metric: np.ndarray) -> float:
        k_trace_sum = 0.0
        
        if len(metric) <= 0:
            return 0.0 
            
        # Reverse history so index 0 is lag 1 (t-1), index 1 is lag 2 (t-2), etc.
        lag_history = history[::-1]
        
        # 1. Clean the incoming metric to prevent inherited overflows
        clean_metric = np.clip(np.nan_to_num(metric), -1e5, 1e5)
        surprise = np.mean(clean_metric, axis=0) if clean_metric.ndim > 1 else clean_metric
        
        model.s_norm_max = np.linalg.norm(history)

        # 2. Safe Shock Calculation (Element-wise)
        # Use absolute maximums and add 1e-8 to physically prevent division by zero
        s_max = np.max(np.abs(history), axis=0)
        actual_state = history[-1]
        
        shock = actual_state / (s_max + 1e-6)
        
        # 3. Safe Error Calculation
        # Add 1e-8 to the denominator. No need for nan_to_num if division by zero is impossible.
        max_surprise = np.max(np.abs(surprise))
        error = surprise / (max_surprise + 1e-8)
        
        # Optional but recommended: Clip the final ratios to prevent runaway momentum
        shock = np.clip(shock, -5.0, 5.0)
        error = np.clip(error, -5.0, 5.0)

        self._store_vars("surprise", surprise)
        self._store_vars("error", error)
        
        for tau in range(1, model.L + 1):
            x_lag = lag_history[tau-1]
            
            current_idx = tau - 1
            start_idx = max(0, current_idx - model.L)
            end_idx = min(model.L, current_idx + model.L + 1)
            
            mu_old = np.mean(model.mu_tensor[:, :, start_idx:end_idx], axis=2)
            sigma_old = np.mean(model.sigma_sq_tensor[:, :, start_idx:end_idx], axis=2)
            
            adaptive_r = np.clip(0.01 + (0.1 * np.exp(-0.5 * shock)), 1e-5, 0.5)
            adaptive_decay = np.clip(1.0 - (0.05 * np.abs(shock)), 0.99, 1.0)
            
            sigma_sq_prior = sigma_old 
            k_gain = sigma_sq_prior / (sigma_sq_prior + adaptive_r)
            
            k_trace_sum += np.mean(k_gain)
            
            # --- CAUSAL MOMENTUM GRADIENT ---
            raw_grad = surprise.reshape(model.N, 1) * x_lag.reshape(1, model.N)
            raw_shock = actual_state.reshape(model.N, 1) * x_lag.reshape(1, model.N)
            model.momentum_beta_1 = np.abs(shock)
            model.momentum_beta_2 = np.abs(error)
            momentum = (model.momentum_beta_1 * raw_shock) + (model.momentum_beta_2 * raw_grad)
            model.gradient_momentum_tensor[:, :, current_idx] = momentum
            
            # --- TENSOR UPDATES ---
            mu_bound = 2/(0.01*(np.mean(self.mu_avg ) - 1)**2 + 1)
            model.mu_tensor[:, :, current_idx] = (self.mu_avg) + (np.clip(momentum, -mu_bound, mu_bound))
            model.mu_tensor[:, :, current_idx] = np.clip(model.mu_tensor[:, :, current_idx], -mu_bound, mu_bound)
            self.mu_avg = (model.L * self.mu_avg + model.mu_tensor[:, :, current_idx] )/(model.L + 1)
            
            model.sigma_sq_tensor[:, :, current_idx] = np.clip((1.0 - k_gain) * sigma_sq_prior, 1e-6, 0.1)


        return float(np.clip(k_trace_sum, -10, 10))