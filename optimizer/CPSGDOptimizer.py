import numpy as np
from .base_optimizer import BaseOptimizer

class CPSGDOptimizer(BaseOptimizer):
    """
    CP-Decomposition Online AdamW Optimizer.
    Uses Adaptive Moment Estimation to boost tiny structural signals 
    out of massive financial noise, escaping the zero-gradient trap.
    """
    def __init__(self):
        super().__init__()
        self.is_initialized = False
        
    def update(self, model, history: np.ndarray, metric: np.ndarray) -> float:
        if len(metric) <= 0:
            return 0.0 
            
        X = history[::-1]
        actual_state = history[-1]
        
        # 1. Clean the incoming gradient
        e_t = np.clip(np.nan_to_num(metric), -5.0, 5.0)
        
        if not self.is_initialized:
            self.R = getattr(model, 'rank', 5) 
            self.eta_asset = getattr(model, 'eta_asset', 0.01) # Lowered safely for Adam
            self.eta_temp = getattr(model, 'eta_temp', 0.001)  
            self.lambda_reg = getattr(model, 'ridge_penalty', 0.01)
            
            init_scale = 0.15
            self.A = np.random.randn(model.N, self.R) * init_scale
            self.B = np.random.randn(model.N, self.R) * init_scale
            self.C = np.random.randn(model.L, self.R) * init_scale
            
            # --- ADAM TRACKERS INITIALIZATION ---
            self.m_A, self.v_A = np.zeros_like(self.A), np.zeros_like(self.A)
            self.m_B, self.v_B = np.zeros_like(self.B), np.zeros_like(self.B)
            self.m_C, self.v_C = np.zeros_like(self.C), np.zeros_like(self.C)
            
            self.beta1 = 0.9      # Momentum decay
            self.beta2 = 0.999    # Variance decay
            self.eps = 1e-8       # Division by zero safety
            self.t_step = 0
            
            self.is_initialized = True

        self.t_step += 1

        # -------------------------------------------------------------------
        # 2. CP-DECOMPOSITION: CONTRACTIONS & RAW GRADIENTS
        # -------------------------------------------------------------------
        X_B = X @ self.B 
        V = np.sum(self.C * X_B, axis=0) 
        S = e_t @ self.A
        
        grad_A = -np.outer(e_t, V)
        X_T_C = X.T @ self.C  
        grad_B = -(X_T_C * S)
        grad_C = -(X_B * S)
        
        # -------------------------------------------------------------------
        # 3. ADAMW UPDATES (Adaptive Learning + Decoupled Weight Decay)
        # -------------------------------------------------------------------
        def adam_update(param, grad, m, v, lr, t):
            # 1. AdamW Decoupled Weight Decay (Safe friction)
            param *= (1.0 - lr * self.lambda_reg)
            
            # 2. Update biased first moment estimate (Momentum)
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            
            # 3. Update biased second raw moment estimate (RMSprop/Variance)
            v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
            
            # 4. Compute bias-corrected estimates
            m_hat = m / (1.0 - self.beta1 ** t)
            v_hat = v / (1.0 - self.beta2 ** t)
            
            # 5. Apply the normalized update
            param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)
            return param, m, v

        self.A, self.m_A, self.v_A = adam_update(self.A, grad_A, self.m_A, self.v_A, self.eta_asset, self.t_step)
        self.B, self.m_B, self.v_B = adam_update(self.B, grad_B, self.m_B, self.v_B, self.eta_asset, self.t_step)
        self.C, self.m_C, self.v_C = adam_update(self.C, grad_C, self.m_C, self.v_C, self.eta_temp, self.t_step)
        
        # Factor Bottleneck (Relaxed to 0.5 because Adam is highly stable)
        f_bound = 0.5
        self.A = np.clip(self.A, -f_bound, f_bound)
        self.B = np.clip(self.B, -f_bound, f_bound)
        self.C = np.clip(self.C, -f_bound, f_bound)
        
        # Reconstruct the massive dense tensor instantly
        model.mu_tensor = np.einsum('ir,jr,tr->ijt', self.A, self.B, self.C)
        
        # -------------------------------------------------------------------
        # 3. KALMAN VARIANCE UPDATES (Fading Memory Filter)
        # -------------------------------------------------------------------
        s_max = np.max(np.abs(history), axis=0)
        shock = actual_state / (s_max + 1e-6)
        
        # Base shock magnitude per target asset
        asset_shock = np.abs(shock).reshape(model.N, 1)
        
        # 1. VIOLENT PROCESS NOISE (Q): Quartic Activation
        # Raising to the 4th power ensures normal market noise (~0.1) is crushed to near zero,
        # but genuine spikes (e.g., 2.0+) explode instantly, injecting massive uncertainty.
        Q = (asset_shock ** 4) * 1e-4
        
        # 2. MEASUREMENT NOISE (R)
        R = asset_shock + 1e-2
        
        # Vectorized variance calculations across all L lags
        sigma_old = np.mean(model.sigma_sq_tensor, axis=2)
        
        # 3. FADING MEMORY PRIOR: The Forced Collapse
        # We multiply sigma_old by 0.90 to artificially drain 10% of the variance every step.
        # This guarantees it rapidly collapses to baseline during quiet periods.
        forgetting_factor = 0.90
        sigma_prior = (sigma_old * forgetting_factor) + Q
        
        # Kalman Gain
        k_gain = sigma_prior / (sigma_prior + R) 
        
        # 4. POSTERIOR VARIANCE
        new_sigma = np.clip((1.0 - k_gain) * sigma_prior, 1e-8, 0.5)
        
        # Update the entire variance tensor across all L dimensions in a single step
        model.sigma_sq_tensor = np.repeat(new_sigma[:, :, None], model.L, axis=2)

        # Return the actual applied magnitude from Adam's momentum tracker
        adaptation_magnitude = float(np.mean(np.abs(self.eta_asset * self.m_A))) * 100
        return np.clip(adaptation_magnitude, -10.0, 10.0)