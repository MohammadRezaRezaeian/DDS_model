import numpy as np
from .base_optimizer import BaseOptimizer

class CPSGDOptimizer(BaseOptimizer):
    """
    CP-Decomposition Online AdamW Optimizer (Zero-Trap Fixed).
    Escapes the flat-line Mean-Zero trap by utilizing microscopic epsilons 
    and conditional "safe zone" weight decay.
    """
    def __init__(self):
        super().__init__()
        self.is_initialized = False
        
    def update(self, model, history: np.ndarray, metric: np.ndarray) -> float:
        if len(metric) <= 0:
            return 0.0 
            
        X = history[::-1]
        actual_state = history[-1]
        
        # Clean the incoming error gradient
        e_t = np.clip(np.nan_to_num(metric), -5.0, 5.0)
        
        if not self.is_initialized:
            self.R = getattr(model, 'rank', 5) 
            self.eta_asset = getattr(model, 'eta_asset', 0.01) 
            self.eta_temp = getattr(model, 'eta_temp', 0.001)  
            
            # Use a microscopic penalty to prevent crushing weak signals
            self.lambda_reg = getattr(model, 'ridge_penalty', 1e-4)
            
            init_scale = 0.15
            self.A = np.random.randn(model.N, self.R) * init_scale
            self.B = np.random.randn(model.N, self.R) * init_scale
            self.C = np.random.randn(model.L, self.R) * init_scale
            
            self.m_A, self.v_A = np.zeros_like(self.A), np.zeros_like(self.A)
            self.m_B, self.v_B = np.zeros_like(self.B), np.zeros_like(self.B)
            self.m_C, self.v_C = np.zeros_like(self.C), np.zeros_like(self.C)
            
            self.beta1 = 0.9      
            self.beta2 = 0.999    
            
            # ---> THE FIX 1: Microscopic Epsilon <---
            # Standard 1e-8 kills gradients for data scaled at 0.001. 
            # 1e-15 allows Adam to normalize and amplify tiny financial correlations.
            self.eps = 1e-15       
            self.t_step = 0
            
            self.is_initialized = True

        self.t_step += 1

        # -------------------------------------------------------------------
        # CP-DECOMPOSITION: CONTRACTIONS & RAW GRADIENTS
        # -------------------------------------------------------------------
        X_B = X @ self.B 
        V = np.sum(self.C * X_B, axis=0) 
        S = e_t @ self.A
        
        grad_A = -np.outer(e_t, V)
        X_T_C = X.T @ self.C  
        grad_B = -(X_T_C * S)
        grad_C = -(X_B * S)
        
        # -------------------------------------------------------------------
        # ADAM UPDATES (With Safe-Zone Penalty)
        # -------------------------------------------------------------------
        def adam_update(param, grad, m, v, lr, t):
            # ---> THE FIX 2: Dynamic Soft-Decay (The Safe Zone) <---
            # Only penalizes the weight if it grows dangerously large (> 1.0).
            # This allows tiny financial signals to freely form correlations around 0.1 
            # without the constant mathematical gravity pulling them down to exactly 0.0.
            penalty = np.where(np.abs(param) > 1.0, self.lambda_reg, 0.0)
            param *= (1.0 - lr * penalty)
            
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
            
            m_hat = m / (1.0 - self.beta1 ** t)
            v_hat = v / (1.0 - self.beta2 ** t)
            
            # Apply normalized step
            param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)
            return param, m, v

        self.A, self.m_A, self.v_A = adam_update(self.A, grad_A, self.m_A, self.v_A, self.eta_asset, self.t_step)
        self.B, self.m_B, self.v_B = adam_update(self.B, grad_B, self.m_B, self.v_B, self.eta_asset, self.t_step)
        self.C, self.m_C, self.v_C = adam_update(self.C, grad_C, self.m_C, self.v_C, self.eta_temp, self.t_step)
        
        # ---> THE FIX 3: Open the Factor Bottleneck <---
        # Bounding at 0.5 was too restrictive to build a strong tensor. 
        # 2.0 allows the tensor ample room to learn actual price-change magnitude.
        f_bound = 2.0
        self.A = np.clip(self.A, -f_bound, f_bound)
        self.B = np.clip(self.B, -f_bound, f_bound)
        self.C = np.clip(self.C, -f_bound, f_bound)
        
        # Reconstruct tensor and strictly bound the final product
        raw_tensor = np.einsum('ir,jr,tr->ijt', self.A, self.B, self.C)
        model.mu_tensor = np.clip(raw_tensor, -5.0, 5.0)
        
        # -------------------------------------------------------------------
        # KALMAN VARIANCE UPDATES 
        # -------------------------------------------------------------------
        s_max = np.max(np.abs(history), axis=0)
        shock = actual_state / (s_max + 1e-8)
        
        asset_shock = np.abs(shock).reshape(model.N, 1)
        Q = (asset_shock ** 4) * 1e-4
        R = asset_shock + 1e-2
        
        sigma_old = np.mean(model.sigma_sq_tensor, axis=2)
        
        forgetting_factor = 0.50
        sigma_prior = (sigma_old * forgetting_factor) + Q
        
        k_gain = sigma_prior / (sigma_prior + R) 
        new_sigma = np.clip((1.0 - k_gain) * sigma_prior, 1e-8, 0.5)
        model.sigma_sq_tensor = np.repeat(new_sigma[:, :, None], model.L, axis=2)

        adaptation_magnitude = float(np.mean(np.abs(self.eta_asset * self.m_A))) * 100
        return np.clip(adaptation_magnitude, -10.0, 10.0)