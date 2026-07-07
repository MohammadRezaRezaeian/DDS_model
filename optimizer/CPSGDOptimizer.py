import numpy as np
from .base_optimizer import BaseOptimizer

class CPSGDOptimizer(BaseOptimizer):
    """
    CP-Decomposition Online Stochastic Gradient Descent (SGD) Optimizer.
    Acts as a shock-absorber by continuously updating Low-Rank Asset and Temporal factors,
    while maintaining Kalman-style variance tracking for uncertainties.
    """
    def __init__(self):
        super().__init__()
        self.is_initialized = False
        
    def update(self, model, history: np.ndarray, metric: np.ndarray) -> float:
        # 'metric' contains the e_t (surprise vector) from CPFrobeniusLoss
        if len(metric) <= 0:
            return 0.0 
            
        # Reverse history so index 0 is lag 1 (t-1)
        X = history[::-1]
        actual_state = history[-1]
        
        # 1. Clean the incoming gradient to prevent explosive updates
        e_t = np.clip(np.nan_to_num(metric), -5.0, 5.0)
        
        # Initialize Latent Factors dynamically on the first pass
        if not self.is_initialized:
            self.R = getattr(model, 'rank', 5) 
            self.eta_asset = getattr(model, 'eta_asset', 0.05) 
            self.eta_temp = getattr(model, 'eta_temp', 0.005)  
            self.lambda_reg = getattr(model, 'ridge_penalty', 1.0)
            
            self.A = np.random.randn(model.N, self.R) * 0.01
            self.B = np.random.randn(model.N, self.R) * 0.01
            self.C = np.random.randn(model.L, self.R) * 0.01
            
            self.is_initialized = True

        # -------------------------------------------------------------------
        # 2. CP-DECOMPOSITION: FAST CONTRACTIONS & GRADIENTS
        # -------------------------------------------------------------------
        X_B = X @ self.B 
        V = np.sum(self.C * X_B, axis=0) 
        S = e_t @ self.A
        
        # -------------------------------------------------------------------
        # 3. GRADIENT COMPUTATION 
        # -------------------------------------------------------------------
        grad_A = -np.outer(e_t, V) + (self.lambda_reg * self.A)
        X_T_C = X.T @ self.C  
        grad_B = -(X_T_C * S) + (self.lambda_reg * self.B)
        grad_C = -(X_B * S) + (self.lambda_reg * self.C)
        
        # ---> THE FIX 1: STRICT GRADIENT CLIPPING <---
        # Prevents a single massive price shock from rewriting the latent space
        grad_A = np.clip(grad_A, -1.0, 1.0)
        grad_B = np.clip(grad_B, -1.0, 1.0)
        grad_C = np.clip(grad_C, -1.0, 1.0)
        
        # -------------------------------------------------------------------
        # 4. DUAL-RATE SGD UPDATES
        # -------------------------------------------------------------------
        self.A -= self.eta_asset * grad_A
        self.B -= self.eta_asset * grad_B
        self.C -= self.eta_temp * grad_C
        
        # ---> THE FIX 2: STRICT FACTOR BOTTLENECK <---
        # If A, B, and C are bounded to 0.3, the max tensor element is (0.027 * Rank).
        # This mathematically forces the Spectral Radius to stay stable (< 1.0)
        f_bound = 0.3
        self.A = np.clip(self.A, -f_bound, f_bound)
        self.B = np.clip(self.B, -f_bound, f_bound)
        self.C = np.clip(self.C, -f_bound, f_bound)
        
        # Reconstruct the massive dense tensor instantly
        model.mu_tensor = np.einsum('ir,jr,tr->ijt', self.A, self.B, self.C)
        
        # -------------------------------------------------------------------
        # 3. KALMAN VARIANCE UPDATES (The Missing Block)
        # -------------------------------------------------------------------
        # Calculate shock exactly as done in KalmanMomentum
        s_max = np.max(np.abs(history), axis=0)
        shock = actual_state / (s_max + 1e-6)
        
        # CRITICAL FIX: Use np.abs() and reshape to (N,1) to ensure Target 'i' 
        # binds properly to its own shock and prevents negative variances.
        adaptive_r = np.abs(shock**2).reshape(model.N, 1) + 1e-8
        
        # Vectorized variance calculations across all L lags simultaneously
        sigma_old = np.mean(model.sigma_sq_tensor, axis=2)
        
        # Your specific adaptive decay logic for the variance floor
        adaptive_s = np.clip(1e-4 * np.exp(-1e4 * sigma_old), 1e-8, 0.5)
        
        k_gain = sigma_old / (sigma_old + adaptive_r) 
        
        # Update the entire variance tensor across all L dimensions in a single step
        new_sigma = np.clip(adaptive_s * (1.0 - k_gain), 1e-9, 0.1)
        model.sigma_sq_tensor = np.repeat(new_sigma[:, :, None], model.L, axis=2)

        # -------------------------------------------------------------------
        # 4. TRACE REPORTING
        # -------------------------------------------------------------------
        # Return the magnitude of the asset gradient as our "adaptation trace"
        adaptation_magnitude = float(np.mean(np.abs(self.eta_asset * grad_A))) * 100
        
        return np.clip(adaptation_magnitude, -10.0, 10.0)