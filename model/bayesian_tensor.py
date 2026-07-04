import numpy as np
from .base_model import BaseModel

class BayesianTensorDDS(BaseModel):
    def __init__(self, metric, optimizer, params: dict):
        self.metric = metric
        self.optimizer = optimizer

        self.N = params.get("n_assets", 58)
        self.L = params.get("lag_depth", 60)
        self.H = params.get("Horizon", 10)
        self.momentum_beta = params.get("momentum_beta", 10)
        self.ridge_penalty = params.get("ridge_penalty", 10)
        
        self.l_rate = 1
        
        self.mu_tensor = np.zeros((self.N, self.N, self.L))
        self.sigma_sq_tensor = np.ones((self.N, self.N, self.L)) 
        self.gradient_momentum_tensor = np.zeros((self.N, self.N, self.L))

    def warmup(self, data: np.ndarray):
        for tau in range(1, self.L + 1):
            y = data[tau:]
            x = data[:-tau]
            for i in range(self.N):
                for j in range(self.N):
                    target_y = y[:, i]
                    predictor_x = x[:, j]
                    
                    sum_x_sq = np.sum(predictor_x ** 2)
                    if sum_x_sq > 0:
                        self.mu_tensor[i, j, tau-1] = np.sum(predictor_x * target_y) / (sum_x_sq + self.ridge_penalty)
                        
                    residuals = target_y - (self.mu_tensor[i, j, tau-1] * predictor_x)
                    self.sigma_sq_tensor[i, j, tau-1] = max(np.mean(residuals ** 2), 1e-5)
                    
        self.mu_tensor = np.clip(self.mu_tensor, -5.0, 5.0)

    def _predict_horizon(self, data: np.ndarray) -> np.ndarray:
        if len(data) < self.L:
            raise ValueError(f"predict_horizon: Input data length {len(data)} is shorter than required lag depth {self.L}")
        
        predictions = np.empty((self.H, self.N))
        current_buffer = np.copy(data[-self.L:])
        sampled_weights = np.random.normal(loc=self.mu_tensor, scale=np.sqrt(self.sigma_sq_tensor))
            
        for h in range(self.H):
            reversed_buffer = current_buffer[::-1]
            weighted_history = sampled_weights * reversed_buffer.T
            step_pred = np.sum(weighted_history, axis=(1, 2))
            
            step_pred = np.nan_to_num(step_pred)
            step_pred[np.abs(step_pred) < 1e-15] = 0.0 
            step_pred = np.clip(step_pred, -10.0, 10.0)
            
            predictions[h] = step_pred
            current_buffer = np.vstack([current_buffer[1:], step_pred])
            
        return np.clip(predictions, a_min=-0.2, a_max=0.2)

    def update(self, data: np.ndarray, diagnostics=None, is_training: bool = True):
        preds = []
        metrics_report = []
        
        for t in range(self.L, len(data)):
            # ---------------------------------------------------------
            # THE FIX: Correctly look backward at the past L steps
            # ---------------------------------------------------------
            history = data[t - self.L : t]
            
            pred = self._predict_horizon(history)
            preds.append(pred)

            r = min(self.H, len(data) - t)
            if r == 0: continue
            
            pred_trunc = pred[:r]
            # ---------------------------------------------------------
            # THE FIX: Correctly look forward at the future r steps
            # ---------------------------------------------------------
            target_trunc = data[t : t + r]

            loss = self.metric.loss(target_trunc, pred_trunc)
            k_trace = 0.0
            if is_training:
                # Pass HISTORY to the optimizer to map past events to future errors
                k_trace = self.optimizer.update(self, history, loss)
                
                self.mu_tensor[np.abs(self.mu_tensor) < 1e-15] = 0.0
                
            if diagnostics is not None:
                mse_error = float(self.metric.metric(target_trunc, pred_trunc))
                mse_var = float(self.metric.var(target_trunc, pred_trunc))
                is_last_step = (t == len(data) - 1)
                
                metrics_report.append({
                    'Mode': 'Train' if is_training else 'Validation', 
                    'MSE_Loss': mse_error,
                    "MSE_var": mse_var,
                    'Kalman_Trace': k_trace,
                    'Spectral_Radius': diagnostics.get_spectral_radius() if is_last_step else 0.0,
                    'Stochastic_Spectral_Radius': diagnostics.get_stochastic_spectral_radius() if is_last_step else 0.0,
                    'BIC': diagnostics.calculate_bic(len(metrics_report) + 1, mse_error) if is_last_step else 0.0
                })
                
        return preds, metrics_report

    def predict(self, data: np.ndarray) -> list:
        preds = []
        for t in range(self.L, len(data) + 1):
            pred = self._predict_horizon(data[t - self.L: t])
            preds.append(pred)
            
        return preds

    def get_weights(self) -> dict:
        return {
            'mu': self.mu_tensor, 
            'sigma': self.sigma_sq_tensor, 
            'mom': self.gradient_momentum_tensor
        }